//! First belief-state root planner.
//!
//! This module is the first event-driven hidden-information planner. It keeps
//! the posterior exact and uniform: non-reveal moves use deterministic belief
//! transitions, reveal moves expand the exact reveal frontier and sample one
//! equal-probability observation per simulation, and leaf values come from the
//! deterministic open-card solver on uniformly sampled full worlds.

use std::time::Instant;

use serde::{Deserialize, Serialize};

mod parallel;
mod support;
#[cfg(test)]
mod tests;

use parallel::recommend_move_belief_uct_root_parallel;
use support::{PlannerRng, StableHash};

use crate::{
    belief::{apply_belief_transition, BeliefTransition, PreparedWorldSampler, RevealFrontier},
    closure::ClosureEngine,
    config::{DeterministicSolverConfig, SolverConfig},
    core::BeliefState,
    deterministic_solver::{
        ordered_macro_moves, DeterministicSearchConfig, DeterministicSearchStats,
        DeterministicSolver, DeterministicTtConfig, EvaluatorWeights, SolveBudget, SolveOutcome,
    },
    error::{SolverError, SolverResult},
    late_exact::{LateExactEvaluationMode, LateExactEvaluator, LateExactResult},
    moves::MacroMove,
    types::{DealSeed, SessionId},
};

/// Leaf continuation mode used by the belief planner.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PlannerLeafEvalMode {
    /// Use the deterministic fast evaluator on sampled worlds.
    Fast,
    /// Use bounded deterministic search on sampled worlds.
    Bounded,
    /// Use proof-oriented deterministic search on sampled worlds.
    Exact,
}

/// Explicit node budget for one root planner call.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PlannerNodeBudget {
    /// Number of root simulations to run.
    pub simulation_budget: usize,
    /// Maximum belief-action depth before leaf evaluation.
    pub max_depth: u8,
}

impl Default for PlannerNodeBudget {
    fn default() -> Self {
        Self {
            simulation_budget: 256,
            max_depth: 8,
        }
    }
}

/// Configuration for the first root-focused belief-state UCT planner.
#[derive(Debug, Copy, Clone, PartialEq, Serialize, Deserialize)]
pub struct BeliefPlannerConfig {
    /// Root simulations to run for non-trivial decisions.
    pub simulation_budget: usize,
    /// Maximum belief-action depth before leaf evaluation.
    pub max_depth: u8,
    /// UCB exploration constant.
    pub exploration_constant: f64,
    /// Number of uniform full worlds sampled for each leaf evaluation.
    pub leaf_world_samples: usize,
    /// Deterministic continuation mode used at leaves.
    pub leaf_eval_mode: PlannerLeafEvalMode,
    /// Reproducible RNG seed for reveal sampling and leaf world sampling.
    pub rng_seed: DealSeed,
    /// Enables confidence-style root early stopping.
    pub enable_early_stop: bool,
    /// Minimum root simulations before early stopping may trigger.
    pub min_simulations_before_stop: usize,
    /// Z multiplier used for root confidence intervals.
    pub confidence_z: f64,
    /// Required CI separation margin between best action and alternatives.
    pub separation_margin: f64,
    /// Simulations to run before conservative root action narrowing.
    pub initial_screen_simulations: usize,
    /// Maximum active root actions after screening; `None` disables cap-based narrowing.
    pub max_active_root_actions: Option<usize>,
    /// Mean-value gap required before an action can be narrowed out.
    pub drop_margin: f64,
    /// Enables a small extra pass for close reveal-causing root contenders.
    pub enable_second_reveal_refinement: bool,
    /// Maximum reveal-causing top root actions eligible for second-reveal refinement.
    pub max_second_reveal_actions: usize,
    /// Root value gap below which second-reveal refinement is considered.
    pub second_reveal_gap_threshold: f64,
    /// Root standard-error threshold above which second-reveal refinement is considered.
    pub second_reveal_uncertainty_threshold: f64,
    /// Additional simulations reserved for selective second-reveal refinement.
    pub second_reveal_refinement_simulations: usize,
    /// Enables microsecond timing breakdowns on planner hot paths.
    pub enable_perf_timing: bool,
    /// Enables independent root workers with end-of-search root-stat aggregation.
    pub enable_root_parallel: bool,
    /// Maximum number of independent root workers.
    pub root_workers: usize,
    /// Optional per-worker simulation budget. If unset, `simulation_budget` is
    /// split across workers as evenly as possible.
    pub worker_simulation_budget: Option<usize>,
    /// Seed stride applied between independent root workers.
    pub worker_seed_stride: u64,
    /// Whether aggregated root statistics refresh confidence bounds after merge.
    pub aggregate_confidence_stats: bool,
}

impl Default for BeliefPlannerConfig {
    fn default() -> Self {
        Self {
            simulation_budget: PlannerNodeBudget::default().simulation_budget,
            max_depth: PlannerNodeBudget::default().max_depth,
            exploration_constant: 1.25,
            leaf_world_samples: 4,
            leaf_eval_mode: PlannerLeafEvalMode::Fast,
            rng_seed: DealSeed(0),
            enable_early_stop: true,
            min_simulations_before_stop: 64,
            confidence_z: 1.96,
            separation_margin: 0.03,
            initial_screen_simulations: 32,
            max_active_root_actions: Some(4),
            drop_margin: 0.20,
            enable_second_reveal_refinement: true,
            max_second_reveal_actions: 2,
            second_reveal_gap_threshold: 0.03,
            second_reveal_uncertainty_threshold: 0.08,
            second_reveal_refinement_simulations: 16,
            enable_perf_timing: false,
            enable_root_parallel: false,
            root_workers: 1,
            worker_simulation_budget: None,
            worker_seed_stride: 1_000_003,
            aggregate_confidence_stats: true,
        }
    }
}

impl BeliefPlannerConfig {
    /// Returns the planner budget as a compact value object.
    pub const fn node_budget(self) -> PlannerNodeBudget {
        PlannerNodeBudget {
            simulation_budget: self.simulation_budget,
            max_depth: self.max_depth,
        }
    }
}

/// Stable structural identity for a belief state.
///
/// This is a lightweight continuation key, not a general belief-state
/// transposition-table entry. It includes the visible state, exact stock/waste
/// state, and unseen tableau card set.
#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct BeliefStateKey(pub u64);

impl BeliefStateKey {
    /// Builds a deterministic key from a belief state.
    pub fn from_belief(belief: &BeliefState) -> Self {
        let mut hash = StableHash::new(0x6275_6374_2d62_656c);
        hash.write_usize(belief.hidden_card_count());

        for top in belief.visible.foundations.top_ranks {
            hash.write_u8(top.map_or(0, |rank| rank.value()));
        }

        for column in &belief.visible.columns {
            hash.write_u8(column.hidden_count);
            hash.write_usize(column.face_up.len());
            for card in &column.face_up {
                hash.write_u8(card.index());
            }
        }

        let stock = &belief.visible.stock;
        hash.write_usize(stock.ring_cards.len());
        hash.write_usize(stock.stock_len);
        hash.write_usize(stock.cursor.unwrap_or(usize::MAX));
        hash.write_u8(stock.accessible_depth);
        hash.write_u32(stock.pass_index);
        hash.write_u32(stock.max_passes.unwrap_or(u32::MAX));
        hash.write_u8(stock.draw_count);
        for card in &stock.ring_cards {
            hash.write_u8(card.index());
        }

        hash.write_usize(belief.unseen_card_count());
        for card in belief.unseen_cards.iter() {
            hash.write_u8(card.index());
        }

        Self(hash.finish())
    }
}

/// Stable fingerprint for the planner/solver controls that affect a cached root.
#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct PlannerConfigFingerprint(pub u64);

impl PlannerConfigFingerprint {
    /// Builds a conservative fingerprint from the public planner inputs.
    pub fn from_configs(
        solver_config: &SolverConfig,
        planner_config: &BeliefPlannerConfig,
        backend_tag: Option<&str>,
        preset_name: Option<&str>,
    ) -> Self {
        let mut hash = StableHash::new(0x7265_7573_652d_6366);
        hash.write_str(backend_tag.unwrap_or(""));
        hash.write_str(preset_name.unwrap_or(""));

        hash.write_u64(solver_config.search.wall_clock_limit_ms);
        hash.write_u64(solver_config.search.node_budget.unwrap_or(u64::MAX));
        hash.write_u64(solver_config.search.rng_seed.unwrap_or(u64::MAX));
        hash.write_usize(solver_config.search.root_workers);

        let closure = solver_config.closure;
        hash.write_u8(closure.max_corridor_steps);
        hash.write_bool(closure.enable_forced_foundation_closure);
        hash.write_bool(closure.enable_single_move_closure);
        hash.write_bool(closure.enable_single_king_placement_closure);
        hash.write_bool(closure.stop_on_reveal);
        hash.write_bool(closure.stop_on_stock_pivot);
        hash.write_bool(closure.debug_validate_each_step);

        let deterministic = &solver_config.deterministic;
        hash.write_u64(u64::from(deterministic.max_macro_depth));
        hash.write_u64(deterministic.exact_node_budget);
        hash.write_u64(deterministic.fast_eval_node_budget);
        hash.write_bool(deterministic.enable_dominance_pruning);
        hash.write_bool(deterministic.enable_foundation_retreats);
        hash.write_bool(deterministic.enable_tt);
        hash.write_usize(deterministic.tt_capacity);
        hash.write_bool(deterministic.tt_store_approx);
        hash.write_u8(match deterministic.leaf_eval_mode {
            crate::ml::LeafEvaluationMode::Heuristic => 0,
            crate::ml::LeafEvaluationMode::VNet => 1,
        });
        hash.write_bool(deterministic.vnet_inference.enable_vnet);
        hash.write_u8(match deterministic.vnet_inference.backend {
            crate::ml::VNetBackend::RustMlpJson => 0,
        });
        hash.write_str(
            deterministic
                .vnet_inference
                .model_path
                .as_ref()
                .and_then(|path| path.to_str())
                .unwrap_or(""),
        );
        hash.write_bool(deterministic.vnet_inference.fallback_to_heuristic);
        hash.write_bool(deterministic.vnet_inference.batch_leaf_eval);

        let late_exact = solver_config.late_exact;
        hash.write_bool(late_exact.enabled);
        hash.write_u8(late_exact.hidden_card_threshold);
        hash.write_usize(late_exact.max_root_actions);
        hash.write_u64(late_exact.assignment_budget.unwrap_or(u64::MAX));
        hash.write_u8(match late_exact.evaluation_mode {
            LateExactEvaluationMode::Exact => 0,
            LateExactEvaluationMode::Bounded => 1,
            LateExactEvaluationMode::Fast => 2,
        });

        hash.write_usize(planner_config.simulation_budget);
        hash.write_u8(planner_config.max_depth);
        hash.write_f64(planner_config.exploration_constant);
        hash.write_usize(planner_config.leaf_world_samples);
        hash.write_u8(match planner_config.leaf_eval_mode {
            PlannerLeafEvalMode::Fast => 0,
            PlannerLeafEvalMode::Bounded => 1,
            PlannerLeafEvalMode::Exact => 2,
        });
        hash.write_u64(planner_config.rng_seed.0);
        hash.write_bool(planner_config.enable_early_stop);
        hash.write_usize(planner_config.min_simulations_before_stop);
        hash.write_f64(planner_config.confidence_z);
        hash.write_f64(planner_config.separation_margin);
        hash.write_usize(planner_config.initial_screen_simulations);
        hash.write_usize(planner_config.max_active_root_actions.unwrap_or(usize::MAX));
        hash.write_f64(planner_config.drop_margin);
        hash.write_bool(planner_config.enable_second_reveal_refinement);
        hash.write_usize(planner_config.max_second_reveal_actions);
        hash.write_f64(planner_config.second_reveal_gap_threshold);
        hash.write_f64(planner_config.second_reveal_uncertainty_threshold);
        hash.write_usize(planner_config.second_reveal_refinement_simulations);
        hash.write_bool(planner_config.enable_perf_timing);
        hash.write_bool(planner_config.enable_root_parallel);
        hash.write_usize(planner_config.root_workers);
        hash.write_usize(
            planner_config
                .worker_simulation_budget
                .unwrap_or(usize::MAX),
        );
        hash.write_u64(planner_config.worker_seed_stride);
        hash.write_bool(planner_config.aggregate_confidence_stats);

        Self(hash.finish())
    }
}

/// Context supplied when asking the planner to continue from a previous root.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct PlannerReuseContext {
    /// Optional session id used to guard session-lineage reuse.
    pub session_id: Option<SessionId>,
    /// Move observed since the previous recommendation, if any.
    pub applied_move: Option<MacroMove>,
    /// Real reveal observed after `applied_move`, if any.
    pub observed_reveal: Option<crate::cards::Card>,
    /// Backend label, such as `belief_uct` or `belief_uct_late_exact`.
    pub backend_tag: Option<String>,
    /// Preset label used to build the configs, if known.
    pub preset_name: Option<String>,
}

/// Reuse mode chosen for a recommendation request.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReuseOutcome {
    /// No continuation was supplied.
    ColdStart,
    /// The current root matched a cached recommendation exactly.
    CurrentRootCache,
    /// The supplied move matched a cached deterministic child.
    FollowedMove,
    /// The supplied move and observed card matched a cached reveal child.
    RevealChild,
    /// The current belief matched a previously cached root by identity.
    HashLookup,
    /// Continuation was present but could not safely apply.
    Fallback,
    /// Continuation belonged to a different planner/solver configuration.
    ConfigMismatch,
}

/// Diagnostics describing how continuation reuse was handled.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReuseDiagnostics {
    /// Whether a previous continuation was inspected.
    pub attempted: bool,
    /// Whether any reusable cache metadata was matched safely.
    pub succeeded: bool,
    /// Reuse mode used for this request.
    pub outcome: ReuseOutcome,
    /// Whether the planner had to run a fresh search after inspecting reuse.
    pub fallback: bool,
    /// Current belief identity.
    pub current_root_key: BeliefStateKey,
    /// Previous root identity, if a continuation was supplied.
    pub previous_root_key: Option<BeliefStateKey>,
    /// Number of cached root actions reused or inspected.
    pub reused_action_count: usize,
    /// Number of cached action-stat records reused or inspected.
    pub reused_stats_count: usize,
    /// Number of cached reveal children matched or available for the path.
    pub reveal_children_reused: usize,
}

impl ReuseDiagnostics {
    fn cold(current_root_key: BeliefStateKey) -> Self {
        Self {
            attempted: false,
            succeeded: false,
            outcome: ReuseOutcome::ColdStart,
            fallback: true,
            current_root_key,
            previous_root_key: None,
            reused_action_count: 0,
            reused_stats_count: 0,
            reveal_children_reused: 0,
        }
    }
}

/// One cached reveal child reached by a root action.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CachedRevealChild {
    /// Card observed at the reveal frontier.
    pub revealed_card: crate::cards::Card,
    /// Belief key after observing this card.
    pub child_key: BeliefStateKey,
}

/// Cached child identities for one root action.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CachedActionChild {
    /// Root action.
    pub action: MacroMove,
    /// Deterministic child key when the action does not reveal.
    pub deterministic_child: Option<BeliefStateKey>,
    /// Exact reveal children when the action reaches a root reveal frontier.
    pub reveal_children: Vec<CachedRevealChild>,
}

/// Bounded cache of one previously searched root recommendation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RootActionCache {
    /// Belief key for the cached root.
    pub root_key: BeliefStateKey,
    /// Config fingerprint used when the cache was built.
    pub config_fingerprint: PlannerConfigFingerprint,
    /// Legal root action list in deterministic generation order.
    pub candidate_actions: Vec<MacroMove>,
    /// Child identities for each cached root action.
    pub action_children: Vec<CachedActionChild>,
    /// Full recommendation produced for this root.
    pub recommendation: PlannerRecommendation,
    /// Optional backend label.
    pub backend_tag: Option<String>,
    /// Optional preset label.
    pub preset_name: Option<String>,
}

/// Cached recommendation payload used by callers that persist only one root.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CachedRecommendation {
    /// Cached root analysis.
    pub root_cache: RootActionCache,
}

/// Lightweight planner continuation carried across turns.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PlannerContinuation {
    /// Optional session id for lineage checks.
    pub session_id: Option<SessionId>,
    /// Belief key for the most recently cached root.
    pub current_root_key: BeliefStateKey,
    /// Config fingerprint for all roots in this continuation.
    pub config_fingerprint: PlannerConfigFingerprint,
    /// Optional backend label.
    pub backend_tag: Option<String>,
    /// Optional preset label.
    pub preset_name: Option<String>,
    /// Most recent root cache.
    pub root_cache: RootActionCache,
    /// Small bounded list of older root caches for deviation/hash lookup.
    pub recent_roots: Vec<RootActionCache>,
    /// Maximum old roots retained.
    pub max_recent_roots: usize,
}

impl PlannerContinuation {
    const DEFAULT_RECENT_ROOT_LIMIT: usize = 8;

    fn from_root_cache(
        session_id: Option<SessionId>,
        root_cache: RootActionCache,
        max_recent_roots: usize,
    ) -> Self {
        Self {
            session_id,
            current_root_key: root_cache.root_key,
            config_fingerprint: root_cache.config_fingerprint,
            backend_tag: root_cache.backend_tag.clone(),
            preset_name: root_cache.preset_name.clone(),
            root_cache,
            recent_roots: Vec::new(),
            max_recent_roots,
        }
    }

    /// Builds a continuation around a freshly produced root recommendation.
    pub fn from_recommendation(
        belief: &BeliefState,
        solver_config: &SolverConfig,
        planner_config: &BeliefPlannerConfig,
        recommendation: PlannerRecommendation,
        context: PlannerReuseContext,
    ) -> SolverResult<Self> {
        let fingerprint = PlannerConfigFingerprint::from_configs(
            solver_config,
            planner_config,
            context.backend_tag.as_deref(),
            context.preset_name.as_deref(),
        );
        let root_cache = build_root_action_cache(
            belief,
            solver_config,
            planner_config,
            fingerprint,
            recommendation,
            context.backend_tag,
            context.preset_name,
        )?;
        Ok(Self::from_root_cache(
            context.session_id,
            root_cache,
            Self::DEFAULT_RECENT_ROOT_LIMIT,
        ))
    }

    fn record_root(&mut self, root_cache: RootActionCache) {
        if self.root_cache.root_key != root_cache.root_key {
            self.recent_roots.insert(0, self.root_cache.clone());
        }
        self.recent_roots
            .retain(|cache| cache.root_key != root_cache.root_key);
        self.recent_roots.truncate(self.max_recent_roots);
        self.current_root_key = root_cache.root_key;
        self.config_fingerprint = root_cache.config_fingerprint;
        self.backend_tag = root_cache.backend_tag.clone();
        self.preset_name = root_cache.preset_name.clone();
        self.root_cache = root_cache;
    }

    fn all_roots(&self) -> impl Iterator<Item = &RootActionCache> {
        std::iter::once(&self.root_cache).chain(self.recent_roots.iter())
    }
}

/// Recommendation plus continuation metadata for the next turn.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ContinuationResult {
    /// Planner recommendation.
    pub recommendation: PlannerRecommendation,
    /// Updated continuation cache to persist in a session.
    pub continuation: PlannerContinuation,
    /// Reuse diagnostics for this call.
    pub reuse: ReuseDiagnostics,
}

/// Running root statistics for one planner action.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PlannerActionStats {
    /// Root action being evaluated.
    pub action: MacroMove,
    /// Number of simulations that selected this action.
    pub visits: usize,
    /// Running mean value in 0..=1.
    pub mean_value: f64,
    /// Running second central moment.
    pub m2: f64,
    /// Sample variance when at least two visits exist.
    pub variance: f64,
    /// Standard error of the mean.
    pub standard_error: f64,
    /// Lower confidence-style bound for root separation checks.
    pub confidence_lower: f64,
    /// Upper confidence-style bound for root separation checks.
    pub confidence_upper: f64,
    /// Simulations that returned a near-certain win value.
    pub win_like_count: usize,
    /// Exact reveal branches expanded while simulating this root action.
    pub reveal_branches_expanded: u64,
    /// Root reveal-frontier child count, when this is a reveal-causing root action.
    pub reveal_frontier_children: usize,
    /// Number of root reveal-frontier children explicitly covered at least once.
    pub reveal_frontier_children_covered: usize,
    /// Whether this action remains active after root screening.
    pub active: bool,
    /// Whether this action was narrowed out after screening.
    pub narrowed_out: bool,
    /// Extra simulations spent during selective second-reveal refinement.
    pub second_reveal_refinement_visits: usize,
    /// Whether late-exact assignment evaluation replaced this action's sampled estimate.
    pub late_exact_evaluated: bool,
    /// Assignments enumerated by late-exact evaluation for this action.
    pub late_exact_assignments_enumerated: u64,
    /// Late-exact expected value for this action, if available.
    pub late_exact_value: Option<f64>,
}

impl PlannerActionStats {
    /// Creates empty stats for one root action.
    pub fn new(action: MacroMove) -> Self {
        Self {
            action,
            visits: 0,
            mean_value: 0.0,
            m2: 0.0,
            variance: 0.0,
            standard_error: 0.0,
            confidence_lower: 0.0,
            confidence_upper: 1.0,
            win_like_count: 0,
            reveal_branches_expanded: 0,
            reveal_frontier_children: 0,
            reveal_frontier_children_covered: 0,
            active: true,
            narrowed_out: false,
            second_reveal_refinement_visits: 0,
            late_exact_evaluated: false,
            late_exact_assignments_enumerated: 0,
            late_exact_value: None,
        }
    }

    fn record(&mut self, value: f32, reveal_branches_expanded: u64) {
        self.visits += 1;
        let value = f64::from(value);
        let delta = value - self.mean_value;
        self.mean_value += delta / self.visits as f64;
        let delta2 = value - self.mean_value;
        self.m2 += delta * delta2;
        self.variance = if self.visits > 1 {
            self.m2 / (self.visits - 1) as f64
        } else {
            0.0
        };
        self.standard_error = if self.visits > 1 {
            (self.variance / self.visits as f64).sqrt()
        } else {
            0.0
        };
        if value >= 0.999 {
            self.win_like_count += 1;
        }
        self.reveal_branches_expanded += reveal_branches_expanded;
    }

    fn refresh_confidence(&mut self, z: f64) {
        let radius = confidence_radius(self, z);
        self.confidence_lower = (self.mean_value - radius).clamp(0.0, 1.0);
        self.confidence_upper = (self.mean_value + radius).clamp(0.0, 1.0);
    }
}

/// Recommendation produced by the belief-state planner.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PlannerRecommendation {
    /// Best legal root move by current root statistics.
    pub best_move: Option<MacroMove>,
    /// Mean value of the selected root move.
    pub best_value: f64,
    /// Per-action root statistics, sorted by value after search.
    pub action_stats: Vec<PlannerActionStats>,
    /// Number of root simulations completed.
    pub simulations_run: usize,
    /// Whether confidence-based early stopping ended the root search.
    pub early_stop_triggered: bool,
    /// Number of actions conservatively narrowed out after screening.
    pub actions_narrowed_out: usize,
    /// Whether selective second-reveal refinement ran.
    pub second_reveal_refinement_ran: bool,
    /// Extra simulations spent on second-reveal refinement.
    pub second_reveal_simulations_run: usize,
    /// Whether the root had no legal moves.
    pub no_legal_moves: bool,
    /// Leaf evaluations performed.
    pub leaf_evaluations: u64,
    /// Deterministic solver nodes used at leaves.
    pub deterministic_nodes: u64,
    /// V-Net inference calls used by deterministic leaf evaluation.
    pub vnet_inferences: u64,
    /// Leaf evaluations that requested V-Net but fell back to the heuristic.
    pub vnet_fallbacks: u64,
    /// Time spent inside V-Net inference, in microseconds.
    pub vnet_inference_elapsed_us: u64,
    /// Closure steps applied while normalizing simulated belief states.
    pub closure_steps_applied: u64,
    /// Exact reveal-frontier branches enumerated during simulations.
    pub reveal_branches_expanded: u64,
    /// Number of root reveal-frontier children explicitly covered at least once.
    pub reveal_frontier_children_covered: u64,
    /// Number of root actions still active after screening/refinement.
    pub active_root_actions: usize,
    /// Whether late-game exact assignment evaluation was used.
    pub late_exact_triggered: bool,
    /// Hidden tableau count when late-exact was considered.
    pub late_exact_hidden_count: usize,
    /// Number of root actions evaluated by late-exact.
    pub late_exact_actions_evaluated: usize,
    /// Hidden assignments enumerated across late-exact action evaluations.
    pub late_exact_assignments_enumerated: u64,
    /// Hidden assignments pruned by conservative late-exact hooks.
    pub late_exact_assignments_pruned: u64,
    /// Deterministic solver nodes consumed by late-exact evaluation.
    pub late_exact_deterministic_nodes: u64,
    /// Late-exact evaluation time in milliseconds.
    pub late_exact_elapsed_ms: u64,
    /// Time spent enumerating late-exact assignments, excluding per-action solver work.
    pub late_exact_assignment_enumeration_elapsed_us: u64,
    /// Time spent in belief transitions during planner simulations, in microseconds.
    pub belief_transition_elapsed_us: u64,
    /// Time spent expanding reveal frontiers during planner simulations, in microseconds.
    pub reveal_expansion_elapsed_us: u64,
    /// Time spent in planner leaf evaluation, including world sampling, in microseconds.
    pub leaf_eval_elapsed_us: u64,
    /// Time spent inside deterministic leaf solver calls, in microseconds.
    pub deterministic_eval_elapsed_us: u64,
    /// Whether the recommendation was produced by independent root-parallel workers.
    pub root_parallel_used: bool,
    /// Number of root workers that contributed to this recommendation.
    pub root_parallel_workers: usize,
    /// Simulations actually completed by each root worker, in worker-index order.
    pub root_parallel_worker_simulations: Vec<usize>,
    /// Elapsed planner wall-clock time in milliseconds.
    pub elapsed_ms: u64,
}

/// Root-focused belief planner.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BeliefPlanner {
    /// Top-level solver configuration.
    pub solver_config: SolverConfig,
    /// Planner controls.
    pub planner_config: BeliefPlannerConfig,
}

impl BeliefPlanner {
    /// Creates a belief planner from explicit configs.
    pub const fn new(solver_config: SolverConfig, planner_config: BeliefPlannerConfig) -> Self {
        Self {
            solver_config,
            planner_config,
        }
    }

    /// Recommends a move from a belief state.
    pub fn recommend(&self, belief: &BeliefState) -> SolverResult<PlannerRecommendation> {
        recommend_move_belief_uct(belief, &self.solver_config, &self.planner_config)
    }

    /// Recommends a move while carrying lightweight root-analysis reuse metadata.
    pub fn recommend_with_reuse(
        &self,
        belief: &BeliefState,
        continuation: Option<&PlannerContinuation>,
        context: PlannerReuseContext,
    ) -> SolverResult<ContinuationResult> {
        recommend_move_belief_uct_with_reuse(
            belief,
            &self.solver_config,
            &self.planner_config,
            continuation,
            context,
        )
    }
}

/// Recommends a root move while reusing bounded continuation metadata when safe.
///
/// This is intentionally lightweight. Exact current-root matches can return the
/// cached recommendation immediately. Followed-move and reveal-child matches
/// preserve and report the cached parent/root metadata, then run a fresh search
/// for the new root unless that new root already exists in the small recent-root
/// cache. Any config or state mismatch falls back to a cold planner call.
pub fn recommend_move_belief_uct_with_reuse(
    belief: &BeliefState,
    solver_config: &SolverConfig,
    planner_config: &BeliefPlannerConfig,
    continuation: Option<&PlannerContinuation>,
    context: PlannerReuseContext,
) -> SolverResult<ContinuationResult> {
    belief.validate_consistency_against_visible()?;

    let current_key = BeliefStateKey::from_belief(belief);
    let fingerprint = PlannerConfigFingerprint::from_configs(
        solver_config,
        planner_config,
        context.backend_tag.as_deref(),
        context.preset_name.as_deref(),
    );

    let mut diagnostics = ReuseDiagnostics::cold(current_key);

    if let Some(continuation) = continuation {
        diagnostics.attempted = true;
        diagnostics.previous_root_key = Some(continuation.current_root_key);

        let session_matches = continuation.session_id.is_none()
            || context.session_id.is_none()
            || continuation.session_id == context.session_id;
        if !session_matches || continuation.config_fingerprint != fingerprint {
            diagnostics.outcome = ReuseOutcome::ConfigMismatch;
        } else if let Some(cache) = find_cached_root(
            continuation,
            current_key,
            fingerprint,
            belief,
            solver_config,
            planner_config,
        )? {
            let mut updated = continuation.clone();
            updated.record_root(cache.clone());
            diagnostics.succeeded = true;
            diagnostics.fallback = false;
            diagnostics.outcome = if cache.root_key == continuation.root_cache.root_key {
                ReuseOutcome::CurrentRootCache
            } else {
                ReuseOutcome::HashLookup
            };
            diagnostics.reused_action_count = cache.candidate_actions.len();
            diagnostics.reused_stats_count = cache.recommendation.action_stats.len();
            diagnostics.reveal_children_reused = cache
                .action_children
                .iter()
                .map(|child| child.reveal_children.len())
                .sum();
            return Ok(ContinuationResult {
                recommendation: cache.recommendation,
                continuation: updated,
                reuse: diagnostics,
            });
        } else if let Some(path) = match_continuation_path(continuation, current_key, &context) {
            diagnostics = path;
        } else {
            diagnostics.outcome = ReuseOutcome::Fallback;
        }
    }

    let recommendation = recommend_move_belief_uct(belief, solver_config, planner_config)?;
    let root_cache = build_root_action_cache(
        belief,
        solver_config,
        planner_config,
        fingerprint,
        recommendation.clone(),
        context.backend_tag.clone(),
        context.preset_name.clone(),
    )?;

    let mut updated = continuation
        .filter(|existing| {
            existing.config_fingerprint == fingerprint
                && (existing.session_id.is_none()
                    || context.session_id.is_none()
                    || existing.session_id == context.session_id)
        })
        .cloned()
        .unwrap_or_else(|| {
            PlannerContinuation::from_root_cache(
                context.session_id,
                root_cache.clone(),
                PlannerContinuation::DEFAULT_RECENT_ROOT_LIMIT,
            )
        });
    updated.session_id = context.session_id.or(updated.session_id);
    updated.record_root(root_cache);

    Ok(ContinuationResult {
        recommendation,
        continuation: updated,
        reuse: diagnostics,
    })
}

/// Recommends a root move with sparse UCT-style belief simulations.
pub fn recommend_move_belief_uct(
    belief: &BeliefState,
    solver_config: &SolverConfig,
    planner_config: &BeliefPlannerConfig,
) -> SolverResult<PlannerRecommendation> {
    if planner_config.enable_root_parallel && planner_config.root_workers.max(1) > 1 {
        recommend_move_belief_uct_parallel(belief, solver_config, planner_config)
    } else {
        recommend_move_belief_uct_single_worker(belief, solver_config, planner_config)
    }
}

/// Recommends a root move by running independent workers from the same belief
/// root and merging only root action statistics.
pub fn recommend_move_belief_uct_parallel(
    belief: &BeliefState,
    solver_config: &SolverConfig,
    planner_config: &BeliefPlannerConfig,
) -> SolverResult<PlannerRecommendation> {
    if planner_config.root_workers.max(1) <= 1 {
        return recommend_move_belief_uct_single_worker(belief, solver_config, planner_config);
    }
    recommend_move_belief_uct_root_parallel(belief, solver_config, planner_config)
}

fn recommend_move_belief_uct_single_worker(
    belief: &BeliefState,
    solver_config: &SolverConfig,
    planner_config: &BeliefPlannerConfig,
) -> SolverResult<PlannerRecommendation> {
    let started = Instant::now();
    belief.validate_consistency_against_visible()?;

    let deterministic_config = deterministic_search_config(solver_config, planner_config);
    let candidates = ordered_macro_moves(&belief.visible, deterministic_config);

    if candidates.is_empty() {
        return Ok(PlannerRecommendation {
            best_move: None,
            best_value: 0.0,
            action_stats: Vec::new(),
            simulations_run: 0,
            early_stop_triggered: false,
            actions_narrowed_out: 0,
            second_reveal_refinement_ran: false,
            second_reveal_simulations_run: 0,
            no_legal_moves: true,
            leaf_evaluations: 0,
            deterministic_nodes: 0,
            vnet_inferences: 0,
            vnet_fallbacks: 0,
            vnet_inference_elapsed_us: 0,
            closure_steps_applied: 0,
            reveal_branches_expanded: 0,
            reveal_frontier_children_covered: 0,
            active_root_actions: 0,
            late_exact_triggered: false,
            late_exact_hidden_count: belief.hidden_card_count(),
            late_exact_actions_evaluated: 0,
            late_exact_assignments_enumerated: 0,
            late_exact_assignments_pruned: 0,
            late_exact_deterministic_nodes: 0,
            late_exact_elapsed_ms: 0,
            late_exact_assignment_enumeration_elapsed_us: 0,
            belief_transition_elapsed_us: 0,
            reveal_expansion_elapsed_us: 0,
            leaf_eval_elapsed_us: 0,
            deterministic_eval_elapsed_us: 0,
            root_parallel_used: false,
            root_parallel_workers: 1,
            root_parallel_worker_simulations: vec![0],
            elapsed_ms: started.elapsed().as_millis() as u64,
        });
    }

    if candidates.len() == 1 {
        return Ok(PlannerRecommendation {
            best_move: candidates.first().cloned(),
            best_value: 1.0,
            action_stats: vec![PlannerActionStats::new(candidates[0].clone())],
            simulations_run: 0,
            early_stop_triggered: false,
            actions_narrowed_out: 0,
            second_reveal_refinement_ran: false,
            second_reveal_simulations_run: 0,
            no_legal_moves: false,
            leaf_evaluations: 0,
            deterministic_nodes: 0,
            vnet_inferences: 0,
            vnet_fallbacks: 0,
            vnet_inference_elapsed_us: 0,
            closure_steps_applied: 0,
            reveal_branches_expanded: 0,
            reveal_frontier_children_covered: 0,
            active_root_actions: 1,
            late_exact_triggered: false,
            late_exact_hidden_count: belief.hidden_card_count(),
            late_exact_actions_evaluated: 0,
            late_exact_assignments_enumerated: 0,
            late_exact_assignments_pruned: 0,
            late_exact_deterministic_nodes: 0,
            late_exact_elapsed_ms: 0,
            late_exact_assignment_enumeration_elapsed_us: 0,
            belief_transition_elapsed_us: 0,
            reveal_expansion_elapsed_us: 0,
            leaf_eval_elapsed_us: 0,
            deterministic_eval_elapsed_us: 0,
            root_parallel_used: false,
            root_parallel_workers: 1,
            root_parallel_worker_simulations: vec![0],
            elapsed_ms: started.elapsed().as_millis() as u64,
        });
    }

    if planner_config.simulation_budget == 0 {
        return Err(SolverError::InvalidState(
            "belief planner simulation_budget must be greater than zero".to_string(),
        ));
    }
    if planner_config.leaf_world_samples == 0 {
        return Err(SolverError::InvalidState(
            "belief planner leaf_world_samples must be greater than zero".to_string(),
        ));
    }

    let solver = DeterministicSolver::new_with_vnet_config(
        deterministic_config,
        &solver_config.deterministic.vnet_inference,
    );
    let mut rng = PlannerRng::new(planner_config.rng_seed.0);
    let mut counters = PlannerCounters {
        timing_enabled: planner_config.enable_perf_timing,
        ..PlannerCounters::default()
    };
    let mut root_actions = candidates
        .into_iter()
        .map(RootActionState::new)
        .collect::<Vec<_>>();
    let mut simulations_run = 0usize;
    let mut actions_narrowed_out = 0usize;
    let mut early_stop_triggered = false;
    let mut screen_applied = false;

    for simulation_index in 0..planner_config.simulation_budget {
        let action_index = select_ucb_action(
            &root_actions,
            simulation_index,
            planner_config.exploration_constant,
        );
        let simulation = simulate_root_action(
            belief,
            &mut root_actions[action_index],
            &mut rng,
            &solver,
            deterministic_config,
            planner_config,
            &mut counters,
        )?;
        root_actions[action_index]
            .stats
            .record(simulation.value, simulation.reveal_branches_expanded);
        root_actions[action_index].refresh_reveal_stats();
        simulations_run += 1;

        if !screen_applied
            && planner_config.initial_screen_simulations > 0
            && simulations_run
                >= planner_config
                    .initial_screen_simulations
                    .max(root_actions.len())
        {
            actions_narrowed_out += narrow_root_actions(&mut root_actions, planner_config);
            screen_applied = true;
        }

        if should_stop_early(&root_actions, simulations_run, planner_config) {
            early_stop_triggered = true;
            break;
        }
    }

    let second_reveal_simulations_run = if !early_stop_triggered {
        run_second_reveal_refinement(
            belief,
            &mut root_actions,
            &mut rng,
            &solver,
            deterministic_config,
            planner_config,
            &mut counters,
        )?
    } else {
        0
    };
    let second_reveal_refinement_ran = second_reveal_simulations_run > 0;
    simulations_run += second_reveal_simulations_run;

    finalize_root_action_states(&mut root_actions, planner_config.confidence_z);
    let late_exact_result = run_late_exact_if_eligible(belief, &mut root_actions, solver_config)?;
    finalize_root_action_states(&mut root_actions, planner_config.confidence_z);

    let reveal_frontier_children_covered = root_actions
        .iter()
        .map(|state| state.stats.reveal_frontier_children_covered as u64)
        .sum::<u64>();
    let active_root_actions = root_actions.iter().filter(|state| state.active).count();
    let mut stats = root_actions
        .into_iter()
        .map(|state| state.stats)
        .collect::<Vec<_>>();

    stats.sort_by(|left, right| {
        right
            .mean_value
            .total_cmp(&left.mean_value)
            .then_with(|| left.action.kind.cmp(&right.action.kind))
            .then_with(|| left.action.id.cmp(&right.action.id))
    });

    let best_move = stats.first().map(|entry| entry.action.clone());
    let best_value = stats
        .first()
        .map(|entry| entry.mean_value)
        .unwrap_or_default();

    Ok(PlannerRecommendation {
        best_move,
        best_value,
        action_stats: stats,
        simulations_run,
        early_stop_triggered,
        actions_narrowed_out,
        second_reveal_refinement_ran,
        second_reveal_simulations_run,
        no_legal_moves: false,
        leaf_evaluations: counters.leaf_evaluations,
        deterministic_nodes: counters.deterministic_nodes,
        vnet_inferences: counters.vnet_inferences,
        vnet_fallbacks: counters.vnet_fallbacks,
        vnet_inference_elapsed_us: counters.vnet_inference_elapsed_us,
        closure_steps_applied: counters.closure_steps_applied,
        reveal_branches_expanded: counters.reveal_branches_expanded,
        reveal_frontier_children_covered,
        active_root_actions,
        late_exact_triggered: late_exact_result.triggered,
        late_exact_hidden_count: late_exact_result.hidden_count,
        late_exact_actions_evaluated: late_exact_result.actions_evaluated,
        late_exact_assignments_enumerated: late_exact_result.assignments_enumerated,
        late_exact_assignments_pruned: late_exact_result.assignments_pruned,
        late_exact_deterministic_nodes: late_exact_result.deterministic_nodes,
        late_exact_elapsed_ms: late_exact_result.elapsed_ms,
        late_exact_assignment_enumeration_elapsed_us: late_exact_result
            .assignment_enumeration_elapsed_us,
        belief_transition_elapsed_us: counters.belief_transition_elapsed_us,
        reveal_expansion_elapsed_us: counters.reveal_expansion_elapsed_us,
        leaf_eval_elapsed_us: counters.leaf_eval_elapsed_us,
        deterministic_eval_elapsed_us: counters.deterministic_eval_elapsed_us,
        root_parallel_used: false,
        root_parallel_workers: 1,
        root_parallel_worker_simulations: vec![simulations_run],
        elapsed_ms: started.elapsed().as_millis() as u64,
    })
}

fn build_root_action_cache(
    belief: &BeliefState,
    solver_config: &SolverConfig,
    planner_config: &BeliefPlannerConfig,
    fingerprint: PlannerConfigFingerprint,
    recommendation: PlannerRecommendation,
    backend_tag: Option<String>,
    preset_name: Option<String>,
) -> SolverResult<RootActionCache> {
    let root_key = BeliefStateKey::from_belief(belief);
    let deterministic_config = deterministic_search_config(solver_config, planner_config);
    let candidate_actions = ordered_macro_moves(&belief.visible, deterministic_config);
    let mut action_children = Vec::with_capacity(candidate_actions.len());

    for action in &candidate_actions {
        let transition = apply_belief_transition(belief, action.atomic)?;
        let child = match transition {
            BeliefTransition::Deterministic { belief, .. } => CachedActionChild {
                action: action.clone(),
                deterministic_child: Some(BeliefStateKey::from_belief(&belief)),
                reveal_children: Vec::new(),
            },
            BeliefTransition::Reveal { frontier } => CachedActionChild {
                action: action.clone(),
                deterministic_child: None,
                reveal_children: frontier
                    .outcomes
                    .iter()
                    .map(|outcome| CachedRevealChild {
                        revealed_card: outcome.revealed_card,
                        child_key: BeliefStateKey::from_belief(&outcome.belief),
                    })
                    .collect(),
            },
        };
        action_children.push(child);
    }

    Ok(RootActionCache {
        root_key,
        config_fingerprint: fingerprint,
        candidate_actions,
        action_children,
        recommendation,
        backend_tag,
        preset_name,
    })
}

fn find_cached_root(
    continuation: &PlannerContinuation,
    current_key: BeliefStateKey,
    fingerprint: PlannerConfigFingerprint,
    belief: &BeliefState,
    solver_config: &SolverConfig,
    planner_config: &BeliefPlannerConfig,
) -> SolverResult<Option<RootActionCache>> {
    for cache in continuation.all_roots() {
        if cache.root_key == current_key
            && cache.config_fingerprint == fingerprint
            && cached_action_surface_matches(cache, belief, solver_config, planner_config)
        {
            return Ok(Some(cache.clone()));
        }
    }
    Ok(None)
}

fn cached_action_surface_matches(
    cache: &RootActionCache,
    belief: &BeliefState,
    solver_config: &SolverConfig,
    planner_config: &BeliefPlannerConfig,
) -> bool {
    let deterministic_config = deterministic_search_config(solver_config, planner_config);
    ordered_macro_moves(&belief.visible, deterministic_config) == cache.candidate_actions
}

fn match_continuation_path(
    continuation: &PlannerContinuation,
    current_key: BeliefStateKey,
    context: &PlannerReuseContext,
) -> Option<ReuseDiagnostics> {
    let mut diagnostics = ReuseDiagnostics {
        attempted: true,
        succeeded: false,
        outcome: ReuseOutcome::Fallback,
        fallback: true,
        current_root_key: current_key,
        previous_root_key: Some(continuation.current_root_key),
        reused_action_count: 0,
        reused_stats_count: 0,
        reveal_children_reused: 0,
    };

    let applied_move = context.applied_move.as_ref()?;
    let child = continuation
        .root_cache
        .action_children
        .iter()
        .find(|child| child.action == *applied_move)?;

    diagnostics.reused_action_count = continuation.root_cache.candidate_actions.len();
    diagnostics.reused_stats_count = continuation.root_cache.recommendation.action_stats.len();

    if let Some(revealed_card) = context.observed_reveal {
        let matching_child = child.reveal_children.iter().find(|reveal| {
            reveal.revealed_card == revealed_card && reveal.child_key == current_key
        });
        if matching_child.is_some() {
            diagnostics.succeeded = true;
            diagnostics.outcome = ReuseOutcome::RevealChild;
            diagnostics.reveal_children_reused = child.reveal_children.len();
            return Some(diagnostics);
        }
        return Some(diagnostics);
    }

    if child.deterministic_child == Some(current_key) {
        diagnostics.succeeded = true;
        diagnostics.outcome = ReuseOutcome::FollowedMove;
        return Some(diagnostics);
    }

    if child
        .reveal_children
        .iter()
        .any(|reveal| reveal.child_key == current_key)
    {
        diagnostics.succeeded = true;
        diagnostics.outcome = ReuseOutcome::HashLookup;
        diagnostics.reveal_children_reused = child.reveal_children.len();
        return Some(diagnostics);
    }

    Some(diagnostics)
}

fn timed_belief_transition(
    belief: &BeliefState,
    action: crate::moves::AtomicMove,
    counters: &mut PlannerCounters,
) -> SolverResult<BeliefTransition> {
    if !counters.timing_enabled {
        return apply_belief_transition(belief, action);
    }

    let started = Instant::now();
    let transition = apply_belief_transition(belief, action)?;
    let elapsed = elapsed_micros(started);
    counters.belief_transition_elapsed_us = counters
        .belief_transition_elapsed_us
        .saturating_add(elapsed);
    if matches!(transition, BeliefTransition::Reveal { .. }) {
        counters.reveal_expansion_elapsed_us =
            counters.reveal_expansion_elapsed_us.saturating_add(elapsed);
    }
    Ok(transition)
}

fn simulate_root_action(
    root_belief: &BeliefState,
    root_action: &mut RootActionState,
    rng: &mut PlannerRng,
    solver: &DeterministicSolver,
    deterministic_config: DeterministicSearchConfig,
    planner_config: &BeliefPlannerConfig,
    counters: &mut PlannerCounters,
) -> SolverResult<SimulationResult> {
    if root_action.stats.action.semantics.causes_reveal {
        let newly_expanded = ensure_root_reveal_frontier(root_belief, root_action, counters)?;
        let frontier = root_action.root_reveal_frontier.as_ref().ok_or_else(|| {
            SolverError::InvalidState("missing cached root reveal frontier".to_string())
        })?;
        if frontier.is_empty() {
            return Err(SolverError::InvalidState(
                "root reveal frontier is empty".to_string(),
            ));
        }

        let child_index = root_action.next_reveal_child % frontier.len();
        root_action.next_reveal_child = root_action.next_reveal_child.saturating_add(1);
        if let Some(covered) = root_action.covered_reveal_children.get_mut(child_index) {
            *covered = true;
        }
        let child = &frontier.outcomes[child_index].belief;
        let mut result = simulate_belief(
            child,
            1,
            rng,
            solver,
            deterministic_config,
            planner_config,
            counters,
        )?;
        result.reveal_branches_expanded += newly_expanded;
        Ok(result)
    } else {
        if root_action.deterministic_child.is_none() {
            match timed_belief_transition(root_belief, root_action.stats.action.atomic, counters)? {
                BeliefTransition::Deterministic { belief, .. } => {
                    root_action.deterministic_child = Some(belief);
                }
                BeliefTransition::Reveal { frontier } => {
                    let branch_count = frontier.len() as u64;
                    counters.reveal_branches_expanded += branch_count;
                    root_action.covered_reveal_children = vec![false; frontier.len()];
                    root_action.root_reveal_frontier = Some(frontier);
                    root_action.stats.reveal_frontier_children =
                        root_action.covered_reveal_children.len();
                    return simulate_root_action(
                        root_belief,
                        root_action,
                        rng,
                        solver,
                        deterministic_config,
                        planner_config,
                        counters,
                    );
                }
            }
        }

        let child = root_action.deterministic_child.as_ref().ok_or_else(|| {
            SolverError::InvalidState("missing cached deterministic root child".to_string())
        })?;
        simulate_belief(
            child,
            1,
            rng,
            solver,
            deterministic_config,
            planner_config,
            counters,
        )
    }
}

fn ensure_root_reveal_frontier(
    root_belief: &BeliefState,
    root_action: &mut RootActionState,
    counters: &mut PlannerCounters,
) -> SolverResult<u64> {
    if root_action.root_reveal_frontier.is_some() {
        return Ok(0);
    }

    match timed_belief_transition(root_belief, root_action.stats.action.atomic, counters)? {
        BeliefTransition::Reveal { frontier } => {
            let branch_count = frontier.len();
            counters.reveal_branches_expanded += branch_count as u64;
            root_action.covered_reveal_children = vec![false; branch_count];
            root_action.stats.reveal_frontier_children = branch_count;
            root_action.root_reveal_frontier = Some(frontier);
            Ok(branch_count as u64)
        }
        BeliefTransition::Deterministic { belief, .. } => {
            root_action.deterministic_child = Some(belief);
            Err(SolverError::InvalidState(
                "root action was tagged as reveal-causing but transitioned deterministically"
                    .to_string(),
            ))
        }
    }
}

fn simulate_belief(
    belief: &BeliefState,
    depth: u8,
    rng: &mut PlannerRng,
    solver: &DeterministicSolver,
    deterministic_config: DeterministicSearchConfig,
    planner_config: &BeliefPlannerConfig,
    counters: &mut PlannerCounters,
) -> SolverResult<SimulationResult> {
    let belief = normalize_belief_for_simulation(belief, planner_config, counters)?;

    if belief.visible.is_structural_win() {
        return Ok(SimulationResult::terminal(1.0));
    }

    let actions = ordered_macro_moves(&belief.visible, deterministic_config);
    if actions.is_empty() {
        return Ok(SimulationResult::terminal(0.0));
    }

    if depth >= planner_config.max_depth {
        return evaluate_leaf_belief(&belief, rng, solver, planner_config, counters);
    }

    let action_index = rng.next_bounded(actions.len());
    simulate_action(
        &belief,
        &actions[action_index],
        depth,
        rng,
        solver,
        deterministic_config,
        planner_config,
        counters,
    )
}

fn simulate_action(
    belief: &BeliefState,
    action: &MacroMove,
    depth: u8,
    rng: &mut PlannerRng,
    solver: &DeterministicSolver,
    deterministic_config: DeterministicSearchConfig,
    planner_config: &BeliefPlannerConfig,
    counters: &mut PlannerCounters,
) -> SolverResult<SimulationResult> {
    match timed_belief_transition(belief, action.atomic, counters)? {
        BeliefTransition::Deterministic { belief, .. } => simulate_belief(
            &belief,
            depth.saturating_add(1),
            rng,
            solver,
            deterministic_config,
            planner_config,
            counters,
        ),
        BeliefTransition::Reveal { frontier } => {
            let branch_count = frontier.len() as u64;
            counters.reveal_branches_expanded += branch_count;
            let outcome = sample_reveal_outcome(&frontier, rng)?;
            let mut result = simulate_belief(
                &outcome.belief,
                depth.saturating_add(1),
                rng,
                solver,
                deterministic_config,
                planner_config,
                counters,
            )?;
            result.reveal_branches_expanded += branch_count;
            Ok(result)
        }
    }
}

fn normalize_belief_for_simulation(
    belief: &BeliefState,
    planner_config: &BeliefPlannerConfig,
    counters: &mut PlannerCounters,
) -> SolverResult<BeliefState> {
    let mut visible = belief.visible.clone();
    let mut closure_config = crate::closure::ClosureConfig {
        max_corridor_steps: planner_config.max_depth,
        ..crate::closure::ClosureConfig::default()
    };
    closure_config.stop_on_reveal = true;

    let closure = ClosureEngine::new(closure_config).run(&mut visible);
    counters.closure_steps_applied += closure.steps as u64;

    let normalized = BeliefState::new(visible, belief.unseen_cards);
    normalized.validate_consistency_against_visible()?;
    Ok(normalized)
}

fn evaluate_leaf_belief(
    belief: &BeliefState,
    rng: &mut PlannerRng,
    solver: &DeterministicSolver,
    planner_config: &BeliefPlannerConfig,
    counters: &mut PlannerCounters,
) -> SolverResult<SimulationResult> {
    let started = counters.timing_enabled.then(Instant::now);
    let mut sampler = PreparedWorldSampler::new(belief, DealSeed(rng.next_u64()))?;

    let mut value_sum = 0.0f32;
    for _ in 0..planner_config.leaf_world_samples {
        let full_state = sampler.sample_full_state()?;
        counters.leaf_evaluations += 1;
        let deterministic_started = counters.timing_enabled.then(Instant::now);
        let leaf = match planner_config.leaf_eval_mode {
            PlannerLeafEvalMode::Fast => {
                let result = solver.evaluate_fast(&full_state)?;
                counters.deterministic_nodes += result.stats.nodes_expanded;
                counters.record_vnet_stats(&result.stats);
                LeafValue {
                    value: result.value,
                    outcome: SolveOutcome::Unknown,
                }
            }
            PlannerLeafEvalMode::Bounded => {
                let result = solver.solve_bounded(&full_state)?;
                counters.deterministic_nodes += result.stats.nodes_expanded;
                counters.record_vnet_stats(&result.stats);
                LeafValue {
                    value: result.estimated_value,
                    outcome: result.outcome,
                }
            }
            PlannerLeafEvalMode::Exact => {
                let result = solver.solve_exact(&full_state)?;
                counters.deterministic_nodes += result.stats.nodes_expanded;
                counters.record_vnet_stats(&result.stats);
                LeafValue {
                    value: result.value,
                    outcome: result.outcome,
                }
            }
        };
        if let Some(deterministic_started) = deterministic_started {
            counters.deterministic_eval_elapsed_us = counters
                .deterministic_eval_elapsed_us
                .saturating_add(elapsed_micros(deterministic_started));
        }
        let _outcome = leaf.outcome;
        value_sum += leaf.value;
    }

    if let Some(started) = started {
        counters.leaf_eval_elapsed_us = counters
            .leaf_eval_elapsed_us
            .saturating_add(elapsed_micros(started));
    }

    Ok(SimulationResult::terminal(
        value_sum / planner_config.leaf_world_samples.max(1) as f32,
    ))
}

fn select_ucb_action(
    root_actions: &[RootActionState],
    simulations_run: usize,
    exploration_constant: f64,
) -> usize {
    if let Some(index) = root_actions
        .iter()
        .position(|entry| entry.active && entry.stats.visits == 0)
    {
        return index;
    }

    let total_visits = simulations_run.max(1) as f64;
    root_actions
        .iter()
        .enumerate()
        .filter(|(_, entry)| entry.active)
        .max_by(|(_, left), (_, right)| {
            ucb_score(&left.stats, total_visits, exploration_constant)
                .total_cmp(&ucb_score(&right.stats, total_visits, exploration_constant))
                .then_with(|| right.stats.action.kind.cmp(&left.stats.action.kind))
                .then_with(|| right.stats.action.id.cmp(&left.stats.action.id))
        })
        .map(|(index, _)| index)
        .unwrap_or_else(|| {
            root_actions
                .iter()
                .position(|entry| entry.active)
                .unwrap_or(0)
        })
}

fn ucb_score(stats: &PlannerActionStats, total_visits: f64, exploration_constant: f64) -> f64 {
    if stats.visits == 0 {
        return f64::INFINITY;
    }
    let exploration = (total_visits.ln() / stats.visits as f64).sqrt();
    stats.mean_value + exploration_constant * exploration
}

fn narrow_root_actions(
    root_actions: &mut [RootActionState],
    planner_config: &BeliefPlannerConfig,
) -> usize {
    let Some(max_active_root_actions) = planner_config.max_active_root_actions else {
        return 0;
    };

    let active_count = root_actions.iter().filter(|entry| entry.active).count();
    if active_count <= 1 || active_count <= max_active_root_actions.max(1) {
        return 0;
    }

    let best_mean = root_actions
        .iter()
        .filter(|entry| entry.active && entry.stats.visits > 0)
        .map(|entry| entry.stats.mean_value)
        .max_by(f64::total_cmp)
        .unwrap_or(0.0);

    let mut ranked = root_actions
        .iter()
        .enumerate()
        .filter(|(_, entry)| entry.active)
        .map(|(index, entry)| (index, entry.stats.mean_value, entry.stats.visits))
        .collect::<Vec<_>>();
    ranked.sort_by(|left, right| {
        right
            .1
            .total_cmp(&left.1)
            .then_with(|| {
                root_actions[left.0]
                    .stats
                    .action
                    .kind
                    .cmp(&root_actions[right.0].stats.action.kind)
            })
            .then_with(|| {
                root_actions[left.0]
                    .stats
                    .action
                    .id
                    .cmp(&root_actions[right.0].stats.action.id)
            })
    });

    let max_keep = max_active_root_actions.max(1);
    let mut narrowed = 0usize;
    for (rank, (index, mean, visits)) in ranked.into_iter().enumerate() {
        if rank < max_keep || visits == 0 {
            continue;
        }
        if best_mean - mean >= planner_config.drop_margin {
            root_actions[index].active = false;
            root_actions[index].stats.active = false;
            root_actions[index].stats.narrowed_out = true;
            narrowed += 1;
        }
    }
    narrowed
}

fn should_stop_early(
    root_actions: &[RootActionState],
    simulations_run: usize,
    planner_config: &BeliefPlannerConfig,
) -> bool {
    if !planner_config.enable_early_stop
        || simulations_run < planner_config.min_simulations_before_stop
    {
        return false;
    }

    let active = root_actions
        .iter()
        .filter(|entry| entry.active)
        .collect::<Vec<_>>();
    if active.is_empty() {
        return false;
    }
    if active.len() == 1 {
        return true;
    }
    if active.iter().any(|entry| entry.stats.visits == 0) {
        return false;
    }

    let (best_index, best) = active
        .iter()
        .enumerate()
        .max_by(|(_, left), (_, right)| {
            left.stats
                .mean_value
                .total_cmp(&right.stats.mean_value)
                .then_with(|| right.stats.action.kind.cmp(&left.stats.action.kind))
                .then_with(|| right.stats.action.id.cmp(&left.stats.action.id))
        })
        .expect("active is not empty");

    let best_lower = confidence_bounds(&best.stats, planner_config.confidence_z).0;
    active
        .iter()
        .enumerate()
        .filter(|(index, _)| *index != best_index)
        .all(|(_, challenger)| {
            let challenger_upper =
                confidence_bounds(&challenger.stats, planner_config.confidence_z).1;
            best_lower > challenger_upper + planner_config.separation_margin
        })
}

fn confidence_bounds(stats: &PlannerActionStats, z: f64) -> (f64, f64) {
    let radius = confidence_radius(stats, z);
    (
        (stats.mean_value - radius).clamp(0.0, 1.0),
        (stats.mean_value + radius).clamp(0.0, 1.0),
    )
}

fn confidence_radius(stats: &PlannerActionStats, z: f64) -> f64 {
    match stats.visits {
        0 => 1.0,
        1 => 0.5,
        _ => z * stats.standard_error,
    }
}

fn run_second_reveal_refinement(
    root_belief: &BeliefState,
    root_actions: &mut [RootActionState],
    rng: &mut PlannerRng,
    solver: &DeterministicSolver,
    deterministic_config: DeterministicSearchConfig,
    planner_config: &BeliefPlannerConfig,
    counters: &mut PlannerCounters,
) -> SolverResult<usize> {
    if !planner_config.enable_second_reveal_refinement
        || planner_config.second_reveal_refinement_simulations == 0
        || planner_config.max_second_reveal_actions == 0
    {
        return Ok(0);
    }

    let candidates = second_reveal_candidates(root_actions, planner_config);
    if candidates.is_empty() {
        return Ok(0);
    }

    let mut refinement_config = *planner_config;
    refinement_config.max_depth = refinement_config.max_depth.saturating_add(1);

    let mut simulations_run = 0usize;
    let mut next_candidate = 0usize;
    for _ in 0..planner_config.second_reveal_refinement_simulations {
        let action_index = candidates[next_candidate % candidates.len()];
        next_candidate += 1;
        let simulation = simulate_root_action(
            root_belief,
            &mut root_actions[action_index],
            rng,
            solver,
            deterministic_config,
            &refinement_config,
            counters,
        )?;
        root_actions[action_index]
            .stats
            .record(simulation.value, simulation.reveal_branches_expanded);
        root_actions[action_index]
            .stats
            .second_reveal_refinement_visits += 1;
        root_actions[action_index].refresh_reveal_stats();
        simulations_run += 1;
    }

    Ok(simulations_run)
}

fn second_reveal_candidates(
    root_actions: &[RootActionState],
    planner_config: &BeliefPlannerConfig,
) -> Vec<usize> {
    let mut ranked = root_actions
        .iter()
        .enumerate()
        .filter(|(_, entry)| entry.active && entry.stats.visits > 0)
        .collect::<Vec<_>>();
    if ranked.is_empty() {
        return Vec::new();
    }

    ranked.sort_by(|(_, left), (_, right)| {
        right
            .stats
            .mean_value
            .total_cmp(&left.stats.mean_value)
            .then_with(|| left.stats.action.kind.cmp(&right.stats.action.kind))
            .then_with(|| left.stats.action.id.cmp(&right.stats.action.id))
    });

    let best_mean = ranked[0].1.stats.mean_value;
    ranked
        .into_iter()
        .take(planner_config.max_second_reveal_actions)
        .filter(|(_, entry)| {
            if !entry.stats.action.semantics.causes_reveal {
                return false;
            }
            let gap = best_mean - entry.stats.mean_value;
            let uncertain =
                entry.stats.standard_error >= planner_config.second_reveal_uncertainty_threshold;
            gap <= planner_config.second_reveal_gap_threshold || uncertain
        })
        .map(|(index, _)| index)
        .collect()
}

fn finalize_root_action_states(root_actions: &mut [RootActionState], confidence_z: f64) {
    for state in root_actions {
        state.refresh_reveal_stats();
        state.stats.refresh_confidence(confidence_z);
    }
}

fn run_late_exact_if_eligible(
    belief: &BeliefState,
    root_actions: &mut [RootActionState],
    solver_config: &SolverConfig,
) -> SolverResult<LateExactResult> {
    let hidden_count = belief.hidden_card_count();
    if !solver_config.late_exact.enabled
        || hidden_count > usize::from(solver_config.late_exact.hidden_card_threshold)
    {
        return Ok(LateExactResult {
            triggered: false,
            hidden_count,
            actions_evaluated: 0,
            assignments_enumerated: 0,
            assignments_pruned: 0,
            exhaustive: false,
            best_move: None,
            best_value: 0.0,
            action_stats: Vec::new(),
            deterministic_nodes: 0,
            elapsed_ms: 0,
            assignment_enumeration_elapsed_us: 0,
        });
    }

    let mut ranked = root_actions
        .iter()
        .enumerate()
        .filter(|(_, state)| state.active && state.stats.visits > 0)
        .collect::<Vec<_>>();
    if ranked.is_empty() {
        return Ok(LateExactResult {
            triggered: false,
            hidden_count,
            actions_evaluated: 0,
            assignments_enumerated: 0,
            assignments_pruned: 0,
            exhaustive: false,
            best_move: None,
            best_value: 0.0,
            action_stats: Vec::new(),
            deterministic_nodes: 0,
            elapsed_ms: 0,
            assignment_enumeration_elapsed_us: 0,
        });
    }

    ranked.sort_by(|(_, left), (_, right)| {
        right
            .stats
            .mean_value
            .total_cmp(&left.stats.mean_value)
            .then_with(|| left.stats.action.kind.cmp(&right.stats.action.kind))
            .then_with(|| left.stats.action.id.cmp(&right.stats.action.id))
    });

    let action_indices = ranked
        .into_iter()
        .take(solver_config.late_exact.max_root_actions.max(1))
        .map(|(index, _)| index)
        .collect::<Vec<_>>();
    let actions = action_indices
        .iter()
        .map(|index| root_actions[*index].stats.action.clone())
        .collect::<Vec<_>>();

    let evaluator = LateExactEvaluator::new(
        solver_config.late_exact,
        late_exact_deterministic_config(solver_config),
    );
    let result = evaluator.evaluate_actions(belief, &actions)?;
    if !result.triggered {
        return Ok(result);
    }

    for exact_stats in &result.action_stats {
        if let Some(root_action) = root_actions.iter_mut().find(|state| {
            state.stats.action.atomic == exact_stats.action.atomic
                && state.stats.action.kind == exact_stats.action.kind
        }) {
            root_action.stats.mean_value = exact_stats.mean_value;
            root_action.stats.m2 = 0.0;
            root_action.stats.variance = 0.0;
            root_action.stats.standard_error = 0.0;
            root_action.stats.confidence_lower = exact_stats.mean_value;
            root_action.stats.confidence_upper = exact_stats.mean_value;
            root_action.stats.late_exact_evaluated = true;
            root_action.stats.late_exact_assignments_enumerated = exact_stats.assignments_evaluated;
            root_action.stats.late_exact_value = Some(exact_stats.mean_value);
        }
    }

    Ok(result)
}

fn sample_reveal_outcome<'a>(
    frontier: &'a RevealFrontier,
    rng: &mut PlannerRng,
) -> SolverResult<&'a crate::belief::RevealOutcome> {
    if frontier.is_empty() {
        return Err(SolverError::InvalidState(
            "cannot sample an empty reveal frontier".to_string(),
        ));
    }

    let mut threshold = rng.next_f64();
    for outcome in frontier.iter() {
        threshold -= f64::from(outcome.probability);
        if threshold <= 0.0 {
            return Ok(outcome);
        }
    }
    frontier.iter().last().ok_or_else(|| {
        SolverError::InvalidState("cannot sample an empty reveal frontier".to_string())
    })
}

fn elapsed_micros(started: Instant) -> u64 {
    started.elapsed().as_micros().min(u128::from(u64::MAX)) as u64
}

fn deterministic_search_config(
    solver_config: &SolverConfig,
    planner_config: &BeliefPlannerConfig,
) -> DeterministicSearchConfig {
    deterministic_search_config_from_parts(&solver_config.deterministic, solver_config)
        .with_leaf_mode_budget(&solver_config.deterministic, planner_config.leaf_eval_mode)
}

fn late_exact_deterministic_config(solver_config: &SolverConfig) -> DeterministicSearchConfig {
    deterministic_search_config_from_parts(&solver_config.deterministic, solver_config)
        .with_late_exact_budget(
            &solver_config.deterministic,
            solver_config.late_exact.evaluation_mode,
        )
}

fn deterministic_search_config_from_parts(
    deterministic: &DeterministicSolverConfig,
    solver_config: &SolverConfig,
) -> DeterministicSearchConfig {
    DeterministicSearchConfig {
        budget: SolveBudget {
            node_budget: Some(deterministic.fast_eval_node_budget),
            depth_budget: Some(deterministic.max_macro_depth),
            wall_clock_limit_ms: None,
        },
        closure: solver_config.closure,
        allow_foundation_retreats: deterministic.enable_foundation_retreats,
        evaluator_weights: EvaluatorWeights::default(),
        tt: DeterministicTtConfig {
            enabled: deterministic.enable_tt,
            capacity: deterministic.tt_capacity,
            store_approx: deterministic.tt_store_approx,
        },
        leaf_eval_mode: deterministic.leaf_eval_mode,
    }
}

trait DeterministicConfigPlannerExt {
    fn with_leaf_mode_budget(
        self,
        deterministic: &DeterministicSolverConfig,
        leaf_mode: PlannerLeafEvalMode,
    ) -> Self;

    fn with_late_exact_budget(
        self,
        deterministic: &DeterministicSolverConfig,
        eval_mode: LateExactEvaluationMode,
    ) -> Self;
}

impl DeterministicConfigPlannerExt for DeterministicSearchConfig {
    fn with_leaf_mode_budget(
        mut self,
        deterministic: &DeterministicSolverConfig,
        leaf_mode: PlannerLeafEvalMode,
    ) -> Self {
        self.budget.node_budget = Some(match leaf_mode {
            PlannerLeafEvalMode::Fast => deterministic.fast_eval_node_budget,
            PlannerLeafEvalMode::Bounded | PlannerLeafEvalMode::Exact => {
                deterministic.exact_node_budget
            }
        });
        self
    }

    fn with_late_exact_budget(
        mut self,
        deterministic: &DeterministicSolverConfig,
        eval_mode: LateExactEvaluationMode,
    ) -> Self {
        self.budget.node_budget = Some(match eval_mode {
            LateExactEvaluationMode::Fast => deterministic.fast_eval_node_budget,
            LateExactEvaluationMode::Bounded | LateExactEvaluationMode::Exact => {
                deterministic.exact_node_budget
            }
        });
        self
    }
}

#[derive(Debug, Clone, PartialEq)]
struct RootActionState {
    stats: PlannerActionStats,
    active: bool,
    deterministic_child: Option<BeliefState>,
    root_reveal_frontier: Option<RevealFrontier>,
    next_reveal_child: usize,
    covered_reveal_children: Vec<bool>,
}

impl RootActionState {
    fn new(action: MacroMove) -> Self {
        Self {
            stats: PlannerActionStats::new(action),
            active: true,
            deterministic_child: None,
            root_reveal_frontier: None,
            next_reveal_child: 0,
            covered_reveal_children: Vec::new(),
        }
    }

    fn refresh_reveal_stats(&mut self) {
        self.stats.active = self.active;
        self.stats.reveal_frontier_children = self
            .root_reveal_frontier
            .as_ref()
            .map_or(0, RevealFrontier::len);
        self.stats.reveal_frontier_children_covered = self
            .covered_reveal_children
            .iter()
            .filter(|covered| **covered)
            .count();
    }
}

#[derive(Debug, Copy, Clone, Default, PartialEq, Eq)]
struct PlannerCounters {
    timing_enabled: bool,
    leaf_evaluations: u64,
    deterministic_nodes: u64,
    vnet_inferences: u64,
    vnet_fallbacks: u64,
    vnet_inference_elapsed_us: u64,
    closure_steps_applied: u64,
    reveal_branches_expanded: u64,
    belief_transition_elapsed_us: u64,
    reveal_expansion_elapsed_us: u64,
    leaf_eval_elapsed_us: u64,
    deterministic_eval_elapsed_us: u64,
}

impl PlannerCounters {
    fn record_vnet_stats(&mut self, stats: &DeterministicSearchStats) {
        self.vnet_inferences = self.vnet_inferences.saturating_add(stats.vnet_inferences);
        self.vnet_fallbacks = self.vnet_fallbacks.saturating_add(stats.vnet_fallbacks);
        self.vnet_inference_elapsed_us = self
            .vnet_inference_elapsed_us
            .saturating_add(stats.vnet_inference_elapsed_us);
    }
}

#[derive(Debug, Copy, Clone, PartialEq)]
struct SimulationResult {
    value: f32,
    reveal_branches_expanded: u64,
}

impl SimulationResult {
    const fn terminal(value: f32) -> Self {
        Self {
            value,
            reveal_branches_expanded: 0,
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq)]
struct LeafValue {
    value: f32,
    outcome: SolveOutcome,
}
