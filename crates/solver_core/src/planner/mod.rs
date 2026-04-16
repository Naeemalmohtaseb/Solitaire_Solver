//! First belief-state root planner.
//!
//! This module is the first event-driven hidden-information planner. It keeps
//! the posterior exact and uniform: non-reveal moves use deterministic belief
//! transitions, reveal moves expand the exact reveal frontier and sample one
//! equal-probability observation per simulation, and leaf values come from the
//! deterministic open-card solver on uniformly sampled full worlds.

use std::time::Instant;

use serde::{Deserialize, Serialize};

use crate::{
    belief::{apply_belief_transition, sample_full_states, BeliefTransition, RevealFrontier},
    closure::ClosureEngine,
    config::{DeterministicSolverConfig, SolverConfig},
    core::BeliefState,
    deterministic_solver::{
        ordered_macro_moves, DeterministicSearchConfig, DeterministicSolver, DeterministicTtConfig,
        EvaluatorWeights, SolveBudget, SolveOutcome,
    },
    error::{SolverError, SolverResult},
    late_exact::{LateExactEvaluationMode, LateExactEvaluator, LateExactResult},
    moves::MacroMove,
    types::DealSeed,
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
}

/// Recommends a root move with sparse UCT-style belief simulations.
pub fn recommend_move_belief_uct(
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

    let solver = DeterministicSolver::new(deterministic_config);
    let mut rng = PlannerRng::new(planner_config.rng_seed.0);
    let mut counters = PlannerCounters::default();
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
        elapsed_ms: started.elapsed().as_millis() as u64,
    })
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
        let child = frontier.outcomes[child_index].belief.clone();
        let mut result = simulate_belief(
            &child,
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
            match apply_belief_transition(root_belief, root_action.stats.action.atomic)? {
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

        let child = root_action
            .deterministic_child
            .as_ref()
            .ok_or_else(|| {
                SolverError::InvalidState("missing cached deterministic root child".to_string())
            })?
            .clone();
        simulate_belief(
            &child,
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

    match apply_belief_transition(root_belief, root_action.stats.action.atomic)? {
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
    match apply_belief_transition(belief, action.atomic)? {
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
    let samples = sample_full_states(
        belief,
        planner_config.leaf_world_samples,
        DealSeed(rng.next_u64()),
    )?;

    let mut value_sum = 0.0f32;
    for sample in &samples {
        counters.leaf_evaluations += 1;
        let leaf = match planner_config.leaf_eval_mode {
            PlannerLeafEvalMode::Fast => {
                let result = solver.evaluate_fast(&sample.full_state)?;
                counters.deterministic_nodes += result.stats.nodes_expanded;
                LeafValue {
                    value: result.value,
                    outcome: SolveOutcome::Unknown,
                }
            }
            PlannerLeafEvalMode::Bounded => {
                let result = solver.solve_bounded(&sample.full_state)?;
                counters.deterministic_nodes += result.stats.nodes_expanded;
                LeafValue {
                    value: result.estimated_value,
                    outcome: result.outcome,
                }
            }
            PlannerLeafEvalMode::Exact => {
                let result = solver.solve_exact(&sample.full_state)?;
                counters.deterministic_nodes += result.stats.nodes_expanded;
                LeafValue {
                    value: result.value,
                    outcome: result.outcome,
                }
            }
        };
        let _outcome = leaf.outcome;
        value_sum += leaf.value;
    }

    Ok(SimulationResult::terminal(
        value_sum / samples.len().max(1) as f32,
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
    leaf_evaluations: u64,
    deterministic_nodes: u64,
    closure_steps_applied: u64,
    reveal_branches_expanded: u64,
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

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
struct PlannerRng {
    state: u64,
}

impl PlannerRng {
    const fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut value = self.state;
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        value ^ (value >> 31)
    }

    fn next_bounded(&mut self, upper_exclusive: usize) -> usize {
        debug_assert!(upper_exclusive > 0);
        (self.next_u64() % upper_exclusive as u64) as usize
    }

    fn next_f64(&mut self) -> f64 {
        let bits = self.next_u64() >> 11;
        bits as f64 / ((1u64 << 53) as f64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        belief::apply_belief_transition,
        cards::Card,
        core::{TableauColumn, UnseenCardSet, VisibleState},
        moves::{AtomicMove, MacroMoveKind},
        stock::CyclicStockState,
        types::ColumnId,
    };

    fn col(index: u8) -> ColumnId {
        ColumnId::new(index).unwrap()
    }

    fn card(text: &str) -> Card {
        text.parse().unwrap()
    }

    fn stock_with_all_except(excluded: &[Card]) -> CyclicStockState {
        let mut cards = Vec::new();
        for index in 0..Card::COUNT {
            let card = Card::new(index as u8).unwrap();
            if !excluded.contains(&card) {
                cards.push(card);
            }
        }
        CyclicStockState::new(cards, None, 0, None, 3)
    }

    fn no_move_belief() -> BeliefState {
        let cards = (0..Card::COUNT)
            .map(|index| Card::new(index as u8).unwrap())
            .collect();
        let mut visible = VisibleState::default();
        visible.stock = CyclicStockState::from_parts(cards, 0, 0, 0, Some(0), 3);
        BeliefState::new(visible, UnseenCardSet::empty())
    }

    fn single_legal_belief() -> BeliefState {
        let cards = (0..Card::COUNT)
            .map(|index| Card::new(index as u8).unwrap())
            .collect();
        let mut visible = VisibleState::default();
        visible.stock = CyclicStockState::new(cards, None, 0, None, 3);
        BeliefState::new(visible, UnseenCardSet::empty())
    }

    fn reveal_belief() -> BeliefState {
        let hidden_cards = [card("Ac"), card("2c")];
        let seven = card("7s");
        let eight = card("8h");
        let mut visible = VisibleState::default();
        visible.columns[0] = TableauColumn::new(2, vec![seven]);
        visible.columns[1] = TableauColumn::new(0, vec![eight]);
        visible.stock = stock_with_all_except(&[hidden_cards[0], hidden_cards[1], seven, eight]);
        BeliefState::new(
            visible,
            UnseenCardSet::from_cards(hidden_cards).expect("unique unseen cards"),
        )
    }

    fn nonreveal_belief() -> BeliefState {
        let hidden = card("Ah");
        let seven = card("7s");
        let six = card("6h");
        let mut visible = VisibleState::default();
        visible.columns[0] = TableauColumn::new(0, vec![seven]);
        visible.columns[1] = TableauColumn::new(0, vec![six]);
        visible.columns[2] = TableauColumn::new(1, Vec::new());
        visible.stock = stock_with_all_except(&[hidden, seven, six]);
        BeliefState::new(
            visible,
            UnseenCardSet::from_cards([hidden]).expect("unique unseen card"),
        )
    }

    fn solver_config() -> SolverConfig {
        let mut config = SolverConfig::default();
        config.deterministic.fast_eval_node_budget = 128;
        config.deterministic.exact_node_budget = 256;
        config.deterministic.max_macro_depth = 4;
        config.deterministic.enable_tt = false;
        config
    }

    fn planner_config(seed: u64) -> BeliefPlannerConfig {
        BeliefPlannerConfig {
            simulation_budget: 6,
            max_depth: 1,
            exploration_constant: 1.0,
            leaf_world_samples: 1,
            leaf_eval_mode: PlannerLeafEvalMode::Fast,
            rng_seed: DealSeed(seed),
            enable_early_stop: false,
            initial_screen_simulations: 0,
            max_active_root_actions: None,
            enable_second_reveal_refinement: false,
            ..BeliefPlannerConfig::default()
        }
    }

    #[test]
    fn planner_returns_only_legal_moves() {
        let belief = reveal_belief();
        let config = solver_config();
        let planner = planner_config(7);
        let recommendation = recommend_move_belief_uct(&belief, &config, &planner).unwrap();
        let legal = ordered_macro_moves(
            &belief.visible,
            deterministic_search_config(&config, &planner),
        );

        assert!(legal
            .iter()
            .any(|legal| Some(legal) == recommendation.best_move.as_ref()));
    }

    #[test]
    fn single_legal_move_returns_immediately() {
        let belief = single_legal_belief();
        let recommendation =
            recommend_move_belief_uct(&belief, &solver_config(), &planner_config(11)).unwrap();

        assert_eq!(recommendation.simulations_run, 0);
        assert!(matches!(
            recommendation.best_move.map(|action| action.kind),
            Some(MacroMoveKind::AdvanceStock)
        ));
        assert_eq!(recommendation.action_stats.len(), 1);
    }

    #[test]
    fn no_legal_moves_returns_no_move_result() {
        let belief = no_move_belief();
        let recommendation =
            recommend_move_belief_uct(&belief, &solver_config(), &planner_config(12)).unwrap();

        assert!(recommendation.no_legal_moves);
        assert!(recommendation.best_move.is_none());
        assert!(recommendation.action_stats.is_empty());
    }

    #[test]
    fn reveal_actions_use_reveal_frontier_logic() {
        let belief = reveal_belief();
        let recommendation =
            recommend_move_belief_uct(&belief, &solver_config(), &planner_config(13)).unwrap();

        let reveal_stats = recommendation
            .action_stats
            .iter()
            .find(|stats| stats.action.semantics.causes_reveal)
            .expect("root should contain a reveal action");

        assert!(reveal_stats.visits > 0);
        assert!(reveal_stats.reveal_branches_expanded >= belief.unseen_card_count() as u64);
        assert!(recommendation.reveal_branches_expanded >= belief.unseen_card_count() as u64);
    }

    #[test]
    fn deterministic_non_reveal_transition_preserves_unseen_set() {
        let belief = nonreveal_belief();
        let transition = apply_belief_transition(
            &belief,
            AtomicMove::TableauToTableau {
                src: col(1),
                dest: col(0),
                run_start: 0,
            },
        )
        .unwrap();

        let BeliefTransition::Deterministic { belief: child, .. } = transition else {
            panic!("expected deterministic transition");
        };
        assert_eq!(child.unseen_cards, belief.unseen_cards);
    }

    #[test]
    fn same_seed_gives_reproducible_planner_output() {
        let belief = reveal_belief();
        let config = solver_config();
        let planner = planner_config(21);

        let first = recommend_move_belief_uct(&belief, &config, &planner).unwrap();
        let second = recommend_move_belief_uct(&belief, &config, &planner).unwrap();

        assert_eq!(first.best_move, second.best_move);
        assert_eq!(first.best_value, second.best_value);
        assert_eq!(first.action_stats, second.action_stats);
        assert_eq!(first.simulations_run, second.simulations_run);
    }

    #[test]
    fn planner_does_not_mutate_input_belief_state() {
        let belief = reveal_belief();
        let before = belief.clone();

        let _recommendation =
            recommend_move_belief_uct(&belief, &solver_config(), &planner_config(31)).unwrap();

        assert_eq!(belief, before);
    }

    #[test]
    fn root_stats_accumulate_one_visit_per_simulation() {
        let belief = reveal_belief();
        let planner = planner_config(44);
        let recommendation =
            recommend_move_belief_uct(&belief, &solver_config(), &planner).unwrap();

        let visits = recommendation
            .action_stats
            .iter()
            .map(|stats| stats.visits)
            .sum::<usize>();

        assert_eq!(recommendation.simulations_run, planner.simulation_budget);
        assert_eq!(visits, planner.simulation_budget);
    }

    #[test]
    fn early_stopping_respects_minimum_sample_threshold() {
        let belief = reveal_belief();
        let config = solver_config();
        let planner = planner_config(55);
        let actions = ordered_macro_moves(
            &belief.visible,
            deterministic_search_config(&config, &planner),
        );
        let mut root_actions = vec![
            RootActionState::new(actions[0].clone()),
            RootActionState::new(actions[1].clone()),
        ];

        for _ in 0..3 {
            root_actions[0].stats.record(1.0, 0);
            root_actions[1].stats.record(0.0, 0);
        }

        let stop_config = BeliefPlannerConfig {
            enable_early_stop: true,
            min_simulations_before_stop: 8,
            confidence_z: 0.0,
            separation_margin: 0.0,
            ..planner
        };

        assert!(!should_stop_early(&root_actions, 7, &stop_config));
        assert!(should_stop_early(&root_actions, 8, &stop_config));
    }

    #[test]
    fn action_narrowing_never_removes_only_legal_move() {
        let belief = single_legal_belief();
        let planner = BeliefPlannerConfig {
            initial_screen_simulations: 1,
            max_active_root_actions: Some(1),
            drop_margin: 0.0,
            ..planner_config(56)
        };
        let recommendation =
            recommend_move_belief_uct(&belief, &solver_config(), &planner).unwrap();

        assert_eq!(recommendation.action_stats.len(), 1);
        assert_eq!(recommendation.actions_narrowed_out, 0);
        assert_eq!(recommendation.active_root_actions, 1);
    }

    #[test]
    fn reveal_root_actions_cover_exact_frontier_children_fairly() {
        let belief = reveal_belief();
        let planner = BeliefPlannerConfig {
            simulation_budget: 4,
            ..planner_config(57)
        };
        let recommendation =
            recommend_move_belief_uct(&belief, &solver_config(), &planner).unwrap();
        let reveal_stats = recommendation
            .action_stats
            .iter()
            .find(|stats| stats.action.semantics.causes_reveal)
            .expect("root should contain reveal action");

        assert_eq!(
            reveal_stats.reveal_frontier_children,
            belief.unseen_card_count()
        );
        assert_eq!(
            reveal_stats.reveal_frontier_children_covered,
            belief.unseen_card_count()
        );
        assert_eq!(
            recommendation.reveal_frontier_children_covered,
            belief.unseen_card_count() as u64
        );
    }

    #[test]
    fn second_reveal_refinement_runs_only_for_close_enabled_reveal_contenders() {
        let belief = reveal_belief();
        let enabled = BeliefPlannerConfig {
            simulation_budget: 4,
            enable_second_reveal_refinement: true,
            max_second_reveal_actions: 2,
            second_reveal_gap_threshold: 1.0,
            second_reveal_uncertainty_threshold: 2.0,
            second_reveal_refinement_simulations: 3,
            ..planner_config(58)
        };
        let enabled_result =
            recommend_move_belief_uct(&belief, &solver_config(), &enabled).unwrap();

        assert!(enabled_result.second_reveal_refinement_ran);
        assert_eq!(enabled_result.second_reveal_simulations_run, 3);
        assert_eq!(
            enabled_result.simulations_run,
            enabled.simulation_budget + 3
        );
        assert_eq!(
            enabled_result
                .action_stats
                .iter()
                .map(|stats| stats.second_reveal_refinement_visits)
                .sum::<usize>(),
            3
        );

        let disabled_by_threshold = BeliefPlannerConfig {
            simulation_budget: 4,
            enable_second_reveal_refinement: true,
            max_second_reveal_actions: 2,
            second_reveal_gap_threshold: -1.0,
            second_reveal_uncertainty_threshold: 2.0,
            second_reveal_refinement_simulations: 3,
            ..planner_config(59)
        };
        let disabled_result =
            recommend_move_belief_uct(&belief, &solver_config(), &disabled_by_threshold).unwrap();

        assert!(!disabled_result.second_reveal_refinement_ran);
        assert_eq!(disabled_result.second_reveal_simulations_run, 0);
        assert_eq!(
            disabled_result.simulations_run,
            disabled_by_threshold.simulation_budget
        );
    }

    #[test]
    fn planner_uses_late_exact_for_eligible_top_actions() {
        let belief = reveal_belief();
        let mut solver = solver_config();
        solver.late_exact.hidden_card_threshold = 2;
        solver.late_exact.max_root_actions = 2;
        solver.late_exact.evaluation_mode = LateExactEvaluationMode::Fast;

        let recommendation =
            recommend_move_belief_uct(&belief, &solver, &planner_config(61)).unwrap();

        assert!(recommendation.late_exact_triggered);
        assert_eq!(
            recommendation.late_exact_hidden_count,
            belief.hidden_card_count()
        );
        assert!(recommendation.late_exact_actions_evaluated <= 2);
        assert!(recommendation.late_exact_assignments_enumerated > 0);
        assert!(recommendation
            .action_stats
            .iter()
            .any(|stats| stats.late_exact_evaluated));
    }

    #[test]
    fn planner_skips_late_exact_above_threshold() {
        let belief = reveal_belief();
        let mut solver = solver_config();
        solver.late_exact.hidden_card_threshold = 1;
        solver.late_exact.max_root_actions = 2;
        solver.late_exact.evaluation_mode = LateExactEvaluationMode::Fast;

        let recommendation =
            recommend_move_belief_uct(&belief, &solver, &planner_config(62)).unwrap();

        assert!(!recommendation.late_exact_triggered);
        assert_eq!(recommendation.late_exact_actions_evaluated, 0);
        assert_eq!(recommendation.late_exact_assignments_enumerated, 0);
    }
}
