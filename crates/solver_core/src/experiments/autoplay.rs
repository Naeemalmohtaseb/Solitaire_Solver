//! Full-game autoplay using hidden-information planner backends.

use serde::{Deserialize, Serialize};

use crate::{
    belief::{
        apply_observed_belief_move, belief_from_full_state, validate_belief_against_full_state,
    },
    cards::Card,
    config::SolverConfig,
    core::{BeliefState, FullState},
    error::SolverResult,
    moves::{apply_atomic_move_full_state, MacroMove},
    planner::{recommend_move_belief_uct, PlannerRecommendation},
};

use super::{
    action_seed, deterministic_search_config_from_solver, recommend_move_pimc, PimcConfig,
};
/// Planner backend used for full-game autoplay.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PlannerBackend {
    /// Root PIMC baseline over uniform determinizations.
    Pimc,
    /// Belief UCT planner with late-exact disabled for the run.
    BeliefUct,
    /// Belief UCT planner with configured late-exact support enabled.
    BeliefUctLateExact,
}

/// Configuration for full-game autoplay.
#[derive(Debug, Copy, Clone, PartialEq, Serialize, Deserialize)]
pub struct AutoplayConfig {
    /// Planner backend to use at each decision.
    pub backend: PlannerBackend,
    /// PIMC configuration used when `backend` is [`PlannerBackend::Pimc`].
    pub pimc: PimcConfig,
    /// Maximum number of chosen moves before terminating.
    pub max_steps: usize,
    /// Optional total planner-time cap in milliseconds.
    pub max_total_planner_time_ms: Option<u64>,
    /// Whether to validate public belief against the true full state after each step.
    pub validate_each_step: bool,
}

impl Default for AutoplayConfig {
    fn default() -> Self {
        Self {
            backend: PlannerBackend::BeliefUctLateExact,
            pimc: PimcConfig::default(),
            max_steps: 500,
            max_total_planner_time_ms: None,
            validate_each_step: true,
        }
    }
}

/// Reason a full autoplay game stopped.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AutoplayTermination {
    /// All foundations are complete.
    Win,
    /// Planner found no legal move.
    NoLegalMove,
    /// Configured step cap was reached.
    StepLimit,
    /// Configured total planner-time cap was reached.
    BudgetExhausted,
}

/// Lightweight planner diagnostics captured at one autoplay step.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AutoplayPlannerSnapshot {
    /// Planner backend used.
    pub backend: PlannerBackend,
    /// Best root value reported by the backend.
    pub best_value: f64,
    /// Planner elapsed time for this step in milliseconds.
    pub elapsed_ms: u64,
    /// Deterministic solver nodes reported by this step.
    pub deterministic_nodes: u64,
    /// Root simulations, samples, or visits reported by the backend.
    pub root_visits: u64,
    /// Whether late-exact triggered during this step.
    pub late_exact_triggered: bool,
}

/// One applied full-game autoplay step.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AutoplayStep {
    /// Zero-based step index.
    pub step_index: usize,
    /// Chosen root move.
    pub chosen_move: MacroMove,
    /// True revealed card, if the move uncovered a hidden tableau card.
    pub revealed_card: Option<Card>,
    /// Public hidden-card count before the move.
    pub hidden_count_before: usize,
    /// Public hidden-card count after the move.
    pub hidden_count_after: usize,
    /// Planner diagnostics for the decision.
    pub planner: AutoplayPlannerSnapshot,
}

/// Ordered trace for one full autoplay game.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct AutoplayTrace {
    /// Applied steps in order.
    pub steps: Vec<AutoplayStep>,
}

impl AutoplayTrace {
    /// Returns number of recorded steps.
    pub fn len(&self) -> usize {
        self.steps.len()
    }

    /// Returns true when no steps were recorded.
    pub fn is_empty(&self) -> bool {
        self.steps.is_empty()
    }
}

/// Result of one full-game autoplay run.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AutoplayResult {
    /// Whether the game ended in a structural win.
    pub won: bool,
    /// Termination reason.
    pub termination: AutoplayTermination,
    /// Applied step trace.
    pub trace: AutoplayTrace,
    /// Final public belief state.
    pub final_belief: BeliefState,
    /// Final true full state.
    pub final_full_state: FullState,
    /// Total planner time in milliseconds.
    pub total_planner_time_ms: u64,
    /// Total deterministic solver nodes reported by planners.
    pub deterministic_nodes: u64,
    /// Total root simulations/samples/visits reported by planners.
    pub root_visits: u64,
    /// Number of steps where late-exact triggered.
    pub late_exact_triggers: usize,
}

/// Plays one true full game using the selected hidden-information backend.
pub fn play_game_with_planner(
    full_state: &FullState,
    solver_config: &SolverConfig,
    autoplay_config: &AutoplayConfig,
) -> SolverResult<AutoplayResult> {
    let mut true_state = full_state.clone();
    let mut belief = belief_from_full_state(&true_state)?;
    let mut trace = AutoplayTrace::default();
    let mut total_planner_time_ms = 0u64;
    let mut deterministic_nodes = 0u64;
    let mut root_visits = 0u64;
    let mut late_exact_triggers = 0usize;

    for step_index in 0..autoplay_config.max_steps {
        if true_state.visible.is_structural_win() {
            return Ok(AutoplayResult {
                won: true,
                termination: AutoplayTermination::Win,
                trace,
                final_belief: belief,
                final_full_state: true_state,
                total_planner_time_ms,
                deterministic_nodes,
                root_visits,
                late_exact_triggers,
            });
        }

        if autoplay_config
            .max_total_planner_time_ms
            .is_some_and(|limit| total_planner_time_ms >= limit)
        {
            return Ok(AutoplayResult {
                won: false,
                termination: AutoplayTermination::BudgetExhausted,
                trace,
                final_belief: belief,
                final_full_state: true_state,
                total_planner_time_ms,
                deterministic_nodes,
                root_visits,
                late_exact_triggers,
            });
        }

        let hidden_count_before = belief.hidden_card_count();
        let decision = recommend_autoplay_move(
            &belief,
            solver_config,
            autoplay_config.backend,
            autoplay_config.pimc,
            step_index,
        )?;

        total_planner_time_ms = total_planner_time_ms.saturating_add(decision.snapshot.elapsed_ms);
        deterministic_nodes =
            deterministic_nodes.saturating_add(decision.snapshot.deterministic_nodes);
        root_visits = root_visits.saturating_add(decision.snapshot.root_visits);
        if decision.snapshot.late_exact_triggered {
            late_exact_triggers += 1;
        }

        let Some(chosen_move) = decision.best_move else {
            return Ok(AutoplayResult {
                won: false,
                termination: AutoplayTermination::NoLegalMove,
                trace,
                final_belief: belief,
                final_full_state: true_state,
                total_planner_time_ms,
                deterministic_nodes,
                root_visits,
                late_exact_triggers,
            });
        };

        let true_transition = apply_atomic_move_full_state(&mut true_state, chosen_move.atomic)?;
        let revealed_card = true_transition.outcome.revealed.map(|reveal| reveal.card);
        let (next_belief, _observed_outcome) =
            apply_observed_belief_move(&belief, chosen_move.atomic, revealed_card)?;
        belief = next_belief;

        if autoplay_config.validate_each_step {
            validate_belief_against_full_state(&belief, &true_state)?;
        }

        let hidden_count_after = belief.hidden_card_count();
        trace.steps.push(AutoplayStep {
            step_index,
            chosen_move,
            revealed_card,
            hidden_count_before,
            hidden_count_after,
            planner: decision.snapshot,
        });
    }

    let termination = if true_state.visible.is_structural_win() {
        AutoplayTermination::Win
    } else {
        AutoplayTermination::StepLimit
    };
    Ok(AutoplayResult {
        won: termination == AutoplayTermination::Win,
        termination,
        trace,
        final_belief: belief,
        final_full_state: true_state,
        total_planner_time_ms,
        deterministic_nodes,
        root_visits,
        late_exact_triggers,
    })
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(crate) struct AutoplayDecision {
    pub(crate) best_move: Option<MacroMove>,
    pub(crate) snapshot: AutoplayPlannerSnapshot,
}

pub(crate) fn recommend_autoplay_move(
    belief: &BeliefState,
    solver_config: &SolverConfig,
    backend: PlannerBackend,
    pimc_config: PimcConfig,
    step_index: usize,
) -> SolverResult<AutoplayDecision> {
    match backend {
        PlannerBackend::Pimc => {
            let mut pimc = pimc_config;
            pimc.rng_seed = action_seed(pimc.rng_seed, step_index);
            let recommendation = recommend_move_pimc(
                belief,
                deterministic_search_config_from_solver(solver_config),
                pimc,
            )?;
            let root_visits = recommendation
                .action_stats
                .iter()
                .map(|stats| stats.visits as u64)
                .sum();
            Ok(AutoplayDecision {
                best_move: recommendation.best_move,
                snapshot: AutoplayPlannerSnapshot {
                    backend,
                    best_value: recommendation.best_value,
                    elapsed_ms: recommendation.elapsed_ms,
                    deterministic_nodes: recommendation.deterministic_nodes,
                    root_visits,
                    late_exact_triggered: false,
                },
            })
        }
        PlannerBackend::BeliefUct | PlannerBackend::BeliefUctLateExact => {
            let mut config = solver_config.clone();
            config.belief_planner.rng_seed =
                action_seed(config.belief_planner.rng_seed, step_index);
            if backend == PlannerBackend::BeliefUct {
                config.late_exact.enabled = false;
            } else {
                config.late_exact.enabled = true;
            }

            let recommendation =
                recommend_move_belief_uct(belief, &config, &config.belief_planner)?;
            let snapshot = planner_snapshot_from_recommendation(backend, &recommendation);
            Ok(AutoplayDecision {
                best_move: recommendation.best_move,
                snapshot,
            })
        }
    }
}

fn planner_snapshot_from_recommendation(
    backend: PlannerBackend,
    recommendation: &PlannerRecommendation,
) -> AutoplayPlannerSnapshot {
    AutoplayPlannerSnapshot {
        backend,
        best_value: recommendation.best_value,
        elapsed_ms: recommendation.elapsed_ms,
        deterministic_nodes: recommendation
            .deterministic_nodes
            .saturating_add(recommendation.late_exact_deterministic_nodes),
        root_visits: recommendation.simulations_run as u64,
        late_exact_triggered: recommendation.late_exact_triggered,
    }
}
