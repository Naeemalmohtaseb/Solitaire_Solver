//! Root-parallel orchestration for the belief planner.
//!
//! Workers are deliberately independent: each receives a cloned root belief and
//! a deterministic seed/budget slice. This module merges only root action
//! statistics after workers complete; it does not share a deep search tree.

use std::{thread, time::Instant};

use crate::{
    core::BeliefState,
    deterministic_solver::ordered_macro_moves,
    error::{SolverError, SolverResult},
    moves::MacroMove,
    types::DealSeed,
    SolverConfig,
};

use super::{
    deterministic_search_config, finalize_root_action_states,
    recommend_move_belief_uct_single_worker, run_late_exact_if_eligible, BeliefPlannerConfig,
    PlannerActionStats, PlannerRecommendation, RootActionState,
};

pub(super) fn recommend_move_belief_uct_root_parallel(
    belief: &BeliefState,
    solver_config: &SolverConfig,
    planner_config: &BeliefPlannerConfig,
) -> SolverResult<PlannerRecommendation> {
    let started = Instant::now();
    belief.validate_consistency_against_visible()?;

    let deterministic_config = deterministic_search_config(solver_config, planner_config);
    let candidates = ordered_macro_moves(&belief.visible, deterministic_config);
    if candidates.len() <= 1 {
        return recommend_move_belief_uct_single_worker(belief, solver_config, planner_config);
    }

    let worker_budgets = root_parallel_worker_budgets(planner_config)?;
    if worker_budgets.len() <= 1 {
        return recommend_move_belief_uct_single_worker(belief, solver_config, planner_config);
    }

    let mut handles = Vec::with_capacity(worker_budgets.len());
    for (worker_index, worker_budget) in worker_budgets.iter().copied().enumerate() {
        let worker_belief = belief.clone();
        let mut worker_solver = solver_config.clone();
        worker_solver.late_exact.enabled = false;

        let mut worker_config = *planner_config;
        worker_config.enable_root_parallel = false;
        worker_config.root_workers = 1;
        worker_config.worker_simulation_budget = None;
        worker_config.simulation_budget = worker_budget;
        worker_config.rng_seed = worker_seed(planner_config, worker_index);

        handles.push(thread::spawn(move || {
            recommend_move_belief_uct_single_worker(&worker_belief, &worker_solver, &worker_config)
        }));
    }

    let mut worker_results = Vec::with_capacity(handles.len());
    for handle in handles {
        match handle.join() {
            Ok(result) => worker_results.push(result?),
            Err(_) => {
                return Err(SolverError::InvalidState(
                    "root-parallel belief planner worker panicked".to_string(),
                ));
            }
        }
    }

    aggregate_parallel_recommendations(
        belief,
        solver_config,
        planner_config,
        candidates,
        worker_results,
        started,
    )
}

fn root_parallel_worker_budgets(planner_config: &BeliefPlannerConfig) -> SolverResult<Vec<usize>> {
    let requested_workers = planner_config.root_workers.max(1);
    if let Some(worker_budget) = planner_config.worker_simulation_budget {
        if worker_budget == 0 {
            return Err(SolverError::InvalidState(
                "belief planner worker_simulation_budget must be greater than zero".to_string(),
            ));
        }
        return Ok(vec![worker_budget; requested_workers]);
    }

    if planner_config.simulation_budget == 0 {
        return Err(SolverError::InvalidState(
            "belief planner simulation_budget must be greater than zero".to_string(),
        ));
    }

    let worker_count = requested_workers.min(planner_config.simulation_budget);
    let base = planner_config.simulation_budget / worker_count;
    let remainder = planner_config.simulation_budget % worker_count;
    Ok((0..worker_count)
        .map(|index| base + usize::from(index < remainder))
        .collect())
}

fn worker_seed(planner_config: &BeliefPlannerConfig, worker_index: usize) -> DealSeed {
    DealSeed(
        planner_config.rng_seed.0.wrapping_add(
            planner_config
                .worker_seed_stride
                .wrapping_mul(worker_index as u64),
        ),
    )
}

fn aggregate_parallel_recommendations(
    belief: &BeliefState,
    solver_config: &SolverConfig,
    planner_config: &BeliefPlannerConfig,
    candidates: Vec<MacroMove>,
    worker_results: Vec<PlannerRecommendation>,
    started: Instant,
) -> SolverResult<PlannerRecommendation> {
    if worker_results.is_empty() {
        return Err(SolverError::InvalidState(
            "root-parallel belief planner produced no worker results".to_string(),
        ));
    }

    let mut merged_stats = candidates
        .into_iter()
        .map(|action| {
            let mut stats = PlannerActionStats::new(action);
            stats.active = false;
            stats.narrowed_out = true;
            stats
        })
        .collect::<Vec<_>>();

    for recommendation in &worker_results {
        for worker_stats in &recommendation.action_stats {
            let target = merged_stats
                .iter_mut()
                .find(|candidate| candidate.action == worker_stats.action)
                .ok_or_else(|| {
                    SolverError::InvalidState(
                        "root-parallel worker returned an unknown root action".to_string(),
                    )
                })?;
            merge_planner_action_stats(target, worker_stats);
        }
    }

    let confidence_z = if planner_config.aggregate_confidence_stats {
        planner_config.confidence_z
    } else {
        0.0
    };
    for stats in &mut merged_stats {
        stats.variance = if stats.visits > 1 {
            stats.m2 / (stats.visits - 1) as f64
        } else {
            0.0
        };
        stats.standard_error = if stats.visits > 1 {
            (stats.variance / stats.visits as f64).sqrt()
        } else {
            0.0
        };
        stats.refresh_confidence(confidence_z);
    }

    let mut root_actions = merged_stats
        .into_iter()
        .map(|stats| RootActionState {
            active: stats.active,
            stats,
            deterministic_child: None,
            root_reveal_frontier: None,
            next_reveal_child: 0,
            covered_reveal_children: Vec::new(),
        })
        .collect::<Vec<_>>();

    let late_exact_result = run_late_exact_if_eligible(belief, &mut root_actions, solver_config)?;
    finalize_root_action_states(&mut root_actions, confidence_z);

    let active_root_actions = root_actions.iter().filter(|state| state.active).count();
    let reveal_frontier_children_covered = root_actions
        .iter()
        .map(|state| state.stats.reveal_frontier_children_covered as u64)
        .sum::<u64>();
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
    let worker_simulations = worker_results
        .iter()
        .map(|result| result.simulations_run)
        .collect::<Vec<_>>();

    Ok(PlannerRecommendation {
        best_move,
        best_value,
        action_stats: stats,
        simulations_run: worker_simulations.iter().sum(),
        early_stop_triggered: worker_results
            .iter()
            .any(|result| result.early_stop_triggered),
        actions_narrowed_out: worker_results
            .iter()
            .map(|result| result.actions_narrowed_out)
            .sum(),
        second_reveal_refinement_ran: worker_results
            .iter()
            .any(|result| result.second_reveal_refinement_ran),
        second_reveal_simulations_run: worker_results
            .iter()
            .map(|result| result.second_reveal_simulations_run)
            .sum(),
        no_legal_moves: false,
        leaf_evaluations: worker_results
            .iter()
            .map(|result| result.leaf_evaluations)
            .sum(),
        deterministic_nodes: worker_results
            .iter()
            .map(|result| result.deterministic_nodes)
            .sum(),
        vnet_inferences: worker_results
            .iter()
            .map(|result| result.vnet_inferences)
            .sum(),
        vnet_fallbacks: worker_results
            .iter()
            .map(|result| result.vnet_fallbacks)
            .sum(),
        vnet_inference_elapsed_us: worker_results
            .iter()
            .map(|result| result.vnet_inference_elapsed_us)
            .sum(),
        closure_steps_applied: worker_results
            .iter()
            .map(|result| result.closure_steps_applied)
            .sum(),
        reveal_branches_expanded: worker_results
            .iter()
            .map(|result| result.reveal_branches_expanded)
            .sum(),
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
        belief_transition_elapsed_us: worker_results
            .iter()
            .map(|result| result.belief_transition_elapsed_us)
            .sum(),
        reveal_expansion_elapsed_us: worker_results
            .iter()
            .map(|result| result.reveal_expansion_elapsed_us)
            .sum(),
        leaf_eval_elapsed_us: worker_results
            .iter()
            .map(|result| result.leaf_eval_elapsed_us)
            .sum(),
        deterministic_eval_elapsed_us: worker_results
            .iter()
            .map(|result| result.deterministic_eval_elapsed_us)
            .sum(),
        root_parallel_used: true,
        root_parallel_workers: worker_results.len(),
        root_parallel_worker_simulations: worker_simulations,
        elapsed_ms: started.elapsed().as_millis() as u64,
    })
}

pub(super) fn merge_planner_action_stats(
    target: &mut PlannerActionStats,
    source: &PlannerActionStats,
) {
    target.active |= source.active;
    target.narrowed_out &= source.narrowed_out;
    target.win_like_count = target.win_like_count.saturating_add(source.win_like_count);
    target.reveal_branches_expanded = target
        .reveal_branches_expanded
        .saturating_add(source.reveal_branches_expanded);
    target.reveal_frontier_children = target
        .reveal_frontier_children
        .max(source.reveal_frontier_children);
    target.reveal_frontier_children_covered = target
        .reveal_frontier_children_covered
        .max(source.reveal_frontier_children_covered);
    target.second_reveal_refinement_visits = target
        .second_reveal_refinement_visits
        .saturating_add(source.second_reveal_refinement_visits);
    target.late_exact_evaluated |= source.late_exact_evaluated;
    target.late_exact_assignments_enumerated = target
        .late_exact_assignments_enumerated
        .saturating_add(source.late_exact_assignments_enumerated);
    target.late_exact_value = target.late_exact_value.or(source.late_exact_value);

    if source.visits == 0 {
        return;
    }
    if target.visits == 0 {
        target.visits = source.visits;
        target.mean_value = source.mean_value;
        target.m2 = source.m2;
        target.variance = source.variance;
        target.standard_error = source.standard_error;
        return;
    }

    let left_visits = target.visits as f64;
    let right_visits = source.visits as f64;
    let total_visits = left_visits + right_visits;
    let delta = source.mean_value - target.mean_value;
    target.mean_value += delta * right_visits / total_visits;
    target.m2 += source.m2 + delta * delta * left_visits * right_visits / total_visits;
    target.visits += source.visits;
}
