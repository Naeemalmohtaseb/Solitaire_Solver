use super::*;
use crate::{
    belief::{apply_belief_transition, apply_observed_belief_move},
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
    let recommendation = recommend_move_belief_uct(&belief, &solver_config(), &planner).unwrap();

    let visits = recommendation
        .action_stats
        .iter()
        .map(|stats| stats.visits)
        .sum::<usize>();

    assert_eq!(recommendation.simulations_run, planner.simulation_budget);
    assert_eq!(visits, planner.simulation_budget);
}

#[test]
fn root_parallel_recommendation_is_reproducible_with_fixed_seed() {
    let belief = reveal_belief();
    let mut planner = planner_config(45);
    planner.enable_root_parallel = true;
    planner.root_workers = 2;
    planner.simulation_budget = 6;
    planner.worker_seed_stride = 17;

    let first = recommend_move_belief_uct(&belief, &solver_config(), &planner).unwrap();
    let second = recommend_move_belief_uct(&belief, &solver_config(), &planner).unwrap();

    assert!(first.root_parallel_used);
    assert_eq!(first.root_parallel_workers, 2);
    assert_eq!(first.root_parallel_worker_simulations, vec![3, 3]);
    assert_eq!(first.best_move, second.best_move);
    assert_eq!(first.action_stats, second.action_stats);
    assert_eq!(first.simulations_run, second.simulations_run);
}

#[test]
fn root_parallel_returns_legal_move() {
    let belief = reveal_belief();
    let config = solver_config();
    let mut planner = planner_config(46);
    planner.enable_root_parallel = true;
    planner.root_workers = 2;
    planner.simulation_budget = 6;
    let recommendation = recommend_move_belief_uct(&belief, &config, &planner).unwrap();
    let legal = ordered_macro_moves(
        &belief.visible,
        deterministic_search_config(&config, &planner),
    );

    assert!(recommendation.root_parallel_used);
    assert!(legal
        .iter()
        .any(|legal| Some(legal) == recommendation.best_move.as_ref()));
}

#[test]
fn root_parallel_single_worker_matches_normal_planner_surface() {
    let belief = reveal_belief();
    let config = solver_config();
    let normal = planner_config(47);
    let mut single_worker = normal;
    single_worker.enable_root_parallel = true;
    single_worker.root_workers = 1;

    let normal_result = recommend_move_belief_uct(&belief, &config, &normal).unwrap();
    let single_worker_result =
        recommend_move_belief_uct_parallel(&belief, &config, &single_worker).unwrap();

    assert!(!single_worker_result.root_parallel_used);
    assert_eq!(normal_result.best_move, single_worker_result.best_move);
    assert_eq!(
        normal_result.action_stats,
        single_worker_result.action_stats
    );
    assert_eq!(
        normal_result.simulations_run,
        single_worker_result.simulations_run
    );
}

#[test]
fn root_parallel_stat_merge_combines_running_variance() {
    let action = ordered_macro_moves(
        &reveal_belief().visible,
        deterministic_search_config(&solver_config(), &planner_config(48)),
    )
    .into_iter()
    .next()
    .unwrap();
    let mut left = PlannerActionStats::new(action.clone());
    left.record(1.0, 2);
    left.record(0.0, 2);
    let mut right = PlannerActionStats::new(action);
    right.record(1.0, 3);

    super::parallel::merge_planner_action_stats(&mut left, &right);
    left.refresh_confidence(1.96);

    assert_eq!(left.visits, 3);
    assert!((left.mean_value - (2.0 / 3.0)).abs() < 0.000_001);
    assert_eq!(left.reveal_branches_expanded, 7);
    assert_eq!(left.win_like_count, 2);
    assert!(left.variance > 0.0);
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
    let recommendation = recommend_move_belief_uct(&belief, &solver_config(), &planner).unwrap();

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
    let recommendation = recommend_move_belief_uct(&belief, &solver_config(), &planner).unwrap();
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
    let enabled_result = recommend_move_belief_uct(&belief, &solver_config(), &enabled).unwrap();

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

    let recommendation = recommend_move_belief_uct(&belief, &solver, &planner_config(61)).unwrap();

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

    let recommendation = recommend_move_belief_uct(&belief, &solver, &planner_config(62)).unwrap();

    assert!(!recommendation.late_exact_triggered);
    assert_eq!(recommendation.late_exact_actions_evaluated, 0);
    assert_eq!(recommendation.late_exact_assignments_enumerated, 0);
}

#[test]
fn exact_current_root_reuses_cached_recommendation() {
    let belief = reveal_belief();
    let solver = solver_config();
    let planner = planner_config(70);
    let context = PlannerReuseContext {
        session_id: Some(crate::types::SessionId(7)),
        backend_tag: Some("belief_uct".to_string()),
        preset_name: Some("unit".to_string()),
        ..PlannerReuseContext::default()
    };

    let first =
        recommend_move_belief_uct_with_reuse(&belief, &solver, &planner, None, context.clone())
            .unwrap();
    let second = recommend_move_belief_uct_with_reuse(
        &belief,
        &solver,
        &planner,
        Some(&first.continuation),
        context,
    )
    .unwrap();

    assert_eq!(second.reuse.outcome, ReuseOutcome::CurrentRootCache);
    assert!(second.reuse.succeeded);
    assert!(!second.reuse.fallback);
    assert_eq!(second.recommendation, first.recommendation);
    assert_eq!(
        second.reuse.reused_stats_count,
        first.recommendation.action_stats.len()
    );
}

#[test]
fn followed_recommended_move_matches_cached_child() {
    let belief = nonreveal_belief();
    let solver = solver_config();
    let planner = planner_config(71);
    let first = recommend_move_belief_uct_with_reuse(
        &belief,
        &solver,
        &planner,
        None,
        PlannerReuseContext::default(),
    )
    .unwrap();
    let action = first
        .recommendation
        .best_move
        .clone()
        .expect("non-reveal belief should have a best move");
    assert!(!action.semantics.causes_reveal);

    let (child_belief, _) = apply_observed_belief_move(&belief, action.atomic, None).unwrap();
    let continued = recommend_move_belief_uct_with_reuse(
        &child_belief,
        &solver,
        &planner,
        Some(&first.continuation),
        PlannerReuseContext {
            applied_move: Some(action),
            ..PlannerReuseContext::default()
        },
    )
    .unwrap();

    assert_eq!(continued.reuse.outcome, ReuseOutcome::FollowedMove);
    assert!(continued.reuse.succeeded);
    assert!(continued.reuse.fallback);
    assert!(continued.reuse.reused_action_count > 0);
}

#[test]
fn observed_reveal_matches_cached_reveal_child() {
    let belief = reveal_belief();
    let solver = solver_config();
    let planner = planner_config(72);
    let first = recommend_move_belief_uct_with_reuse(
        &belief,
        &solver,
        &planner,
        None,
        PlannerReuseContext::default(),
    )
    .unwrap();
    let reveal_action = first
        .recommendation
        .action_stats
        .iter()
        .find(|stats| stats.action.semantics.causes_reveal)
        .map(|stats| stats.action.clone())
        .expect("root should contain reveal action");
    let revealed_card = belief.unseen_cards.iter().next().unwrap();
    let (child_belief, _) =
        apply_observed_belief_move(&belief, reveal_action.atomic, Some(revealed_card)).unwrap();

    let continued = recommend_move_belief_uct_with_reuse(
        &child_belief,
        &solver,
        &planner,
        Some(&first.continuation),
        PlannerReuseContext {
            applied_move: Some(reveal_action),
            observed_reveal: Some(revealed_card),
            ..PlannerReuseContext::default()
        },
    )
    .unwrap();

    assert_eq!(continued.reuse.outcome, ReuseOutcome::RevealChild);
    assert!(continued.reuse.succeeded);
    assert!(continued.reuse.reveal_children_reused >= belief.unseen_card_count());
}

#[test]
fn deviation_falls_back_safely() {
    let belief = reveal_belief();
    let other = no_move_belief();
    let solver = solver_config();
    let planner = planner_config(73);
    let first = recommend_move_belief_uct_with_reuse(
        &belief,
        &solver,
        &planner,
        None,
        PlannerReuseContext::default(),
    )
    .unwrap();

    let continued = recommend_move_belief_uct_with_reuse(
        &other,
        &solver,
        &planner,
        Some(&first.continuation),
        PlannerReuseContext::default(),
    )
    .unwrap();

    assert!(matches!(
        continued.reuse.outcome,
        ReuseOutcome::Fallback | ReuseOutcome::ColdStart
    ));
    assert!(!continued.reuse.succeeded);
    assert!(continued.reuse.fallback);
    assert!(continued.recommendation.no_legal_moves);
}

#[test]
fn autoplay_benchmark_accepts_root_parallel_config() {
    let mut preset = crate::experiments::fast_benchmark();
    preset.autoplay.backend = crate::experiments::PlannerBackend::BeliefUct;
    preset.autoplay.max_steps = 1;
    preset.solver.belief_planner.enable_root_parallel = true;
    preset.solver.belief_planner.root_workers = 2;
    preset.solver.belief_planner.simulation_budget = 4;
    preset.solver.belief_planner.enable_second_reveal_refinement = false;
    let suite = crate::experiments::BenchmarkSuite::from_base_seed("parallel-smoke", 88, 1);
    let result =
        crate::experiments::run_autoplay_benchmark(&suite, &preset.autoplay_benchmark_config())
            .unwrap();

    assert_eq!(result.games, 1);
    assert!(result.records[0].root_parallel_steps <= result.records[0].moves_played);
}
