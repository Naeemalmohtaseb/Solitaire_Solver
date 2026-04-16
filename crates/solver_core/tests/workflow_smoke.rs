use solver_core::{
    collect_vnet_examples_from_autoplay_suite, play_game_with_planner,
    regression_pack_from_benchmark_suite, replay_session, run_autoplay_benchmark,
    run_autoplay_paired_comparison, run_regression_pack, save_session, BenchmarkSuite,
    DeterministicSearchConfig, ExperimentRunner, RegressionRunConfig, SessionId, SessionMetadata,
    SessionRecord, SolveBudget, VNetExportConfig,
};

#[test]
fn public_workflow_smoke_exercises_exports_sessions_benchmarks_and_regressions() {
    let suite = BenchmarkSuite::from_base_seed("workflow-smoke", 901, 1);
    let mut preset = solver_core::fast_benchmark();
    preset.autoplay.max_steps = 0;
    preset.solver.belief_planner.simulation_budget = 2;
    preset.solver.belief_planner.enable_second_reveal_refinement = false;

    let benchmark = run_autoplay_benchmark(&suite, &preset.autoplay_benchmark_config()).unwrap();
    assert_eq!(benchmark.games, 1);

    let comparison = run_autoplay_paired_comparison(
        &suite,
        &preset.autoplay_benchmark_config(),
        &preset.autoplay_benchmark_config(),
    )
    .unwrap();
    assert_eq!(comparison.same_outcome_count, 1);

    let dataset = collect_vnet_examples_from_autoplay_suite(
        &suite,
        &preset,
        &VNetExportConfig {
            max_steps: Some(0),
            ..VNetExportConfig::default()
        },
    )
    .unwrap();
    assert_eq!(dataset.metadata.games, 1);
    assert!(!dataset.examples.is_empty());

    let deal = ExperimentRunner
        .generate_deal(suite.seeds[0])
        .expect("seeded deal should generate");
    let autoplay =
        play_game_with_planner(&deal.full_state, &preset.solver, &preset.autoplay).unwrap();
    let session = SessionRecord::from_autoplay_result(
        SessionMetadata::new(SessionId(901), Some("workflow-smoke".to_string())),
        deal.full_state,
        &autoplay,
    )
    .unwrap();
    let path = std::env::temp_dir().join(format!(
        "solitaire-workflow-smoke-{}-{}.json",
        std::process::id(),
        solver_core::VERSION
    ));
    let _ = std::fs::remove_file(&path);
    save_session(&path, &session).unwrap();
    let loaded = solver_core::load_session(&path).unwrap();
    let replay = replay_session(&loaded).unwrap();
    assert!(replay.matched);
    let _ = std::fs::remove_file(path);

    let regression_config = RegressionRunConfig {
        preset,
        deterministic_mode: solver_core::OracleEvaluationMode::Fast,
        deterministic_override: Some(DeterministicSearchConfig {
            budget: SolveBudget {
                node_budget: Some(8),
                depth_budget: Some(1),
                wall_clock_limit_ms: None,
            },
            ..DeterministicSearchConfig::default()
        }),
    };
    let pack = regression_pack_from_benchmark_suite(
        &suite,
        &regression_config,
        "workflow-smoke",
        vec!["smoke".to_string()],
    )
    .unwrap();
    let regression = run_regression_pack(&pack, &regression_config).unwrap();
    assert_eq!(regression.failed, 0);
}
