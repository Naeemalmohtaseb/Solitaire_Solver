//! Core library for a Draw-3 Klondike Solitaire solver under hidden tableau information.
//!
//! This crate intentionally starts as architecture scaffolding. It defines the module
//! boundaries, configuration surfaces, and domain-facing types that later prompts will
//! extend with move generation, deterministic solving, belief planning, and experiments.

#![deny(unsafe_code)]

pub mod belief;
pub mod cards;
pub mod closure;
pub mod config;
pub mod core;
pub mod deterministic_solver;
pub mod error;
pub mod experiments;
pub mod hashing_tt;
pub mod late_exact;
pub mod ml;
pub mod moves;
pub mod planner;
pub mod session;
pub mod stock;
pub mod types;

pub use belief::{
    apply_atomic_move_belief_nonreveal, apply_belief_transition, apply_observed_belief_move,
    belief_from_full_state, expand_reveal_frontier, hidden_slots_for_belief,
    requires_reveal_from_belief, sample_full_state, sample_full_states,
    validate_belief_against_full_state, validate_sample_against_belief,
    BeliefActionEvaluationContext, BeliefTransition, BeliefTransitionKind, DeterminizationSample,
    PreparedWorldSampler, RevealEvent, RevealFrontier, RevealOutcome, WorldSampler,
};
pub use cards::{Card, Color, Rank, Suit};
pub use closure::{
    ClosureConfig, ClosureEngine, ClosureReason, ClosureResult, ClosureStep, ClosureStopReason,
    ClosureTranscript,
};
pub use config::{DeterministicSolverConfig, ExperimentConfig, SearchConfig, SolverConfig};
pub use core::{
    BeliefState, FoundationState, FullState, HiddenAssignment, HiddenAssignments, HiddenSlot,
    TableauColumn, UnseenCardSet, VisibleState,
};
pub use deterministic_solver::{
    evaluate_fast, solve_bounded, solve_exact, BoundedSolveResult, DeterministicBound,
    DeterministicHashKey, DeterministicSearchConfig, DeterministicSearchStats, DeterministicSolver,
    DeterministicTt, DeterministicTtConfig, DeterministicTtEntry, DeterministicTtValue,
    EvaluatorWeights, ExactSolveResult, FastEvalResult, OpenCardRecommendation, ProofStatus,
    SolveBudget, SolveMode, SolveOutcome,
};
pub use error::{SolverError, SolverResult};
pub use experiments::{
    balanced_benchmark, balanced_vnet_benchmark, belief_uct_default, belief_uct_late_exact,
    compare_experiment_presets_on_suite, compare_named_presets_on_suite,
    compare_oracle_reference_results, compare_oracle_results, compare_vnet_leaf_mode_on_suite,
    evaluate_oracle_cases, experiment_preset_by_name, export_autoplay_benchmark_csv,
    export_autoplay_benchmark_json, export_autoplay_comparison_csv,
    export_autoplay_comparison_json, export_autoplay_game_csv, fast_benchmark, fast_vnet_benchmark,
    load_oracle_case_pack, load_oracle_local_evaluation, load_oracle_reference_results,
    load_regression_pack, oracle_cases_from_autoplay_trace, oracle_cases_from_full_states,
    oracle_cases_from_seeded_suite, parse_oracle_reference_results, pimc_baseline,
    play_game_with_planner, quality_benchmark, quality_vnet_benchmark, recommend_move_pimc,
    recommend_move_pimc_with_vnet, regression_case_from_autoplay_state,
    regression_case_from_belief_state, regression_case_from_full_state,
    regression_case_from_oracle_comparison, regression_pack_from_autoplay_result,
    regression_pack_from_benchmark_suite, regression_pack_from_session, run_autoplay_benchmark,
    run_autoplay_paired_comparison, run_autoplay_repeated_comparison, run_regression_pack,
    save_oracle_case_pack, save_oracle_local_evaluation, save_regression_pack,
    AutoplayBenchmarkConfig, AutoplayBenchmarkRecord, AutoplayBenchmarkResult,
    AutoplayBenchmarkSummaryReport, AutoplayComparisonResult, AutoplayComparisonSummaryReport,
    AutoplayConfig, AutoplayPairedGameResult, AutoplayPlannerSnapshot,
    AutoplayRepeatedComparisonResult, AutoplayRepetitionSummary, AutoplayResult, AutoplayStep,
    AutoplaySuiteSummary, AutoplayTermination, AutoplayTerminationCount, AutoplayTrace,
    BenchmarkConfig, BenchmarkDeal, BenchmarkRecord, BenchmarkResult, BenchmarkSuite,
    BenchmarkSuiteDescription, BenchmarkSummary, BenchmarkSummaryReport, EngineConfigLabel,
    ExperimentPreset, ExperimentRunner, OracleCase, OracleCasePack, OracleCaseProvenance,
    OracleCaseResult, OracleComparisonRecord, OracleComparisonSummary, OracleEvaluationConfig,
    OracleEvaluationMode, OracleLocalEvaluation, OracleMismatchCount, OracleMismatchKind,
    OracleReferenceResult, OracleReferenceResultSet, PairedComparisonResult,
    PairedComparisonSummary, PairedDealResult, PimcActionStats, PimcActionValue, PimcConfig,
    PimcEvaluationMode, PimcEvaluator, PimcRecommendation, PimcWorldBatch, PlannerBackend,
    PresetComparisonEntry, PresetComparisonSummary, PresetRankingMetric, RegressionCase,
    RegressionCaseData, RegressionCaseKind, RegressionCaseProvenance, RegressionCaseResult,
    RegressionExpectation, RegressionMismatch, RegressionMismatchCount, RegressionMismatchKind,
    RegressionObserved, RegressionPack, RegressionPackMetadata, RegressionPackSummary,
    RegressionRunConfig, RegressionRunResult, RepeatedComparisonResult, RepetitionSummary,
    VNetImpactSummary, EXPERIMENT_PRESET_NAMES, ORACLE_CASE_SCHEMA_VERSION,
    ORACLE_RESULT_SCHEMA_VERSION, REGRESSION_PACK_SCHEMA_VERSION,
};
pub use late_exact::{
    assignment_count_for_belief, enumerate_hidden_assignments, AssignmentPrefix,
    LateExactActionStats, LateExactConfig, LateExactEvaluationMode, LateExactEvaluator,
    LateExactResult,
};
pub use ml::{
    collect_vnet_examples_from_autoplay_suite, export_vnet_dataset_from_autoplay_suite,
    vnet_example_from_deterministic_solve, vnet_example_from_full_state, DatasetFormat,
    DatasetMetadata, DatasetSplit, DatasetSplitStrategy, EncodedFullState, EncodedStateShape,
    LeafEvaluationMode, ModelRole, TrainingExampleKind, VNetActivation, VNetBackend,
    VNetDataSource, VNetDataset, VNetDatasetMetadata, VNetDatasetRecord, VNetDatasetWriter,
    VNetEvaluator, VNetExample, VNetExportConfig, VNetInferenceArtifact, VNetInferenceConfig,
    VNetLabelMode, VNetLayerArtifact, VNetProvenance, VNetStateEncoding,
};
pub use planner::{
    recommend_move_belief_uct, recommend_move_belief_uct_parallel,
    recommend_move_belief_uct_with_reuse, BeliefPlanner, BeliefPlannerConfig, BeliefStateKey,
    CachedActionChild, CachedRecommendation, CachedRevealChild, ContinuationResult,
    PlannerActionStats, PlannerConfigFingerprint, PlannerContinuation, PlannerLeafEvalMode,
    PlannerNodeBudget, PlannerRecommendation, PlannerReuseContext, ReuseDiagnostics, ReuseOutcome,
    RootActionCache,
};
pub use session::{
    load_current_game_session, load_session, replay_session, save_current_game_session,
    save_session, ReplayMismatch, ReplayResult, SavedSession, SessionMetadata, SessionMoveRecord,
    SessionPlannerSnapshot, SessionRecord, SessionRevealRecord, SessionSnapshot, SessionStep,
    SessionSummary, SubtreeReuseHint, SESSION_SCHEMA_VERSION,
};
pub use types::{
    ActionEvaluation, ColumnId, ConfidenceInterval, DealSeed, FoundationId, MoveId,
    MoveRecommendation, SearchNodeId, SearchSummary, SessionId, ValueEstimate,
};

/// Crate version reported by the CLI and diagnostic output.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// A concise architecture summary suitable for CLI diagnostics.
pub fn architecture_summary() -> &'static str {
    "Draw-3 Klondike solver backend: Rust core, exact known stock/waste order, hidden \
     uncertainty only in face-down tableau cards, deterministic open-card solver at the \
     center, belief-state reveal planning above it, late-game exact assignment mode, and \
     built-in benchmarking."
}
