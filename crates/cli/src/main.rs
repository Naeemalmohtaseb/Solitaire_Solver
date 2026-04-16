//! Command-line entry point for diagnostics and reproducible benchmarks.

use std::{
    error::Error,
    fs,
    io::{Error as IoError, ErrorKind},
    path::{Path, PathBuf},
};

use clap::{Args, Parser, Subcommand, ValueEnum};

use solver_core::{
    collect_vnet_examples_from_autoplay_suite, compare_experiment_presets_on_suite,
    compare_oracle_results, evaluate_oracle_cases, experiment_preset_by_name,
    load_oracle_case_pack, load_oracle_local_evaluation, load_oracle_reference_results,
    load_regression_pack, load_session, oracle_cases_from_seeded_suite, play_game_with_planner,
    regression_pack_from_benchmark_suite, regression_pack_from_session, replay_session,
    run_autoplay_benchmark, run_autoplay_paired_comparison, run_autoplay_repeated_comparison,
    run_regression_pack, save_oracle_case_pack, save_oracle_local_evaluation, save_regression_pack,
    save_session, AutoplayBenchmarkResult, AutoplayComparisonResult,
    AutoplayRepeatedComparisonResult, BenchmarkSuite, DatasetFormat, DealSeed,
    DeterministicSearchConfig, ExperimentPreset, ExperimentRunner, LeafEvaluationMode,
    OracleComparisonSummary, OracleEvaluationConfig, OracleEvaluationMode, PresetComparisonSummary,
    PresetRankingMetric, RegressionPack, RegressionPackSummary, RegressionRunConfig,
    RegressionRunResult, SessionMetadata, SessionRecord, SessionSummary, SolveBudget, VNetDataset,
    VNetDatasetWriter, VNetExportConfig, VNetLabelMode, EXPERIMENT_PRESET_NAMES,
};

type CliResult<T> = Result<T, Box<dyn Error>>;

/// Command-line entry point for the Solitaire solver backend.
#[derive(Debug, Parser)]
#[command(name = "solitaire-cli")]
#[command(about = "Draw-3 Klondike solver backend CLI")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

/// Supported CLI commands.
#[derive(Debug, Subcommand)]
enum Command {
    /// Print basic environment and crate health information.
    Doctor,
    /// Print the solver crate version.
    Version,
    /// Print the high-level architecture summary.
    PrintArchitecture,
    /// Run reproducible benchmark commands.
    Benchmark {
        #[command(subcommand)]
        command: BenchmarkCommand,
    },
    /// Export machine-learning datasets.
    Dataset {
        #[command(subcommand)]
        command: DatasetCommand,
    },
    /// Save, inspect, and replay persisted game sessions.
    Session {
        #[command(subcommand)]
        command: SessionCommand,
    },
    /// Export and compare deterministic-solver oracle cases.
    Oracle {
        #[command(subcommand)]
        command: OracleCommand,
    },
    /// Create and run curated regression packs.
    Regression {
        #[command(subcommand)]
        command: RegressionCommand,
    },
}

/// Benchmark subcommands.
#[derive(Debug, Subcommand)]
enum BenchmarkCommand {
    /// Run one full-game autoplay benchmark.
    Autoplay(BenchmarkAutoplayArgs),
    /// Compare two presets on the same full-game deal suite.
    Compare(BenchmarkCompareArgs),
    /// Compare two presets over repeated deterministic full-game suites.
    RepeatedCompare(BenchmarkRepeatedCompareArgs),
    /// Compare several presets on one deterministic full-game suite.
    ComparePresets(BenchmarkComparePresetsArgs),
}

/// Dataset export subcommands.
#[derive(Debug, Subcommand)]
enum DatasetCommand {
    /// Export V-Net full-state supervised examples.
    ExportVnet(DatasetExportVnetArgs),
}

/// Session persistence subcommands.
#[derive(Debug, Subcommand)]
enum SessionCommand {
    /// Save a reproducible autoplay/debug session from a preset and seed.
    Save(SessionSaveArgs),
    /// Inspect a saved session.
    Inspect(SessionPathArgs),
    /// Print a compact session summary.
    Summary(SessionPathArgs),
    /// Replay a saved session and report mismatches.
    Replay(SessionPathArgs),
}

/// Offline oracle validation subcommands.
#[derive(Debug, Subcommand)]
enum OracleCommand {
    /// Export fully known deterministic cases for an external oracle.
    ExportCases(OracleExportCasesArgs),
    /// Evaluate exported cases with the local deterministic solver.
    EvaluateLocal(OracleEvaluateLocalArgs),
    /// Compare local deterministic results with external reference results.
    Compare(OracleCompareArgs),
}

/// Curated regression-pack subcommands.
#[derive(Debug, Subcommand)]
enum RegressionCommand {
    /// Create a deterministic open-card pack from a seeded benchmark suite.
    CreateFromBenchmark(RegressionCreateFromBenchmarkArgs),
    /// Create a replay pack from an existing session JSON file.
    CreateFromSession(RegressionCreateFromSessionArgs),
    /// Run a regression pack against the current solver.
    Run(RegressionRunArgs),
    /// Summarize a regression pack without running it.
    Summarize(RegressionSummarizeArgs),
}

/// Arguments shared by single-suite benchmark commands.
#[derive(Debug, Clone, Args)]
struct SuiteArgs {
    /// Number of seeded games to run.
    #[arg(long, default_value_t = 10)]
    games: usize,
    /// Base seed for deterministic suite generation.
    #[arg(long, default_value_t = 0)]
    seed: u64,
    /// Optional autoplay step cap override for smoke tests and quick runs.
    #[arg(long = "max-steps")]
    max_steps: Option<usize>,
}

/// File export arguments shared by summary-style commands.
#[derive(Debug, Clone, Default, Args)]
struct ExportArgs {
    /// Path for a JSON summary export.
    #[arg(long)]
    json: Option<PathBuf>,
    /// Path for a CSV summary export.
    #[arg(long)]
    csv: Option<PathBuf>,
}

/// Optional V-Net leaf evaluation overrides.
#[derive(Debug, Clone, Default, Args)]
struct LeafEvalArgs {
    /// Approximate leaf evaluator for deterministic cutoffs.
    #[arg(long = "leaf-eval-mode", value_enum)]
    leaf_eval_mode: Option<CliLeafEvalMode>,
    /// Rust-native V-Net inference artifact path.
    #[arg(long = "vnet-model")]
    vnet_model: Option<PathBuf>,
}

/// Arguments for `benchmark autoplay`.
#[derive(Debug, Clone, Args)]
struct BenchmarkAutoplayArgs {
    /// Preset name.
    #[arg(long)]
    preset: String,
    #[command(flatten)]
    suite: SuiteArgs,
    #[command(flatten)]
    export: ExportArgs,
    #[command(flatten)]
    leaf: LeafEvalArgs,
    /// Path for per-game CSV rows.
    #[arg(long = "game-csv")]
    game_csv: Option<PathBuf>,
}

/// Arguments for `benchmark compare`.
#[derive(Debug, Clone, Args)]
struct BenchmarkCompareArgs {
    /// Baseline preset name.
    #[arg(long)]
    baseline: String,
    /// Candidate preset name.
    #[arg(long)]
    candidate: String,
    #[command(flatten)]
    suite: SuiteArgs,
    #[command(flatten)]
    export: ExportArgs,
    /// Baseline leaf evaluator override.
    #[arg(long = "baseline-leaf-eval-mode", value_enum)]
    baseline_leaf_eval_mode: Option<CliLeafEvalMode>,
    /// Baseline V-Net inference artifact path.
    #[arg(long = "baseline-vnet-model")]
    baseline_vnet_model: Option<PathBuf>,
    /// Candidate leaf evaluator override.
    #[arg(long = "candidate-leaf-eval-mode", value_enum)]
    candidate_leaf_eval_mode: Option<CliLeafEvalMode>,
    /// Candidate V-Net inference artifact path.
    #[arg(long = "candidate-vnet-model")]
    candidate_vnet_model: Option<PathBuf>,
}

/// Arguments for `benchmark repeated-compare`.
#[derive(Debug, Clone, Args)]
struct BenchmarkRepeatedCompareArgs {
    /// Baseline preset name.
    #[arg(long)]
    baseline: String,
    /// Candidate preset name.
    #[arg(long)]
    candidate: String,
    /// Games per repetition.
    #[arg(long, default_value_t = 10)]
    games: usize,
    /// Number of deterministic repeated suites.
    #[arg(long, default_value_t = 3)]
    repetitions: usize,
    /// Base seed for deterministic repeated suite generation.
    #[arg(long, default_value_t = 0)]
    seed: u64,
    /// Optional autoplay step cap override for smoke tests and quick runs.
    #[arg(long = "max-steps")]
    max_steps: Option<usize>,
    #[command(flatten)]
    export: ExportArgs,
    /// Baseline leaf evaluator override.
    #[arg(long = "baseline-leaf-eval-mode", value_enum)]
    baseline_leaf_eval_mode: Option<CliLeafEvalMode>,
    /// Baseline V-Net inference artifact path.
    #[arg(long = "baseline-vnet-model")]
    baseline_vnet_model: Option<PathBuf>,
    /// Candidate leaf evaluator override.
    #[arg(long = "candidate-leaf-eval-mode", value_enum)]
    candidate_leaf_eval_mode: Option<CliLeafEvalMode>,
    /// Candidate V-Net inference artifact path.
    #[arg(long = "candidate-vnet-model")]
    candidate_vnet_model: Option<PathBuf>,
}

/// Ranking metric accepted by `benchmark compare-presets`.
#[derive(Debug, Copy, Clone, PartialEq, Eq, ValueEnum)]
enum CliRankingMetric {
    /// Rank by highest win rate.
    WinRate,
    /// Rank by lowest average planner time per game.
    Time,
    /// Rank by highest win rate per planner-second.
    Efficiency,
}

/// Approximate leaf evaluator accepted by benchmark commands.
#[derive(Debug, Copy, Clone, PartialEq, Eq, ValueEnum)]
enum CliLeafEvalMode {
    /// Existing handcrafted heuristic.
    Heuristic,
    /// Loaded V-Net inference artifact, with heuristic fallback.
    Vnet,
}

impl From<CliLeafEvalMode> for LeafEvaluationMode {
    fn from(value: CliLeafEvalMode) -> Self {
        match value {
            CliLeafEvalMode::Heuristic => Self::Heuristic,
            CliLeafEvalMode::Vnet => Self::VNet,
        }
    }
}

impl From<CliRankingMetric> for PresetRankingMetric {
    fn from(value: CliRankingMetric) -> Self {
        match value {
            CliRankingMetric::WinRate => Self::WinRate,
            CliRankingMetric::Time => Self::TimePerGame,
            CliRankingMetric::Efficiency => Self::Efficiency,
        }
    }
}

/// Arguments for `benchmark compare-presets`.
#[derive(Debug, Clone, Args)]
struct BenchmarkComparePresetsArgs {
    /// Comma-separated preset names. Defaults to every registered preset.
    #[arg(long)]
    presets: Option<String>,
    /// Number of seeded games to run per preset.
    #[arg(long, default_value_t = 10)]
    games: usize,
    /// Base seed for deterministic suite generation.
    #[arg(long, default_value_t = 0)]
    seed: u64,
    /// Optional autoplay step cap override for smoke tests and quick runs.
    #[arg(long = "max-steps")]
    max_steps: Option<usize>,
    /// Ranking metric for the printed/exported order.
    #[arg(long = "rank-by", value_enum, default_value_t = CliRankingMetric::Efficiency)]
    rank_by: CliRankingMetric,
    #[command(flatten)]
    export: ExportArgs,
    #[command(flatten)]
    leaf: LeafEvalArgs,
}

/// Label modes accepted by `dataset export-vnet`.
#[derive(Debug, Copy, Clone, PartialEq, Eq, ValueEnum)]
enum CliVNetLabelMode {
    /// Label examples by the final win/loss of the configured autoplay run.
    #[value(alias = "terminal")]
    TerminalOutcome,
    /// Label examples using the deterministic open-card solver's bounded value.
    #[value(alias = "deterministic")]
    DeterministicSolverValue,
    /// Label examples using planner root values where available.
    #[value(alias = "planner")]
    PlannerApproximateValue,
}

impl From<CliVNetLabelMode> for VNetLabelMode {
    fn from(value: CliVNetLabelMode) -> Self {
        match value {
            CliVNetLabelMode::TerminalOutcome => Self::TerminalOutcome,
            CliVNetLabelMode::DeterministicSolverValue => Self::DeterministicSolverValue,
            CliVNetLabelMode::PlannerApproximateValue => Self::PlannerBackedApproximateValue,
        }
    }
}

/// Dataset output formats accepted by `dataset export-vnet`.
#[derive(Debug, Copy, Clone, PartialEq, Eq, ValueEnum)]
enum CliDatasetFormat {
    /// JSON Lines output.
    Jsonl,
}

impl From<CliDatasetFormat> for DatasetFormat {
    fn from(value: CliDatasetFormat) -> Self {
        match value {
            CliDatasetFormat::Jsonl => Self::Jsonl,
        }
    }
}

/// Arguments for `dataset export-vnet`.
#[derive(Debug, Clone, Args)]
struct DatasetExportVnetArgs {
    /// Preset name.
    #[arg(long)]
    preset: String,
    /// Number of seeded games to run.
    #[arg(long, default_value_t = 10)]
    games: usize,
    /// Base seed for deterministic suite generation.
    #[arg(long, default_value_t = 0)]
    seed: u64,
    /// Output path.
    #[arg(long)]
    out: PathBuf,
    /// Label mode.
    #[arg(long = "label-mode", value_enum, default_value_t = CliVNetLabelMode::TerminalOutcome)]
    label_mode: CliVNetLabelMode,
    /// Output format.
    #[arg(long = "format", value_enum, default_value_t = CliDatasetFormat::Jsonl)]
    format: CliDatasetFormat,
    /// Optional autoplay step cap override for smoke-test dataset exports.
    #[arg(long = "max-steps")]
    max_steps: Option<usize>,
    /// Export every Nth decision state.
    #[arg(long = "decision-stride", default_value_t = 1)]
    decision_stride: usize,
}

/// Arguments for `session save`.
#[derive(Debug, Clone, Args)]
struct SessionSaveArgs {
    /// Preset name used to create an autoplay/debug session.
    #[arg(long)]
    preset: String,
    /// Deal seed.
    #[arg(long, default_value_t = 0)]
    seed: u64,
    /// Output session JSON path.
    #[arg(long)]
    out: PathBuf,
    /// Optional autoplay step cap.
    #[arg(long = "max-steps")]
    max_steps: Option<usize>,
    /// Optional human-readable label.
    #[arg(long)]
    label: Option<String>,
}

/// Path-only session command arguments.
#[derive(Debug, Clone, Args)]
struct SessionPathArgs {
    /// Session JSON path.
    #[arg(long)]
    path: PathBuf,
}

/// Oracle local evaluation modes accepted by CLI.
#[derive(Debug, Copy, Clone, PartialEq, Eq, ValueEnum)]
enum CliOracleEvaluationMode {
    /// Proof-oriented deterministic search.
    Exact,
    /// Bounded deterministic search.
    Bounded,
    /// Fast deterministic value evaluation.
    Fast,
}

impl From<CliOracleEvaluationMode> for OracleEvaluationMode {
    fn from(value: CliOracleEvaluationMode) -> Self {
        match value {
            CliOracleEvaluationMode::Exact => Self::Exact,
            CliOracleEvaluationMode::Bounded => Self::Bounded,
            CliOracleEvaluationMode::Fast => Self::Fast,
        }
    }
}

/// Arguments for `oracle export-cases`.
#[derive(Debug, Clone, Args)]
struct OracleExportCasesArgs {
    /// Preset name used for provenance metadata.
    #[arg(long)]
    preset: String,
    /// Number of seeded cases to export.
    #[arg(long, default_value_t = 10)]
    games: usize,
    /// Base seed for deterministic suite generation.
    #[arg(long, default_value_t = 0)]
    seed: u64,
    /// Output oracle case-pack JSON path.
    #[arg(long)]
    out: PathBuf,
}

/// Arguments for `oracle evaluate-local`.
#[derive(Debug, Clone, Args)]
struct OracleEvaluateLocalArgs {
    /// Input oracle case-pack JSON path.
    #[arg(long)]
    cases: PathBuf,
    /// Output local evaluation JSON path.
    #[arg(long)]
    out: PathBuf,
    /// Local deterministic evaluation mode.
    #[arg(long, value_enum, default_value_t = CliOracleEvaluationMode::Exact)]
    mode: CliOracleEvaluationMode,
    /// Optional node budget override.
    #[arg(long = "node-budget")]
    node_budget: Option<u64>,
    /// Optional depth budget override.
    #[arg(long = "depth-budget")]
    depth_budget: Option<u16>,
}

/// Arguments for `oracle compare`.
#[derive(Debug, Clone, Args)]
struct OracleCompareArgs {
    /// Local evaluation JSON path produced by `oracle evaluate-local`.
    #[arg(long)]
    local: PathBuf,
    /// External reference result file, JSON or JSONL.
    #[arg(long)]
    reference: PathBuf,
    #[command(flatten)]
    export: ExportArgs,
}

/// Arguments for `regression create-from-benchmark`.
#[derive(Debug, Clone, Args)]
struct RegressionCreateFromBenchmarkArgs {
    /// Preset name used to snapshot current expectations.
    #[arg(long)]
    preset: String,
    /// Number of seeded deterministic cases to capture.
    #[arg(long, default_value_t = 10)]
    games: usize,
    /// Base seed for deterministic suite generation.
    #[arg(long, default_value_t = 0)]
    seed: u64,
    /// Output regression-pack JSON path.
    #[arg(long)]
    out: PathBuf,
    /// Stable pack name.
    #[arg(long)]
    name: Option<String>,
    /// Case tag. May be repeated.
    #[arg(long = "tag")]
    tags: Vec<String>,
    /// Deterministic evaluation mode used to snapshot expectations.
    #[arg(long, value_enum, default_value_t = CliOracleEvaluationMode::Fast)]
    mode: CliOracleEvaluationMode,
    /// Optional deterministic node budget override.
    #[arg(long = "node-budget")]
    node_budget: Option<u64>,
    /// Optional deterministic depth budget override.
    #[arg(long = "depth-budget")]
    depth_budget: Option<u16>,
}

/// Arguments for `regression create-from-session`.
#[derive(Debug, Clone, Args)]
struct RegressionCreateFromSessionArgs {
    /// Input session JSON path.
    #[arg(long)]
    path: PathBuf,
    /// Output regression-pack JSON path.
    #[arg(long)]
    out: PathBuf,
    /// Stable pack name.
    #[arg(long)]
    name: Option<String>,
    /// Case tag. May be repeated.
    #[arg(long = "tag")]
    tags: Vec<String>,
}

/// Arguments for `regression run`.
#[derive(Debug, Clone, Args)]
struct RegressionRunArgs {
    /// Input regression-pack JSON path.
    #[arg(long)]
    pack: PathBuf,
    /// Preset name used for current solver checks.
    #[arg(long)]
    preset: String,
    /// Deterministic evaluation mode.
    #[arg(long, value_enum, default_value_t = CliOracleEvaluationMode::Fast)]
    mode: CliOracleEvaluationMode,
    /// Optional deterministic node budget override.
    #[arg(long = "node-budget")]
    node_budget: Option<u64>,
    /// Optional deterministic depth budget override.
    #[arg(long = "depth-budget")]
    depth_budget: Option<u16>,
    #[command(flatten)]
    export: ExportArgs,
}

/// Arguments for `regression summarize`.
#[derive(Debug, Clone, Args)]
struct RegressionSummarizeArgs {
    /// Input regression-pack JSON path.
    #[arg(long)]
    pack: PathBuf,
}

fn main() -> CliResult<()> {
    let cli = Cli::parse();

    match cli.command {
        Command::Doctor => print_doctor(),
        Command::Version => {
            println!("{}", solver_core::VERSION);
            Ok(())
        }
        Command::PrintArchitecture => {
            println!("{}", solver_core::architecture_summary());
            Ok(())
        }
        Command::Benchmark { command } => run_benchmark_command(command),
        Command::Dataset { command } => run_dataset_command(command),
        Command::Session { command } => run_session_command(command),
        Command::Oracle { command } => run_oracle_command(command),
        Command::Regression { command } => run_regression_command(command),
    }
}

fn print_doctor() -> CliResult<()> {
    println!("solitaire-cli {}", solver_core::VERSION);
    println!("solver_core: available");
    println!("architecture: {}", solver_core::architecture_summary());
    println!("benchmark presets: {}", EXPERIMENT_PRESET_NAMES.join(", "));
    Ok(())
}

fn run_benchmark_command(command: BenchmarkCommand) -> CliResult<()> {
    match command {
        BenchmarkCommand::Autoplay(args) => {
            let result = run_autoplay_benchmark_command(&args)?;
            print_autoplay_summary(&result);
            Ok(())
        }
        BenchmarkCommand::Compare(args) => {
            let result = run_compare_command(&args)?;
            print_comparison_summary(&result);
            Ok(())
        }
        BenchmarkCommand::RepeatedCompare(args) => {
            let result = run_repeated_compare_command(&args)?;
            print_repeated_summary(&result);
            Ok(())
        }
        BenchmarkCommand::ComparePresets(args) => {
            let result = run_compare_presets_command(&args)?;
            print_preset_comparison_summary(&result);
            Ok(())
        }
    }
}

fn run_dataset_command(command: DatasetCommand) -> CliResult<()> {
    match command {
        DatasetCommand::ExportVnet(args) => {
            let dataset = run_dataset_export_vnet_command(&args)?;
            print_vnet_dataset_summary(&dataset, &args.out);
            Ok(())
        }
    }
}

fn run_session_command(command: SessionCommand) -> CliResult<()> {
    match command {
        SessionCommand::Save(args) => {
            let session = run_session_save_command(&args)?;
            print_session_summary(&session.summary());
            println!("  saved: {}", args.out.display());
            Ok(())
        }
        SessionCommand::Inspect(args) => {
            let session = load_session(&args.path)?;
            print_session_inspect(&session);
            Ok(())
        }
        SessionCommand::Summary(args) => {
            let session = load_session(&args.path)?;
            print_session_summary(&session.summary());
            Ok(())
        }
        SessionCommand::Replay(args) => {
            let session = load_session(&args.path)?;
            let replay = replay_session(&session)?;
            println!("Session replay");
            println!("  matched: {}", replay.matched);
            println!("  replayed steps: {}", replay.replayed_steps);
            println!("  mismatches: {}", replay.mismatches.len());
            for mismatch in &replay.mismatches {
                println!(
                    "  - step {}: {}",
                    mismatch
                        .step_index
                        .map(|step| step.to_string())
                        .unwrap_or_else(|| "n/a".to_string()),
                    mismatch.message
                );
            }
            Ok(())
        }
    }
}

fn run_oracle_command(command: OracleCommand) -> CliResult<()> {
    match command {
        OracleCommand::ExportCases(args) => {
            let pack = run_oracle_export_cases_command(&args)?;
            println!("Oracle case export");
            println!("  cases: {}", pack.cases.len());
            println!("  schema: {}", pack.schema_version);
            println!("  out: {}", args.out.display());
            Ok(())
        }
        OracleCommand::EvaluateLocal(args) => {
            let evaluation = run_oracle_evaluate_local_command(&args)?;
            println!("Oracle local evaluation");
            println!("  cases: {}", evaluation.results.len());
            println!("  mode: {:?}", evaluation.mode);
            println!("  out: {}", args.out.display());
            Ok(())
        }
        OracleCommand::Compare(args) => {
            let summary = run_oracle_compare_command(&args)?;
            print_oracle_comparison_summary(&summary);
            Ok(())
        }
    }
}

fn run_regression_command(command: RegressionCommand) -> CliResult<()> {
    match command {
        RegressionCommand::CreateFromBenchmark(args) => {
            let pack = run_regression_create_from_benchmark_command(&args)?;
            print_regression_pack_summary(&pack.summary());
            println!("  out: {}", args.out.display());
            Ok(())
        }
        RegressionCommand::CreateFromSession(args) => {
            let pack = run_regression_create_from_session_command(&args)?;
            print_regression_pack_summary(&pack.summary());
            println!("  out: {}", args.out.display());
            Ok(())
        }
        RegressionCommand::Run(args) => {
            let result = run_regression_run_command(&args)?;
            print_regression_run_summary(&result);
            Ok(())
        }
        RegressionCommand::Summarize(args) => {
            let pack = load_regression_pack(&args.pack)?;
            print_regression_pack_summary(&pack.summary());
            Ok(())
        }
    }
}

fn run_oracle_export_cases_command(
    args: &OracleExportCasesArgs,
) -> CliResult<solver_core::OracleCasePack> {
    let preset = lookup_preset(&args.preset)?;
    let suite = BenchmarkSuite::from_base_seed(
        format!("cli-oracle-{}", preset.name),
        args.seed,
        args.games,
    );
    let pack = oracle_cases_from_seeded_suite(&suite, Some(&preset))?;
    ensure_parent_dir(&args.out)?;
    save_oracle_case_pack(&args.out, &pack)?;
    Ok(pack)
}

fn run_oracle_evaluate_local_command(
    args: &OracleEvaluateLocalArgs,
) -> CliResult<solver_core::OracleLocalEvaluation> {
    let pack = load_oracle_case_pack(&args.cases)?;
    let mut deterministic = DeterministicSearchConfig::default();
    if args.node_budget.is_some() || args.depth_budget.is_some() {
        deterministic.budget = SolveBudget {
            node_budget: args.node_budget.or(deterministic.budget.node_budget),
            depth_budget: args.depth_budget.or(deterministic.budget.depth_budget),
            wall_clock_limit_ms: deterministic.budget.wall_clock_limit_ms,
        };
    }
    let evaluation = evaluate_oracle_cases(
        &pack.cases,
        OracleEvaluationConfig {
            deterministic,
            mode: args.mode.into(),
        },
    )?;
    ensure_parent_dir(&args.out)?;
    save_oracle_local_evaluation(&args.out, &evaluation)?;
    Ok(evaluation)
}

fn run_oracle_compare_command(args: &OracleCompareArgs) -> CliResult<OracleComparisonSummary> {
    let local = load_oracle_local_evaluation(&args.local)?;
    let reference = load_oracle_reference_results(&args.reference)?;
    let summary = compare_oracle_results(&local.results, &reference);
    write_optional(&args.export.json, summary.to_json_summary()?)?;
    write_optional(&args.export.csv, summary.to_csv_summary())?;
    Ok(summary)
}

fn run_regression_create_from_benchmark_command(
    args: &RegressionCreateFromBenchmarkArgs,
) -> CliResult<RegressionPack> {
    let preset = lookup_preset(&args.preset)?;
    let suite = BenchmarkSuite::from_base_seed(
        format!("cli-regression-{}", preset.name),
        args.seed,
        args.games,
    );
    let config =
        regression_config_from_parts(preset, args.mode, args.node_budget, args.depth_budget);
    let pack = regression_pack_from_benchmark_suite(
        &suite,
        &config,
        args.name
            .clone()
            .unwrap_or_else(|| format!("{}-regression", config.preset.name)),
        args.tags.clone(),
    )?;
    ensure_parent_dir(&args.out)?;
    save_regression_pack(&args.out, &pack)?;
    Ok(pack)
}

fn run_regression_create_from_session_command(
    args: &RegressionCreateFromSessionArgs,
) -> CliResult<RegressionPack> {
    let session = load_session(&args.path)?;
    let pack = regression_pack_from_session(
        session,
        args.name
            .clone()
            .unwrap_or_else(|| "session-regression".to_string()),
        args.tags.clone(),
    );
    ensure_parent_dir(&args.out)?;
    save_regression_pack(&args.out, &pack)?;
    Ok(pack)
}

fn run_regression_run_command(args: &RegressionRunArgs) -> CliResult<RegressionRunResult> {
    let pack = load_regression_pack(&args.pack)?;
    let preset = lookup_preset(&args.preset)?;
    let config =
        regression_config_from_parts(preset, args.mode, args.node_budget, args.depth_budget);
    let result = run_regression_pack(&pack, &config)?;
    write_optional(&args.export.json, result.to_json_summary()?)?;
    write_optional(&args.export.csv, result.to_csv_summary())?;
    Ok(result)
}

fn regression_config_from_parts(
    preset: ExperimentPreset,
    mode: CliOracleEvaluationMode,
    node_budget: Option<u64>,
    depth_budget: Option<u16>,
) -> RegressionRunConfig {
    let mut config = RegressionRunConfig::from_preset(preset);
    config.deterministic_mode = mode.into();
    if node_budget.is_some() || depth_budget.is_some() {
        let mut deterministic = DeterministicSearchConfig::default();
        deterministic.budget = SolveBudget {
            node_budget: node_budget.or(deterministic.budget.node_budget),
            depth_budget: depth_budget.or(deterministic.budget.depth_budget),
            wall_clock_limit_ms: deterministic.budget.wall_clock_limit_ms,
        };
        config.deterministic_override = Some(deterministic);
    }
    config
}

fn run_autoplay_benchmark_command(
    args: &BenchmarkAutoplayArgs,
) -> CliResult<AutoplayBenchmarkResult> {
    let mut preset = lookup_preset(&args.preset)?;
    apply_leaf_eval_override(&mut preset, &args.leaf);
    let suite = BenchmarkSuite::from_base_seed(
        format!("cli-autoplay-{}", preset.name),
        args.suite.seed,
        args.suite.games,
    );
    let config = benchmark_config_from_preset(preset, args.suite.max_steps);
    let result = run_autoplay_benchmark(&suite, &config)?;

    write_optional(&args.export.json, result.to_json_summary()?)?;
    write_optional(&args.export.csv, result.to_csv_summary())?;
    write_optional(&args.game_csv, result.to_game_csv())?;

    Ok(result)
}

fn run_compare_command(args: &BenchmarkCompareArgs) -> CliResult<AutoplayComparisonResult> {
    let mut baseline_preset = lookup_preset(&args.baseline)?;
    apply_leaf_eval_override(
        &mut baseline_preset,
        &LeafEvalArgs {
            leaf_eval_mode: args.baseline_leaf_eval_mode,
            vnet_model: args.baseline_vnet_model.clone(),
        },
    );
    let mut candidate_preset = lookup_preset(&args.candidate)?;
    apply_leaf_eval_override(
        &mut candidate_preset,
        &LeafEvalArgs {
            leaf_eval_mode: args.candidate_leaf_eval_mode,
            vnet_model: args.candidate_vnet_model.clone(),
        },
    );
    let baseline = benchmark_config_from_preset(baseline_preset, args.suite.max_steps);
    let candidate = benchmark_config_from_preset(candidate_preset, args.suite.max_steps);
    let suite = BenchmarkSuite::from_base_seed(
        format!(
            "cli-compare-{}-vs-{}",
            baseline.label.name, candidate.label.name
        ),
        args.suite.seed,
        args.suite.games,
    );
    let result = run_autoplay_paired_comparison(&suite, &baseline, &candidate)?;

    write_optional(&args.export.json, result.to_json_summary()?)?;
    write_optional(&args.export.csv, result.to_csv_summary())?;

    Ok(result)
}

fn run_repeated_compare_command(
    args: &BenchmarkRepeatedCompareArgs,
) -> CliResult<AutoplayRepeatedComparisonResult> {
    let mut baseline_preset = lookup_preset(&args.baseline)?;
    apply_leaf_eval_override(
        &mut baseline_preset,
        &LeafEvalArgs {
            leaf_eval_mode: args.baseline_leaf_eval_mode,
            vnet_model: args.baseline_vnet_model.clone(),
        },
    );
    let mut candidate_preset = lookup_preset(&args.candidate)?;
    apply_leaf_eval_override(
        &mut candidate_preset,
        &LeafEvalArgs {
            leaf_eval_mode: args.candidate_leaf_eval_mode,
            vnet_model: args.candidate_vnet_model.clone(),
        },
    );
    let baseline = benchmark_config_from_preset(baseline_preset, args.max_steps);
    let candidate = benchmark_config_from_preset(candidate_preset, args.max_steps);
    let result = run_autoplay_repeated_comparison(
        "cli-repeated",
        args.seed,
        args.games,
        args.repetitions,
        &baseline,
        &candidate,
    )?;

    write_optional(&args.export.json, result.to_json_summary()?)?;
    write_optional(&args.export.csv, result.to_csv_summary())?;

    Ok(result)
}

fn run_compare_presets_command(
    args: &BenchmarkComparePresetsArgs,
) -> CliResult<PresetComparisonSummary> {
    let mut presets = preset_list_from_arg(args.presets.as_deref())?;
    for preset in &mut presets {
        apply_compare_presets_leaf_eval_override(preset, &args.leaf);
    }
    if let Some(max_steps) = args.max_steps {
        for preset in &mut presets {
            preset.autoplay.max_steps = max_steps;
        }
    }
    let suite = BenchmarkSuite::from_base_seed("cli-compare-presets", args.seed, args.games);
    let result = compare_experiment_presets_on_suite(&suite, &presets, args.rank_by.into())?;

    write_optional(&args.export.json, result.to_json_summary()?)?;
    write_optional(&args.export.csv, result.to_csv_summary())?;

    Ok(result)
}

fn run_dataset_export_vnet_command(args: &DatasetExportVnetArgs) -> CliResult<VNetDataset> {
    let preset = lookup_preset(&args.preset)?;
    let suite =
        BenchmarkSuite::from_base_seed(format!("cli-vnet-{}", preset.name), args.seed, args.games);
    let export_config = VNetExportConfig {
        label_mode: args.label_mode.into(),
        max_steps: args.max_steps,
        decision_stride: args.decision_stride,
        format: args.format.into(),
        ..VNetExportConfig::default()
    };
    let dataset = collect_vnet_examples_from_autoplay_suite(&suite, &preset, &export_config)?;

    ensure_parent_dir(&args.out)?;
    match export_config.format {
        DatasetFormat::Jsonl => VNetDatasetWriter::write_jsonl(&args.out, &dataset)?,
    }

    Ok(dataset)
}

fn run_session_save_command(args: &SessionSaveArgs) -> CliResult<SessionRecord> {
    let mut preset = lookup_preset(&args.preset)?;
    if let Some(max_steps) = args.max_steps {
        preset.autoplay.max_steps = max_steps;
    }
    let deal = ExperimentRunner.generate_deal(DealSeed(args.seed))?;
    let result = play_game_with_planner(&deal.full_state, &preset.solver, &preset.autoplay)?;
    let metadata = SessionMetadata::generated(args.label.clone())
        .with_solver_provenance(Some(preset.name.clone()), Some(preset.autoplay.backend));
    let session = SessionRecord::from_autoplay_result(metadata, deal.full_state, &result)?;
    save_session(&args.out, &session)?;
    Ok(session)
}

fn lookup_preset(name: &str) -> CliResult<ExperimentPreset> {
    experiment_preset_by_name(name).ok_or_else(|| {
        invalid_input(format!(
            "unknown preset {name:?}; expected one of: {}",
            EXPERIMENT_PRESET_NAMES.join(", ")
        ))
    })
}

fn preset_list_from_arg(names: Option<&str>) -> CliResult<Vec<ExperimentPreset>> {
    let names = match names {
        Some(names) => names
            .split(',')
            .map(str::trim)
            .filter(|name| !name.is_empty())
            .collect::<Vec<_>>(),
        None => EXPERIMENT_PRESET_NAMES.to_vec(),
    };
    if names.is_empty() {
        return Err(invalid_input("at least one preset is required".to_string()));
    }
    names.into_iter().map(lookup_preset).collect()
}

fn benchmark_config_from_preset(
    preset: ExperimentPreset,
    max_steps: Option<usize>,
) -> solver_core::AutoplayBenchmarkConfig {
    let mut config = preset.autoplay_benchmark_config();
    if let Some(max_steps) = max_steps {
        config.autoplay.max_steps = max_steps;
    }
    config
}

fn apply_leaf_eval_override(preset: &mut ExperimentPreset, args: &LeafEvalArgs) {
    if let Some(mode) = args.leaf_eval_mode {
        preset.solver.deterministic.leaf_eval_mode = mode.into();
    }
    if let Some(path) = &args.vnet_model {
        preset.solver.deterministic.leaf_eval_mode = LeafEvaluationMode::VNet;
        preset.solver.deterministic.vnet_inference.enable_vnet = true;
        preset.solver.deterministic.vnet_inference.model_path = Some(path.clone());
    }
}

fn apply_compare_presets_leaf_eval_override(preset: &mut ExperimentPreset, args: &LeafEvalArgs) {
    if args.leaf_eval_mode.is_some() {
        apply_leaf_eval_override(preset, args);
        return;
    }

    if let Some(path) = &args.vnet_model {
        if preset.solver.deterministic.leaf_eval_mode == LeafEvaluationMode::VNet {
            preset.solver.deterministic.vnet_inference.enable_vnet = true;
            preset.solver.deterministic.vnet_inference.model_path = Some(path.clone());
        }
    }
}

fn print_autoplay_summary(result: &AutoplayBenchmarkResult) {
    println!("Autoplay benchmark");
    println!("  preset: {}", result.config.name);
    println!("  backend: {:?}", result.backend);
    println!(
        "  suite: {} games={} base_seed={}",
        result.suite_name,
        result.games,
        result
            .suite
            .base_seed
            .map(|seed| seed.0.to_string())
            .unwrap_or_else(|| "n/a".to_string())
    );
    println!(
        "  wins/losses: {}/{}  win_rate: {:.3}",
        result.wins, result.losses, result.win_rate
    );
    println!(
        "  avg moves/game: {:.2}  avg time/game: {:.2}ms  avg time/move: {:.2}ms",
        result.average_moves_per_game,
        result.average_total_planner_time_per_game_ms,
        result.average_planner_time_per_move_ms
    );
    println!(
        "  avg nodes/game: {:.1}  avg root visits/game: {:.1}  late-exact triggers: {}",
        result.average_deterministic_nodes,
        result.average_root_visits,
        result.late_exact_trigger_count
    );
    println!(
        "  root-parallel steps: {}  avg workers/game: {:.1}  avg worker sims/game: {:.1}",
        result.root_parallel_step_count,
        result.average_root_parallel_workers,
        result.average_root_parallel_simulations
    );
    println!(
        "  leaf: {:?}  vnet model: {}",
        result.leaf_eval_mode,
        result.vnet_model_path.as_deref().unwrap_or("n/a")
    );
    println!(
        "  vnet inferences: {}  fallbacks: {}  inference time: {}us",
        result.vnet_inferences, result.vnet_fallbacks, result.vnet_inference_elapsed_us
    );
}

fn print_comparison_summary(result: &AutoplayComparisonResult) {
    println!("Paired autoplay comparison");
    println!(
        "  baseline: {} ({:?})  wins: {}",
        result.baseline.config.name, result.baseline.backend, result.baseline_wins
    );
    println!(
        "  candidate: {} ({:?})  wins: {}",
        result.candidate.config.name, result.candidate.backend, result.candidate_wins
    );
    println!(
        "  games: {}  same outcome: {}  candidate-only: {}  baseline-only: {}",
        result.baseline.games,
        result.same_outcome_count,
        result.candidate_only_wins,
        result.baseline_only_wins
    );
    println!(
        "  leaf modes: baseline {:?}  candidate {:?}",
        result.baseline.leaf_eval_mode, result.candidate.leaf_eval_mode
    );
    println!(
        "  vnet inferences A/B: {}/{}  fallbacks A/B: {}/{}",
        result.baseline.vnet_inferences,
        result.candidate.vnet_inferences,
        result.baseline.vnet_fallbacks,
        result.candidate.vnet_fallbacks
    );
    println!(
        "  paired delta: {:.3}  stderr: {:.3}  ci: [{:.3}, {:.3}]",
        result.paired_win_rate_delta,
        result.paired_standard_error,
        result.ci_lower,
        result.ci_upper
    );
}

fn print_repeated_summary(result: &AutoplayRepeatedComparisonResult) {
    println!("Repeated paired autoplay comparison");
    println!("  repetitions: {}", result.repetitions.len());
    println!(
        "  mean paired delta: {:.3}  stderr: {:.3}  ci: [{:.3}, {:.3}]",
        result.mean_paired_win_rate_delta,
        result.paired_standard_error,
        result.ci_lower,
        result.ci_upper
    );
    for repetition in &result.repetitions {
        println!(
            "  rep {}: delta {:.3}  wins A/B {}/{}",
            repetition.repetition_index,
            repetition.comparison.paired_win_rate_delta,
            repetition.comparison.baseline_wins,
            repetition.comparison.candidate_wins
        );
    }
}

fn print_preset_comparison_summary(result: &PresetComparisonSummary) {
    println!("Preset comparison");
    println!(
        "  suite: {} games={} base_seed={} rank_by={:?}",
        result.suite.name,
        result.suite.seed_count,
        result
            .suite
            .base_seed
            .map(|seed| seed.0.to_string())
            .unwrap_or_else(|| "n/a".to_string()),
        result.ranking_metric
    );
    println!("  rank  preset                    backend             leaf       win_rate  time/game  time/move  vnet/fb     efficiency");
    for (index, entry) in result.entries.iter().enumerate() {
        println!(
            "  {:>4}  {:<24}  {:<18?}  {:<9?}  {:>7.3}  {:>8.2}ms  {:>8.2}ms  {:>4}/{:<4}  {:>10.3}",
            index + 1,
            entry.preset_name,
            entry.backend,
            entry.leaf_eval_mode,
            entry.win_rate,
            entry.average_time_per_game_ms,
            entry.average_time_per_move_ms,
            entry.vnet_inferences,
            entry.vnet_fallbacks,
            entry.win_rate_per_second
        );
    }
}

fn print_vnet_dataset_summary(dataset: &VNetDataset, path: &Path) {
    println!("V-Net dataset export");
    println!("  preset: {}", dataset.metadata.preset_name);
    println!("  label_mode: {:?}", dataset.metadata.label_mode);
    println!(
        "  suite: {} games={} base_seed={}",
        dataset.metadata.suite_name,
        dataset.metadata.games,
        dataset
            .metadata
            .suite
            .base_seed
            .map(|seed| seed.0.to_string())
            .unwrap_or_else(|| "n/a".to_string())
    );
    println!("  examples: {}", dataset.metadata.example_count);
    println!(
        "  format: {:?}  out: {}",
        dataset.metadata.format,
        path.display()
    );
}

fn print_session_summary(summary: &SessionSummary) {
    println!("Session summary");
    println!("  id: {}", summary.id.0);
    println!("  schema: {}", summary.schema_version);
    println!("  label: {}", summary.label.as_deref().unwrap_or("n/a"));
    println!(
        "  preset/backend: {}/{}",
        summary.preset_name.as_deref().unwrap_or("n/a"),
        summary
            .backend
            .map(|backend| format!("{backend:?}"))
            .unwrap_or_else(|| "n/a".to_string())
    );
    println!(
        "  steps: {}  reveals: {}  hidden_remaining: {}",
        summary.steps, summary.reveals, summary.hidden_cards_remaining
    );
    println!(
        "  full_state: {}  structural_win: {}  planner_reuse: {}",
        summary.has_full_state, summary.structural_win, summary.has_planner_continuation
    );
}

fn print_session_inspect(session: &SessionRecord) {
    print_session_summary(&session.summary());
    println!("  engine_version: {}", session.metadata.engine_version);
    println!(
        "  created_unix_secs: {}",
        session.metadata.created_unix_secs
    );
    println!("  current visible:");
    print!("{}", session.current_snapshot.visible);
}

fn print_oracle_comparison_summary(summary: &OracleComparisonSummary) {
    println!("Oracle comparison");
    println!(
        "  cases: {}  matches: {}  mismatches: {}",
        summary.cases_compared, summary.matches, summary.mismatches
    );
    println!(
        "  exact win/loss agreements: {}  best-move agreements: {}/{}",
        summary.exact_win_loss_agreements,
        summary.best_move_agreements,
        summary.best_move_comparisons
    );
    for count in &summary.mismatch_counts {
        println!("  {:?}: {}", count.kind, count.count);
    }
}

fn print_regression_pack_summary(summary: &RegressionPackSummary) {
    println!("Regression pack");
    println!("  name: {}", summary.name);
    println!("  schema: {}", summary.schema_version);
    println!("  engine_version: {}", summary.engine_version);
    println!("  cases: {}", summary.case_count);
    if !summary.kind_counts.is_empty() {
        println!("  kinds:");
        for (kind, count) in &summary.kind_counts {
            println!("    {kind}: {count}");
        }
    }
    if !summary.tag_counts.is_empty() {
        println!("  tags:");
        for (tag, count) in &summary.tag_counts {
            println!("    {tag}: {count}");
        }
    }
}

fn print_regression_run_summary(result: &RegressionRunResult) {
    println!("Regression run");
    println!("  pack: {}", result.pack_name);
    println!("  preset: {}", result.preset_name);
    println!(
        "  cases: {}  passed: {}  failed: {}",
        result.total_cases, result.passed, result.failed
    );
    for count in &result.mismatch_counts {
        println!("  {:?}: {}", count.kind, count.count);
    }
}

fn write_optional(path: &Option<PathBuf>, contents: String) -> CliResult<()> {
    if let Some(path) = path {
        write_export(path, contents)?;
    }
    Ok(())
}

fn write_export(path: &Path, contents: String) -> CliResult<()> {
    ensure_parent_dir(path)?;
    fs::write(path, contents)?;
    Ok(())
}

fn ensure_parent_dir(path: &Path) -> CliResult<()> {
    if let Some(parent) = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        fs::create_dir_all(parent)?;
    }
    Ok(())
}

fn invalid_input(message: String) -> Box<dyn Error> {
    Box::new(IoError::new(ErrorKind::InvalidInput, message))
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    #[test]
    fn cli_parses_autoplay_benchmark_command() {
        let cli = Cli::try_parse_from([
            "solitaire-cli",
            "benchmark",
            "autoplay",
            "--preset",
            "fast_benchmark",
            "--games",
            "2",
            "--seed",
            "10",
            "--max-steps",
            "0",
            "--json",
            "summary.json",
            "--csv",
            "summary.csv",
            "--game-csv",
            "games.csv",
            "--leaf-eval-mode",
            "vnet",
            "--vnet-model",
            "best_vnet_inference.json",
        ])
        .unwrap();

        match cli.command {
            Command::Benchmark {
                command: BenchmarkCommand::Autoplay(args),
            } => {
                assert_eq!(args.preset, "fast_benchmark");
                assert_eq!(args.suite.games, 2);
                assert_eq!(args.suite.seed, 10);
                assert_eq!(args.suite.max_steps, Some(0));
                assert_eq!(args.export.json, Some(PathBuf::from("summary.json")));
                assert_eq!(args.game_csv, Some(PathBuf::from("games.csv")));
                assert_eq!(args.leaf.leaf_eval_mode, Some(CliLeafEvalMode::Vnet));
                assert_eq!(
                    args.leaf.vnet_model,
                    Some(PathBuf::from("best_vnet_inference.json"))
                );
            }
            _ => panic!("wrong command parsed"),
        }
    }

    #[test]
    fn cli_parses_compare_commands() {
        let compare = Cli::try_parse_from([
            "solitaire-cli",
            "benchmark",
            "compare",
            "--baseline",
            "pimc_baseline",
            "--candidate",
            "belief_uct_late_exact",
            "--games",
            "3",
            "--seed",
            "42",
            "--baseline-leaf-eval-mode",
            "heuristic",
            "--candidate-leaf-eval-mode",
            "vnet",
            "--candidate-vnet-model",
            "best_vnet_inference.json",
        ])
        .unwrap();
        match compare.command {
            Command::Benchmark {
                command: BenchmarkCommand::Compare(args),
            } => {
                assert_eq!(
                    args.baseline_leaf_eval_mode,
                    Some(CliLeafEvalMode::Heuristic)
                );
                assert_eq!(args.candidate_leaf_eval_mode, Some(CliLeafEvalMode::Vnet));
                assert_eq!(
                    args.candidate_vnet_model,
                    Some(PathBuf::from("best_vnet_inference.json"))
                );
            }
            _ => panic!("wrong command parsed"),
        }

        let repeated = Cli::try_parse_from([
            "solitaire-cli",
            "benchmark",
            "repeated-compare",
            "--baseline",
            "belief_uct_default",
            "--candidate",
            "belief_uct_late_exact",
            "--games",
            "2",
            "--repetitions",
            "2",
            "--seed",
            "99",
        ])
        .unwrap();
        assert!(matches!(
            repeated.command,
            Command::Benchmark {
                command: BenchmarkCommand::RepeatedCompare(_)
            }
        ));
    }

    #[test]
    fn cli_parses_compare_presets_command() {
        let cli = Cli::try_parse_from([
            "solitaire-cli",
            "benchmark",
            "compare-presets",
            "--presets",
            "fast_benchmark,balanced_benchmark",
            "--games",
            "2",
            "--seed",
            "11",
            "--max-steps",
            "0",
            "--rank-by",
            "win-rate",
            "--json",
            "presets.json",
            "--csv",
            "presets.csv",
            "--leaf-eval-mode",
            "vnet",
            "--vnet-model",
            "best_vnet_inference.json",
        ])
        .unwrap();

        match cli.command {
            Command::Benchmark {
                command: BenchmarkCommand::ComparePresets(args),
            } => {
                assert_eq!(
                    args.presets,
                    Some("fast_benchmark,balanced_benchmark".to_string())
                );
                assert_eq!(args.games, 2);
                assert_eq!(args.seed, 11);
                assert_eq!(args.max_steps, Some(0));
                assert_eq!(args.rank_by, CliRankingMetric::WinRate);
                assert_eq!(args.export.csv, Some(PathBuf::from("presets.csv")));
                assert_eq!(args.leaf.leaf_eval_mode, Some(CliLeafEvalMode::Vnet));
                assert_eq!(
                    args.leaf.vnet_model,
                    Some(PathBuf::from("best_vnet_inference.json"))
                );
            }
            _ => panic!("wrong command parsed"),
        }
    }

    #[test]
    fn cli_parses_dataset_export_vnet_command() {
        let cli = Cli::try_parse_from([
            "solitaire-cli",
            "dataset",
            "export-vnet",
            "--preset",
            "fast_benchmark",
            "--games",
            "1",
            "--seed",
            "22",
            "--out",
            "vnet.jsonl",
            "--label-mode",
            "planner-approximate-value",
            "--format",
            "jsonl",
            "--max-steps",
            "0",
            "--decision-stride",
            "2",
        ])
        .unwrap();

        match cli.command {
            Command::Dataset {
                command: DatasetCommand::ExportVnet(args),
            } => {
                assert_eq!(args.preset, "fast_benchmark");
                assert_eq!(args.games, 1);
                assert_eq!(args.seed, 22);
                assert_eq!(args.out, PathBuf::from("vnet.jsonl"));
                assert_eq!(args.label_mode, CliVNetLabelMode::PlannerApproximateValue);
                assert_eq!(args.format, CliDatasetFormat::Jsonl);
                assert_eq!(args.max_steps, Some(0));
                assert_eq!(args.decision_stride, 2);
            }
            _ => panic!("wrong command parsed"),
        }
    }

    #[test]
    fn cli_parses_session_commands() {
        let save = Cli::try_parse_from([
            "solitaire-cli",
            "session",
            "save",
            "--preset",
            "fast_benchmark",
            "--seed",
            "44",
            "--out",
            "session.json",
            "--max-steps",
            "0",
            "--label",
            "smoke",
        ])
        .unwrap();
        assert!(matches!(
            save.command,
            Command::Session {
                command: SessionCommand::Save(_)
            }
        ));

        let replay = Cli::try_parse_from([
            "solitaire-cli",
            "session",
            "replay",
            "--path",
            "session.json",
        ])
        .unwrap();
        assert!(matches!(
            replay.command,
            Command::Session {
                command: SessionCommand::Replay(_)
            }
        ));
    }

    #[test]
    fn cli_parses_oracle_commands() {
        let export = Cli::try_parse_from([
            "solitaire-cli",
            "oracle",
            "export-cases",
            "--preset",
            "fast_benchmark",
            "--games",
            "2",
            "--seed",
            "50",
            "--out",
            "cases.json",
        ])
        .unwrap();
        assert!(matches!(
            export.command,
            Command::Oracle {
                command: OracleCommand::ExportCases(_)
            }
        ));

        let evaluate = Cli::try_parse_from([
            "solitaire-cli",
            "oracle",
            "evaluate-local",
            "--cases",
            "cases.json",
            "--out",
            "local.json",
            "--mode",
            "fast",
            "--node-budget",
            "10",
            "--depth-budget",
            "1",
        ])
        .unwrap();
        match evaluate.command {
            Command::Oracle {
                command: OracleCommand::EvaluateLocal(args),
            } => {
                assert_eq!(args.mode, CliOracleEvaluationMode::Fast);
                assert_eq!(args.node_budget, Some(10));
                assert_eq!(args.depth_budget, Some(1));
            }
            _ => panic!("wrong command parsed"),
        }

        let compare = Cli::try_parse_from([
            "solitaire-cli",
            "oracle",
            "compare",
            "--local",
            "local.json",
            "--reference",
            "reference.jsonl",
            "--json",
            "summary.json",
            "--csv",
            "summary.csv",
        ])
        .unwrap();
        assert!(matches!(
            compare.command,
            Command::Oracle {
                command: OracleCommand::Compare(_)
            }
        ));
    }

    #[test]
    fn cli_parses_regression_commands() {
        let create = Cli::try_parse_from([
            "solitaire-cli",
            "regression",
            "create-from-benchmark",
            "--preset",
            "fast_benchmark",
            "--games",
            "2",
            "--seed",
            "77",
            "--out",
            "pack.json",
            "--tag",
            "reveal-heavy",
            "--tag",
            "stock-pivot",
            "--mode",
            "fast",
            "--node-budget",
            "8",
            "--depth-budget",
            "1",
        ])
        .unwrap();
        match create.command {
            Command::Regression {
                command: RegressionCommand::CreateFromBenchmark(args),
            } => {
                assert_eq!(args.preset, "fast_benchmark");
                assert_eq!(args.games, 2);
                assert_eq!(args.seed, 77);
                assert_eq!(args.out, PathBuf::from("pack.json"));
                assert_eq!(
                    args.tags,
                    vec!["reveal-heavy".to_string(), "stock-pivot".to_string()]
                );
                assert_eq!(args.mode, CliOracleEvaluationMode::Fast);
                assert_eq!(args.node_budget, Some(8));
                assert_eq!(args.depth_budget, Some(1));
            }
            _ => panic!("wrong command parsed"),
        }

        let run = Cli::try_parse_from([
            "solitaire-cli",
            "regression",
            "run",
            "--pack",
            "pack.json",
            "--preset",
            "fast_benchmark",
            "--json",
            "run.json",
            "--csv",
            "run.csv",
        ])
        .unwrap();
        assert!(matches!(
            run.command,
            Command::Regression {
                command: RegressionCommand::Run(_)
            }
        ));

        let from_session = Cli::try_parse_from([
            "solitaire-cli",
            "regression",
            "create-from-session",
            "--path",
            "session.json",
            "--out",
            "session-pack.json",
            "--tag",
            "replay",
        ])
        .unwrap();
        assert!(matches!(
            from_session.command,
            Command::Regression {
                command: RegressionCommand::CreateFromSession(_)
            }
        ));

        let summarize = Cli::try_parse_from([
            "solitaire-cli",
            "regression",
            "summarize",
            "--pack",
            "pack.json",
        ])
        .unwrap();
        assert!(matches!(
            summarize.command,
            Command::Regression {
                command: RegressionCommand::Summarize(_)
            }
        ));
    }

    #[test]
    fn preset_lookup_accepts_known_names() {
        for name in EXPERIMENT_PRESET_NAMES {
            assert!(lookup_preset(name).is_ok(), "{name}");
        }
        assert!(lookup_preset("definitely_not_a_preset").is_err());
    }

    #[test]
    fn export_path_writes_parent_directories() {
        let path = std::env::temp_dir()
            .join("solitaire_solver_cli_tests")
            .join(format!(
                "export-{}-{}.csv",
                std::process::id(),
                solver_core::VERSION
            ));
        let _ = fs::remove_file(&path);

        write_export(&path, "hello,world\n".to_string()).unwrap();

        let contents = fs::read_to_string(&path).unwrap();
        assert_eq!(contents, "hello,world\n");
        let _ = fs::remove_file(path);
    }

    #[test]
    fn benchmark_command_wiring_is_reproducible_on_small_suite() {
        let args = BenchmarkAutoplayArgs {
            preset: "fast_benchmark".to_string(),
            suite: SuiteArgs {
                games: 1,
                seed: 7,
                max_steps: Some(0),
            },
            export: ExportArgs::default(),
            leaf: LeafEvalArgs::default(),
            game_csv: None,
        };

        let first = run_autoplay_benchmark_command(&args).unwrap();
        let second = run_autoplay_benchmark_command(&args).unwrap();

        assert_eq!(first.records, second.records);
        assert_eq!(first.config.name, "fast_benchmark");
        assert_eq!(first.games, 1);
    }

    #[test]
    fn compare_presets_command_wiring_is_reproducible_on_small_suite() {
        let args = BenchmarkComparePresetsArgs {
            presets: Some("fast_benchmark,balanced_benchmark".to_string()),
            games: 1,
            seed: 17,
            max_steps: Some(0),
            rank_by: CliRankingMetric::Efficiency,
            export: ExportArgs::default(),
            leaf: LeafEvalArgs::default(),
        };

        let first = run_compare_presets_command(&args).unwrap();
        let second = run_compare_presets_command(&args).unwrap();

        assert_eq!(first, second);
        assert_eq!(first.entries.len(), 2);
        assert_eq!(first.entries[0].preset_name, "balanced_benchmark");
    }

    #[test]
    fn compare_presets_shared_vnet_model_only_updates_vnet_presets() {
        let args = BenchmarkComparePresetsArgs {
            presets: Some("fast_benchmark,fast_vnet_benchmark".to_string()),
            games: 1,
            seed: 18,
            max_steps: Some(0),
            rank_by: CliRankingMetric::WinRate,
            export: ExportArgs::default(),
            leaf: LeafEvalArgs {
                leaf_eval_mode: None,
                vnet_model: Some(PathBuf::from("shared-vnet.json")),
            },
        };

        let result = run_compare_presets_command(&args).unwrap();
        let fast = result
            .entries
            .iter()
            .find(|entry| entry.preset_name == "fast_benchmark")
            .unwrap();
        let fast_vnet = result
            .entries
            .iter()
            .find(|entry| entry.preset_name == "fast_vnet_benchmark")
            .unwrap();

        assert_eq!(fast.leaf_eval_mode, LeafEvaluationMode::Heuristic);
        assert_eq!(fast.vnet_model_path, None);
        assert_eq!(fast_vnet.leaf_eval_mode, LeafEvaluationMode::VNet);
        assert_eq!(
            fast_vnet.vnet_model_path,
            Some("shared-vnet.json".to_string())
        );
    }

    #[test]
    fn dataset_export_vnet_command_writes_reproducible_jsonl() {
        let path = std::env::temp_dir().join(format!(
            "solitaire-cli-vnet-{}-{}.jsonl",
            std::process::id(),
            solver_core::VERSION
        ));
        let _ = fs::remove_file(&path);
        let args = DatasetExportVnetArgs {
            preset: "fast_benchmark".to_string(),
            games: 1,
            seed: 31,
            out: path.clone(),
            label_mode: CliVNetLabelMode::TerminalOutcome,
            format: CliDatasetFormat::Jsonl,
            max_steps: Some(0),
            decision_stride: 1,
        };

        let first = run_dataset_export_vnet_command(&args).unwrap();
        let first_contents = fs::read_to_string(&path).unwrap();
        let second = run_dataset_export_vnet_command(&args).unwrap();
        let second_contents = fs::read_to_string(&path).unwrap();

        assert_eq!(first, second);
        assert_eq!(first_contents, second_contents);
        assert!(first_contents.contains("\"record_type\":\"metadata\""));
        assert!(first_contents.contains("\"record_type\":\"example\""));
        let _ = fs::remove_file(path);
    }

    #[test]
    fn session_save_command_writes_replayable_session() {
        let path = std::env::temp_dir().join(format!(
            "solitaire-cli-session-{}-{}.json",
            std::process::id(),
            solver_core::VERSION
        ));
        let _ = fs::remove_file(&path);
        let args = SessionSaveArgs {
            preset: "fast_benchmark".to_string(),
            seed: 41,
            out: path.clone(),
            max_steps: Some(0),
            label: Some("cli-test".to_string()),
        };

        let session = run_session_save_command(&args).unwrap();
        let loaded = load_session(&path).unwrap();
        let replay = replay_session(&loaded).unwrap();

        assert_eq!(loaded.metadata.label, Some("cli-test".to_string()));
        assert_eq!(loaded.summary().steps, 0);
        assert!(replay.matched);
        assert_eq!(session.current_snapshot, loaded.current_snapshot);
        let _ = fs::remove_file(path);
    }

    #[test]
    fn oracle_commands_export_evaluate_and_compare() {
        let temp = std::env::temp_dir().join(format!(
            "solitaire-cli-oracle-{}-{}",
            std::process::id(),
            solver_core::VERSION
        ));
        let cases_path = temp.join("cases.json");
        let local_path = temp.join("local.json");
        let reference_path = temp.join("reference.json");
        let summary_path = temp.join("summary.json");
        let csv_path = temp.join("summary.csv");
        let _ = fs::remove_dir_all(&temp);

        let cases = run_oracle_export_cases_command(&OracleExportCasesArgs {
            preset: "fast_benchmark".to_string(),
            games: 1,
            seed: 61,
            out: cases_path.clone(),
        })
        .unwrap();
        assert_eq!(cases.cases.len(), 1);

        let local = run_oracle_evaluate_local_command(&OracleEvaluateLocalArgs {
            cases: cases_path,
            out: local_path.clone(),
            mode: CliOracleEvaluationMode::Fast,
            node_budget: Some(8),
            depth_budget: Some(1),
        })
        .unwrap();
        let reference = solver_core::OracleReferenceResultSet::new(
            Some("local-copy".to_string()),
            local.as_reference_results(),
        );
        fs::write(&reference_path, reference.to_json().unwrap()).unwrap();

        let summary = run_oracle_compare_command(&OracleCompareArgs {
            local: local_path,
            reference: reference_path,
            export: ExportArgs {
                json: Some(summary_path.clone()),
                csv: Some(csv_path.clone()),
            },
        })
        .unwrap();

        assert_eq!(summary.mismatches, 0);
        assert!(summary.matches > 0);
        assert!(fs::read_to_string(summary_path)
            .unwrap()
            .contains("cases_compared"));
        assert!(fs::read_to_string(csv_path)
            .unwrap()
            .starts_with("case_id,"));
        let _ = fs::remove_dir_all(temp);
    }

    #[test]
    fn regression_commands_create_run_and_summarize() {
        let temp = std::env::temp_dir().join(format!(
            "solitaire-cli-regression-{}-{}",
            std::process::id(),
            solver_core::VERSION
        ));
        let pack_path = temp.join("pack.json");
        let run_json = temp.join("run.json");
        let run_csv = temp.join("run.csv");
        let _ = fs::remove_dir_all(&temp);

        let pack =
            run_regression_create_from_benchmark_command(&RegressionCreateFromBenchmarkArgs {
                preset: "fast_benchmark".to_string(),
                games: 1,
                seed: 71,
                out: pack_path.clone(),
                name: Some("cli-regression".to_string()),
                tags: vec!["reveal-heavy".to_string()],
                mode: CliOracleEvaluationMode::Fast,
                node_budget: Some(8),
                depth_budget: Some(1),
            })
            .unwrap();
        assert_eq!(pack.cases.len(), 1);

        let loaded = load_regression_pack(&pack_path).unwrap();
        let summary = loaded.summary();
        assert_eq!(summary.name, "cli-regression");
        assert_eq!(summary.tag_counts.get("reveal-heavy"), Some(&1));

        let result = run_regression_run_command(&RegressionRunArgs {
            pack: pack_path,
            preset: "fast_benchmark".to_string(),
            mode: CliOracleEvaluationMode::Fast,
            node_budget: Some(8),
            depth_budget: Some(1),
            export: ExportArgs {
                json: Some(run_json.clone()),
                csv: Some(run_csv.clone()),
            },
        })
        .unwrap();

        assert_eq!(result.total_cases, 1);
        assert_eq!(result.failed, 0);
        assert!(fs::read_to_string(run_json)
            .unwrap()
            .contains("case_results"));
        assert!(fs::read_to_string(run_csv).unwrap().starts_with("case_id,"));
        let _ = fs::remove_dir_all(temp);
    }
}
