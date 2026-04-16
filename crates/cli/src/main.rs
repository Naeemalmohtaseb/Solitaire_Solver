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
    experiment_preset_by_name, run_autoplay_benchmark, run_autoplay_paired_comparison,
    run_autoplay_repeated_comparison, AutoplayBenchmarkResult, AutoplayComparisonResult,
    AutoplayRepeatedComparisonResult, BenchmarkSuite, DatasetFormat, ExperimentPreset,
    PresetComparisonSummary, PresetRankingMetric, VNetDataset, VNetDatasetWriter, VNetExportConfig,
    VNetLabelMode, EXPERIMENT_PRESET_NAMES,
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
}

/// Label modes accepted by `dataset export-vnet`.
#[derive(Debug, Copy, Clone, PartialEq, Eq, ValueEnum)]
enum CliVNetLabelMode {
    /// Label examples by the final win/loss of the configured autoplay run.
    TerminalOutcome,
    /// Label examples using the deterministic open-card solver's bounded value.
    DeterministicSolverValue,
    /// Label examples using planner root values where available.
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

fn run_autoplay_benchmark_command(
    args: &BenchmarkAutoplayArgs,
) -> CliResult<AutoplayBenchmarkResult> {
    let preset = lookup_preset(&args.preset)?;
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
    let baseline =
        benchmark_config_from_preset(lookup_preset(&args.baseline)?, args.suite.max_steps);
    let candidate =
        benchmark_config_from_preset(lookup_preset(&args.candidate)?, args.suite.max_steps);
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
    let baseline = benchmark_config_from_preset(lookup_preset(&args.baseline)?, args.max_steps);
    let candidate = benchmark_config_from_preset(lookup_preset(&args.candidate)?, args.max_steps);
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
    println!("  rank  preset                    backend             win_rate  time/game  time/move  efficiency");
    for (index, entry) in result.entries.iter().enumerate() {
        println!(
            "  {:>4}  {:<24}  {:<18?}  {:>7.3}  {:>8.2}ms  {:>8.2}ms  {:>10.3}",
            index + 1,
            entry.preset_name,
            entry.backend,
            entry.win_rate,
            entry.average_time_per_game_ms,
            entry.average_time_per_move_ms,
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
        ])
        .unwrap();
        assert!(matches!(
            compare.command,
            Command::Benchmark {
                command: BenchmarkCommand::Compare(_)
            }
        ));

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
            }
            _ => panic!("wrong command parsed"),
        }
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
        };

        let first = run_compare_presets_command(&args).unwrap();
        let second = run_compare_presets_command(&args).unwrap();

        assert_eq!(first, second);
        assert_eq!(first.entries.len(), 2);
        assert_eq!(first.entries[0].preset_name, "balanced_benchmark");
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
}
