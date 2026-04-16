//! Machine-friendly benchmark report exports.

use serde::{Deserialize, Serialize};

use crate::error::SolverResult;

use super::{
    csv_table, optional_seed_string, termination_counts_string, to_pretty_json,
    AutoplayBenchmarkConfig, AutoplayBenchmarkResult, AutoplayComparisonResult,
    AutoplayRepeatedComparisonResult, AutoplayTerminationCount, BenchmarkResult, BenchmarkSuite,
    BenchmarkSuiteDescription, ExperimentRunner, PlannerBackend,
};
/// Machine-friendly root-only benchmark summary report.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BenchmarkSummaryReport {
    /// Benchmark layer.
    pub benchmark_kind: String,
    /// Backend name.
    pub backend: String,
    /// Stable config/preset name.
    pub config_preset_name: String,
    /// Suite metadata.
    pub suite: BenchmarkSuiteDescription,
    /// Deals attempted.
    pub games_played: usize,
    /// Wins counted by the root evaluator.
    pub wins: usize,
    /// Losses counted by the root evaluator.
    pub losses: usize,
    /// Win rate.
    pub win_rate: f64,
    /// Mean decision time in milliseconds.
    pub avg_planner_time_per_move_ms: f64,
    /// Mean deterministic nodes per root decision.
    pub avg_deterministic_nodes: f64,
    /// Mean sampled worlds per root decision.
    pub avg_root_visits_or_samples: f64,
}

/// Machine-friendly full-game autoplay benchmark summary report.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AutoplayBenchmarkSummaryReport {
    /// Benchmark layer.
    pub benchmark_kind: String,
    /// Backend used.
    pub backend: PlannerBackend,
    /// Stable config/preset name.
    pub config_preset_name: String,
    /// Suite metadata.
    pub suite: BenchmarkSuiteDescription,
    /// Games attempted.
    pub games_played: usize,
    /// Wins.
    pub wins: usize,
    /// Losses.
    pub losses: usize,
    /// Win rate.
    pub win_rate: f64,
    /// Average moves per game.
    pub avg_moves_per_game: f64,
    /// Average planner time per applied move.
    pub avg_planner_time_per_move_ms: f64,
    /// Average total planner time per game.
    pub avg_planner_time_per_game_ms: f64,
    /// Average deterministic nodes per game.
    pub avg_deterministic_nodes: f64,
    /// Average root visits/samples per game.
    pub avg_root_visits_or_samples: f64,
    /// Total late-exact triggers.
    pub late_exact_trigger_count: usize,
    /// Termination reason counts.
    pub termination_counts: Vec<AutoplayTerminationCount>,
}

/// Machine-friendly paired autoplay comparison report.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AutoplayComparisonSummaryReport {
    /// Baseline backend.
    pub baseline_backend: PlannerBackend,
    /// Candidate backend.
    pub candidate_backend: PlannerBackend,
    /// Baseline config/preset name.
    pub baseline_config_name: String,
    /// Candidate config/preset name.
    pub candidate_config_name: String,
    /// Suite metadata.
    pub suite: BenchmarkSuiteDescription,
    /// Games attempted per config.
    pub games_played: usize,
    /// Baseline wins.
    pub wins_a: usize,
    /// Candidate wins.
    pub wins_b: usize,
    /// Candidate minus baseline win-rate delta.
    pub paired_win_difference: f64,
    /// Candidate-only wins.
    pub candidate_only_wins: usize,
    /// Baseline-only wins.
    pub baseline_only_wins: usize,
    /// Same-outcome seeds.
    pub same_outcome_count: usize,
    /// Paired-delta standard error.
    pub paired_standard_error: f64,
    /// Lower CI-like bound.
    pub ci_lower: f64,
    /// Upper CI-like bound.
    pub ci_upper: f64,
}

impl BenchmarkResult {
    /// Builds a compact summary report for export.
    pub fn summary_report(&self) -> BenchmarkSummaryReport {
        BenchmarkSummaryReport {
            benchmark_kind: "root".to_string(),
            backend: "Pimc".to_string(),
            config_preset_name: self.config.name.clone(),
            suite: self.suite.clone(),
            games_played: self.deals,
            wins: self.wins,
            losses: self.losses,
            win_rate: self.win_rate,
            avg_planner_time_per_move_ms: self.mean_time_ms,
            avg_deterministic_nodes: self.mean_nodes,
            avg_root_visits_or_samples: self.mean_samples,
        }
    }

    /// Exports a deterministic JSON summary.
    pub fn to_json_summary(&self) -> SolverResult<String> {
        to_pretty_json(&self.summary_report())
    }

    /// Exports a deterministic one-row CSV summary.
    pub fn to_csv_summary(&self) -> String {
        csv_table(
            &[
                "benchmark_kind",
                "backend",
                "config_preset_name",
                "suite_name",
                "base_seed",
                "games_played",
                "wins",
                "losses",
                "win_rate",
                "avg_planner_time_per_move_ms",
                "avg_deterministic_nodes",
                "avg_root_visits_or_samples",
            ],
            &[vec![
                "root".to_string(),
                "Pimc".to_string(),
                self.config.name.clone(),
                self.suite.name.clone(),
                optional_seed_string(self.suite.base_seed),
                self.deals.to_string(),
                self.wins.to_string(),
                self.losses.to_string(),
                self.win_rate.to_string(),
                self.mean_time_ms.to_string(),
                self.mean_nodes.to_string(),
                self.mean_samples.to_string(),
            ]],
        )
    }
}

impl AutoplayBenchmarkResult {
    /// Builds a compact summary report for export.
    pub fn summary_report(&self) -> AutoplayBenchmarkSummaryReport {
        AutoplayBenchmarkSummaryReport {
            benchmark_kind: "autoplay".to_string(),
            backend: self.backend,
            config_preset_name: self.config.name.clone(),
            suite: self.suite.clone(),
            games_played: self.games,
            wins: self.wins,
            losses: self.losses,
            win_rate: self.win_rate,
            avg_moves_per_game: self.average_moves_per_game,
            avg_planner_time_per_move_ms: self.average_planner_time_per_move_ms,
            avg_planner_time_per_game_ms: self.average_total_planner_time_per_game_ms,
            avg_deterministic_nodes: self.average_deterministic_nodes,
            avg_root_visits_or_samples: self.average_root_visits,
            late_exact_trigger_count: self.late_exact_trigger_count,
            termination_counts: self.terminations.clone(),
        }
    }

    /// Exports a deterministic JSON summary.
    pub fn to_json_summary(&self) -> SolverResult<String> {
        to_pretty_json(&self.summary_report())
    }

    /// Exports a deterministic one-row CSV summary.
    pub fn to_csv_summary(&self) -> String {
        csv_table(
            &[
                "benchmark_kind",
                "backend",
                "config_preset_name",
                "suite_name",
                "base_seed",
                "games_played",
                "wins",
                "losses",
                "win_rate",
                "avg_moves_per_game",
                "avg_planner_time_per_move_ms",
                "avg_planner_time_per_game_ms",
                "avg_deterministic_nodes",
                "avg_root_visits_or_samples",
                "late_exact_trigger_count",
                "termination_counts",
            ],
            &[vec![
                "autoplay".to_string(),
                format!("{:?}", self.backend),
                self.config.name.clone(),
                self.suite.name.clone(),
                optional_seed_string(self.suite.base_seed),
                self.games.to_string(),
                self.wins.to_string(),
                self.losses.to_string(),
                self.win_rate.to_string(),
                self.average_moves_per_game.to_string(),
                self.average_planner_time_per_move_ms.to_string(),
                self.average_total_planner_time_per_game_ms.to_string(),
                self.average_deterministic_nodes.to_string(),
                self.average_root_visits.to_string(),
                self.late_exact_trigger_count.to_string(),
                termination_counts_string(&self.terminations),
            ]],
        )
    }

    /// Exports per-game autoplay rows as deterministic CSV.
    pub fn to_game_csv(&self) -> String {
        let rows = self
            .records
            .iter()
            .map(|record| {
                vec![
                    format!("{:?}", self.backend),
                    self.config.name.clone(),
                    self.suite.name.clone(),
                    record.seed.0.to_string(),
                    record.won.to_string(),
                    format!("{:?}", record.termination),
                    record.moves_played.to_string(),
                    record.total_planner_time_ms.to_string(),
                    record.mean_planner_time_per_move_ms.to_string(),
                    record.deterministic_nodes.to_string(),
                    record.root_visits.to_string(),
                    record.late_exact_triggers.to_string(),
                ]
            })
            .collect::<Vec<_>>();
        csv_table(
            &[
                "backend",
                "config_preset_name",
                "suite_name",
                "seed",
                "won",
                "termination",
                "moves_played",
                "total_planner_time_ms",
                "mean_planner_time_per_move_ms",
                "deterministic_nodes",
                "root_visits",
                "late_exact_triggers",
            ],
            &rows,
        )
    }
}

impl AutoplayComparisonResult {
    /// Builds a compact paired comparison report for export.
    pub fn summary_report(&self) -> AutoplayComparisonSummaryReport {
        AutoplayComparisonSummaryReport {
            baseline_backend: self.baseline.backend,
            candidate_backend: self.candidate.backend,
            baseline_config_name: self.baseline.config.name.clone(),
            candidate_config_name: self.candidate.config.name.clone(),
            suite: self.baseline.suite.clone(),
            games_played: self.baseline.games,
            wins_a: self.baseline_wins,
            wins_b: self.candidate_wins,
            paired_win_difference: self.paired_win_rate_delta,
            candidate_only_wins: self.candidate_only_wins,
            baseline_only_wins: self.baseline_only_wins,
            same_outcome_count: self.same_outcome_count,
            paired_standard_error: self.paired_standard_error,
            ci_lower: self.ci_lower,
            ci_upper: self.ci_upper,
        }
    }

    /// Exports a deterministic JSON summary.
    pub fn to_json_summary(&self) -> SolverResult<String> {
        to_pretty_json(&self.summary_report())
    }

    /// Exports a deterministic one-row CSV summary.
    pub fn to_csv_summary(&self) -> String {
        csv_table(
            &[
                "baseline_backend",
                "candidate_backend",
                "baseline_config_name",
                "candidate_config_name",
                "suite_name",
                "base_seed",
                "games_played",
                "wins_a",
                "wins_b",
                "paired_win_difference",
                "candidate_only_wins",
                "baseline_only_wins",
                "same_outcome_count",
                "paired_standard_error",
                "ci_lower",
                "ci_upper",
            ],
            &[vec![
                format!("{:?}", self.baseline.backend),
                format!("{:?}", self.candidate.backend),
                self.baseline.config.name.clone(),
                self.candidate.config.name.clone(),
                self.baseline.suite.name.clone(),
                optional_seed_string(self.baseline.suite.base_seed),
                self.baseline.games.to_string(),
                self.baseline_wins.to_string(),
                self.candidate_wins.to_string(),
                self.paired_win_rate_delta.to_string(),
                self.candidate_only_wins.to_string(),
                self.baseline_only_wins.to_string(),
                self.same_outcome_count.to_string(),
                self.paired_standard_error.to_string(),
                self.ci_lower.to_string(),
                self.ci_upper.to_string(),
            ]],
        )
    }
}

impl AutoplayRepeatedComparisonResult {
    /// Exports deterministic JSON containing all repetition summaries.
    pub fn to_json_summary(&self) -> SolverResult<String> {
        to_pretty_json(self)
    }

    /// Exports one CSV row per repetition.
    pub fn to_csv_summary(&self) -> String {
        let rows = self
            .repetitions
            .iter()
            .map(|summary| {
                vec![
                    summary.repetition_index.to_string(),
                    summary.suite.name.clone(),
                    optional_seed_string(summary.suite.description().base_seed),
                    summary.comparison.baseline.config.name.clone(),
                    summary.comparison.candidate.config.name.clone(),
                    summary.comparison.baseline_wins.to_string(),
                    summary.comparison.candidate_wins.to_string(),
                    summary.comparison.paired_win_rate_delta.to_string(),
                    summary.comparison.paired_standard_error.to_string(),
                    summary.comparison.ci_lower.to_string(),
                    summary.comparison.ci_upper.to_string(),
                ]
            })
            .collect::<Vec<_>>();
        csv_table(
            &[
                "repetition_index",
                "suite_name",
                "base_seed",
                "baseline_config_name",
                "candidate_config_name",
                "wins_a",
                "wins_b",
                "paired_win_difference",
                "paired_standard_error",
                "ci_lower",
                "ci_upper",
            ],
            &rows,
        )
    }
}

/// Runs one full-game autoplay benchmark with the default experiment runner.
pub fn run_autoplay_benchmark(
    suite: &BenchmarkSuite,
    config: &AutoplayBenchmarkConfig,
) -> SolverResult<AutoplayBenchmarkResult> {
    ExperimentRunner.run_autoplay_benchmark(suite, config)
}

/// Runs one paired full-game autoplay comparison with the default experiment runner.
pub fn run_autoplay_paired_comparison(
    suite: &BenchmarkSuite,
    baseline: &AutoplayBenchmarkConfig,
    candidate: &AutoplayBenchmarkConfig,
) -> SolverResult<AutoplayComparisonResult> {
    ExperimentRunner.run_autoplay_paired_comparison(suite, baseline, candidate)
}

/// Runs repeated paired full-game autoplay comparisons with deterministic suites.
pub fn run_autoplay_repeated_comparison(
    suite_name: &str,
    base_seed: u64,
    suite_size: usize,
    repetitions: usize,
    baseline: &AutoplayBenchmarkConfig,
    candidate: &AutoplayBenchmarkConfig,
) -> SolverResult<AutoplayRepeatedComparisonResult> {
    ExperimentRunner.run_autoplay_repeated_comparison(
        suite_name,
        base_seed,
        suite_size,
        repetitions,
        baseline,
        candidate,
    )
}

/// Exports a full-game autoplay benchmark as a JSON summary.
pub fn export_autoplay_benchmark_json(result: &AutoplayBenchmarkResult) -> SolverResult<String> {
    result.to_json_summary()
}

/// Exports a full-game autoplay benchmark as a one-row CSV summary.
pub fn export_autoplay_benchmark_csv(result: &AutoplayBenchmarkResult) -> String {
    result.to_csv_summary()
}

/// Exports per-game full-game autoplay rows as CSV.
pub fn export_autoplay_game_csv(result: &AutoplayBenchmarkResult) -> String {
    result.to_game_csv()
}

/// Exports a paired full-game autoplay comparison as a JSON summary.
pub fn export_autoplay_comparison_json(result: &AutoplayComparisonResult) -> SolverResult<String> {
    result.to_json_summary()
}

/// Exports a paired full-game autoplay comparison as a one-row CSV summary.
pub fn export_autoplay_comparison_csv(result: &AutoplayComparisonResult) -> String {
    result.to_csv_summary()
}
