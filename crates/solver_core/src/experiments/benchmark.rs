//! Seeded deal suites, benchmark records, comparisons, and runners.

use serde::{Deserialize, Serialize};

use crate::{
    config::SolverConfig,
    core::{BeliefState, FullState},
    deterministic_solver::DeterministicSearchConfig,
    error::SolverResult,
    ml::{LeafEvaluationMode, VNetInferenceConfig},
    types::DealSeed,
};

use super::{
    generate_benchmark_deal, mean, play_game_with_planner, standard_error,
    summarize_autoplay_benchmark, summarize_benchmark, vnet_model_path_string, AutoplayConfig,
    AutoplayTermination, PimcConfig, PlannerBackend,
};
/// A reproducible suite of deals identified by seed.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BenchmarkSuite {
    /// Human-readable suite name.
    pub name: String,
    /// Seeds included in the suite.
    pub seeds: Vec<DealSeed>,
}

impl BenchmarkSuite {
    /// Creates a benchmark suite from explicit seeds.
    pub fn new(name: impl Into<String>, seeds: Vec<DealSeed>) -> Self {
        Self {
            name: name.into(),
            seeds,
        }
    }

    /// Creates a benchmark suite from explicit seeds.
    pub fn from_seeds(name: impl Into<String>, seeds: impl IntoIterator<Item = DealSeed>) -> Self {
        Self {
            name: name.into(),
            seeds: seeds.into_iter().collect(),
        }
    }

    /// Creates a deterministic contiguous suite of seeds.
    pub fn from_seed_range(name: impl Into<String>, start: u64, count: usize) -> Self {
        Self {
            name: name.into(),
            seeds: (0..count)
                .map(|offset| DealSeed(start + offset as u64))
                .collect(),
        }
    }

    /// Creates a deterministic contiguous suite from a base seed and count.
    pub fn from_base_seed(name: impl Into<String>, base_seed: u64, count: usize) -> Self {
        Self::from_seed_range(name, base_seed, count)
    }

    /// Creates repeated deterministic suites from a base seed.
    pub fn repeated_from_base_seed(
        name: impl AsRef<str>,
        base_seed: u64,
        suite_size: usize,
        repetitions: usize,
    ) -> Vec<Self> {
        (0..repetitions)
            .map(|repetition_index| {
                Self::from_base_seed(
                    format!("{}-{repetition_index}", name.as_ref()),
                    base_seed + (repetition_index as u64 * 1_000_000),
                    suite_size,
                )
            })
            .collect()
    }

    /// Returns a compact machine-friendly description for reports.
    pub fn description(&self) -> BenchmarkSuiteDescription {
        BenchmarkSuiteDescription::from_suite(self)
    }
}

/// Compact suite metadata used by machine-readable benchmark reports.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BenchmarkSuiteDescription {
    /// Suite name.
    pub name: String,
    /// Number of seeds in the suite.
    pub seed_count: usize,
    /// First seed, if the suite is non-empty.
    pub first_seed: Option<DealSeed>,
    /// Last seed, if the suite is non-empty.
    pub last_seed: Option<DealSeed>,
    /// Base seed for contiguous generated suites.
    pub base_seed: Option<DealSeed>,
    /// Whether seeds are contiguous with step 1.
    pub contiguous: bool,
}

impl BenchmarkSuiteDescription {
    /// Builds a description from a benchmark suite.
    pub fn from_suite(suite: &BenchmarkSuite) -> Self {
        Self::from_seed_slice(suite.name.clone(), &suite.seeds)
    }

    fn from_seed_slice(name: String, seeds: &[DealSeed]) -> Self {
        let contiguous = seeds
            .windows(2)
            .all(|window| window[1].0 == window[0].0.saturating_add(1));
        Self {
            name,
            seed_count: seeds.len(),
            first_seed: seeds.first().copied(),
            last_seed: seeds.last().copied(),
            base_seed: if contiguous {
                seeds.first().copied()
            } else {
                None
            },
            contiguous,
        }
    }
}

/// Named solver parameter configuration used in A/B comparisons.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EngineConfigLabel {
    /// Stable configuration name.
    pub name: String,
}

impl EngineConfigLabel {
    /// Creates a label.
    pub fn new(name: impl Into<String>) -> Self {
        Self { name: name.into() }
    }
}

/// Benchmarkable PIMC engine configuration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BenchmarkConfig {
    /// Stable label for reports.
    pub label: EngineConfigLabel,
    /// PIMC baseline configuration.
    pub pimc: PimcConfig,
    /// Deterministic continuation configuration.
    pub deterministic: DeterministicSearchConfig,
    /// Optional V-Net inference controls used by deterministic continuations.
    pub vnet_inference: VNetInferenceConfig,
}

/// Deterministic deal generated from a seed.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BenchmarkDeal {
    /// Deal seed.
    pub seed: DealSeed,
    /// True perfect-information deal.
    pub full_state: FullState,
    /// Human-visible belief state with hidden tableau cards unknown.
    pub belief_state: BeliefState,
}

/// Per-deal benchmark record.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BenchmarkRecord {
    /// Deal seed.
    pub seed: DealSeed,
    /// Whether the benchmark counted the root recommendation as a win.
    pub win: bool,
    /// Recommended root value.
    pub value: f64,
    /// Recommendation elapsed milliseconds.
    pub elapsed_ms: u64,
    /// PIMC samples per evaluated action.
    pub sample_count: usize,
    /// Deterministic continuation nodes used.
    pub deterministic_nodes: u64,
    /// Leaf evaluation mode configured for deterministic continuations.
    pub leaf_eval_mode: LeafEvaluationMode,
    /// V-Net model path configured for this decision, if any.
    pub vnet_model_path: Option<String>,
    /// V-Net inference calls used.
    pub vnet_inferences: u64,
    /// V-Net fallback count.
    pub vnet_fallbacks: u64,
    /// V-Net inference time in microseconds.
    pub vnet_inference_elapsed_us: u64,
}

/// Result of one single-config benchmark run.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BenchmarkResult {
    /// Configuration label.
    pub config: EngineConfigLabel,
    /// Suite name.
    pub suite_name: String,
    /// Machine-friendly suite metadata.
    pub suite: BenchmarkSuiteDescription,
    /// Per-deal records in suite order.
    pub records: Vec<BenchmarkRecord>,
    /// Number of deals attempted.
    pub deals: usize,
    /// Number of wins.
    pub wins: usize,
    /// Number of losses.
    pub losses: usize,
    /// Win rate.
    pub win_rate: f64,
    /// Mean recommendation time in milliseconds.
    pub mean_time_ms: f64,
    /// Mean deterministic continuation nodes per deal.
    pub mean_nodes: f64,
    /// Mean PIMC samples per root decision.
    pub mean_samples: f64,
    /// Leaf evaluation mode configured for deterministic continuations.
    pub leaf_eval_mode: LeafEvaluationMode,
    /// V-Net model path configured for this run, if any.
    pub vnet_model_path: Option<String>,
    /// Total V-Net inference calls used.
    pub vnet_inferences: u64,
    /// Total V-Net fallback count.
    pub vnet_fallbacks: u64,
    /// Total V-Net inference time in microseconds.
    pub vnet_inference_elapsed_us: u64,
}

/// Backward-compatible compact benchmark summary.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BenchmarkSummary {
    /// Configuration label.
    pub config: EngineConfigLabel,
    /// Number of deals attempted.
    pub deals: usize,
    /// Number of wins.
    pub wins: usize,
    /// Mean recommendation or play time in milliseconds.
    pub mean_time_ms: f64,
    /// Mean expanded nodes per deal.
    pub mean_nodes: f64,
}

impl From<&BenchmarkResult> for BenchmarkSummary {
    fn from(result: &BenchmarkResult) -> Self {
        Self {
            config: result.config.clone(),
            deals: result.deals,
            wins: result.wins,
            mean_time_ms: result.mean_time_ms,
            mean_nodes: result.mean_nodes,
        }
    }
}

/// Per-seed paired A/B result.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PairedDealResult {
    /// Shared deal seed.
    pub seed: DealSeed,
    /// Baseline win bit.
    pub baseline_win: bool,
    /// Candidate win bit.
    pub candidate_win: bool,
    /// Candidate minus baseline result for this seed.
    pub paired_delta: i8,
}

/// Paired comparison summary for two configurations on the same deal suite.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PairedComparisonResult {
    /// Baseline configuration result.
    pub baseline: BenchmarkResult,
    /// Candidate configuration result.
    pub candidate: BenchmarkResult,
    /// Per-seed paired outcomes.
    pub paired_records: Vec<PairedDealResult>,
    /// Candidate win-rate minus baseline win-rate.
    pub paired_win_rate_delta: f64,
    /// Candidate-only wins.
    pub candidate_only_wins: usize,
    /// Baseline-only wins.
    pub baseline_only_wins: usize,
    /// Baseline wins.
    pub baseline_wins: usize,
    /// Candidate wins.
    pub candidate_wins: usize,
    /// Seeds where both configurations had the same outcome.
    pub same_outcome_count: usize,
    /// Standard error of paired per-seed deltas.
    pub paired_standard_error: f64,
    /// Lower 95%-style confidence bound for the paired delta.
    pub ci_lower: f64,
    /// Upper 95%-style confidence bound for the paired delta.
    pub ci_upper: f64,
}

/// Backward-compatible compact paired comparison summary.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PairedComparisonSummary {
    /// Baseline configuration summary.
    pub baseline: BenchmarkSummary,
    /// Candidate configuration summary.
    pub candidate: BenchmarkSummary,
    /// Candidate win-rate minus baseline win-rate.
    pub paired_win_rate_delta: f64,
    /// Lower confidence bound for the paired difference.
    pub ci_lower: f64,
    /// Upper confidence bound for the paired difference.
    pub ci_upper: f64,
}

/// One repeated-suite comparison summary.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RepetitionSummary {
    /// Repetition index.
    pub repetition_index: usize,
    /// Suite used for this repetition.
    pub suite: BenchmarkSuite,
    /// Paired comparison result.
    pub comparison: PairedComparisonResult,
}

/// Result of repeated paired comparisons.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RepeatedComparisonResult {
    /// Per-repetition summaries.
    pub repetitions: Vec<RepetitionSummary>,
    /// Mean paired win-rate delta across repetitions.
    pub mean_paired_win_rate_delta: f64,
    /// Standard error of repetition-level paired deltas.
    pub paired_standard_error: f64,
    /// Lower 95%-style confidence bound for the mean repetition delta.
    pub ci_lower: f64,
    /// Upper 95%-style confidence bound for the mean repetition delta.
    pub ci_upper: f64,
}

/// Benchmarkable full-game autoplay configuration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AutoplayBenchmarkConfig {
    /// Stable label for reports.
    pub label: EngineConfigLabel,
    /// Full solver configuration used by the selected backend.
    pub solver: SolverConfig,
    /// Autoplay controls.
    pub autoplay: AutoplayConfig,
}

/// Per-game full autoplay benchmark record.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AutoplayBenchmarkRecord {
    /// Deal seed.
    pub seed: DealSeed,
    /// Whether autoplay won this game.
    pub won: bool,
    /// Termination reason.
    pub termination: AutoplayTermination,
    /// Applied moves.
    pub moves_played: usize,
    /// Total planner time for the game.
    pub total_planner_time_ms: u64,
    /// Mean planner time per applied move.
    pub mean_planner_time_per_move_ms: f64,
    /// Deterministic solver nodes reported by planners.
    pub deterministic_nodes: u64,
    /// Root simulations/samples/visits reported by planners.
    pub root_visits: u64,
    /// Number of moves that used root-parallel planner workers.
    pub root_parallel_steps: usize,
    /// Sum of root workers used across this game.
    pub root_parallel_worker_count: usize,
    /// Total root-parallel worker simulations for this game.
    pub root_parallel_simulations: usize,
    /// Number of autoplay steps where late-exact triggered.
    pub late_exact_triggers: usize,
    /// Leaf evaluation mode configured for this game.
    pub leaf_eval_mode: LeafEvaluationMode,
    /// V-Net model path configured for this game, if any.
    pub vnet_model_path: Option<String>,
    /// V-Net inference calls used.
    pub vnet_inferences: u64,
    /// V-Net fallback count.
    pub vnet_fallbacks: u64,
    /// V-Net inference time in microseconds.
    pub vnet_inference_elapsed_us: u64,
}

/// Count for one autoplay termination reason.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AutoplayTerminationCount {
    /// Termination reason.
    pub termination: AutoplayTermination,
    /// Number of games that ended this way.
    pub count: usize,
}

/// Full-game autoplay benchmark result.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AutoplayBenchmarkResult {
    /// Configuration label.
    pub config: EngineConfigLabel,
    /// Planner backend used for this run.
    pub backend: PlannerBackend,
    /// Suite name.
    pub suite_name: String,
    /// Machine-friendly suite metadata.
    pub suite: BenchmarkSuiteDescription,
    /// Per-seed game records.
    pub records: Vec<AutoplayBenchmarkRecord>,
    /// Number of games attempted.
    pub games: usize,
    /// Number of wins.
    pub wins: usize,
    /// Number of losses.
    pub losses: usize,
    /// Win rate.
    pub win_rate: f64,
    /// Average moves per game.
    pub average_moves_per_game: f64,
    /// Average planner time per move.
    pub average_planner_time_per_move_ms: f64,
    /// Average total planner time per game.
    pub average_total_planner_time_per_game_ms: f64,
    /// Average deterministic nodes per game.
    pub average_deterministic_nodes: f64,
    /// Average root visits per game.
    pub average_root_visits: f64,
    /// Total moves that used root-parallel planner workers.
    pub root_parallel_step_count: usize,
    /// Average root-parallel workers per game.
    pub average_root_parallel_workers: f64,
    /// Average root-parallel worker simulations per game.
    pub average_root_parallel_simulations: f64,
    /// Total late-exact trigger count.
    pub late_exact_trigger_count: usize,
    /// Leaf evaluation mode configured for this run.
    pub leaf_eval_mode: LeafEvaluationMode,
    /// V-Net model path configured for this run, if any.
    pub vnet_model_path: Option<String>,
    /// Total V-Net inference calls used.
    pub vnet_inferences: u64,
    /// Total V-Net fallback count.
    pub vnet_fallbacks: u64,
    /// Total V-Net inference time in microseconds.
    pub vnet_inference_elapsed_us: u64,
    /// Termination reason counts.
    pub terminations: Vec<AutoplayTerminationCount>,
}

/// Compact full-game autoplay suite summary.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AutoplaySuiteSummary {
    /// Configuration label.
    pub config: EngineConfigLabel,
    /// Games attempted.
    pub games: usize,
    /// Wins.
    pub wins: usize,
    /// Win rate.
    pub win_rate: f64,
    /// Average moves per game.
    pub average_moves_per_game: f64,
    /// Average planner time per move.
    pub average_planner_time_per_move_ms: f64,
}

impl From<&AutoplayBenchmarkResult> for AutoplaySuiteSummary {
    fn from(result: &AutoplayBenchmarkResult) -> Self {
        Self {
            config: result.config.clone(),
            games: result.games,
            wins: result.wins,
            win_rate: result.win_rate,
            average_moves_per_game: result.average_moves_per_game,
            average_planner_time_per_move_ms: result.average_planner_time_per_move_ms,
        }
    }
}

/// Per-seed paired full-game autoplay result.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AutoplayPairedGameResult {
    /// Shared deal seed.
    pub seed: DealSeed,
    /// Baseline win bit.
    pub baseline_win: bool,
    /// Candidate win bit.
    pub candidate_win: bool,
    /// Candidate minus baseline outcome for this seed.
    pub paired_delta: i8,
}

/// Paired full-game autoplay comparison.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AutoplayComparisonResult {
    /// Baseline benchmark.
    pub baseline: AutoplayBenchmarkResult,
    /// Candidate benchmark.
    pub candidate: AutoplayBenchmarkResult,
    /// Per-seed paired outcomes.
    pub paired_records: Vec<AutoplayPairedGameResult>,
    /// Candidate win-rate minus baseline win-rate.
    pub paired_win_rate_delta: f64,
    /// Candidate-only wins.
    pub candidate_only_wins: usize,
    /// Baseline-only wins.
    pub baseline_only_wins: usize,
    /// Baseline wins.
    pub baseline_wins: usize,
    /// Candidate wins.
    pub candidate_wins: usize,
    /// Seeds where both configurations had the same outcome.
    pub same_outcome_count: usize,
    /// Standard error of paired per-seed deltas.
    pub paired_standard_error: f64,
    /// Lower 95%-style confidence bound for the paired delta.
    pub ci_lower: f64,
    /// Upper 95%-style confidence bound for the paired delta.
    pub ci_upper: f64,
}

/// One repeated full-game autoplay comparison.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AutoplayRepetitionSummary {
    /// Repetition index.
    pub repetition_index: usize,
    /// Suite used for this repetition.
    pub suite: BenchmarkSuite,
    /// Paired full-game comparison.
    pub comparison: AutoplayComparisonResult,
}

/// Repeated full-game autoplay comparison result.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AutoplayRepeatedComparisonResult {
    /// Per-repetition summaries.
    pub repetitions: Vec<AutoplayRepetitionSummary>,
    /// Mean paired win-rate delta across repetitions.
    pub mean_paired_win_rate_delta: f64,
    /// Standard error of repetition-level paired deltas.
    pub paired_standard_error: f64,
    /// Lower 95%-style confidence bound for the mean repetition delta.
    pub ci_lower: f64,
    /// Upper 95%-style confidence bound for the mean repetition delta.
    pub ci_upper: f64,
}

/// Experiment runner for seeded PIMC baseline benchmarks.
#[derive(Debug, Copy, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExperimentRunner;

impl ExperimentRunner {
    /// Generates a deterministic deal from a seed.
    pub fn generate_deal(&self, seed: DealSeed) -> SolverResult<BenchmarkDeal> {
        generate_benchmark_deal(seed)
    }

    /// Runs one configuration over a suite of seeds.
    pub fn run_benchmark(
        &self,
        suite: &BenchmarkSuite,
        config: &BenchmarkConfig,
    ) -> SolverResult<BenchmarkResult> {
        let mut records = Vec::with_capacity(suite.seeds.len());
        for seed in &suite.seeds {
            let deal = self.generate_deal(*seed)?;
            let recommendation = super::recommend_move_pimc_with_vnet(
                &deal.belief_state,
                config.deterministic,
                config.pimc,
                config.vnet_inference.clone(),
            )?;
            let win = recommendation.best_value >= 0.5;
            records.push(BenchmarkRecord {
                seed: *seed,
                win,
                value: recommendation.best_value,
                elapsed_ms: recommendation.elapsed_ms,
                sample_count: recommendation.sample_count,
                deterministic_nodes: recommendation.deterministic_nodes,
                leaf_eval_mode: config.deterministic.leaf_eval_mode,
                vnet_model_path: config
                    .vnet_inference
                    .model_path
                    .as_ref()
                    .map(|path| path.display().to_string()),
                vnet_inferences: recommendation.vnet_inferences,
                vnet_fallbacks: recommendation.vnet_fallbacks,
                vnet_inference_elapsed_us: recommendation.vnet_inference_elapsed_us,
            });
        }
        Ok(summarize_benchmark(config.label.clone(), suite, records))
    }

    /// Runs A/B configs on the exact same seed suite.
    pub fn run_paired_comparison(
        &self,
        suite: &BenchmarkSuite,
        baseline: &BenchmarkConfig,
        candidate: &BenchmarkConfig,
    ) -> SolverResult<PairedComparisonResult> {
        let baseline_result = self.run_benchmark(suite, baseline)?;
        let candidate_result = self.run_benchmark(suite, candidate)?;
        let mut paired_records = Vec::with_capacity(suite.seeds.len());
        let mut candidate_only_wins = 0usize;
        let mut baseline_only_wins = 0usize;

        for (baseline_record, candidate_record) in baseline_result
            .records
            .iter()
            .zip(candidate_result.records.iter())
        {
            debug_assert_eq!(baseline_record.seed, candidate_record.seed);
            let paired_delta = match (baseline_record.win, candidate_record.win) {
                (false, true) => {
                    candidate_only_wins += 1;
                    1
                }
                (true, false) => {
                    baseline_only_wins += 1;
                    -1
                }
                _ => 0,
            };
            paired_records.push(PairedDealResult {
                seed: baseline_record.seed,
                baseline_win: baseline_record.win,
                candidate_win: candidate_record.win,
                paired_delta,
            });
        }

        let paired_win_rate_delta = candidate_result.win_rate - baseline_result.win_rate;
        let paired_standard_error = standard_error(
            paired_records
                .iter()
                .map(|record| f64::from(record.paired_delta)),
        );
        let ci_lower = paired_win_rate_delta - 1.96 * paired_standard_error;
        let ci_upper = paired_win_rate_delta + 1.96 * paired_standard_error;
        let same_outcome_count = paired_records
            .iter()
            .filter(|record| record.paired_delta == 0)
            .count();
        let baseline_wins = baseline_result.wins;
        let candidate_wins = candidate_result.wins;

        Ok(PairedComparisonResult {
            paired_win_rate_delta,
            baseline: baseline_result,
            candidate: candidate_result,
            paired_records,
            candidate_only_wins,
            baseline_only_wins,
            baseline_wins,
            candidate_wins,
            same_outcome_count,
            paired_standard_error,
            ci_lower,
            ci_upper,
        })
    }

    /// Runs paired comparison over multiple independent deterministic suites.
    pub fn run_repeated_comparison(
        &self,
        suite_name: &str,
        base_seed: u64,
        suite_size: usize,
        repetitions: usize,
        baseline: &BenchmarkConfig,
        candidate: &BenchmarkConfig,
    ) -> SolverResult<RepeatedComparisonResult> {
        let mut summaries = Vec::with_capacity(repetitions);
        for (repetition_index, suite) in
            BenchmarkSuite::repeated_from_base_seed(suite_name, base_seed, suite_size, repetitions)
                .into_iter()
                .enumerate()
        {
            let comparison = self.run_paired_comparison(&suite, baseline, candidate)?;
            summaries.push(RepetitionSummary {
                repetition_index,
                suite,
                comparison,
            });
        }
        let deltas = summaries
            .iter()
            .map(|summary| summary.comparison.paired_win_rate_delta)
            .collect::<Vec<_>>();
        let mean_paired_win_rate_delta = mean(deltas.iter().copied());
        let paired_standard_error = standard_error(deltas.iter().copied());
        let ci_lower = mean_paired_win_rate_delta - 1.96 * paired_standard_error;
        let ci_upper = mean_paired_win_rate_delta + 1.96 * paired_standard_error;
        Ok(RepeatedComparisonResult {
            repetitions: summaries,
            mean_paired_win_rate_delta,
            paired_standard_error,
            ci_lower,
            ci_upper,
        })
    }

    /// Runs one full-game autoplay configuration over a seed suite.
    pub fn run_autoplay_benchmark(
        &self,
        suite: &BenchmarkSuite,
        config: &AutoplayBenchmarkConfig,
    ) -> SolverResult<AutoplayBenchmarkResult> {
        let mut records = Vec::with_capacity(suite.seeds.len());
        for seed in &suite.seeds {
            let deal = self.generate_deal(*seed)?;
            let result =
                play_game_with_planner(&deal.full_state, &config.solver, &config.autoplay)?;
            let moves_played = result.trace.len();
            records.push(AutoplayBenchmarkRecord {
                seed: *seed,
                won: result.won,
                termination: result.termination,
                moves_played,
                total_planner_time_ms: result.total_planner_time_ms,
                mean_planner_time_per_move_ms: if moves_played == 0 {
                    0.0
                } else {
                    result.total_planner_time_ms as f64 / moves_played as f64
                },
                deterministic_nodes: result.deterministic_nodes,
                root_visits: result.root_visits,
                root_parallel_steps: result.root_parallel_steps,
                root_parallel_worker_count: result.root_parallel_worker_count,
                root_parallel_simulations: result.root_parallel_simulations,
                late_exact_triggers: result.late_exact_triggers,
                leaf_eval_mode: config.solver.deterministic.leaf_eval_mode,
                vnet_model_path: vnet_model_path_string(&config.solver),
                vnet_inferences: result.vnet_inferences,
                vnet_fallbacks: result.vnet_fallbacks,
                vnet_inference_elapsed_us: result.vnet_inference_elapsed_us,
            });
        }
        Ok(summarize_autoplay_benchmark(
            config.label.clone(),
            config.autoplay.backend,
            suite,
            records,
        ))
    }

    /// Runs full-game A/B configs on the exact same seed suite.
    pub fn run_autoplay_paired_comparison(
        &self,
        suite: &BenchmarkSuite,
        baseline: &AutoplayBenchmarkConfig,
        candidate: &AutoplayBenchmarkConfig,
    ) -> SolverResult<AutoplayComparisonResult> {
        let baseline_result = self.run_autoplay_benchmark(suite, baseline)?;
        let candidate_result = self.run_autoplay_benchmark(suite, candidate)?;
        let mut paired_records = Vec::with_capacity(suite.seeds.len());
        let mut candidate_only_wins = 0usize;
        let mut baseline_only_wins = 0usize;

        for (baseline_record, candidate_record) in baseline_result
            .records
            .iter()
            .zip(candidate_result.records.iter())
        {
            debug_assert_eq!(baseline_record.seed, candidate_record.seed);
            let paired_delta = match (baseline_record.won, candidate_record.won) {
                (false, true) => {
                    candidate_only_wins += 1;
                    1
                }
                (true, false) => {
                    baseline_only_wins += 1;
                    -1
                }
                _ => 0,
            };
            paired_records.push(AutoplayPairedGameResult {
                seed: baseline_record.seed,
                baseline_win: baseline_record.won,
                candidate_win: candidate_record.won,
                paired_delta,
            });
        }

        let paired_win_rate_delta = candidate_result.win_rate - baseline_result.win_rate;
        let paired_standard_error = standard_error(
            paired_records
                .iter()
                .map(|record| f64::from(record.paired_delta)),
        );
        let ci_lower = paired_win_rate_delta - 1.96 * paired_standard_error;
        let ci_upper = paired_win_rate_delta + 1.96 * paired_standard_error;
        let same_outcome_count = paired_records
            .iter()
            .filter(|record| record.paired_delta == 0)
            .count();
        let baseline_wins = baseline_result.wins;
        let candidate_wins = candidate_result.wins;

        Ok(AutoplayComparisonResult {
            paired_win_rate_delta,
            baseline: baseline_result,
            candidate: candidate_result,
            paired_records,
            candidate_only_wins,
            baseline_only_wins,
            baseline_wins,
            candidate_wins,
            same_outcome_count,
            paired_standard_error,
            ci_lower,
            ci_upper,
        })
    }

    /// Runs repeated paired full-game autoplay comparisons.
    pub fn run_autoplay_repeated_comparison(
        &self,
        suite_name: &str,
        base_seed: u64,
        suite_size: usize,
        repetitions: usize,
        baseline: &AutoplayBenchmarkConfig,
        candidate: &AutoplayBenchmarkConfig,
    ) -> SolverResult<AutoplayRepeatedComparisonResult> {
        let mut summaries = Vec::with_capacity(repetitions);
        for (repetition_index, suite) in
            BenchmarkSuite::repeated_from_base_seed(suite_name, base_seed, suite_size, repetitions)
                .into_iter()
                .enumerate()
        {
            let comparison = self.run_autoplay_paired_comparison(&suite, baseline, candidate)?;
            summaries.push(AutoplayRepetitionSummary {
                repetition_index,
                suite,
                comparison,
            });
        }
        let deltas = summaries
            .iter()
            .map(|summary| summary.comparison.paired_win_rate_delta)
            .collect::<Vec<_>>();
        let mean_paired_win_rate_delta = mean(deltas.iter().copied());
        let paired_standard_error = standard_error(deltas.iter().copied());
        let ci_lower = mean_paired_win_rate_delta - 1.96 * paired_standard_error;
        let ci_upper = mean_paired_win_rate_delta + 1.96 * paired_standard_error;
        Ok(AutoplayRepeatedComparisonResult {
            repetitions: summaries,
            mean_paired_win_rate_delta,
            paired_standard_error,
            ci_lower,
            ci_upper,
        })
    }
}
