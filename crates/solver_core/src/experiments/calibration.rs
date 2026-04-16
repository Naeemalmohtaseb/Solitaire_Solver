//! Preset comparison and calibration helpers built on autoplay benchmarks.

use serde::{Deserialize, Serialize};

use std::path::PathBuf;

use crate::{
    error::{SolverError, SolverResult},
    ml::LeafEvaluationMode,
};

use super::{
    csv_table, experiment_preset_by_name, optional_seed_string, to_pretty_json,
    AutoplayBenchmarkConfig, AutoplayBenchmarkResult, AutoplayComparisonResult, BenchmarkSuite,
    BenchmarkSuiteDescription, ExperimentPreset, ExperimentRunner, PlannerBackend,
};

/// Ranking metric used by preset comparison reports.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PresetRankingMetric {
    /// Higher win rate is better.
    WinRate,
    /// Lower average time per game is better.
    TimePerGame,
    /// Higher win rate per planner-second is better.
    Efficiency,
}

impl Default for PresetRankingMetric {
    fn default() -> Self {
        Self::Efficiency
    }
}

/// One preset's compact benchmark row inside a preset comparison.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PresetComparisonEntry {
    /// Stable preset/config name.
    pub preset_name: String,
    /// Planner backend used by this preset.
    pub backend: PlannerBackend,
    /// Games attempted.
    pub games: usize,
    /// Wins.
    pub wins: usize,
    /// Losses.
    pub losses: usize,
    /// Win rate.
    pub win_rate: f64,
    /// Average moves per game.
    pub average_moves_per_game: f64,
    /// Average planner time per game.
    pub average_time_per_game_ms: f64,
    /// Average planner time per move.
    pub average_time_per_move_ms: f64,
    /// Average deterministic nodes per game.
    pub average_deterministic_nodes: f64,
    /// Average root visits/samples per game.
    pub average_root_visits: f64,
    /// Total late-exact triggers.
    pub late_exact_trigger_count: usize,
    /// Leaf evaluation mode actually configured for this preset.
    pub leaf_eval_mode: LeafEvaluationMode,
    /// V-Net model path or artifact id, if configured.
    pub vnet_model_path: Option<String>,
    /// Total V-Net inference calls.
    pub vnet_inferences: u64,
    /// Total V-Net fallback count.
    pub vnet_fallbacks: u64,
    /// Total V-Net inference time in microseconds.
    pub vnet_inference_elapsed_us: u64,
    /// Win-rate per planner-second.
    pub win_rate_per_second: f64,
    /// Rank by win rate.
    pub win_rate_rank: usize,
    /// Rank by time per game.
    pub time_rank: usize,
    /// Rank by win-rate per second.
    pub efficiency_rank: usize,
}

impl PresetComparisonEntry {
    fn from_result(result: &AutoplayBenchmarkResult) -> Self {
        let win_rate_per_second = efficiency_score(
            result.win_rate,
            result.average_total_planner_time_per_game_ms,
        );
        Self {
            preset_name: result.config.name.clone(),
            backend: result.backend,
            games: result.games,
            wins: result.wins,
            losses: result.losses,
            win_rate: result.win_rate,
            average_moves_per_game: result.average_moves_per_game,
            average_time_per_game_ms: result.average_total_planner_time_per_game_ms,
            average_time_per_move_ms: result.average_planner_time_per_move_ms,
            average_deterministic_nodes: result.average_deterministic_nodes,
            average_root_visits: result.average_root_visits,
            late_exact_trigger_count: result.late_exact_trigger_count,
            leaf_eval_mode: result.leaf_eval_mode,
            vnet_model_path: result.vnet_model_path.clone(),
            vnet_inferences: result.vnet_inferences,
            vnet_fallbacks: result.vnet_fallbacks,
            vnet_inference_elapsed_us: result.vnet_inference_elapsed_us,
            win_rate_per_second,
            win_rate_rank: 0,
            time_rank: 0,
            efficiency_rank: 0,
        }
    }
}

/// Compact ranked comparison across several presets on the same suite.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PresetComparisonSummary {
    /// Suite metadata shared by every preset.
    pub suite: BenchmarkSuiteDescription,
    /// Ranking metric used for entry order.
    pub ranking_metric: PresetRankingMetric,
    /// Ranked preset rows.
    pub entries: Vec<PresetComparisonEntry>,
}

impl PresetComparisonSummary {
    /// Exports a deterministic JSON summary.
    pub fn to_json_summary(&self) -> SolverResult<String> {
        to_pretty_json(self)
    }

    /// Exports a deterministic CSV summary.
    pub fn to_csv_summary(&self) -> String {
        let rows = self
            .entries
            .iter()
            .map(|entry| {
                vec![
                    format!("{:?}", self.ranking_metric),
                    entry.preset_name.clone(),
                    format!("{:?}", entry.backend),
                    self.suite.name.clone(),
                    optional_seed_string(self.suite.base_seed),
                    entry.games.to_string(),
                    entry.wins.to_string(),
                    entry.losses.to_string(),
                    entry.win_rate.to_string(),
                    entry.average_time_per_game_ms.to_string(),
                    entry.average_time_per_move_ms.to_string(),
                    entry.average_deterministic_nodes.to_string(),
                    entry.average_root_visits.to_string(),
                    entry.late_exact_trigger_count.to_string(),
                    format!("{:?}", entry.leaf_eval_mode),
                    entry.vnet_model_path.clone().unwrap_or_default(),
                    entry.vnet_inferences.to_string(),
                    entry.vnet_fallbacks.to_string(),
                    entry.vnet_inference_elapsed_us.to_string(),
                    entry.win_rate_per_second.to_string(),
                    entry.win_rate_rank.to_string(),
                    entry.time_rank.to_string(),
                    entry.efficiency_rank.to_string(),
                ]
            })
            .collect::<Vec<_>>();
        csv_table(
            &[
                "ranking_metric",
                "preset_name",
                "backend",
                "suite_name",
                "base_seed",
                "games",
                "wins",
                "losses",
                "win_rate",
                "avg_time_per_game_ms",
                "avg_time_per_move_ms",
                "avg_deterministic_nodes",
                "avg_root_visits",
                "late_exact_trigger_count",
                "leaf_eval_mode",
                "vnet_model_path",
                "vnet_inferences",
                "vnet_fallbacks",
                "vnet_inference_elapsed_us",
                "win_rate_per_second",
                "win_rate_rank",
                "time_rank",
                "efficiency_rank",
            ],
            &rows,
        )
    }
}

/// Compact summary of V-Net leaf-evaluation impact for one paired comparison.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VNetImpactSummary {
    /// Non-neural baseline benchmark.
    pub baseline: AutoplayBenchmarkResult,
    /// V-Net-assisted benchmark.
    pub vnet: AutoplayBenchmarkResult,
    /// Full paired comparison result.
    pub comparison: AutoplayComparisonResult,
    /// V-Net minus baseline win-rate difference.
    pub win_rate_delta: f64,
    /// V-Net minus baseline average time per game, in milliseconds.
    pub time_per_game_delta_ms: f64,
    /// V-Net minus baseline efficiency score.
    pub efficiency_delta: f64,
    /// V-Net fallback count divided by V-Net inference attempts.
    pub fallback_rate: f64,
}

/// Compares benchmark presets on the same autoplay suite.
pub fn compare_experiment_presets_on_suite(
    suite: &BenchmarkSuite,
    presets: &[ExperimentPreset],
    ranking_metric: PresetRankingMetric,
) -> SolverResult<PresetComparisonSummary> {
    let runner = ExperimentRunner;
    let configs = presets
        .iter()
        .map(ExperimentPreset::autoplay_benchmark_config)
        .collect::<Vec<_>>();
    compare_preset_configs_on_suite(suite, &configs, ranking_metric, &runner)
}

/// Compares named benchmark presets on the same autoplay suite.
pub fn compare_named_presets_on_suite(
    suite: &BenchmarkSuite,
    preset_names: &[impl AsRef<str>],
    ranking_metric: PresetRankingMetric,
) -> SolverResult<PresetComparisonSummary> {
    let mut presets = Vec::with_capacity(preset_names.len());
    for name in preset_names {
        let name = name.as_ref();
        let preset = experiment_preset_by_name(name).ok_or_else(|| {
            SolverError::InvalidState(format!("unknown experiment preset {name:?}"))
        })?;
        presets.push(preset);
    }
    compare_experiment_presets_on_suite(suite, &presets, ranking_metric)
}

/// Compares one preset with heuristic leaves against the same preset with V-Net leaves.
pub fn compare_vnet_leaf_mode_on_suite(
    suite: &BenchmarkSuite,
    preset: &ExperimentPreset,
    vnet_model_path: impl Into<PathBuf>,
) -> SolverResult<VNetImpactSummary> {
    let baseline = heuristic_leaf_variant(preset);
    let vnet = vnet_leaf_variant(preset, vnet_model_path.into());
    let runner = ExperimentRunner;
    let baseline_config = baseline.autoplay_benchmark_config();
    let vnet_config = vnet.autoplay_benchmark_config();
    let comparison =
        runner.run_autoplay_paired_comparison(suite, &baseline_config, &vnet_config)?;
    let fallback_rate = if comparison.candidate.vnet_inferences == 0 {
        0.0
    } else {
        comparison.candidate.vnet_fallbacks as f64 / comparison.candidate.vnet_inferences as f64
    };
    Ok(VNetImpactSummary {
        baseline: comparison.baseline.clone(),
        vnet: comparison.candidate.clone(),
        win_rate_delta: comparison.paired_win_rate_delta,
        time_per_game_delta_ms: comparison.candidate.average_total_planner_time_per_game_ms
            - comparison.baseline.average_total_planner_time_per_game_ms,
        efficiency_delta: efficiency_score(
            comparison.candidate.win_rate,
            comparison.candidate.average_total_planner_time_per_game_ms,
        ) - efficiency_score(
            comparison.baseline.win_rate,
            comparison.baseline.average_total_planner_time_per_game_ms,
        ),
        fallback_rate,
        comparison,
    })
}

fn heuristic_leaf_variant(preset: &ExperimentPreset) -> ExperimentPreset {
    let mut baseline = preset.clone();
    baseline.name = format!("{}_heuristic", preset.name);
    baseline.solver.deterministic.leaf_eval_mode = LeafEvaluationMode::Heuristic;
    baseline.solver.deterministic.vnet_inference.enable_vnet = false;
    baseline.solver.deterministic.vnet_inference.model_path = None;
    baseline
}

fn vnet_leaf_variant(preset: &ExperimentPreset, model_path: PathBuf) -> ExperimentPreset {
    let mut vnet = preset.clone();
    vnet.name = format!("{}_vnet", preset.name);
    vnet.solver.deterministic.leaf_eval_mode = LeafEvaluationMode::VNet;
    vnet.solver.deterministic.vnet_inference.enable_vnet = true;
    vnet.solver.deterministic.vnet_inference.model_path = Some(model_path);
    vnet
}

fn compare_preset_configs_on_suite(
    suite: &BenchmarkSuite,
    configs: &[AutoplayBenchmarkConfig],
    ranking_metric: PresetRankingMetric,
    runner: &ExperimentRunner,
) -> SolverResult<PresetComparisonSummary> {
    let mut entries = Vec::with_capacity(configs.len());
    for config in configs {
        let result = runner.run_autoplay_benchmark(suite, config)?;
        entries.push(PresetComparisonEntry::from_result(&result));
    }
    assign_ranks(&mut entries);
    sort_entries(&mut entries, ranking_metric);
    Ok(PresetComparisonSummary {
        suite: suite.description(),
        ranking_metric,
        entries,
    })
}

fn assign_ranks(entries: &mut [PresetComparisonEntry]) {
    let mut order = (0..entries.len()).collect::<Vec<_>>();

    order.sort_by(|left, right| {
        entries[*right]
            .win_rate
            .total_cmp(&entries[*left].win_rate)
            .then_with(|| entries[*left].preset_name.cmp(&entries[*right].preset_name))
    });
    for (rank, index) in order.iter().enumerate() {
        entries[*index].win_rate_rank = rank + 1;
    }

    order.sort_by(|left, right| {
        entries[*left]
            .average_time_per_game_ms
            .total_cmp(&entries[*right].average_time_per_game_ms)
            .then_with(|| entries[*left].preset_name.cmp(&entries[*right].preset_name))
    });
    for (rank, index) in order.iter().enumerate() {
        entries[*index].time_rank = rank + 1;
    }

    order.sort_by(|left, right| {
        entries[*right]
            .win_rate_per_second
            .total_cmp(&entries[*left].win_rate_per_second)
            .then_with(|| entries[*left].preset_name.cmp(&entries[*right].preset_name))
    });
    for (rank, index) in order.iter().enumerate() {
        entries[*index].efficiency_rank = rank + 1;
    }
}

fn sort_entries(entries: &mut [PresetComparisonEntry], ranking_metric: PresetRankingMetric) {
    match ranking_metric {
        PresetRankingMetric::WinRate => entries.sort_by(|left, right| {
            left.win_rate_rank
                .cmp(&right.win_rate_rank)
                .then_with(|| left.preset_name.cmp(&right.preset_name))
        }),
        PresetRankingMetric::TimePerGame => entries.sort_by(|left, right| {
            left.time_rank
                .cmp(&right.time_rank)
                .then_with(|| left.preset_name.cmp(&right.preset_name))
        }),
        PresetRankingMetric::Efficiency => entries.sort_by(|left, right| {
            left.efficiency_rank
                .cmp(&right.efficiency_rank)
                .then_with(|| left.preset_name.cmp(&right.preset_name))
        }),
    }
}

const fn efficiency_score(win_rate: f64, average_time_per_game_ms: f64) -> f64 {
    if average_time_per_game_ms <= f64::EPSILON {
        if win_rate > 0.0 {
            f64::INFINITY
        } else {
            0.0
        }
    } else {
        win_rate / (average_time_per_game_ms / 1_000.0)
    }
}
