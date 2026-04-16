//! Benchmark-ready configuration presets.

use serde::{Deserialize, Serialize};

use crate::{
    config::{ExperimentConfig, SolverConfig},
    late_exact::LateExactEvaluationMode,
    planner::{BeliefPlannerConfig, PlannerLeafEvalMode},
    types::DealSeed,
};

use super::{
    deterministic_search_config_from_solver, AutoplayBenchmarkConfig, AutoplayConfig,
    BenchmarkConfig, EngineConfigLabel, PimcConfig, PimcEvaluationMode, PlannerBackend,
};

/// Stable preset names accepted by library and CLI benchmark entry points.
pub const EXPERIMENT_PRESET_NAMES: &[&str] = &[
    "pimc_baseline",
    "belief_uct_default",
    "belief_uct_late_exact",
    "fast_benchmark",
    "balanced_benchmark",
    "quality_benchmark",
];

/// Looks up a benchmark preset by its stable name.
pub fn experiment_preset_by_name(name: &str) -> Option<ExperimentPreset> {
    match name {
        "pimc_baseline" => Some(pimc_baseline()),
        "belief_uct_default" => Some(belief_uct_default()),
        "belief_uct_late_exact" => Some(belief_uct_late_exact()),
        "fast_benchmark" => Some(fast_benchmark()),
        "balanced_benchmark" => Some(balanced_benchmark()),
        "quality_benchmark" => Some(quality_benchmark()),
        _ => None,
    }
}

/// Complete benchmark-ready configuration bundle.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExperimentPreset {
    /// Stable preset name used in reports.
    pub name: String,
    /// Solver configuration used by planner backends.
    pub solver: SolverConfig,
    /// Full-game autoplay controls.
    pub autoplay: AutoplayConfig,
    /// Experiment-suite defaults.
    pub experiments: ExperimentConfig,
}

impl ExperimentPreset {
    /// Converts this preset into a full-game autoplay benchmark config.
    pub fn autoplay_benchmark_config(&self) -> AutoplayBenchmarkConfig {
        AutoplayBenchmarkConfig {
            label: EngineConfigLabel::new(self.name.clone()),
            solver: self.solver.clone(),
            autoplay: self.autoplay,
        }
    }

    /// Converts this preset into a root-only PIMC benchmark config.
    pub fn root_benchmark_config(&self) -> BenchmarkConfig {
        BenchmarkConfig {
            label: EngineConfigLabel::new(self.name.clone()),
            pimc: self.autoplay.pimc,
            deterministic: deterministic_search_config_from_solver(&self.solver),
        }
    }
}

/// Baseline preset using uniform-determinization PIMC for full-game autoplay.
pub fn pimc_baseline() -> ExperimentPreset {
    let mut solver = SolverConfig::default();
    solver.deterministic.fast_eval_node_budget = 25_000;
    solver.belief_planner.rng_seed = DealSeed(0);

    let autoplay = AutoplayConfig {
        backend: PlannerBackend::Pimc,
        pimc: PimcConfig {
            sample_count: 32,
            deterministic_mode: PimcEvaluationMode::Bounded,
            shared_world_batch: true,
            rng_seed: DealSeed(0),
            report_standard_error: true,
            ..PimcConfig::default()
        },
        ..AutoplayConfig::default()
    };

    ExperimentPreset {
        name: "pimc_baseline".to_string(),
        solver,
        autoplay,
        experiments: ExperimentConfig::default(),
    }
}

/// Default belief-UCT preset with late-exact assignment evaluation disabled.
pub fn belief_uct_default() -> ExperimentPreset {
    let mut solver = SolverConfig::default();
    solver.late_exact.enabled = false;
    solver.belief_planner = BeliefPlannerConfig {
        simulation_budget: 192,
        max_depth: 8,
        leaf_world_samples: 3,
        leaf_eval_mode: PlannerLeafEvalMode::Fast,
        rng_seed: DealSeed(0),
        min_simulations_before_stop: 48,
        initial_screen_simulations: 24,
        second_reveal_refinement_simulations: 12,
        ..BeliefPlannerConfig::default()
    };

    let autoplay = AutoplayConfig {
        backend: PlannerBackend::BeliefUct,
        pimc: PimcConfig::default(),
        ..AutoplayConfig::default()
    };

    ExperimentPreset {
        name: "belief_uct_default".to_string(),
        solver,
        autoplay,
        experiments: ExperimentConfig::default(),
    }
}

/// Belief-UCT preset with the small-hidden-card late-exact evaluator enabled.
pub fn belief_uct_late_exact() -> ExperimentPreset {
    let mut preset = belief_uct_default();
    preset.name = "belief_uct_late_exact".to_string();
    preset.autoplay.backend = PlannerBackend::BeliefUctLateExact;
    preset.solver.late_exact.enabled = true;
    preset.solver.late_exact.hidden_card_threshold = 8;
    preset.solver.late_exact.max_root_actions = 2;
    preset.solver.late_exact.evaluation_mode = LateExactEvaluationMode::Bounded;
    preset
}

/// Small deterministic preset intended for smoke tests and quick local comparisons.
pub fn fast_benchmark() -> ExperimentPreset {
    let mut preset = belief_uct_late_exact();
    preset.name = "fast_benchmark".to_string();
    preset.solver.deterministic.max_macro_depth = 8;
    preset.solver.deterministic.exact_node_budget = 2_000;
    preset.solver.deterministic.fast_eval_node_budget = 250;
    preset.solver.deterministic.tt_capacity = 4_096;
    preset.solver.belief_planner.simulation_budget = 24;
    preset.solver.belief_planner.initial_screen_simulations = 8;
    preset.solver.belief_planner.min_simulations_before_stop = 12;
    preset.solver.belief_planner.leaf_world_samples = 1;
    preset.solver.belief_planner.enable_second_reveal_refinement = false;
    preset.solver.late_exact.evaluation_mode = LateExactEvaluationMode::Fast;
    preset.solver.late_exact.assignment_budget = Some(256);
    preset.autoplay.max_steps = 80;
    preset.autoplay.pimc.sample_count = 4;
    preset.autoplay.pimc.deterministic_mode = PimcEvaluationMode::Fast;
    preset.autoplay.pimc.per_world_node_budget_override = Some(250);
    preset.autoplay.pimc.max_candidate_actions = Some(4);
    preset.experiments.default_suite_size = 32;
    preset.experiments.repetitions = 2;
    preset
}

/// Middle-ground preset for practical local tuning runs.
pub fn balanced_benchmark() -> ExperimentPreset {
    let mut preset = belief_uct_late_exact();
    preset.name = "balanced_benchmark".to_string();
    preset.solver.deterministic.max_macro_depth = 32;
    preset.solver.deterministic.exact_node_budget = 100_000;
    preset.solver.deterministic.fast_eval_node_budget = 5_000;
    preset.solver.deterministic.tt_capacity = 32_768;
    preset.solver.belief_planner.simulation_budget = 128;
    preset.solver.belief_planner.initial_screen_simulations = 24;
    preset.solver.belief_planner.min_simulations_before_stop = 48;
    preset.solver.belief_planner.leaf_world_samples = 2;
    preset
        .solver
        .belief_planner
        .second_reveal_refinement_simulations = 8;
    preset.solver.late_exact.evaluation_mode = LateExactEvaluationMode::Fast;
    preset.solver.late_exact.assignment_budget = Some(1_024);
    preset.autoplay.max_steps = 200;
    preset.autoplay.pimc.sample_count = 32;
    preset.autoplay.pimc.deterministic_mode = PimcEvaluationMode::Fast;
    preset.autoplay.pimc.per_world_node_budget_override = Some(5_000);
    preset.experiments.default_suite_size = 250;
    preset.experiments.repetitions = 3;
    preset
}

/// Heavier preset intended for higher-quality local benchmark runs.
pub fn quality_benchmark() -> ExperimentPreset {
    let mut preset = belief_uct_late_exact();
    preset.name = "quality_benchmark".to_string();
    preset.solver.belief_planner.simulation_budget = 512;
    preset.solver.belief_planner.initial_screen_simulations = 64;
    preset.solver.belief_planner.min_simulations_before_stop = 128;
    preset.solver.belief_planner.leaf_world_samples = 8;
    preset.solver.belief_planner.enable_second_reveal_refinement = true;
    preset
        .solver
        .belief_planner
        .second_reveal_refinement_simulations = 32;
    preset.autoplay.pimc.sample_count = 128;
    preset.autoplay.pimc.deterministic_mode = PimcEvaluationMode::Bounded;
    preset.experiments.default_suite_size = 1_000;
    preset.experiments.repetitions = 5;
    preset
}
