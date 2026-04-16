//! Configuration surfaces for search, solving, planning, and experiments.

use serde::{Deserialize, Serialize};

use crate::{closure::ClosureConfig, late_exact::LateExactConfig, planner::BeliefPlannerConfig};

/// Top-level configuration object passed into future recommendation calls.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SolverConfig {
    /// Generic budget and execution controls shared by solver layers.
    pub search: SearchConfig,
    /// Controls for deterministic closure and corridor compression.
    pub closure: ClosureConfig,
    /// Controls for the deterministic perfect-information solver.
    pub deterministic: DeterministicSolverConfig,
    /// Controls for the hidden-information belief planner.
    pub belief_planner: BeliefPlannerConfig,
    /// Controls for exact assignment search in small-hidden-card regimes.
    pub late_exact: LateExactConfig,
    /// Controls for benchmark and parameter-comparison runs.
    pub experiments: ExperimentConfig,
}

impl Default for SolverConfig {
    fn default() -> Self {
        Self {
            search: SearchConfig::default(),
            closure: ClosureConfig::default(),
            deterministic: DeterministicSolverConfig::default(),
            belief_planner: BeliefPlannerConfig::default(),
            late_exact: LateExactConfig::default(),
            experiments: ExperimentConfig::default(),
        }
    }
}

/// Shared search budget and execution controls.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SearchConfig {
    /// Wall-clock budget for a recommendation request, in milliseconds.
    pub wall_clock_limit_ms: u64,
    /// Optional global node budget for bounded searches.
    pub node_budget: Option<u64>,
    /// Optional deterministic seed for reproducible planner behavior.
    pub rng_seed: Option<u64>,
    /// Number of independent root workers to use when parallel search is enabled.
    pub root_workers: usize,
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            wall_clock_limit_ms: 15_000,
            node_budget: Some(1_000_000),
            rng_seed: None,
            root_workers: 1,
        }
    }
}

/// Configuration for the future deterministic open-card solver.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeterministicSolverConfig {
    /// Default macro-depth limit for bounded solve attempts.
    pub max_macro_depth: u16,
    /// Node budget for exact or proof-oriented deterministic searches.
    pub exact_node_budget: u64,
    /// Node budget for fast deterministic value estimates.
    pub fast_eval_node_budget: u64,
    /// Whether validated hard dominance pruning is enabled.
    pub enable_dominance_pruning: bool,
    /// Whether foundation-to-tableau retreat moves are generated.
    pub enable_foundation_retreats: bool,
    /// Whether the deterministic open-card transposition table is enabled.
    pub enable_tt: bool,
    /// Number of slots in the deterministic open-card transposition table.
    pub tt_capacity: usize,
    /// Whether budget-limited heuristic values may be stored in the TT.
    pub tt_store_approx: bool,
}

impl Default for DeterministicSolverConfig {
    fn default() -> Self {
        Self {
            max_macro_depth: 128,
            exact_node_budget: 2_000_000,
            fast_eval_node_budget: 50_000,
            enable_dominance_pruning: true,
            enable_foundation_retreats: true,
            enable_tt: true,
            tt_capacity: 65_536,
            tt_store_approx: true,
        }
    }
}

/// Configuration for seeded simulation and A/B benchmark runs.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExperimentConfig {
    /// Number of deals in a default generated benchmark suite.
    pub default_suite_size: usize,
    /// Number of independent suite repetitions for variance checks.
    pub repetitions: usize,
    /// Whether machine-readable JSON reports should be emitted.
    pub emit_json: bool,
    /// Whether compact CSV summaries should be emitted.
    pub emit_csv: bool,
}

impl Default for ExperimentConfig {
    fn default() -> Self {
        Self {
            default_suite_size: 1_000,
            repetitions: 5,
            emit_json: true,
            emit_csv: true,
        }
    }
}
