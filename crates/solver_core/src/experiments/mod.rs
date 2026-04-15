//! Seeded deals, A/B benchmarking, and regression experiment surfaces.

use serde::{Deserialize, Serialize};

use crate::types::DealSeed;

/// A reproducible suite of deals identified by seed.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BenchmarkSuite {
    /// Human-readable suite name.
    pub name: String,
    /// Seeds included in the suite.
    pub seeds: Vec<DealSeed>,
}

/// Named solver parameter configuration used in A/B comparisons.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EngineConfigLabel {
    /// Stable configuration name.
    pub name: String,
}

/// Summary for one benchmark run.
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

/// Paired comparison summary for two configurations on the same deal suite.
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
