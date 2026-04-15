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

pub use cards::{Card, Color, Rank, Suit};
pub use config::{
    BeliefPlannerConfig, DeterministicSolverConfig, ExperimentConfig, LateExactConfig,
    SearchConfig, SolverConfig,
};
pub use error::{SolverError, SolverResult};
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
