//! Interfaces for the future perfect-information open-card solver.

use serde::{Deserialize, Serialize};

use crate::types::{MoveId, SearchSummary, ValueEstimate};

/// Deterministic solver operating mode.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SolveMode {
    /// Attempt to prove win/loss exactly within budget.
    Exact,
    /// Search within a bounded horizon and return a fallback estimate if unresolved.
    Bounded,
    /// Prefer a fast approximate value estimate.
    FastEvaluate,
}

/// Proof status returned by deterministic solving.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProofStatus {
    /// A win is proven.
    ProvenWin,
    /// A loss is proven.
    ProvenLoss,
    /// Search did not prove the state within the supplied budget.
    Unknown,
}

/// Result of a future exact or bounded deterministic solve.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExactSolveResult {
    /// Proof status reached by the solver.
    pub status: ProofStatus,
    /// Best known move when available.
    pub best_move: Option<MoveId>,
    /// Value estimate associated with the result.
    pub value: ValueEstimate,
    /// Diagnostics for the solve.
    pub summary: SearchSummary,
}

/// Recommendation from the open-card solver.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OpenCardRecommendation {
    /// Suggested move for the fully instantiated state.
    pub best_move: Option<MoveId>,
    /// Estimated value after choosing the move.
    pub value: ValueEstimate,
    /// Search diagnostics.
    pub summary: SearchSummary,
}
