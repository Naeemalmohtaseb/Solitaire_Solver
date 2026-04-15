//! Small shared types and result surfaces used across solver modules.

use serde::{Deserialize, Serialize};

use crate::error::{SolverError, SolverResult};

/// Number of tableau columns in Klondike.
pub const TABLEAU_COLUMN_COUNT: usize = 7;

/// Number of foundations in Klondike.
pub const FOUNDATION_COUNT: usize = 4;

/// Stable identifier for a macro or atomic move at a node.
#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct MoveId(pub u32);

impl MoveId {
    /// Creates a move id from a compact integer.
    pub const fn new(value: u32) -> Self {
        Self(value)
    }
}

/// Tableau column id in the range 0..=6.
#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct ColumnId(u8);

impl ColumnId {
    /// Creates a checked column id.
    pub fn new(index: u8) -> SolverResult<Self> {
        if usize::from(index) < TABLEAU_COLUMN_COUNT {
            Ok(Self(index))
        } else {
            Err(SolverError::InvalidColumnId(index))
        }
    }

    /// Returns the zero-based column index.
    pub const fn index(self) -> u8 {
        self.0
    }
}

/// Foundation id in the range 0..=3.
#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct FoundationId(u8);

impl FoundationId {
    /// Creates a checked foundation id.
    pub fn new(index: u8) -> SolverResult<Self> {
        if usize::from(index) < FOUNDATION_COUNT {
            Ok(Self(index))
        } else {
            Err(SolverError::InvalidFoundationId(index))
        }
    }

    /// Returns the zero-based foundation index.
    pub const fn index(self) -> u8 {
        self.0
    }
}

/// Stable session identifier for a real or simulated game.
#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct SessionId(pub u128);

/// Reproducible random deal seed.
#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct DealSeed(pub u64);

/// Stable identifier for a future planner or deterministic search node.
#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct SearchNodeId(pub u64);

/// Closed interval used for confidence reporting.
#[derive(Debug, Copy, Clone, PartialEq, Serialize, Deserialize)]
pub struct ConfidenceInterval {
    /// Lower bound of the interval.
    pub lower: f32,
    /// Upper bound of the interval.
    pub upper: f32,
}

/// Value estimate for a state or action.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ValueEstimate {
    /// Estimated probability of eventually winning from this point.
    pub win_probability: f32,
    /// Optional confidence interval around the estimate.
    pub confidence: Option<ConfidenceInterval>,
}

/// Diagnostic summary shared by deterministic and belief searches.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SearchSummary {
    /// Number of nodes expanded.
    pub nodes_expanded: u64,
    /// Number of transposition-table hits.
    pub tt_hits: u64,
    /// Elapsed wall-clock time in milliseconds.
    pub elapsed_ms: u64,
    /// Number of reveal branches explicitly expanded.
    pub reveal_branches_expanded: u64,
    /// Whether late-game exact assignment mode was entered.
    pub late_exact_triggered: bool,
}

impl SearchSummary {
    /// Returns an empty summary suitable for unstarted placeholder results.
    pub const fn empty() -> Self {
        Self {
            nodes_expanded: 0,
            tt_hits: 0,
            elapsed_ms: 0,
            reveal_branches_expanded: 0,
            late_exact_triggered: false,
        }
    }
}

/// Evaluation summary for one candidate action.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ActionEvaluation {
    /// Candidate move id.
    pub move_id: MoveId,
    /// Estimated value for choosing this action.
    pub value: ValueEstimate,
    /// Number of visits or samples backing this action.
    pub visits: u64,
}

/// Public recommendation result returned by the future planner.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MoveRecommendation {
    /// Best move selected by the planner.
    pub best_move: MoveId,
    /// Value estimate for the selected move.
    pub expected_value: ValueEstimate,
    /// Alternative actions worth showing to callers.
    pub alternatives: Vec<ActionEvaluation>,
    /// Search diagnostics for the recommendation request.
    pub summary: SearchSummary,
}
