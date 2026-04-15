//! Exact hidden-assignment search surfaces for late-game positions.

use serde::{Deserialize, Serialize};

use crate::types::{MoveId, ValueEstimate};

/// Assignment enumeration strategy.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AssignmentSearchMode {
    /// Enumerate all assignments up to the configured budget.
    Exhaustive,
    /// Enumerate in value-guided order once bounds exist.
    BranchAndBound,
}

/// Request for exact hidden-assignment evaluation.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AssignmentSearchRequest {
    /// Hidden-card threshold that made this regime eligible.
    pub hidden_threshold: u8,
    /// Search mode.
    pub mode: AssignmentSearchMode,
    /// Optional assignment budget.
    pub max_assignments: Option<u64>,
}

/// Aggregate value over hidden assignments for one candidate action.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AssignmentAggregate {
    /// Root action being evaluated.
    pub move_id: MoveId,
    /// Number of assignments evaluated.
    pub assignments_evaluated: u64,
    /// Expected value over evaluated assignments.
    pub expected_value: ValueEstimate,
    /// Whether all consistent assignments were evaluated.
    pub exhaustive: bool,
}
