//! Deterministic closure and corridor-compression interfaces.

use serde::{Deserialize, Serialize};

use crate::types::MoveId;

/// Reason a future closure pass stopped.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ClosureStopReason {
    /// No additional automatic action was available.
    Stable,
    /// A hidden tableau card was revealed.
    RevealEvent,
    /// Multiple meaningful macro choices remain.
    BranchingDecision,
    /// Empty-column commitment needs explicit planner choice.
    EmptyColumnDecision,
    /// Stock/waste position reached a meaningful pivot.
    StockPivot,
    /// Configured corridor depth was reached.
    DepthLimit,
    /// Repeated equivalent state risk was detected.
    LoopRisk,
    /// Terminal win or loss state was reached.
    Terminal,
}

/// One event in a future closure debug transcript.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ClosureEvent {
    /// Move applied during closure, if the event corresponds to a move.
    pub move_id: Option<MoveId>,
    /// Human-readable event label for logs and debugging.
    pub label: String,
}

/// Debug transcript emitted by deterministic closure.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ClosureTranscript {
    /// Ordered closure events.
    pub events: Vec<ClosureEvent>,
}

/// Summary of a future closure pass.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ClosureOutcome {
    /// Why closure stopped.
    pub stop_reason: ClosureStopReason,
    /// Whether a hidden card was revealed.
    pub revealed_card: bool,
    /// Whether a terminal state was reached.
    pub terminal: bool,
    /// Debug transcript for diagnostics and UX.
    pub transcript: ClosureTranscript,
}
