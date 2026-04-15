//! Event-driven belief-state sparse UCT planner surfaces.

use serde::{Deserialize, Serialize};

use crate::{
    config::BeliefPlannerConfig,
    types::{ActionEvaluation, MoveId, SearchNodeId, SearchSummary},
};

/// Logical node kinds for the future event-driven planner.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PlannerNodeKind {
    /// Meaningful belief-state action choice.
    Decision,
    /// Compressed deterministic transition region.
    DeterministicCorridor,
    /// Chance node representing hidden-card reveal.
    ChanceReveal,
}

/// Per-root-action statistics tracked by the planner.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RootActionStats {
    /// Action evaluation surface.
    pub evaluation: ActionEvaluation,
    /// Online variance accumulator.
    pub variance_m2: f64,
    /// Whether first-reveal frontier expansion is complete.
    pub exact_frontier_complete: bool,
    /// Number of reveal branches expanded under this action.
    pub reveal_branches_expanded: u64,
    /// Best known continuation move.
    pub best_continuation: Option<MoveId>,
}

/// Request object for a future planner run.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PlannerRunRequest {
    /// Planner-specific configuration.
    pub config: BeliefPlannerConfig,
}

/// Summary returned by a future root-search worker.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorkerSummary {
    /// Worker node id or arena root id.
    pub worker_root: Option<SearchNodeId>,
    /// Diagnostics accumulated by this worker.
    pub summary: SearchSummary,
}
