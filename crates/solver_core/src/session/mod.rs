//! Persistence, replay logs, and session metadata surfaces.

use serde::{Deserialize, Serialize};

use crate::{
    core::BeliefState,
    types::{MoveId, SessionId},
};

/// Metadata saved with a real or simulated game session.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SessionMetadata {
    /// Session identifier.
    pub id: SessionId,
    /// Format or engine version string.
    pub version: String,
    /// Human-readable label.
    pub label: Option<String>,
}

/// Persistable session shell.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SavedSession {
    /// Metadata for the session.
    pub metadata: SessionMetadata,
    /// Current belief state.
    pub belief: BeliefState,
    /// Macro moves applied so far.
    pub move_history: Vec<MoveId>,
    /// Revealed-card history encoded as user-facing text until reveal logs are implemented.
    pub reveal_history: Vec<String>,
}

/// Reuse hint for carrying search work across real user moves.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SubtreeReuseHint {
    /// Root move that may align with the next reusable subtree.
    pub root_move: Option<MoveId>,
    /// Whether a transposition lookup should be attempted if direct reuse fails.
    pub allow_transposition_lookup: bool,
}
