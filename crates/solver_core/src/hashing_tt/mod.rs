//! Hashing, canonicalization, and transposition-table surfaces.

use serde::{Deserialize, Serialize};

use crate::types::{MoveId, SearchNodeId, ValueEstimate};

/// Zobrist-style state hash.
#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct StateHash(pub u64);

/// Distinguishes deterministic and belief-state hash domains.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum HashDomain {
    /// Fully instantiated perfect-information state.
    Deterministic,
    /// Belief state with visible structure and hidden-card set.
    Belief,
}

/// Canonicalization mode for future hashing and transposition lookups.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum CanonicalizationMode {
    /// No canonicalization beyond direct structural encoding.
    None,
    /// Apply only symmetry transformations that preserve exact equivalence.
    SafeSymmetry,
}

/// Bound type stored in deterministic transposition-table entries.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum BoundType {
    /// Exact solved or searched value.
    Exact,
    /// Lower bound.
    Lower,
    /// Upper bound.
    Upper,
}

/// Future deterministic transposition-table entry.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DeterministicTtEntry {
    /// State hash.
    pub hash: StateHash,
    /// Stored value estimate.
    pub value: ValueEstimate,
    /// Bound type for alpha/beta-style consumers.
    pub bound: BoundType,
    /// Best known move from this state.
    pub best_move: Option<MoveId>,
    /// Depth or horizon searched.
    pub depth: u16,
    /// Replacement generation.
    pub generation: u32,
    /// Whether the entry proves win/loss status.
    pub proven: bool,
}

/// Future belief planner cache entry.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BeliefTtEntry {
    /// Belief-state hash.
    pub hash: StateHash,
    /// Planner node id in the owning arena.
    pub node_id: SearchNodeId,
    /// Visit count.
    pub visits: u64,
    /// Mean value estimate at this node.
    pub mean_value: ValueEstimate,
    /// Best action hint.
    pub best_action: Option<MoveId>,
    /// Replacement generation.
    pub generation: u32,
}

/// Lightweight transposition-table diagnostics.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TranspositionTableStats {
    /// Entries currently occupied.
    pub entries: usize,
    /// Probe count.
    pub probes: u64,
    /// Successful lookup count.
    pub hits: u64,
    /// Replacement count.
    pub replacements: u64,
}
