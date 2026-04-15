//! Hidden-information transitions, reveal handling, and world sampling surfaces.
//!
//! Belief logic must preserve the exact uniform posterior over assignments of
//! unseen cards to face-down tableau slots. The stock/waste cycle is fully known,
//! and this module should not grow unjustified weighted-posterior machinery.

use serde::{Deserialize, Serialize};

use crate::{
    cards::Card,
    types::{ColumnId, DealSeed},
};

/// Reveal observation produced when a hidden tableau card is exposed.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RevealEvent {
    /// Column where the card was revealed.
    pub column: ColumnId,
    /// Revealed card identity.
    pub card: Card,
}

/// One branch of an exact reveal chance node.
#[derive(Debug, Copy, Clone, PartialEq, Serialize, Deserialize)]
pub struct RevealBranch {
    /// Card assigned to the revealed slot.
    pub card: Card,
    /// Branch probability under the exact uniform posterior.
    pub probability: f32,
}

/// Belief transition category.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum BeliefTransitionKind {
    /// A visible move with no reveal.
    Deterministic,
    /// A move that exposes one hidden tableau card.
    Reveal,
}

/// Request for future uniform full-world sampling.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorldSampleRequest {
    /// Reproducible seed for the sampler.
    pub seed: DealSeed,
    /// Number of sampled worlds requested.
    pub samples: usize,
}

/// Summary of sampled worlds.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorldSampleSummary {
    /// Number of samples produced.
    pub samples: usize,
    /// Number of samples rejected by validation.
    pub rejected: usize,
}
