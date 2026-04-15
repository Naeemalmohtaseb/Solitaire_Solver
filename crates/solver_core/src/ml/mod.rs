//! Lightweight machine-learning data and adapter surfaces.

use serde::{Deserialize, Serialize};

/// Future model role.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModelRole {
    /// Full-state deterministic value network.
    VNet,
    /// Visible belief-state policy/value prior network.
    PNet,
}

/// Dataset split marker for exported training examples.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum DatasetSplit {
    /// Training data.
    Train,
    /// Validation data.
    Validation,
    /// Frozen test data.
    Test,
}

/// High-level exported example kind.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum TrainingExampleKind {
    /// Full deterministic state labeled by eventual win/loss or value.
    DeterministicValue,
    /// Belief root state labeled by planner statistics.
    BeliefPolicy,
}

/// Shape metadata for encoded state tensors.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EncodedStateShape {
    /// Number of scalar features.
    pub feature_count: usize,
    /// Number of planes for future spatial encodings.
    pub plane_count: usize,
}
