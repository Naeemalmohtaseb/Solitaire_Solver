//! Project-wide error and result types.

use thiserror::Error;

/// Convenient result alias used throughout the solver.
pub type SolverResult<T> = Result<T, SolverError>;

/// Errors produced by validation, parsing, persistence, and future solver layers.
#[derive(Debug, Error)]
pub enum SolverError {
    /// A card index outside the 0..=51 encoding range was requested.
    #[error("invalid card index {0}; expected 0..=51")]
    InvalidCardIndex(u16),

    /// A tableau column id outside the 0..=6 range was requested.
    #[error("invalid tableau column id {0}; expected 0..=6")]
    InvalidColumnId(u8),

    /// A foundation id outside the 0..=3 range was requested.
    #[error("invalid foundation id {0}; expected 0..=3")]
    InvalidFoundationId(u8),

    /// A state failed a structural invariant.
    #[error("invalid state: {0}")]
    InvalidState(String),

    /// A requested capability is part of the architecture but not implemented yet.
    #[error("unsupported operation: {0}")]
    Unsupported(&'static str),

    /// Persistence or file-system failure.
    #[error(transparent)]
    Io(#[from] std::io::Error),
}
