//! Project-wide error and result types.

use thiserror::Error;

use crate::cards::Card;

/// Convenient result alias used throughout the solver.
pub type SolverResult<T> = Result<T, SolverError>;

/// Errors produced by validation, parsing, persistence, and future solver layers.
#[derive(Debug, Error)]
pub enum SolverError {
    /// A card index outside the 0..=51 encoding range was requested.
    #[error("invalid card index {0}; expected 0..=51")]
    InvalidCardIndex(u16),

    /// A card string could not be parsed.
    #[error("invalid card text {0:?}; expected rank+suit such as Ah, Td, or Ks")]
    InvalidCardText(String),

    /// A rank value outside Ace=1 through King=13 was requested.
    #[error("invalid rank value {0}; expected 1..=13")]
    InvalidRankValue(u8),

    /// A suit index outside 0..=3 was requested.
    #[error("invalid suit index {0}; expected 0..=3")]
    InvalidSuitIndex(u8),

    /// A tableau column id outside the 0..=6 range was requested.
    #[error("invalid tableau column id {0}; expected 0..=6")]
    InvalidColumnId(u8),

    /// A foundation id outside the 0..=3 range was requested.
    #[error("invalid foundation id {0}; expected 0..=3")]
    InvalidFoundationId(u8),

    /// A tableau column failed a local structural check.
    #[error("invalid tableau column: {0}")]
    InvalidTableauColumn(String),

    /// The stock/waste state failed a local structural check.
    #[error("invalid stock/waste state: {0}")]
    InvalidStockState(String),

    /// A requested move is not legal in the current visible state.
    #[error("illegal move: {0}")]
    IllegalMove(String),

    /// A tableau uncover requires the caller to provide the observed revealed card.
    #[error("move requires a revealed card for column {column}")]
    RevealCardRequired {
        /// Column where the reveal occurs.
        column: crate::types::ColumnId,
    },

    /// A revealed card was provided for a move that does not uncover hidden tableau cards.
    #[error("unexpected revealed card {card}")]
    UnexpectedRevealCard {
        /// Revealed card that was not needed.
        card: Card,
    },

    /// A card appears more than once in a state component where cards must be unique.
    #[error("duplicate card {0}")]
    DuplicateCard(Card),

    /// A card appears both visibly and in a hidden/unseen representation.
    #[error("card {0} overlaps visible and hidden/unseen state")]
    VisibleHiddenCardOverlap(Card),

    /// A belief state's unseen card count does not match the visible hidden-slot count.
    #[error("unseen card count mismatch: expected {expected}, actual {actual}")]
    UnseenCountMismatch {
        /// Expected unseen card count.
        expected: usize,
        /// Actual unseen card count.
        actual: usize,
    },

    /// A full state's hidden assignments do not match the visible hidden-slot count.
    #[error("hidden assignment count mismatch: expected {expected}, actual {actual}")]
    HiddenAssignmentCountMismatch {
        /// Expected assignment count.
        expected: usize,
        /// Actual assignment count.
        actual: usize,
    },

    /// A hidden assignment contains the same hidden slot more than once.
    #[error("duplicate hidden slot {0}")]
    DuplicateHiddenSlot(String),

    /// A hidden slot does not exist in the visible tableau structure.
    #[error("hidden slot {slot} is out of range for column hidden count {hidden_count}")]
    HiddenSlotOutOfRange {
        /// Display form of the offending slot.
        slot: String,
        /// Hidden count in that column.
        hidden_count: u8,
    },

    /// A full or belief state does not account for all 52 cards exactly once.
    #[error("card accounting mismatch: expected 52 total cards, actual {actual}")]
    CardAccountingMismatch {
        /// Actual unique/accounted card count.
        actual: usize,
    },

    /// A state failed a structural invariant.
    #[error("invalid state: {0}")]
    InvalidState(String),

    /// A requested capability is part of the architecture but not implemented yet.
    #[error("unsupported operation: {0}")]
    Unsupported(&'static str),

    /// Machine-readable benchmark/report serialization failed.
    #[error("serialization failed: {0}")]
    Serialization(String),

    /// Persistence or file-system failure.
    #[error(transparent)]
    Io(#[from] std::io::Error),
}
