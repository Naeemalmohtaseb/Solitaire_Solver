//! Tableau column representation.

use serde::{Deserialize, Serialize};

use crate::cards::Card;

/// Maximum practical tableau column length for a standard Klondike deal.
pub const MAX_TABLEAU_COLUMN_LEN: usize = 19;

/// Visible and hidden portion of one tableau column.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TableauColumn {
    /// Number of face-down cards below the visible run.
    pub hidden_count: u8,
    /// Face-up cards in bottom-to-top order within the visible segment.
    pub face_up: Vec<Card>,
}

impl TableauColumn {
    /// Creates a tableau column from a hidden count and visible cards.
    pub fn new(hidden_count: u8, face_up: Vec<Card>) -> Self {
        Self {
            hidden_count,
            face_up,
        }
    }

    /// Returns true if the column has no face-up cards.
    pub fn is_face_up_empty(&self) -> bool {
        self.face_up.is_empty()
    }
}

impl Default for TableauColumn {
    fn default() -> Self {
        Self {
            hidden_count: 0,
            face_up: Vec::new(),
        }
    }
}
