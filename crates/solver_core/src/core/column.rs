//! Tableau column representation.

use std::fmt;

use serde::{Deserialize, Serialize};

use crate::{
    cards::Card,
    error::{SolverError, SolverResult},
};

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

    /// Returns true if the column contains no hidden or face-up cards.
    pub fn is_empty(&self) -> bool {
        self.hidden_count == 0 && self.face_up.is_empty()
    }

    /// Returns true if the column has no face-up cards.
    pub fn is_face_up_empty(&self) -> bool {
        self.face_up.is_empty()
    }

    /// Returns the number of face-up cards.
    pub fn face_up_len(&self) -> usize {
        self.face_up.len()
    }

    /// Returns the total column length, including hidden and face-up cards.
    pub fn total_len(&self) -> usize {
        usize::from(self.hidden_count) + self.face_up.len()
    }

    /// Returns the top visible card, if any.
    pub fn top_face_up(&self) -> Option<Card> {
        self.face_up.last().copied()
    }

    /// Validates local tableau structure.
    pub fn validate_structure(&self) -> SolverResult<()> {
        if self.total_len() > MAX_TABLEAU_COLUMN_LEN {
            return Err(SolverError::InvalidTableauColumn(format!(
                "column length {} exceeds maximum {}",
                self.total_len(),
                MAX_TABLEAU_COLUMN_LEN
            )));
        }

        for (left_index, left) in self.face_up.iter().enumerate() {
            if self
                .face_up
                .iter()
                .skip(left_index + 1)
                .any(|right| right == left)
            {
                return Err(SolverError::DuplicateCard(*left));
            }
        }

        for window in self.face_up.windows(2) {
            let lower = window[0];
            let upper = window[1];
            if !upper.can_tableau_stack_on(lower) {
                return Err(SolverError::InvalidTableauColumn(format!(
                    "face-up run is not descending alternating at {upper} on {lower}"
                )));
            }
        }

        Ok(())
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

impl fmt::Display for TableauColumn {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("[")?;
        for index in 0..self.hidden_count {
            if index > 0 {
                f.write_str(" ")?;
            }
            f.write_str("##")?;
        }
        if self.hidden_count > 0 && !self.face_up.is_empty() {
            f.write_str(" | ")?;
        }
        for (index, card) in self.face_up.iter().enumerate() {
            if index > 0 {
                f.write_str(" ")?;
            }
            write!(f, "{card}")?;
        }
        f.write_str("]")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tableau_column_helpers_report_shape() {
        let column = TableauColumn::new(2, vec!["7s".parse().unwrap(), "6h".parse().unwrap()]);

        assert!(!column.is_empty());
        assert_eq!(column.face_up_len(), 2);
        assert_eq!(column.total_len(), 4);
        assert_eq!(column.top_face_up().unwrap().to_string(), "6h");
        assert_eq!(column.to_string(), "[## ## | 7s 6h]");
        column.validate_structure().unwrap();
    }

    #[test]
    fn tableau_column_rejects_bad_face_up_run() {
        let column = TableauColumn::new(0, vec!["7s".parse().unwrap(), "6c".parse().unwrap()]);

        assert!(column.validate_structure().is_err());
    }
}
