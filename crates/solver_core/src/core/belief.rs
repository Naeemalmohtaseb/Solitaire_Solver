//! Belief-state shell for hidden tableau uncertainty.

use serde::{Deserialize, Serialize};

use crate::{cards::Card, types::TABLEAU_COLUMN_COUNT};

use super::state::VisibleState;

/// Hidden-information state with an exact uniform posterior over tableau assignments.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BeliefState {
    /// Current visible game state.
    pub visible: VisibleState,
    /// Cards that are not visible and must occupy face-down tableau slots.
    pub unseen_cards: Vec<Card>,
    /// Hidden card count for each tableau column.
    pub hidden_counts: [u8; TABLEAU_COLUMN_COUNT],
}

impl BeliefState {
    /// Creates a belief state from visible state and the exact unseen tableau set.
    pub fn new(
        visible: VisibleState,
        unseen_cards: Vec<Card>,
        hidden_counts: [u8; TABLEAU_COLUMN_COUNT],
    ) -> Self {
        Self {
            visible,
            unseen_cards,
            hidden_counts,
        }
    }

    /// Returns the number of currently hidden tableau cards.
    pub fn hidden_card_count(&self) -> usize {
        self.unseen_cards.len()
    }
}
