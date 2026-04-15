//! Core belief-state shell for hidden tableau uncertainty.
//!
//! The stock/waste order is fully known in this project. Hidden uncertainty exists
//! only in face-down tableau slots, and the posterior over assignments of
//! `unseen_cards` to those slots is exactly uniform over all states consistent with
//! the visible tableau hidden counts. This type intentionally contains no weighted
//! posterior machinery.

use std::fmt;

use serde::{Deserialize, Serialize};

use crate::{
    cards::Card,
    error::{SolverError, SolverResult},
    types::TABLEAU_COLUMN_COUNT,
};

use super::{state::VisibleState, unseen::UnseenCardSet};

/// Hidden-information state with an exact uniform posterior over tableau assignments.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BeliefState {
    /// Current visible game state, including the only source of truth for per-column
    /// hidden tableau counts.
    pub visible: VisibleState,
    /// Cards that are not visible and must occupy face-down tableau slots.
    pub unseen_cards: UnseenCardSet,
}

impl BeliefState {
    /// Creates a belief state from visible state and the exact unseen tableau set.
    pub fn new(visible: VisibleState, unseen_cards: UnseenCardSet) -> Self {
        Self {
            visible,
            unseen_cards,
        }
    }

    /// Returns per-column hidden tableau counts derived from the visible columns.
    pub fn hidden_counts(&self) -> [u8; TABLEAU_COLUMN_COUNT] {
        self.visible.hidden_counts()
    }

    /// Returns the number of currently hidden tableau slots.
    pub fn hidden_card_count(&self) -> usize {
        self.visible.hidden_slot_count()
    }

    /// Returns the number of unseen card identities carried by this belief state.
    pub fn unseen_card_count(&self) -> usize {
        self.unseen_cards.count()
    }

    /// Validates local belief-state structure.
    pub fn validate_structure(&self) -> SolverResult<()> {
        self.visible.validate_structure()?;
        self.unseen_cards.validate_structure()
    }

    /// Validates the exact hidden-tableau belief against visible card accounting.
    pub fn validate_consistency_against_visible(&self) -> SolverResult<()> {
        self.validate_structure()?;
        self.visible.validate_consistency()?;

        let expected = self.hidden_card_count();
        let actual = self.unseen_card_count();
        if expected != actual {
            return Err(SolverError::UnseenCountMismatch { expected, actual });
        }

        let mut visible_mask = 0u64;
        let mut visible_count = 0usize;
        for card in self.visible.iter_visible_cards() {
            visible_mask |= card_bit(card);
            visible_count += 1;
        }

        for card in self.unseen_cards.iter() {
            if (visible_mask & card_bit(card)) != 0 {
                return Err(SolverError::VisibleHiddenCardOverlap(card));
            }
        }

        let total_cards = visible_count + self.unseen_card_count();
        if total_cards != Card::COUNT {
            return Err(SolverError::CardAccountingMismatch {
                actual: total_cards,
            });
        }

        Ok(())
    }

    /// Runs full validation in debug builds and becomes a no-op in release builds.
    ///
    /// This is intended for future mutation-heavy code paths that want cheap
    /// invariant assertions during development.
    pub fn debug_validate(&self) -> SolverResult<()> {
        #[cfg(debug_assertions)]
        {
            self.validate_consistency_against_visible()
        }
        #[cfg(not(debug_assertions))]
        {
            Ok(())
        }
    }
}

impl fmt::Display for BeliefState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "Belief[hidden_slots={} unseen={}]",
            self.hidden_card_count(),
            self.unseen_cards
        )?;
        write!(f, "{}", self.visible)
    }
}

const fn card_bit(card: Card) -> u64 {
    1u64 << card.index()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{core::TableauColumn, stock::CyclicStockState};

    fn stock_with_all_except(excluded: &[Card]) -> CyclicStockState {
        let mut cards = Vec::new();
        for index in 0..Card::COUNT {
            let card = Card::new(index as u8).unwrap();
            if !excluded.contains(&card) {
                cards.push(card);
            }
        }
        CyclicStockState::new(cards, None, 0, None, 3)
    }

    #[test]
    fn belief_state_rejects_unseen_visible_overlap() {
        let overlap: Card = "Ah".parse().unwrap();
        let hidden: Card = "2h".parse().unwrap();
        let mut visible = VisibleState::default();
        visible.columns[0] = TableauColumn::new(2, vec![overlap]);
        visible.stock = stock_with_all_except(&[overlap, hidden]);
        let unseen = UnseenCardSet::from_cards([overlap, hidden]).unwrap();
        let belief = BeliefState::new(visible, unseen);

        assert!(matches!(
            belief.validate_consistency_against_visible(),
            Err(SolverError::VisibleHiddenCardOverlap(card)) if card == overlap
        ));
    }

    #[test]
    fn belief_state_rejects_unseen_count_mismatch() {
        let hidden: Card = "Ah".parse().unwrap();
        let mut visible = VisibleState::default();
        visible.columns[0] = TableauColumn::new(2, Vec::new());
        visible.stock = stock_with_all_except(&[hidden]);
        let belief = BeliefState::new(visible, UnseenCardSet::from_cards([hidden]).unwrap());

        assert!(matches!(
            belief.validate_consistency_against_visible(),
            Err(SolverError::UnseenCountMismatch {
                expected: 2,
                actual: 1
            })
        ));
    }
}
