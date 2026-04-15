//! Compact unseen-card set for belief states.

use std::fmt;

use serde::{Deserialize, Serialize};

use crate::{
    cards::Card,
    error::{SolverError, SolverResult},
};

const VALID_CARD_MASK: u64 = (1u64 << Card::COUNT) - 1;

/// Bitmask-backed set of unseen tableau-hidden cards.
///
/// This set stores card identities only. It does not store hidden-slot counts;
/// those live in `VisibleState.columns[*].hidden_count`.
#[derive(Debug, Copy, Clone, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct UnseenCardSet {
    mask: u64,
}

impl UnseenCardSet {
    /// Creates an empty unseen-card set.
    pub const fn empty() -> Self {
        Self { mask: 0 }
    }

    /// Creates a set from a raw bitmask after validating that only card bits are set.
    pub fn from_mask(mask: u64) -> SolverResult<Self> {
        let set = Self { mask };
        set.validate_structure()?;
        Ok(set)
    }

    /// Creates a set from unique cards.
    pub fn from_cards(cards: impl IntoIterator<Item = Card>) -> SolverResult<Self> {
        let mut set = Self::empty();
        for card in cards {
            if !set.insert(card) {
                return Err(SolverError::DuplicateCard(card));
            }
        }
        Ok(set)
    }

    /// Returns the raw bitmask.
    pub const fn mask(self) -> u64 {
        self.mask
    }

    /// Returns true if the set contains no cards.
    pub const fn is_empty(self) -> bool {
        self.mask == 0
    }

    /// Returns the number of cards in the set.
    pub const fn count(self) -> usize {
        self.mask.count_ones() as usize
    }

    /// Returns true if the card is present.
    pub const fn contains(self, card: Card) -> bool {
        (self.mask & card_bit(card)) != 0
    }

    /// Inserts a card and returns true if it was not already present.
    pub fn insert(&mut self, card: Card) -> bool {
        let bit = card_bit(card);
        let was_absent = (self.mask & bit) == 0;
        self.mask |= bit;
        was_absent
    }

    /// Removes a card and returns true if it was present.
    pub fn remove(&mut self, card: Card) -> bool {
        let bit = card_bit(card);
        let was_present = (self.mask & bit) != 0;
        self.mask &= !bit;
        was_present
    }

    /// Iterates cards in compact-card order.
    pub const fn iter(self) -> UnseenCardIter {
        UnseenCardIter {
            mask: self.mask,
            next_index: 0,
        }
    }

    /// Returns cards in deterministic compact-card order.
    pub fn to_sorted_vec(self) -> Vec<Card> {
        self.iter().collect()
    }

    /// Validates that the set contains only bits for the 52-card deck.
    pub fn validate_structure(self) -> SolverResult<()> {
        if (self.mask & !VALID_CARD_MASK) != 0 {
            return Err(SolverError::InvalidState(
                "unseen card set contains bits outside the 52-card deck".to_string(),
            ));
        }
        Ok(())
    }
}

impl IntoIterator for UnseenCardSet {
    type IntoIter = UnseenCardIter;
    type Item = Card;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl fmt::Display for UnseenCardSet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("{")?;
        for (index, card) in self.iter().enumerate() {
            if index > 0 {
                f.write_str(" ")?;
            }
            write!(f, "{card}")?;
        }
        f.write_str("}")
    }
}

/// Deterministic iterator over an `UnseenCardSet`.
#[derive(Debug, Copy, Clone)]
pub struct UnseenCardIter {
    mask: u64,
    next_index: u8,
}

impl Iterator for UnseenCardIter {
    type Item = Card;

    fn next(&mut self) -> Option<Self::Item> {
        while usize::from(self.next_index) < Card::COUNT {
            let index = self.next_index;
            self.next_index += 1;
            let card = Card::new(index).expect("iterator index is always a valid card");
            if (self.mask & card_bit(card)) != 0 {
                return Some(card);
            }
        }
        None
    }
}

const fn card_bit(card: Card) -> u64 {
    1u64 << card.index()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unseen_card_set_insert_remove_contains_and_iterate() {
        let ace_clubs: Card = "Ac".parse().unwrap();
        let king_spades: Card = "Ks".parse().unwrap();
        let ten_diamonds: Card = "Td".parse().unwrap();
        let mut set = UnseenCardSet::empty();

        assert!(set.is_empty());
        assert!(set.insert(king_spades));
        assert!(!set.insert(king_spades));
        assert!(set.insert(ace_clubs));
        assert!(set.insert(ten_diamonds));
        assert_eq!(set.count(), 3);
        assert!(set.contains(king_spades));
        assert_eq!(
            set.to_sorted_vec(),
            vec![ace_clubs, ten_diamonds, king_spades]
        );
        assert_eq!(set.to_string(), "{Ac Td Ks}");

        assert!(set.remove(ten_diamonds));
        assert!(!set.remove(ten_diamonds));
        assert_eq!(set.to_sorted_vec(), vec![ace_clubs, king_spades]);
    }

    #[test]
    fn unseen_card_set_rejects_duplicate_input() {
        let ace_hearts: Card = "Ah".parse().unwrap();
        assert!(UnseenCardSet::from_cards([ace_hearts, ace_hearts]).is_err());
    }
}
