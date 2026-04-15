//! Visible and full deterministic game state shells.
//!
//! Stock/waste order is represented as fully known and deterministic. Hidden
//! uncertainty is restricted to face-down tableau cards.

use std::fmt;

use serde::{Deserialize, Serialize};

use crate::{
    cards::{Card, Rank, Suit},
    error::{SolverError, SolverResult},
    stock::CyclicStockState,
    types::{ColumnId, FOUNDATION_COUNT, TABLEAU_COLUMN_COUNT},
};

use super::{
    column::TableauColumn,
    hidden::{HiddenAssignments, HiddenSlot},
};

/// Suit-indexed foundation top ranks.
///
/// Each entry stores the top rank currently present for that suit. Lower cards
/// of the same suit are implied to be in the foundation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FoundationState {
    /// Top rank per suit, indexed by `Suit::offset()`.
    pub top_ranks: [Option<Rank>; FOUNDATION_COUNT],
}

impl FoundationState {
    /// Creates an empty foundation state.
    pub const fn empty() -> Self {
        Self {
            top_ranks: [None; FOUNDATION_COUNT],
        }
    }

    /// Returns the top rank for a suit.
    pub fn top_rank(&self, suit: Suit) -> Option<Rank> {
        self.top_ranks[usize::from(suit.offset())]
    }

    /// Sets the top rank for a suit.
    pub fn set_top_rank(&mut self, suit: Suit, rank: Option<Rank>) {
        self.top_ranks[usize::from(suit.offset())] = rank;
    }

    /// Returns all cards implied by the foundation tops in deterministic order.
    pub fn cards(&self) -> Vec<Card> {
        self.iter_cards().collect()
    }

    /// Iterates all cards implied by the foundation tops in deterministic order.
    pub fn iter_cards(&self) -> FoundationCardIter {
        FoundationCardIter {
            top_ranks: self.top_ranks,
            suit_index: 0,
            next_value: 1,
        }
    }

    /// Returns the number of cards implied by the foundation tops.
    pub fn card_count(&self) -> usize {
        self.top_ranks
            .iter()
            .flatten()
            .map(|rank| usize::from(rank.value()))
            .sum()
    }

    /// Returns true if all foundations are complete.
    pub fn is_complete(&self) -> bool {
        self.top_ranks
            .iter()
            .all(|rank| matches!(rank, Some(Rank::King)))
    }

    /// Validates local foundation structure.
    pub fn validate_structure(&self) -> SolverResult<()> {
        Ok(())
    }
}

/// Non-allocating iterator over cards implied by suit-indexed foundation tops.
#[derive(Debug, Copy, Clone)]
pub struct FoundationCardIter {
    top_ranks: [Option<Rank>; FOUNDATION_COUNT],
    suit_index: usize,
    next_value: u8,
}

impl Iterator for FoundationCardIter {
    type Item = Card;

    fn next(&mut self) -> Option<Self::Item> {
        while self.suit_index < FOUNDATION_COUNT {
            if let Some(top_rank) = self.top_ranks[self.suit_index] {
                if self.next_value <= top_rank.value() {
                    let suit = Suit::from_index(self.suit_index as u8)
                        .expect("foundation indices are valid suit indices");
                    let rank =
                        Rank::from_value(self.next_value).expect("foundation rank value is valid");
                    self.next_value += 1;
                    return Some(Card::from_suit_rank(suit, rank));
                }
            }

            self.suit_index += 1;
            self.next_value = 1;
        }

        None
    }
}

impl Default for FoundationState {
    fn default() -> Self {
        Self::empty()
    }
}

impl fmt::Display for FoundationState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("F[")?;
        for suit_index in 0..FOUNDATION_COUNT {
            if suit_index > 0 {
                f.write_str(" ")?;
            }
            let suit =
                Suit::from_index(suit_index as u8).expect("foundation index is a valid suit");
            match self.top_ranks[suit_index] {
                Some(rank) => write!(f, "{suit}:{rank}")?,
                None => write!(f, "{suit}:-")?,
            }
        }
        f.write_str("]")
    }
}

/// User-visible Draw-3 Klondike state.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VisibleState {
    /// Suit-indexed foundation top ranks.
    pub foundations: FoundationState,
    /// The seven tableau columns. Each column owns its hidden-card count; belief
    /// states derive hidden counts from this field instead of duplicating them.
    pub columns: [TableauColumn; TABLEAU_COLUMN_COUNT],
    /// Fully known draw-3 stock/waste cycle state.
    pub stock: CyclicStockState,
}

impl VisibleState {
    /// Returns per-column hidden tableau counts.
    pub fn hidden_counts(&self) -> [u8; TABLEAU_COLUMN_COUNT] {
        std::array::from_fn(|index| self.columns[index].hidden_count)
    }

    /// Returns the total number of face-down tableau slots.
    pub fn hidden_slot_count(&self) -> usize {
        self.columns
            .iter()
            .map(|column| usize::from(column.hidden_count))
            .sum()
    }

    /// Returns deterministic hidden-slot identifiers derived from tableau columns.
    pub fn hidden_slots(&self) -> Vec<HiddenSlot> {
        let mut slots = Vec::with_capacity(self.hidden_slot_count());
        for column_index in 0..TABLEAU_COLUMN_COUNT {
            let column = ColumnId::new(column_index as u8).expect("tableau index is valid");
            for depth in 0..self.columns[column_index].hidden_count {
                slots.push(HiddenSlot::new(column, depth));
            }
        }
        slots
    }

    /// Returns the number of currently visible or otherwise known cards.
    pub fn visible_card_count(&self) -> usize {
        self.iter_visible_cards().count()
    }

    /// Returns all cards represented by the visible state in deterministic groups.
    pub fn all_visible_cards(&self) -> Vec<Card> {
        self.iter_visible_cards().collect()
    }

    /// Iterates all cards represented by the visible state without allocating.
    pub fn iter_visible_cards(&self) -> impl Iterator<Item = Card> + '_ {
        self.foundations
            .iter_cards()
            .chain(
                self.columns
                    .iter()
                    .flat_map(|column| column.face_up.iter().copied()),
            )
            .chain(self.stock.ring_cards.iter().copied())
    }

    /// Returns true if the foundations are structurally complete.
    pub fn is_structural_win(&self) -> bool {
        self.foundations.is_complete()
    }

    /// Validates local structure without cross-card accounting.
    pub fn validate_structure(&self) -> SolverResult<()> {
        self.foundations.validate_structure()?;
        for column in &self.columns {
            column.validate_structure()?;
        }
        self.stock.validate_structure()?;
        Ok(())
    }

    /// Validates cross-state consistency such as duplicate visible cards.
    pub fn validate_consistency(&self) -> SolverResult<()> {
        self.validate_structure()?;
        ensure_unique_cards(self.iter_visible_cards())
    }

    /// Runs full validation in debug builds and becomes a no-op in release builds.
    ///
    /// Future mutation-heavy apply/undo code can call this after state changes
    /// without paying validation cost in optimized builds.
    pub fn debug_validate(&self) -> SolverResult<()> {
        #[cfg(debug_assertions)]
        {
            self.validate_consistency()
        }
        #[cfg(not(debug_assertions))]
        {
            Ok(())
        }
    }
}

impl Default for VisibleState {
    fn default() -> Self {
        Self {
            foundations: FoundationState::default(),
            columns: std::array::from_fn(|_| TableauColumn::default()),
            stock: CyclicStockState::default(),
        }
    }
}

/// Perfect-information state used by deterministic solving and sampled worlds.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FullState {
    /// Visible portion shared with belief states.
    pub visible: VisibleState,
    /// Concrete identities of all hidden tableau slots.
    pub hidden_assignments: HiddenAssignments,
}

impl FullState {
    /// Creates a full state from visible information and hidden assignments.
    pub fn new(visible: VisibleState, hidden_assignments: HiddenAssignments) -> Self {
        Self {
            visible,
            hidden_assignments,
        }
    }

    /// Validates local structure.
    pub fn validate_structure(&self) -> SolverResult<()> {
        self.visible.validate_structure()?;
        self.hidden_assignments.validate_structure()
    }

    /// Validates cross-state card accounting and hidden-slot compatibility.
    pub fn validate_consistency(&self) -> SolverResult<()> {
        self.visible.validate_consistency()?;
        self.hidden_assignments
            .validate_against_visible(&self.visible)?;

        let mut visible_mask = 0u64;
        let mut visible_count = 0usize;
        for card in self.visible.iter_visible_cards() {
            visible_mask |= card_bit(card);
            visible_count += 1;
        }

        for entry in self.hidden_assignments.iter() {
            if (visible_mask & card_bit(entry.card)) != 0 {
                return Err(SolverError::VisibleHiddenCardOverlap(entry.card));
            }
        }

        let total_cards = visible_count + self.hidden_assignments.len();
        if total_cards != Card::COUNT {
            return Err(SolverError::CardAccountingMismatch {
                actual: total_cards,
            });
        }

        Ok(())
    }

    /// Runs full validation in debug builds and becomes a no-op in release builds.
    pub fn debug_validate(&self) -> SolverResult<()> {
        #[cfg(debug_assertions)]
        {
            self.validate_consistency()
        }
        #[cfg(not(debug_assertions))]
        {
            Ok(())
        }
    }
}

impl fmt::Display for VisibleState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "{} {}", self.foundations, self.stock)?;
        for (index, column) in self.columns.iter().enumerate() {
            writeln!(f, "T{index}: {column}")?;
        }
        Ok(())
    }
}

fn ensure_unique_cards(cards: impl IntoIterator<Item = Card>) -> SolverResult<()> {
    let mut mask = 0u64;
    for card in cards {
        let bit = card_bit(card);
        if (mask & bit) != 0 {
            return Err(SolverError::DuplicateCard(card));
        }
        mask |= bit;
    }
    Ok(())
}

pub(crate) const fn card_bit(card: Card) -> u64 {
    1u64 << card.index()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::HiddenAssignment;

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
    fn foundation_state_implies_suit_indexed_cards() {
        let mut foundations = FoundationState::default();
        foundations.set_top_rank(Suit::Hearts, Some(Rank::Three));

        assert_eq!(
            foundations.cards(),
            vec![
                Card::from_suit_rank(Suit::Hearts, Rank::Ace),
                Card::from_suit_rank(Suit::Hearts, Rank::Two),
                Card::from_suit_rank(Suit::Hearts, Rank::Three),
            ]
        );
        assert_eq!(foundations.card_count(), 3);
        assert_eq!(foundations.to_string(), "F[c:- d:- h:3 s:-]");
    }

    #[test]
    fn visible_state_iterates_visible_cards_without_collection() {
        let mut visible = VisibleState::default();
        visible
            .foundations
            .set_top_rank(Suit::Clubs, Some(Rank::Two));
        visible.columns[0] =
            TableauColumn::new(0, vec!["7s".parse().unwrap(), "6h".parse().unwrap()]);
        visible.stock = CyclicStockState::new(vec!["Kd".parse().unwrap()], Some(0), 0, None, 3);

        let cards: Vec<String> = visible
            .iter_visible_cards()
            .map(|card| card.to_string())
            .collect();

        assert_eq!(cards, vec!["Ac", "2c", "7s", "6h", "Kd"]);
        assert_eq!(visible.visible_card_count(), 5);
    }

    #[test]
    fn visible_state_detects_duplicate_visible_cards() {
        let duplicate: Card = "7s".parse().unwrap();
        let mut visible = VisibleState::default();
        visible.columns[0] = TableauColumn::new(0, vec![duplicate]);
        visible.columns[1] = TableauColumn::new(0, vec![duplicate]);

        assert!(matches!(
            visible.validate_consistency(),
            Err(SolverError::DuplicateCard(card)) if card == duplicate
        ));
    }

    #[test]
    fn visible_state_derives_hidden_slots_in_order() {
        let mut visible = VisibleState::default();
        visible.columns[0].hidden_count = 2;
        visible.columns[2].hidden_count = 1;

        let slots = visible.hidden_slots();

        assert_eq!(
            slots,
            vec![
                HiddenSlot::new(ColumnId::new(0).unwrap(), 0),
                HiddenSlot::new(ColumnId::new(0).unwrap(), 1),
                HiddenSlot::new(ColumnId::new(2).unwrap(), 0),
            ]
        );
    }

    #[test]
    fn full_state_detects_duplicate_hidden_assignment_cards() {
        let hidden_card: Card = "Ah".parse().unwrap();
        let visible_card: Card = "Ks".parse().unwrap();
        let mut visible = VisibleState::default();
        visible.columns[0] = TableauColumn::new(2, vec![visible_card]);
        visible.stock = stock_with_all_except(&[hidden_card, visible_card]);
        let assignments = HiddenAssignments::new(vec![
            HiddenAssignment::new(HiddenSlot::new(ColumnId::new(0).unwrap(), 0), hidden_card),
            HiddenAssignment::new(HiddenSlot::new(ColumnId::new(0).unwrap(), 1), hidden_card),
        ]);
        let full = FullState::new(visible, assignments);

        assert!(matches!(
            full.validate_consistency(),
            Err(SolverError::DuplicateCard(card)) if card == hidden_card
        ));
    }

    #[test]
    fn full_state_detects_visible_hidden_overlap() {
        let overlap: Card = "Ah".parse().unwrap();
        let other_hidden: Card = "2h".parse().unwrap();
        let mut visible = VisibleState::default();
        visible.columns[0] = TableauColumn::new(2, vec![overlap]);
        visible.stock = stock_with_all_except(&[overlap, other_hidden]);
        let assignments = HiddenAssignments::new(vec![
            HiddenAssignment::new(HiddenSlot::new(ColumnId::new(0).unwrap(), 0), overlap),
            HiddenAssignment::new(HiddenSlot::new(ColumnId::new(0).unwrap(), 1), other_hidden),
        ]);
        let full = FullState::new(visible, assignments);

        assert!(matches!(
            full.validate_consistency(),
            Err(SolverError::VisibleHiddenCardOverlap(card)) if card == overlap
        ));
    }
}
