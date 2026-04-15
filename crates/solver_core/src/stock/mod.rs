//! Exact Draw-3 stock/waste cycle representation.
//!
//! The stock/waste order is known from the start of the game. This module stores
//! that exact order and the current accessibility marker. It does not model any
//! hidden uncertainty over stock order.
//!
//! This module owns the small transition primitives needed by move application:
//! draw advancement, accessible waste removal, and recycle/reset. It deliberately
//! does not generate moves or apply tableau/foundation logic.

use serde::{Deserialize, Serialize};

use crate::{
    cards::Card,
    error::{SolverError, SolverResult},
};

/// Known cyclic stock/waste state for Draw-3 Klondike.
///
/// `ring_cards` is normalized as `stock prefix` followed by `waste bottom-to-top`.
/// `stock_len` is the length of the stock prefix. Accessibility is constrained
/// by Draw-3 rules: only the current cursor card is playable from the waste,
/// even though the complete order is known.
///
/// When a draw happens, up to `draw_count` cards rotate from the stock prefix to
/// the top of the waste suffix. `accessible_depth` tracks how many cards from
/// that latest draw group can still surface as the top waste card if the current
/// accessible card is played.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CyclicStockState {
    /// Fully known ordered ring of cards remaining in the stock/waste cycle.
    pub ring_cards: Vec<Card>,
    /// Number of cards at the front of `ring_cards` that remain in the stock.
    pub stock_len: usize,
    /// Index of the currently accessible waste card in `ring_cards`, if one exists.
    pub cursor: Option<usize>,
    /// Number of cards in the current draw group that can still become accessible.
    pub accessible_depth: u8,
    /// Number of completed stock passes.
    pub pass_index: u32,
    /// Optional cap on stock passes for rule variants.
    pub max_passes: Option<u32>,
    /// Draw count, normally 3.
    pub draw_count: u8,
}

impl Default for CyclicStockState {
    fn default() -> Self {
        Self {
            ring_cards: Vec::new(),
            stock_len: 0,
            cursor: None,
            accessible_depth: 0,
            pass_index: 0,
            max_passes: None,
            draw_count: 3,
        }
    }
}

impl CyclicStockState {
    /// Creates a known stock/waste cycle shell.
    pub fn new(
        ring_cards: Vec<Card>,
        cursor: Option<usize>,
        pass_index: u32,
        max_passes: Option<u32>,
        draw_count: u8,
    ) -> Self {
        let accessible_depth = u8::from(cursor.is_some());
        let stock_len = cursor.map_or(ring_cards.len(), |cursor| cursor);
        Self {
            ring_cards,
            stock_len,
            cursor,
            accessible_depth,
            pass_index,
            max_passes,
            draw_count,
        }
    }

    /// Creates a normalized stock/waste state from explicit accessibility parts.
    pub fn from_parts(
        ring_cards: Vec<Card>,
        stock_len: usize,
        accessible_depth: u8,
        pass_index: u32,
        max_passes: Option<u32>,
        draw_count: u8,
    ) -> Self {
        let cursor = if accessible_depth > 0 {
            ring_cards.len().checked_sub(1)
        } else {
            None
        };
        Self {
            ring_cards,
            stock_len,
            cursor,
            accessible_depth,
            pass_index,
            max_passes,
            draw_count,
        }
    }

    /// Returns true if the known stock/waste ring contains no cards.
    pub fn is_empty(&self) -> bool {
        self.ring_cards.is_empty()
    }

    /// Returns the number of cards in the known stock/waste ring.
    pub fn len(&self) -> usize {
        self.ring_cards.len()
    }

    /// Returns the number of cards still in the stock prefix.
    pub fn stock_len(&self) -> usize {
        self.stock_len
    }

    /// Returns the number of cards currently in the waste suffix.
    pub fn waste_len(&self) -> usize {
        self.ring_cards.len().saturating_sub(self.stock_len)
    }

    /// Returns the currently accessible waste card, if one exists.
    pub fn accessible_card(&self) -> Option<Card> {
        self.cursor
            .and_then(|index| self.ring_cards.get(index).copied())
    }

    /// Returns true if a stock advance can draw at least one card.
    pub fn can_advance(&self) -> bool {
        self.stock_len > 0
    }

    /// Returns true if the waste suffix can be recycled into the stock prefix.
    pub fn can_recycle(&self) -> bool {
        self.stock_len == 0
            && !self.ring_cards.is_empty()
            && self
                .max_passes
                .is_none_or(|max_passes| self.pass_index < max_passes)
    }

    /// Advances the known stock by Draw-N and exposes the new top waste card.
    pub fn advance_draw(&mut self) -> SolverResult<()> {
        self.validate_structure()?;
        if !self.can_advance() {
            return Err(SolverError::IllegalMove(
                "cannot advance stock when stock prefix is empty".to_string(),
            ));
        }

        let draw_count = usize::from(self.draw_count).min(self.stock_len);
        self.ring_cards.rotate_left(draw_count);
        self.stock_len -= draw_count;
        self.accessible_depth = draw_count as u8;
        self.cursor = self.ring_cards.len().checked_sub(1);
        self.validate_structure()
    }

    /// Removes and returns the currently accessible waste card.
    pub fn remove_accessible_card(&mut self) -> SolverResult<Card> {
        self.validate_structure()?;
        if self.accessible_depth == 0 {
            return Err(SolverError::IllegalMove(
                "no accessible waste card is available".to_string(),
            ));
        }

        let cursor = self.cursor.ok_or_else(|| {
            SolverError::InvalidStockState("accessible depth set without cursor".to_string())
        })?;
        let card = self.ring_cards.remove(cursor);
        self.accessible_depth -= 1;
        self.cursor = if self.accessible_depth > 0 {
            self.ring_cards.len().checked_sub(1)
        } else {
            None
        };
        self.validate_structure()?;
        Ok(card)
    }

    /// Recycles the waste suffix back into the stock prefix.
    pub fn recycle(&mut self) -> SolverResult<()> {
        self.validate_structure()?;
        if !self.can_recycle() {
            return Err(SolverError::IllegalMove(
                "cannot recycle stock under current stock/waste state".to_string(),
            ));
        }

        self.stock_len = self.ring_cards.len();
        self.cursor = None;
        self.accessible_depth = 0;
        self.pass_index += 1;
        self.validate_structure()
    }

    /// Validates local stock/waste structure.
    pub fn validate_structure(&self) -> SolverResult<()> {
        if self.draw_count == 0 {
            return Err(SolverError::InvalidStockState(
                "draw count must be at least 1".to_string(),
            ));
        }

        if usize::from(self.draw_count) > Card::COUNT {
            return Err(SolverError::InvalidStockState(format!(
                "draw count {} exceeds deck size {}",
                self.draw_count,
                Card::COUNT
            )));
        }

        if self.stock_len > self.ring_cards.len() {
            return Err(SolverError::InvalidStockState(format!(
                "stock_len {} exceeds ring length {}",
                self.stock_len,
                self.ring_cards.len()
            )));
        }

        if self.stock_len + self.waste_len() != self.ring_cards.len() {
            return Err(SolverError::InvalidStockState(format!(
                "stock_len {} plus waste_len {} does not equal ring length {}",
                self.stock_len,
                self.waste_len(),
                self.ring_cards.len()
            )));
        }

        if self.ring_cards.is_empty()
            && (self.cursor.is_some() || self.stock_len != 0 || self.accessible_depth != 0)
        {
            return Err(SolverError::InvalidStockState(
                "empty stock ring cannot have stock or waste accessibility".to_string(),
            ));
        }

        if usize::from(self.accessible_depth) > self.waste_len() {
            return Err(SolverError::InvalidStockState(format!(
                "accessible depth {} exceeds waste length {}",
                self.accessible_depth,
                self.waste_len()
            )));
        }

        if self.accessible_depth > self.draw_count {
            return Err(SolverError::InvalidStockState(format!(
                "accessible depth {} exceeds draw count {}",
                self.accessible_depth, self.draw_count
            )));
        }

        match (self.cursor, self.accessible_depth) {
            (None, 0) => {}
            (Some(cursor), depth) => {
                if depth == 0 {
                    return Err(SolverError::InvalidStockState(
                        "cursor cannot be set when accessible depth is zero".to_string(),
                    ));
                }
                let expected = self.ring_cards.len() - 1;
                if cursor != expected {
                    return Err(SolverError::InvalidStockState(format!(
                        "cursor {cursor} must point to top waste index {expected}"
                    )));
                }
            }
            (None, depth) => {
                return Err(SolverError::InvalidStockState(format!(
                    "accessible depth {depth} requires a cursor"
                )));
            }
        }

        if self
            .max_passes
            .is_some_and(|max_passes| self.pass_index > max_passes)
        {
            return Err(SolverError::InvalidStockState(format!(
                "pass index {} exceeds max passes {}",
                self.pass_index,
                self.max_passes.expect("checked above")
            )));
        }

        let mut mask = 0u64;
        for card in &self.ring_cards {
            let bit = 1u64 << card.index();
            if (mask & bit) != 0 {
                return Err(SolverError::DuplicateCard(*card));
            }
            mask |= bit;
        }

        Ok(())
    }

    /// Runs structure validation in debug builds and becomes a no-op in release builds.
    pub fn debug_validate(&self) -> SolverResult<()> {
        #[cfg(debug_assertions)]
        {
            self.validate_structure()
        }
        #[cfg(not(debug_assertions))]
        {
            Ok(())
        }
    }
}

impl std::fmt::Display for CyclicStockState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Stock[len={} cursor=", self.ring_cards.len(),)?;
        match self.accessible_card() {
            Some(card) => write!(f, "{card}")?,
            None => f.write_str("-")?,
        }
        write!(
            f,
            " stock_len={} depth={} pass={} draw={}]",
            self.stock_len, self.accessible_depth, self.pass_index, self.draw_count
        )
    }
}

/// Future stock action categories used by macro generation.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum StockActionKind {
    /// Advance by the configured draw count.
    AdvanceDraw,
    /// Advance until a target card becomes accessible.
    AdvanceToTarget(Card),
    /// Recycle the stock/waste cycle according to the rule configuration.
    Recycle,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stock_accessible_card_uses_cursor() {
        let stock = CyclicStockState::from_parts(
            vec!["Ac".parse().unwrap(), "2c".parse().unwrap()],
            0,
            1,
            0,
            None,
            3,
        );

        assert_eq!(stock.len(), 2);
        assert_eq!(stock.accessible_card().unwrap().to_string(), "2c");
        stock.validate_structure().unwrap();
    }

    #[test]
    fn stock_validation_rejects_bad_cursor_and_duplicates() {
        let bad_cursor = CyclicStockState::new(vec!["Ac".parse().unwrap()], Some(1), 0, None, 3);
        assert!(bad_cursor.validate_structure().is_err());

        let duplicate = CyclicStockState::new(
            vec!["Ac".parse().unwrap(), "Ac".parse().unwrap()],
            None,
            0,
            None,
            3,
        );
        assert!(matches!(
            duplicate.validate_structure(),
            Err(SolverError::DuplicateCard(card)) if card.to_string() == "Ac"
        ));
    }

    #[test]
    fn stock_validation_rejects_inconsistent_accessibility_fields() {
        let depth_exceeds_draw = CyclicStockState::from_parts(
            vec![
                "Ac".parse().unwrap(),
                "2c".parse().unwrap(),
                "3c".parse().unwrap(),
                "4c".parse().unwrap(),
            ],
            0,
            4,
            0,
            None,
            3,
        );
        assert!(depth_exceeds_draw.validate_structure().is_err());

        let cursor_without_depth = CyclicStockState {
            ring_cards: vec!["Ac".parse().unwrap()],
            stock_len: 0,
            cursor: Some(0),
            accessible_depth: 0,
            pass_index: 0,
            max_passes: None,
            draw_count: 3,
        };
        assert!(cursor_without_depth.debug_validate().is_err());

        let cursor_not_top_waste = CyclicStockState {
            ring_cards: vec!["Ac".parse().unwrap(), "2c".parse().unwrap()],
            stock_len: 0,
            cursor: Some(0),
            accessible_depth: 1,
            pass_index: 0,
            max_passes: None,
            draw_count: 3,
        };
        assert!(cursor_not_top_waste.validate_structure().is_err());
    }

    #[test]
    fn stock_advance_remove_and_recycle_follow_draw_window() {
        let mut stock = CyclicStockState::from_parts(
            vec![
                "Ac".parse().unwrap(),
                "2c".parse().unwrap(),
                "3c".parse().unwrap(),
                "4c".parse().unwrap(),
            ],
            4,
            0,
            0,
            None,
            3,
        );

        stock.advance_draw().unwrap();
        assert_eq!(stock.stock_len(), 1);
        assert_eq!(stock.accessible_card().unwrap().to_string(), "3c");

        assert_eq!(stock.remove_accessible_card().unwrap().to_string(), "3c");
        assert_eq!(stock.accessible_card().unwrap().to_string(), "2c");

        stock.advance_draw().unwrap();
        assert_eq!(stock.stock_len(), 0);
        assert_eq!(stock.accessible_card().unwrap().to_string(), "4c");

        stock.recycle().unwrap();
        assert_eq!(stock.stock_len(), 3);
        assert!(stock.accessible_card().is_none());
        assert_eq!(stock.pass_index, 1);
    }
}
