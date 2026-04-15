//! Exact draw-3 stock/waste cycle representation.

use serde::{Deserialize, Serialize};

use crate::cards::Card;

/// Known cyclic stock/waste state for Draw-3 Klondike.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CyclicStockState {
    /// Ordered ring of cards remaining in the stock/waste cycle.
    pub ring_cards: Vec<Card>,
    /// Index of the currently accessible waste card, if one exists.
    pub cursor: Option<usize>,
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
            cursor: None,
            pass_index: 0,
            max_passes: None,
            draw_count: 3,
        }
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
