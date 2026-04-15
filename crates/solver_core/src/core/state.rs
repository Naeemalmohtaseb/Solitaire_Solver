//! Visible and full deterministic game state shells.

use serde::{Deserialize, Serialize};

use crate::{
    cards::Card,
    stock::CyclicStockState,
    types::{ColumnId, FOUNDATION_COUNT, TABLEAU_COLUMN_COUNT},
};

use super::column::TableauColumn;

/// User-visible Draw-3 Klondike state.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VisibleState {
    /// Top card of each foundation, if any.
    pub foundations: [Option<Card>; FOUNDATION_COUNT],
    /// The seven tableau columns with hidden counts and visible runs.
    pub columns: [TableauColumn; TABLEAU_COLUMN_COUNT],
    /// Fully known draw-3 stock/waste cycle state.
    pub stock: CyclicStockState,
}

impl Default for VisibleState {
    fn default() -> Self {
        Self {
            foundations: [None; FOUNDATION_COUNT],
            columns: std::array::from_fn(|_| TableauColumn::default()),
            stock: CyclicStockState::default(),
        }
    }
}

/// One hidden tableau slot assigned to a concrete card in a full deterministic world.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HiddenTableauCard {
    /// Column containing this hidden card.
    pub column: ColumnId,
    /// Zero-based hidden slot index within the column.
    pub hidden_slot: u8,
    /// Card assigned to the hidden slot.
    pub card: Card,
}

/// Perfect-information state used by deterministic solving and sampled worlds.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FullState {
    /// Visible portion shared with belief states.
    pub visible: VisibleState,
    /// Concrete identities of the hidden tableau cards.
    pub hidden_tableau: Vec<HiddenTableauCard>,
}

impl FullState {
    /// Creates a full state from visible information and hidden assignments.
    pub fn new(visible: VisibleState, hidden_tableau: Vec<HiddenTableauCard>) -> Self {
        Self {
            visible,
            hidden_tableau,
        }
    }
}
