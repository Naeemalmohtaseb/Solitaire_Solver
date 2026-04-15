//! Move identity, atomic move, and macro move surfaces.

use serde::{Deserialize, Serialize};

use crate::{
    cards::Card,
    stock::StockActionKind,
    types::{ColumnId, FoundationId, MoveId},
};

/// Low-level legal move families.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AtomicMove {
    /// Move the accessible waste card to a tableau column.
    WasteToTableau {
        /// Destination tableau column.
        dest: ColumnId,
    },
    /// Move the accessible waste card to a foundation.
    WasteToFoundation {
        /// Destination foundation.
        dest: FoundationId,
    },
    /// Move a face-up tableau run between columns.
    TableauRunToTableau {
        /// Source tableau column.
        src: ColumnId,
        /// Zero-based start index inside the face-up run.
        start_index: u8,
        /// Destination tableau column.
        dest: ColumnId,
    },
    /// Move a tableau top card to a foundation.
    TableauTopToFoundation {
        /// Source tableau column.
        src: ColumnId,
        /// Destination foundation.
        dest: FoundationId,
    },
    /// Move a foundation top card back to a tableau column.
    FoundationToTableau {
        /// Source foundation.
        src: FoundationId,
        /// Destination tableau column.
        dest: ColumnId,
    },
    /// Apply a stock/waste transition.
    Stock(StockActionKind),
}

/// Planner-level macro move families.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum MacroMoveKind {
    /// Play the accessible waste card to a tableau column.
    PlayWasteToTableau {
        /// Destination tableau column.
        dest: ColumnId,
    },
    /// Play the accessible waste card to a foundation.
    PlayWasteToFoundation,
    /// Move a tableau run.
    MoveTableauRun {
        /// Source tableau column.
        src: ColumnId,
        /// Zero-based start index inside the face-up run.
        start_index: u8,
        /// Destination tableau column.
        dest: ColumnId,
    },
    /// Move a tableau top card to a foundation.
    MoveTableauTopToFoundation {
        /// Source tableau column.
        src: ColumnId,
    },
    /// Place a king-headed run into an empty column.
    PlaceKingRunInEmpty {
        /// Source tableau column.
        src: ColumnId,
        /// Zero-based start index inside the face-up run.
        start_index: u8,
        /// Empty destination column.
        dest: ColumnId,
    },
    /// Advance stock/waste to a target card or frontier.
    AdvanceStock(StockActionKind),
    /// Move a foundation card back to the tableau.
    FoundationRetreat {
        /// Card being retreated.
        card: Card,
        /// Destination tableau column.
        dest: ColumnId,
    },
}

/// Semantic tags used for ordering, diagnostics, and future pruning audits.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum MoveTag {
    /// The move can uncover a hidden tableau card.
    RevealsCard,
    /// The move changes waste accessibility.
    ChangesWasteAccess,
    /// The move creates or fills an empty tableau column.
    EmptyColumnInteraction,
    /// The move advances foundation progress.
    FoundationProgress,
    /// The move increases tableau mobility.
    MobilityImprovement,
    /// The move may be reversible and needs inverse-suppression handling.
    Reversible,
}

/// Planner macro with stable identity and semantic metadata.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MacroMove {
    /// Stable move id at the node where this macro was generated.
    pub id: MoveId,
    /// Macro move family.
    pub kind: MacroMoveKind,
    /// Number of atomic moves represented by the macro.
    pub atomic_len: usize,
    /// Semantic tags for ordering and explanation.
    pub tags: Vec<MoveTag>,
}
