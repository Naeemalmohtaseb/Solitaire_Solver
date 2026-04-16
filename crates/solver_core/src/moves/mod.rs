//! Legal move generation and reversible visible-state transitions.
//!
//! This module implements atomic move correctness for visible states. It does
//! not perform search, closure, dominance pruning, or heuristic evaluation.

use serde::{Deserialize, Serialize};

use crate::{
    cards::{Card, Rank, Suit},
    core::{FullState, HiddenAssignment, HiddenSlot, TableauColumn, VisibleState},
    error::{SolverError, SolverResult},
    types::{ColumnId, MoveId, TABLEAU_COLUMN_COUNT},
};

/// Move-generation knobs that affect legal move surfaces without adding search logic.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MoveGenerationConfig {
    /// Whether legal foundation-to-tableau retreat moves are generated.
    pub allow_foundation_retreats: bool,
}

impl Default for MoveGenerationConfig {
    fn default() -> Self {
        Self {
            allow_foundation_retreats: true,
        }
    }
}

/// Low-level legal move families.
#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum AtomicMove {
    /// Move the accessible waste card to a tableau column.
    WasteToTableau {
        /// Destination tableau column.
        dest: ColumnId,
    },
    /// Move the accessible waste card to its suit foundation.
    WasteToFoundation,
    /// Move a tableau top card to its suit foundation.
    TableauToFoundation {
        /// Source tableau column.
        src: ColumnId,
    },
    /// Move a face-up tableau suffix between columns.
    TableauToTableau {
        /// Source tableau column.
        src: ColumnId,
        /// Destination tableau column.
        dest: ColumnId,
        /// Zero-based start index inside the source face-up run.
        run_start: usize,
    },
    /// Move a foundation top card back to a tableau column.
    FoundationToTableau {
        /// Foundation suit.
        suit: Suit,
        /// Destination tableau column.
        dest: ColumnId,
    },
    /// Advance the stock by the configured Draw-N amount.
    StockAdvance,
    /// Recycle the waste back into the stock when legal.
    StockRecycle,
}

/// Planner-level macro move families.
#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum MacroMoveKind {
    /// Play the accessible waste card to a tableau column.
    PlayWasteToTableau {
        /// Destination tableau column.
        dest: ColumnId,
    },
    /// Play the accessible waste card to its suit foundation.
    PlayWasteToFoundation,
    /// Move a tableau run.
    MoveRun {
        /// Source tableau column.
        src: ColumnId,
        /// Destination tableau column.
        dest: ColumnId,
        /// Zero-based start index inside the source face-up run.
        run_start: usize,
    },
    /// Move a tableau top card to foundation.
    MoveTopToFoundation {
        /// Source tableau column.
        src: ColumnId,
    },
    /// Place a king-headed run into an empty column.
    PlaceKingRun {
        /// Source tableau column.
        src: ColumnId,
        /// Empty destination tableau column.
        dest: ColumnId,
        /// Zero-based start index inside the source face-up run.
        run_start: usize,
    },
    /// Move a foundation card back to the tableau.
    FoundationRetreat {
        /// Foundation suit.
        suit: Suit,
        /// Destination tableau column.
        dest: ColumnId,
    },
    /// Advance the stock.
    AdvanceStock,
    /// Recycle the stock.
    RecycleStock,
}

/// Semantic tags used for ordering, diagnostics, and future pruning audits.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum MoveTag {
    /// The move can uncover a hidden tableau card.
    RevealsCard,
    /// The move uses the accessible waste card.
    UsesWaste,
    /// The move advances foundation progress.
    MovesToFoundation,
    /// The move retreats a foundation card.
    MovesFromFoundation,
    /// The move creates an empty tableau column.
    CreatesEmptyColumn,
    /// The move fills an empty tableau column.
    FillsEmptyColumn,
    /// The move changes stock/waste accessibility.
    AffectsStockCycle,
}

/// Semantic move descriptor for future search and diagnostics.
#[derive(Debug, Copy, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct MoveSemantics {
    /// The move uncovers a hidden tableau card.
    pub causes_reveal: bool,
    /// The move consumes the accessible waste card.
    pub uses_waste: bool,
    /// The move moves a card to a foundation.
    pub moves_to_foundation: bool,
    /// The move moves a card out of a foundation.
    pub moves_from_foundation: bool,
    /// The move creates an empty tableau column.
    pub creates_empty_column: bool,
    /// The move fills an empty tableau column.
    pub fills_empty_column: bool,
    /// The move changes stock/waste cycle state.
    pub affects_stock_cycle: bool,
}

impl MoveSemantics {
    /// Converts semantic booleans into stable tags.
    pub fn tags(self) -> Vec<MoveTag> {
        let mut tags = Vec::new();
        if self.causes_reveal {
            tags.push(MoveTag::RevealsCard);
        }
        if self.uses_waste {
            tags.push(MoveTag::UsesWaste);
        }
        if self.moves_to_foundation {
            tags.push(MoveTag::MovesToFoundation);
        }
        if self.moves_from_foundation {
            tags.push(MoveTag::MovesFromFoundation);
        }
        if self.creates_empty_column {
            tags.push(MoveTag::CreatesEmptyColumn);
        }
        if self.fills_empty_column {
            tags.push(MoveTag::FillsEmptyColumn);
        }
        if self.affects_stock_cycle {
            tags.push(MoveTag::AffectsStockCycle);
        }
        tags
    }
}

/// Planner macro with stable identity and semantic metadata.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MacroMove {
    /// Stable move id at the node where this macro was generated.
    pub id: MoveId,
    /// Macro move family.
    pub kind: MacroMoveKind,
    /// Atomic move represented by this first macro implementation.
    pub atomic: AtomicMove,
    /// Semantic summary for ordering and explanation.
    pub semantics: MoveSemantics,
}

/// Reveal side effect produced by applying a move.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RevealRecord {
    /// Column where the reveal occurred.
    pub column: ColumnId,
    /// Card that became visible.
    pub card: Card,
}

/// Result summary for one applied move.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MoveOutcome {
    /// Move that was applied.
    pub applied_move: AtomicMove,
    /// Semantic summary computed from the pre-move state.
    pub semantics: MoveSemantics,
    /// Cards physically moved by the move, excluding auto-revealed cards.
    pub moved_cards: Vec<Card>,
    /// Reveal side effect, if one occurred.
    pub revealed: Option<RevealRecord>,
    /// Whether the transition created an empty column.
    pub created_empty_column: bool,
    /// Whether the transition filled an empty column.
    pub filled_empty_column: bool,
    /// Whether stock/waste state changed.
    pub stock_changed: bool,
}

/// Column-level delta recorded for undo.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ColumnUndo {
    /// Column changed by the move.
    pub column: ColumnId,
    /// Hidden count before the move.
    pub previous_hidden_count: u8,
    /// Cards removed from the top/suffix of this column.
    pub removed_cards: Vec<Card>,
    /// Cards added to the top/suffix of this column.
    pub added_cards: Vec<Card>,
}

/// Foundation-level delta recorded for undo.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FoundationUndo {
    /// Foundation suit changed by the move.
    pub suit: Suit,
    /// Top rank before the move.
    pub previous_top: Option<Rank>,
}

/// Stock operation delta recorded for undo.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum StockUndoKind {
    /// A draw advanced `drawn` cards from stock to waste.
    Advance {
        /// Number of cards advanced.
        drawn: usize,
    },
    /// The accessible waste card was removed.
    RemoveAccessible {
        /// Removed card.
        card: Card,
        /// Previous index of the removed card.
        index: usize,
    },
    /// Waste was recycled into stock.
    Recycle,
}

/// Stock-level delta recorded for undo.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StockUndo {
    /// Previous cursor.
    pub previous_cursor: Option<usize>,
    /// Previous stock prefix length.
    pub previous_stock_len: usize,
    /// Previous accessible draw-window depth.
    pub previous_accessible_depth: u8,
    /// Previous pass index.
    pub previous_pass_index: u32,
    /// Operation-specific stock delta.
    pub kind: StockUndoKind,
}

/// Explicit delta undo record sufficient to restore the exact previous visible state.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MoveUndo {
    /// Move that was applied.
    pub applied_move: AtomicMove,
    /// Cards physically moved by the move.
    pub moved_cards: Vec<Card>,
    /// Column deltas, in deterministic order of recording.
    pub column_changes: Vec<ColumnUndo>,
    /// Foundation delta, if any.
    pub foundation_change: Option<FoundationUndo>,
    /// Stock delta, if any.
    pub stock_change: Option<StockUndo>,
    /// Reveal side effect that must be undone, if any.
    pub revealed: Option<RevealRecord>,
}

/// Applied transition containing both diagnostic outcome and undo data.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MoveTransition {
    /// Diagnostic transition outcome.
    pub outcome: MoveOutcome,
    /// Undo record that restores the exact previous visible state.
    pub undo: MoveUndo,
}

/// Full-state undo record for a move that may reveal a concrete hidden card.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FullStateMoveUndo {
    /// Visible-state undo record from the move engine.
    pub visible_undo: MoveUndo,
    /// Hidden assignment removed because the move auto-revealed a card.
    pub revealed_assignment: Option<HiddenAssignment>,
}

/// Applied full-state transition with visible outcome and hidden-assignment undo.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FullStateMoveTransition {
    /// Diagnostic transition outcome.
    pub outcome: MoveOutcome,
    /// Undo record that restores both visible state and hidden assignments.
    pub undo: FullStateMoveUndo,
}

/// Generates legal atomic moves with default move-generation config.
pub fn generate_legal_atomic_moves(state: &VisibleState) -> Vec<AtomicMove> {
    generate_legal_atomic_moves_with_config(state, MoveGenerationConfig::default())
}

/// Generates legal atomic moves with explicit move-generation config.
pub fn generate_legal_atomic_moves_with_config(
    state: &VisibleState,
    config: MoveGenerationConfig,
) -> Vec<AtomicMove> {
    let mut moves = Vec::new();

    if let Some(waste_card) = state.stock.accessible_card() {
        for dest_index in 0..TABLEAU_COLUMN_COUNT {
            let dest = ColumnId::new(dest_index as u8).expect("tableau index is valid");
            if can_place_on_tableau(waste_card, &state.columns[dest_index]) {
                moves.push(AtomicMove::WasteToTableau { dest });
            }
        }

        if can_move_to_foundation(state, waste_card) {
            moves.push(AtomicMove::WasteToFoundation);
        }
    }

    for src_index in 0..TABLEAU_COLUMN_COUNT {
        let src = ColumnId::new(src_index as u8).expect("tableau index is valid");
        let src_column = &state.columns[src_index];

        if let Some(top_card) = src_column.top_face_up() {
            if can_move_to_foundation(state, top_card) {
                moves.push(AtomicMove::TableauToFoundation { src });
            }
        }

        for run_start in movable_run_starts(src_column) {
            let head = src_column.face_up[run_start];
            for dest_index in 0..TABLEAU_COLUMN_COUNT {
                if dest_index == src_index {
                    continue;
                }
                let dest = ColumnId::new(dest_index as u8).expect("tableau index is valid");
                if can_place_on_tableau(head, &state.columns[dest_index]) {
                    moves.push(AtomicMove::TableauToTableau {
                        src,
                        dest,
                        run_start,
                    });
                }
            }
        }
    }

    if config.allow_foundation_retreats {
        for suit_index in 0..crate::types::FOUNDATION_COUNT {
            let suit = Suit::from_index(suit_index as u8).expect("foundation index is valid");
            if let Some(rank) = state.foundations.top_rank(suit) {
                let card = Card::from_suit_rank(suit, rank);
                for dest_index in 0..TABLEAU_COLUMN_COUNT {
                    let dest = ColumnId::new(dest_index as u8).expect("tableau index is valid");
                    if can_place_on_tableau(card, &state.columns[dest_index]) {
                        moves.push(AtomicMove::FoundationToTableau { suit, dest });
                    }
                }
            }
        }
    }

    if state.stock.can_advance() {
        moves.push(AtomicMove::StockAdvance);
    }
    if state.stock.can_recycle() {
        moves.push(AtomicMove::StockRecycle);
    }

    normalize_atomic_moves(&mut moves);
    moves
}

/// Generates macro moves by wrapping currently legal atomic moves.
pub fn generate_legal_macro_moves(state: &VisibleState) -> Vec<MacroMove> {
    generate_legal_macro_moves_with_config(state, MoveGenerationConfig::default())
}

/// Generates macro moves by wrapping currently legal atomic moves with explicit config.
pub fn generate_legal_macro_moves_with_config(
    state: &VisibleState,
    config: MoveGenerationConfig,
) -> Vec<MacroMove> {
    generate_legal_atomic_moves_with_config(state, config)
        .into_iter()
        .enumerate()
        .map(|(index, atomic)| MacroMove {
            id: MoveId::new(index as u32),
            kind: macro_kind_for_atomic(state, atomic),
            atomic,
            semantics: semantics_for_atomic_move(state, atomic),
        })
        .collect()
}

/// Applies an atomic move, optionally supplying a card for an auto-reveal side effect.
pub fn apply_atomic_move(
    state: &mut VisibleState,
    atomic: AtomicMove,
    revealed_card: Option<Card>,
) -> SolverResult<MoveTransition> {
    enforce_reveal_contract(state, atomic, revealed_card)?;

    let rollback_state = state.clone();
    let undo_base = MoveUndo {
        applied_move: atomic,
        moved_cards: Vec::new(),
        column_changes: Vec::new(),
        foundation_change: None,
        stock_change: None,
        revealed: None,
    };

    match apply_atomic_move_inner(state, atomic, revealed_card, undo_base) {
        Ok(transition) => Ok(transition),
        Err(err) => {
            *state = rollback_state;
            Err(err)
        }
    }
}

/// Restores a visible state using a prior undo record.
pub fn undo_atomic_move(state: &mut VisibleState, undo: MoveUndo) -> SolverResult<()> {
    if let Some(stock_change) = undo.stock_change {
        undo_stock_change(&mut state.stock, stock_change)?;
    }

    if let Some(foundation_change) = undo.foundation_change {
        state
            .foundations
            .set_top_rank(foundation_change.suit, foundation_change.previous_top);
    }

    for column_change in undo.column_changes.into_iter().rev() {
        undo_column_change(
            &mut state.columns[column_index(column_change.column)],
            &column_change,
        )?;
    }

    state.validate_consistency()
}

/// Returns the hidden slot that would be revealed by a legal visible move.
pub fn next_hidden_slot_to_reveal(state: &VisibleState, atomic: AtomicMove) -> Option<HiddenSlot> {
    let column = match atomic {
        AtomicMove::TableauToFoundation { src } => {
            let tableau = &state.columns[column_index(src)];
            (tableau.face_up_len() == 1 && tableau.hidden_count > 0).then_some(src)
        }
        AtomicMove::TableauToTableau { src, run_start, .. } => {
            let tableau = &state.columns[column_index(src)];
            (run_start == 0 && !tableau.face_up.is_empty() && tableau.hidden_count > 0)
                .then_some(src)
        }
        _ => None,
    }?;

    Some(HiddenSlot::new(
        column,
        state.columns[column_index(column)].hidden_count - 1,
    ))
}

/// Returns the concrete card that a full-state move would reveal.
pub fn revealed_card_for_move(
    full_state: &FullState,
    atomic: AtomicMove,
) -> SolverResult<Option<Card>> {
    match next_hidden_slot_to_reveal(&full_state.visible, atomic) {
        Some(slot) => full_state
            .hidden_assignments
            .card_for_slot(slot)
            .map(Some)
            .ok_or_else(|| {
                SolverError::InvalidState(format!(
                    "missing hidden assignment for reveal slot {slot}"
                ))
            }),
        None => Ok(None),
    }
}

/// Applies an atomic move to a full deterministic state.
///
/// Reveal identities are read from `hidden_assignments`, then removed after the
/// visible transition succeeds. This keeps perfect-information search aligned
/// with the visible move engine's reveal contract.
pub fn apply_atomic_move_full_state(
    full_state: &mut FullState,
    atomic: AtomicMove,
) -> SolverResult<FullStateMoveTransition> {
    let reveal_slot = next_hidden_slot_to_reveal(&full_state.visible, atomic);
    let revealed_assignment = match reveal_slot {
        Some(slot) => Some(
            full_state
                .hidden_assignments
                .assignment_for_slot(slot)
                .ok_or_else(|| {
                    SolverError::InvalidState(format!(
                        "missing hidden assignment for reveal slot {slot}"
                    ))
                })?,
        ),
        None => None,
    };

    let visible_transition = apply_atomic_move(
        &mut full_state.visible,
        atomic,
        revealed_assignment.map(|assignment| assignment.card),
    )?;

    let removed_assignment = match revealed_assignment {
        Some(assignment) => {
            let removed = full_state.hidden_assignments.remove_slot(assignment.slot)?;
            debug_assert_eq!(removed, assignment);
            Some(removed)
        }
        None => None,
    };

    full_state.debug_validate()?;

    Ok(FullStateMoveTransition {
        outcome: visible_transition.outcome,
        undo: FullStateMoveUndo {
            visible_undo: visible_transition.undo,
            revealed_assignment: removed_assignment,
        },
    })
}

/// Undoes a full deterministic move.
pub fn undo_atomic_move_full_state(
    full_state: &mut FullState,
    undo: FullStateMoveUndo,
) -> SolverResult<()> {
    undo_atomic_move(&mut full_state.visible, undo.visible_undo)?;
    if let Some(assignment) = undo.revealed_assignment {
        full_state.hidden_assignments.insert(assignment)?;
    }
    full_state.validate_consistency()
}

fn apply_atomic_move_inner(
    state: &mut VisibleState,
    atomic: AtomicMove,
    revealed_card: Option<Card>,
    mut undo: MoveUndo,
) -> SolverResult<MoveTransition> {
    let semantics = semantics_for_atomic_move(state, atomic);
    let mut moved_cards = Vec::new();
    let mut revealed = None;
    let mut stock_changed = false;
    let mut revealed_card = revealed_card;

    match atomic {
        AtomicMove::WasteToTableau { dest } => {
            let card = state.stock.accessible_card().ok_or_else(|| {
                SolverError::IllegalMove("waste-to-tableau requires accessible waste".to_string())
            })?;
            ensure_can_place_on_tableau(card, &state.columns[column_index(dest)])?;
            let stock_undo = stock_undo_for_remove_accessible(state)?;
            let removed = state.stock.remove_accessible_card()?;
            debug_assert_eq!(card, removed);
            undo.stock_change = Some(stock_undo);
            undo.column_changes.push(ColumnUndo {
                column: dest,
                previous_hidden_count: state.columns[column_index(dest)].hidden_count,
                removed_cards: Vec::new(),
                added_cards: vec![card],
            });
            state.columns[column_index(dest)].face_up.push(card);
            moved_cards.push(card);
            stock_changed = true;
        }
        AtomicMove::WasteToFoundation => {
            let card = state.stock.accessible_card().ok_or_else(|| {
                SolverError::IllegalMove(
                    "waste-to-foundation requires accessible waste".to_string(),
                )
            })?;
            ensure_can_move_to_foundation(state, card)?;
            let stock_undo = stock_undo_for_remove_accessible(state)?;
            let removed = state.stock.remove_accessible_card()?;
            debug_assert_eq!(card, removed);
            undo.stock_change = Some(stock_undo);
            undo.foundation_change = Some(FoundationUndo {
                suit: card.suit(),
                previous_top: state.foundations.top_rank(card.suit()),
            });
            state
                .foundations
                .set_top_rank(card.suit(), Some(card.rank()));
            moved_cards.push(card);
            stock_changed = true;
        }
        AtomicMove::TableauToFoundation { src } => {
            let src_index = column_index(src);
            let card = state.columns[src_index].top_face_up().ok_or_else(|| {
                SolverError::IllegalMove(
                    "tableau-to-foundation requires a face-up card".to_string(),
                )
            })?;
            ensure_can_move_to_foundation(state, card)?;
            undo.column_changes.push(ColumnUndo {
                column: src,
                previous_hidden_count: state.columns[src_index].hidden_count,
                removed_cards: vec![card],
                added_cards: Vec::new(),
            });
            undo.foundation_change = Some(FoundationUndo {
                suit: card.suit(),
                previous_top: state.foundations.top_rank(card.suit()),
            });
            state.columns[src_index].face_up.pop();
            state
                .foundations
                .set_top_rank(card.suit(), Some(card.rank()));
            moved_cards.push(card);
            revealed = auto_reveal_if_needed(state, src, &mut revealed_card, &mut undo)?;
        }
        AtomicMove::TableauToTableau {
            src,
            dest,
            run_start,
        } => {
            if src == dest {
                return Err(SolverError::IllegalMove(
                    "cannot move a tableau run onto the same column".to_string(),
                ));
            }

            let src_index = column_index(src);
            let dest_index = column_index(dest);
            ensure_valid_run_start(&state.columns[src_index], run_start)?;
            let head = state.columns[src_index].face_up[run_start];
            ensure_can_place_on_tableau(head, &state.columns[dest_index])?;

            let run = state.columns[src_index].face_up.split_off(run_start);
            moved_cards.extend(run.iter().copied());
            undo.column_changes.push(ColumnUndo {
                column: src,
                previous_hidden_count: state.columns[src_index].hidden_count,
                removed_cards: run.clone(),
                added_cards: Vec::new(),
            });
            undo.column_changes.push(ColumnUndo {
                column: dest,
                previous_hidden_count: state.columns[dest_index].hidden_count,
                removed_cards: Vec::new(),
                added_cards: run.clone(),
            });
            state.columns[dest_index].face_up.extend(run);
            revealed = auto_reveal_if_needed(state, src, &mut revealed_card, &mut undo)?;
        }
        AtomicMove::FoundationToTableau { suit, dest } => {
            let rank = state.foundations.top_rank(suit).ok_or_else(|| {
                SolverError::IllegalMove(
                    "foundation retreat requires a foundation card".to_string(),
                )
            })?;
            let card = Card::from_suit_rank(suit, rank);
            ensure_can_place_on_tableau(card, &state.columns[column_index(dest)])?;
            undo.foundation_change = Some(FoundationUndo {
                suit,
                previous_top: state.foundations.top_rank(suit),
            });
            undo.column_changes.push(ColumnUndo {
                column: dest,
                previous_hidden_count: state.columns[column_index(dest)].hidden_count,
                removed_cards: Vec::new(),
                added_cards: vec![card],
            });
            state.foundations.set_top_rank(suit, rank.predecessor());
            state.columns[column_index(dest)].face_up.push(card);
            moved_cards.push(card);
        }
        AtomicMove::StockAdvance => {
            undo.stock_change = Some(stock_undo_for_advance(state)?);
            state.stock.advance_draw()?;
            stock_changed = true;
        }
        AtomicMove::StockRecycle => {
            undo.stock_change = Some(stock_undo_for_recycle(state)?);
            state.stock.recycle()?;
            stock_changed = true;
        }
    }

    state.validate_consistency()?;

    undo.moved_cards = moved_cards.clone();
    undo.revealed = revealed;

    let outcome = MoveOutcome {
        applied_move: atomic,
        semantics,
        moved_cards,
        revealed,
        created_empty_column: semantics.creates_empty_column,
        filled_empty_column: semantics.fills_empty_column,
        stock_changed,
    };

    Ok(MoveTransition { outcome, undo })
}

/// Returns true if applying this move would uncover a hidden tableau card.
pub fn requires_reveal(state: &VisibleState, atomic: AtomicMove) -> bool {
    match atomic {
        AtomicMove::TableauToFoundation { src } => {
            let src_col = &state.columns[column_index(src)];
            src_col.face_up_len() == 1 && src_col.hidden_count > 0
        }
        AtomicMove::TableauToTableau { src, run_start, .. } => {
            let src_col = &state.columns[column_index(src)];
            run_start == 0 && !src_col.face_up.is_empty() && src_col.hidden_count > 0
        }
        AtomicMove::WasteToTableau { .. }
        | AtomicMove::WasteToFoundation
        | AtomicMove::FoundationToTableau { .. }
        | AtomicMove::StockAdvance
        | AtomicMove::StockRecycle => false,
    }
}

fn enforce_reveal_contract(
    state: &VisibleState,
    atomic: AtomicMove,
    revealed_card: Option<Card>,
) -> SolverResult<()> {
    match (requires_reveal(state, atomic), revealed_card) {
        (true, None) => Err(SolverError::RevealCardRequired {
            column: reveal_source_column(atomic).expect("reveal moves have source columns"),
        }),
        (false, Some(card)) => Err(SolverError::UnexpectedRevealCard { card }),
        _ => Ok(()),
    }
}

fn reveal_source_column(atomic: AtomicMove) -> Option<ColumnId> {
    match atomic {
        AtomicMove::TableauToFoundation { src } | AtomicMove::TableauToTableau { src, .. } => {
            Some(src)
        }
        _ => None,
    }
}

fn auto_reveal_if_needed(
    state: &mut VisibleState,
    column: ColumnId,
    revealed_card: &mut Option<Card>,
    undo: &mut MoveUndo,
) -> SolverResult<Option<RevealRecord>> {
    let column_index = column_index(column);
    let tableau = &mut state.columns[column_index];
    if tableau.face_up.is_empty() && tableau.hidden_count > 0 {
        let card = revealed_card
            .take()
            .ok_or(SolverError::RevealCardRequired { column })?;
        tableau.hidden_count -= 1;
        tableau.face_up.push(card);
        if let Some(change) = undo
            .column_changes
            .iter_mut()
            .find(|change| change.column == column)
        {
            change.added_cards.push(card);
        } else {
            undo.column_changes.push(ColumnUndo {
                column,
                previous_hidden_count: tableau.hidden_count + 1,
                removed_cards: Vec::new(),
                added_cards: vec![card],
            });
        }
        Ok(Some(RevealRecord { column, card }))
    } else {
        Ok(None)
    }
}

fn undo_column_change(column: &mut TableauColumn, change: &ColumnUndo) -> SolverResult<()> {
    if !change.added_cards.is_empty() {
        if column.face_up.len() < change.added_cards.len() {
            return Err(SolverError::InvalidState(format!(
                "cannot undo column {}: added-card suffix is longer than face-up run",
                change.column
            )));
        }
        let suffix_start = column.face_up.len() - change.added_cards.len();
        if column.face_up[suffix_start..] != change.added_cards {
            return Err(SolverError::InvalidState(format!(
                "cannot undo column {}: added-card suffix does not match",
                change.column
            )));
        }
        column.face_up.truncate(suffix_start);
    }

    column.face_up.extend(change.removed_cards.iter().copied());
    column.hidden_count = change.previous_hidden_count;
    column.validate_structure()
}

fn stock_undo_for_advance(state: &VisibleState) -> SolverResult<StockUndo> {
    state.stock.validate_structure()?;
    if !state.stock.can_advance() {
        return Err(SolverError::IllegalMove(
            "cannot advance stock when stock prefix is empty".to_string(),
        ));
    }
    Ok(StockUndo {
        previous_cursor: state.stock.cursor,
        previous_stock_len: state.stock.stock_len,
        previous_accessible_depth: state.stock.accessible_depth,
        previous_pass_index: state.stock.pass_index,
        kind: StockUndoKind::Advance {
            drawn: usize::from(state.stock.draw_count).min(state.stock.stock_len),
        },
    })
}

fn stock_undo_for_remove_accessible(state: &VisibleState) -> SolverResult<StockUndo> {
    state.stock.validate_structure()?;
    let index = state.stock.cursor.ok_or_else(|| {
        SolverError::IllegalMove("no accessible waste card is available".to_string())
    })?;
    let card = state.stock.ring_cards[index];
    Ok(StockUndo {
        previous_cursor: state.stock.cursor,
        previous_stock_len: state.stock.stock_len,
        previous_accessible_depth: state.stock.accessible_depth,
        previous_pass_index: state.stock.pass_index,
        kind: StockUndoKind::RemoveAccessible { card, index },
    })
}

fn stock_undo_for_recycle(state: &VisibleState) -> SolverResult<StockUndo> {
    state.stock.validate_structure()?;
    if !state.stock.can_recycle() {
        return Err(SolverError::IllegalMove(
            "cannot recycle stock under current stock/waste state".to_string(),
        ));
    }
    Ok(StockUndo {
        previous_cursor: state.stock.cursor,
        previous_stock_len: state.stock.stock_len,
        previous_accessible_depth: state.stock.accessible_depth,
        previous_pass_index: state.stock.pass_index,
        kind: StockUndoKind::Recycle,
    })
}

fn undo_stock_change(
    stock: &mut crate::stock::CyclicStockState,
    change: StockUndo,
) -> SolverResult<()> {
    match change.kind {
        StockUndoKind::Advance { drawn } => {
            stock.ring_cards.rotate_right(drawn);
        }
        StockUndoKind::RemoveAccessible { card, index } => {
            if index > stock.ring_cards.len() {
                return Err(SolverError::InvalidStockState(format!(
                    "cannot undo stock removal at index {index}; ring length is {}",
                    stock.ring_cards.len()
                )));
            }
            stock.ring_cards.insert(index, card);
        }
        StockUndoKind::Recycle => {}
    }

    stock.cursor = change.previous_cursor;
    stock.stock_len = change.previous_stock_len;
    stock.accessible_depth = change.previous_accessible_depth;
    stock.pass_index = change.previous_pass_index;
    stock.validate_structure()
}

fn normalize_atomic_moves(moves: &mut Vec<AtomicMove>) {
    moves.sort();
    moves.dedup();
}

fn macro_kind_for_atomic(state: &VisibleState, atomic: AtomicMove) -> MacroMoveKind {
    match atomic {
        AtomicMove::WasteToTableau { dest } => MacroMoveKind::PlayWasteToTableau { dest },
        AtomicMove::WasteToFoundation => MacroMoveKind::PlayWasteToFoundation,
        AtomicMove::TableauToFoundation { src } => MacroMoveKind::MoveTopToFoundation { src },
        AtomicMove::TableauToTableau {
            src,
            dest,
            run_start,
        } if state.columns[column_index(dest)].is_empty()
            && state.columns[column_index(src)].face_up[run_start].rank() == Rank::King =>
        {
            MacroMoveKind::PlaceKingRun {
                src,
                dest,
                run_start,
            }
        }
        AtomicMove::TableauToTableau {
            src,
            dest,
            run_start,
        } => MacroMoveKind::MoveRun {
            src,
            dest,
            run_start,
        },
        AtomicMove::FoundationToTableau { suit, dest } => {
            MacroMoveKind::FoundationRetreat { suit, dest }
        }
        AtomicMove::StockAdvance => MacroMoveKind::AdvanceStock,
        AtomicMove::StockRecycle => MacroMoveKind::RecycleStock,
    }
}

fn semantics_for_atomic_move(state: &VisibleState, atomic: AtomicMove) -> MoveSemantics {
    match atomic {
        AtomicMove::WasteToTableau { dest } => MoveSemantics {
            uses_waste: true,
            fills_empty_column: state.columns[column_index(dest)].is_empty(),
            affects_stock_cycle: true,
            ..MoveSemantics::default()
        },
        AtomicMove::WasteToFoundation => MoveSemantics {
            uses_waste: true,
            moves_to_foundation: true,
            affects_stock_cycle: true,
            ..MoveSemantics::default()
        },
        AtomicMove::TableauToFoundation { src } => {
            let src_col = &state.columns[column_index(src)];
            MoveSemantics {
                causes_reveal: src_col.face_up_len() == 1 && src_col.hidden_count > 0,
                moves_to_foundation: true,
                creates_empty_column: src_col.face_up_len() == 1 && src_col.hidden_count == 0,
                ..MoveSemantics::default()
            }
        }
        AtomicMove::TableauToTableau {
            src,
            dest,
            run_start,
        } => {
            let src_col = &state.columns[column_index(src)];
            MoveSemantics {
                causes_reveal: run_start == 0 && src_col.hidden_count > 0,
                creates_empty_column: run_start == 0 && src_col.hidden_count == 0,
                fills_empty_column: state.columns[column_index(dest)].is_empty(),
                ..MoveSemantics::default()
            }
        }
        AtomicMove::FoundationToTableau { dest, .. } => MoveSemantics {
            moves_from_foundation: true,
            fills_empty_column: state.columns[column_index(dest)].is_empty(),
            ..MoveSemantics::default()
        },
        AtomicMove::StockAdvance | AtomicMove::StockRecycle => MoveSemantics {
            affects_stock_cycle: true,
            ..MoveSemantics::default()
        },
    }
}

fn can_move_to_foundation(state: &VisibleState, card: Card) -> bool {
    card.can_move_to_foundation(state.foundations.top_rank(card.suit()))
}

fn ensure_can_move_to_foundation(state: &VisibleState, card: Card) -> SolverResult<()> {
    if can_move_to_foundation(state, card) {
        Ok(())
    } else {
        Err(SolverError::IllegalMove(format!(
            "{card} cannot move to foundation"
        )))
    }
}

/// Returns true if `card` can be placed onto a tableau destination column.
pub fn can_place_on_tableau(card: Card, dest: &TableauColumn) -> bool {
    match dest.top_face_up() {
        Some(top) => card.can_tableau_stack_on(top),
        None => dest.is_empty() && card.rank() == Rank::King,
    }
}

fn ensure_can_place_on_tableau(card: Card, dest: &TableauColumn) -> SolverResult<()> {
    if can_place_on_tableau(card, dest) {
        Ok(())
    } else {
        Err(SolverError::IllegalMove(format!(
            "{card} cannot be placed on tableau destination {dest}"
        )))
    }
}

/// Returns all legal source indices for movable face-up suffixes in a column.
pub fn movable_run_starts(column: &TableauColumn) -> impl Iterator<Item = usize> + '_ {
    0..column.face_up_len()
}

fn ensure_valid_run_start(column: &TableauColumn, run_start: usize) -> SolverResult<()> {
    if run_start < column.face_up_len() {
        Ok(())
    } else {
        Err(SolverError::IllegalMove(format!(
            "run start {run_start} is out of range for face-up length {}",
            column.face_up_len()
        )))
    }
}

fn column_index(column: ColumnId) -> usize {
    usize::from(column.index())
}

#[cfg(test)]
mod tests;
