use super::*;
use crate::core::HiddenAssignments;
use crate::stock::CyclicStockState;

fn col(index: u8) -> ColumnId {
    ColumnId::new(index).unwrap()
}

fn card(text: &str) -> Card {
    text.parse().unwrap()
}

fn visible_with_stock_card(waste: Card) -> VisibleState {
    let mut state = VisibleState::default();
    state.stock = CyclicStockState::from_parts(vec![waste], 0, 1, 0, None, 3);
    state
}

fn assert_round_trip(mut state: VisibleState, mv: AtomicMove, reveal: Option<Card>) {
    let before = state.clone();
    let transition = apply_atomic_move(&mut state, mv, reveal).unwrap();
    undo_atomic_move(&mut state, transition.undo).unwrap();
    assert_eq!(state, before);
}

#[test]
fn waste_to_tableau_legality_apply_and_undo() {
    let mut state = visible_with_stock_card(card("6h"));
    state.columns[0] = TableauColumn::new(0, vec![card("7s")]);
    let before = state.clone();

    let moves = generate_legal_atomic_moves(&state);
    assert!(moves.contains(&AtomicMove::WasteToTableau { dest: col(0) }));

    let transition = apply_atomic_move(
        &mut state,
        AtomicMove::WasteToTableau { dest: col(0) },
        None,
    )
    .unwrap();
    assert_eq!(state.columns[0].top_face_up(), Some(card("6h")));
    assert!(state.stock.accessible_card().is_none());
    assert!(transition.outcome.semantics.uses_waste);
    assert!(transition.outcome.stock_changed);

    undo_atomic_move(&mut state, transition.undo).unwrap();
    assert_eq!(state, before);
}

#[test]
fn waste_to_foundation_legality_and_application() {
    let mut state = visible_with_stock_card(card("Ac"));
    let transition = apply_atomic_move(&mut state, AtomicMove::WasteToFoundation, None).unwrap();

    assert_eq!(state.foundations.top_rank(Suit::Clubs), Some(Rank::Ace));
    assert_eq!(transition.outcome.moved_cards, vec![card("Ac")]);
    assert!(state.stock.is_empty());
}

#[test]
fn tableau_to_foundation_apply_and_undo() {
    let mut state = VisibleState::default();
    state
        .foundations
        .set_top_rank(Suit::Spades, Some(Rank::Six));
    state.columns[0] = TableauColumn::new(0, vec![card("7s")]);
    let before = state.clone();

    let transition = apply_atomic_move(
        &mut state,
        AtomicMove::TableauToFoundation { src: col(0) },
        None,
    )
    .unwrap();

    assert_eq!(state.foundations.top_rank(Suit::Spades), Some(Rank::Seven));
    assert!(state.columns[0].is_empty());
    assert!(transition.outcome.created_empty_column);

    undo_atomic_move(&mut state, transition.undo).unwrap();
    assert_eq!(state, before);
}

#[test]
fn tableau_run_to_tableau_supports_partial_runs() {
    let mut state = VisibleState::default();
    state.columns[0] = TableauColumn::new(0, vec![card("9s"), card("8h"), card("7c")]);
    state.columns[1] = TableauColumn::new(0, vec![card("9c")]);

    let mv = AtomicMove::TableauToTableau {
        src: col(0),
        dest: col(1),
        run_start: 1,
    };
    assert!(generate_legal_atomic_moves(&state).contains(&mv));

    apply_atomic_move(&mut state, mv, None).unwrap();

    assert_eq!(state.columns[0].face_up, vec![card("9s")]);
    assert_eq!(
        state.columns[1].face_up,
        vec![card("9c"), card("8h"), card("7c")]
    );
}

#[test]
fn empty_column_requires_king_head() {
    let mut state = VisibleState::default();
    state.columns[0] = TableauColumn::new(0, vec![card("Qh")]);
    state.columns[1] = TableauColumn::new(0, Vec::new());

    assert!(
        !generate_legal_atomic_moves(&state).contains(&AtomicMove::TableauToTableau {
            src: col(0),
            dest: col(1),
            run_start: 0,
        })
    );
    assert!(apply_atomic_move(
        &mut state,
        AtomicMove::TableauToTableau {
            src: col(0),
            dest: col(1),
            run_start: 0,
        },
        None,
    )
    .is_err());

    state.columns[0] = TableauColumn::new(0, vec![card("Ks"), card("Qh")]);
    assert!(
        generate_legal_atomic_moves(&state).contains(&AtomicMove::TableauToTableau {
            src: col(0),
            dest: col(1),
            run_start: 0,
        })
    );
}

#[test]
fn auto_flip_on_uncover_is_transition_side_effect() {
    let mut state = VisibleState::default();
    state.columns[0] = TableauColumn::new(1, vec![card("7s")]);
    state.columns[1] = TableauColumn::new(0, vec![card("8h")]);

    let transition = apply_atomic_move(
        &mut state,
        AtomicMove::TableauToTableau {
            src: col(0),
            dest: col(1),
            run_start: 0,
        },
        Some(card("Ac")),
    )
    .unwrap();

    assert_eq!(state.columns[0].hidden_count, 0);
    assert_eq!(state.columns[0].face_up, vec![card("Ac")]);
    assert_eq!(
        transition.outcome.revealed,
        Some(RevealRecord {
            column: col(0),
            card: card("Ac"),
        })
    );
    assert!(transition.outcome.semantics.causes_reveal);
}

#[test]
fn reveal_required_when_uncovering_hidden_tableau() {
    let mut state = VisibleState::default();
    state.columns[0] = TableauColumn::new(1, vec![card("7s")]);
    state.columns[1] = TableauColumn::new(0, vec![card("8h")]);

    assert!(matches!(
        apply_atomic_move(
            &mut state,
            AtomicMove::TableauToTableau {
                src: col(0),
                dest: col(1),
                run_start: 0,
            },
            None,
        ),
        Err(SolverError::RevealCardRequired { column }) if column == col(0)
    ));
}

#[test]
fn reveal_card_is_rejected_when_move_does_not_uncover() {
    let mut state = VisibleState::default();
    state.columns[0] = TableauColumn::new(0, vec![card("7s")]);
    state.columns[1] = TableauColumn::new(0, vec![card("8h")]);

    assert!(matches!(
        apply_atomic_move(
            &mut state,
            AtomicMove::TableauToTableau {
                src: col(0),
                dest: col(1),
                run_start: 0,
            },
            Some(card("Ac")),
        ),
        Err(SolverError::UnexpectedRevealCard { card: unexpected }) if unexpected == card("Ac")
    ));
}

#[test]
fn requires_reveal_reports_uncovering_moves_only() {
    let mut state = VisibleState::default();
    state.columns[0] = TableauColumn::new(1, vec![card("7s")]);
    state.columns[1] = TableauColumn::new(0, vec![card("8h")]);

    assert!(requires_reveal(
        &state,
        AtomicMove::TableauToTableau {
            src: col(0),
            dest: col(1),
            run_start: 0,
        }
    ));
    assert!(!requires_reveal(&state, AtomicMove::StockAdvance));
}

#[test]
fn full_state_reveal_lookup_handles_whole_multi_card_runs() {
    let mut visible = VisibleState::default();
    visible.columns[0] = TableauColumn::new(1, vec![card("6h"), card("5s")]);
    visible.columns[1] = TableauColumn::new(0, vec![card("7s")]);
    let used = [card("Ac"), card("6h"), card("5s"), card("7s")];
    let stock_cards = (0..Card::COUNT)
        .map(|index| Card::new(index as u8).unwrap())
        .filter(|candidate| !used.contains(candidate))
        .collect();
    visible.stock = CyclicStockState::new(stock_cards, None, 0, None, 3);
    let mut full_state = FullState::new(
        visible,
        HiddenAssignments::new(vec![HiddenAssignment::new(
            HiddenSlot::new(col(0), 0),
            card("Ac"),
        )]),
    );

    let transition = apply_atomic_move_full_state(
        &mut full_state,
        AtomicMove::TableauToTableau {
            src: col(0),
            dest: col(1),
            run_start: 0,
        },
    )
    .unwrap();

    assert_eq!(
        transition.outcome.revealed,
        Some(RevealRecord {
            column: col(0),
            card: card("Ac"),
        })
    );
    assert_eq!(full_state.visible.columns[0].face_up, vec![card("Ac")]);
    assert!(full_state.hidden_assignments.is_empty());
}

#[test]
fn foundation_to_tableau_round_trip() {
    let mut state = VisibleState::default();
    state
        .foundations
        .set_top_rank(Suit::Spades, Some(Rank::Seven));
    state.columns[0] = TableauColumn::new(0, vec![card("8h")]);
    let before = state.clone();

    let mv = AtomicMove::FoundationToTableau {
        suit: Suit::Spades,
        dest: col(0),
    };
    assert!(generate_legal_atomic_moves(&state).contains(&mv));

    let transition = apply_atomic_move(&mut state, mv, None).unwrap();
    assert_eq!(state.columns[0].top_face_up(), Some(card("7s")));
    assert_eq!(state.foundations.top_rank(Suit::Spades), Some(Rank::Six));

    undo_atomic_move(&mut state, transition.undo).unwrap();
    assert_eq!(state, before);
}

#[test]
fn stock_advance_and_recycle_apply() {
    let mut state = VisibleState::default();
    state.stock = CyclicStockState::from_parts(
        vec![card("Ac"), card("2c"), card("3c")],
        3,
        0,
        0,
        Some(1),
        3,
    );

    let advance = apply_atomic_move(&mut state, AtomicMove::StockAdvance, None).unwrap();
    assert_eq!(state.stock.accessible_card(), Some(card("3c")));
    assert_eq!(state.stock.stock_len(), 0);

    let recycle = apply_atomic_move(&mut state, AtomicMove::StockRecycle, None).unwrap();
    assert_eq!(state.stock.stock_len(), 3);
    assert_eq!(state.stock.pass_index, 1);
    assert!(state.stock.accessible_card().is_none());

    undo_atomic_move(&mut state, recycle.undo).unwrap();
    assert_eq!(state.stock.stock_len(), 0);
    undo_atomic_move(&mut state, advance.undo).unwrap();
    assert_eq!(state.stock.stock_len(), 3);
}

#[test]
fn generated_moves_are_unique_and_stably_sorted() {
    let mut state = visible_with_stock_card(card("Kh"));
    state.columns[0] = TableauColumn::new(0, Vec::new());
    state.columns[1] = TableauColumn::new(0, vec![card("7s"), card("6h")]);
    state.columns[2] = TableauColumn::new(0, vec![card("7c")]);
    state
        .foundations
        .set_top_rank(Suit::Spades, Some(Rank::Six));

    let moves = generate_legal_atomic_moves(&state);
    let mut sorted = moves.clone();
    sorted.sort();
    sorted.dedup();

    assert_eq!(moves, sorted);
}

#[test]
fn macro_move_ids_are_deterministic_from_sorted_atomic_order() {
    let mut state = visible_with_stock_card(card("Kh"));
    state.columns[0] = TableauColumn::new(0, Vec::new());
    state.columns[1] = TableauColumn::new(0, vec![card("7s"), card("6h")]);

    let first = generate_legal_macro_moves(&state);
    let second = generate_legal_macro_moves(&state);

    assert_eq!(first, second);
    for (index, macro_move) in first.iter().enumerate() {
        assert_eq!(macro_move.id, MoveId::new(index as u32));
    }
}

#[test]
fn delta_undo_records_only_changed_components() {
    let mut state = visible_with_stock_card(card("6h"));
    state.columns[0] = TableauColumn::new(0, vec![card("7s")]);

    let transition = apply_atomic_move(
        &mut state,
        AtomicMove::WasteToTableau { dest: col(0) },
        None,
    )
    .unwrap();

    assert_eq!(transition.undo.column_changes.len(), 1);
    assert_eq!(transition.undo.column_changes[0].column, col(0));
    assert_eq!(
        transition.undo.column_changes[0].added_cards,
        vec![card("6h")]
    );
    assert!(transition.undo.stock_change.is_some());
    assert!(transition.undo.foundation_change.is_none());
}

#[test]
fn apply_undo_symmetry_for_representative_move_types() {
    let mut waste_tableau = visible_with_stock_card(card("6h"));
    waste_tableau.columns[0] = TableauColumn::new(0, vec![card("7s")]);
    assert_round_trip(
        waste_tableau,
        AtomicMove::WasteToTableau { dest: col(0) },
        None,
    );

    assert_round_trip(
        visible_with_stock_card(card("Ac")),
        AtomicMove::WasteToFoundation,
        None,
    );

    let mut tableau_foundation = VisibleState::default();
    tableau_foundation
        .foundations
        .set_top_rank(Suit::Spades, Some(Rank::Six));
    tableau_foundation.columns[0] = TableauColumn::new(0, vec![card("7s")]);
    assert_round_trip(
        tableau_foundation,
        AtomicMove::TableauToFoundation { src: col(0) },
        None,
    );

    let mut reveal_foundation = VisibleState::default();
    reveal_foundation.columns[0] = TableauColumn::new(1, vec![card("Ac")]);
    assert_round_trip(
        reveal_foundation,
        AtomicMove::TableauToFoundation { src: col(0) },
        Some(card("2d")),
    );

    let mut partial_run = VisibleState::default();
    partial_run.columns[0] = TableauColumn::new(0, vec![card("9s"), card("8h"), card("7c")]);
    partial_run.columns[1] = TableauColumn::new(0, vec![card("9c")]);
    assert_round_trip(
        partial_run,
        AtomicMove::TableauToTableau {
            src: col(0),
            dest: col(1),
            run_start: 1,
        },
        None,
    );

    let mut foundation_tableau = VisibleState::default();
    foundation_tableau
        .foundations
        .set_top_rank(Suit::Spades, Some(Rank::Seven));
    foundation_tableau.columns[0] = TableauColumn::new(0, vec![card("8h")]);
    assert_round_trip(
        foundation_tableau,
        AtomicMove::FoundationToTableau {
            suit: Suit::Spades,
            dest: col(0),
        },
        None,
    );

    let mut stock_advance = VisibleState::default();
    stock_advance.stock =
        CyclicStockState::from_parts(vec![card("Ac"), card("2c"), card("3c")], 3, 0, 0, None, 3);
    assert_round_trip(stock_advance, AtomicMove::StockAdvance, None);

    let mut stock_recycle = VisibleState::default();
    stock_recycle.stock =
        CyclicStockState::from_parts(vec![card("Ac"), card("2c"), card("3c")], 0, 1, 0, None, 3);
    assert_round_trip(stock_recycle, AtomicMove::StockRecycle, None);
}

#[test]
fn macro_generation_wraps_atomic_moves_with_semantics() {
    let mut state = visible_with_stock_card(card("Kh"));
    state.columns[0] = TableauColumn::new(0, Vec::new());

    let macros = generate_legal_macro_moves(&state);
    let king_macro = macros
        .iter()
        .find(|macro_move| macro_move.atomic == AtomicMove::WasteToTableau { dest: col(0) })
        .expect("waste king should be playable to empty tableau");

    assert_eq!(
        king_macro.kind,
        MacroMoveKind::PlayWasteToTableau { dest: col(0) }
    );
    assert!(king_macro.semantics.uses_waste);
    assert!(king_macro.semantics.fills_empty_column);
}
