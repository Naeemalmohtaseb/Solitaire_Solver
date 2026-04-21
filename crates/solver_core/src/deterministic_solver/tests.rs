use super::*;
use crate::{
    cards::Suit,
    core::{HiddenAssignment, HiddenAssignments, HiddenSlot, TableauColumn},
    ml::{
        LeafEvaluationMode, VNetActivation, VNetBackend, VNetEvaluator, VNetInferenceArtifact,
        VNetLayerArtifact,
    },
    stock::CyclicStockState,
    types::ColumnId,
};

fn col(index: u8) -> ColumnId {
    ColumnId::new(index).unwrap()
}

fn card(text: &str) -> Card {
    text.parse().unwrap()
}

fn complete_foundations() -> crate::core::FoundationState {
    let mut foundations = crate::core::FoundationState::default();
    for suit in [Suit::Clubs, Suit::Diamonds, Suit::Hearts, Suit::Spades] {
        foundations.set_top_rank(suit, Some(Rank::King));
    }
    foundations
}

fn foundations_missing_king(suit: Suit) -> crate::core::FoundationState {
    let mut foundations = complete_foundations();
    foundations.set_top_rank(suit, Some(Rank::Queen));
    foundations
}

fn full_from_visible(visible: VisibleState) -> FullState {
    FullState::new(visible, HiddenAssignments::empty())
}

fn one_move_to_win_state() -> FullState {
    let mut visible = VisibleState::default();
    visible.foundations = foundations_missing_king(Suit::Spades);
    visible.columns[0] = TableauColumn::new(0, vec![card("Ks")]);
    full_from_visible(visible)
}

fn dead_end_state() -> FullState {
    let mut visible = VisibleState::default();
    let cards = (0..Card::COUNT)
        .map(|index| Card::new(index as u8).unwrap())
        .collect();
    visible.stock = CyclicStockState::from_parts(cards, 0, 0, 0, Some(0), 3);
    full_from_visible(visible)
}

fn reveal_state() -> FullState {
    let mut visible = VisibleState::default();
    visible.foundations = complete_foundations();
    visible
        .foundations
        .set_top_rank(Suit::Spades, Some(Rank::Jack));
    visible.columns[0] = TableauColumn::new(1, vec![card("Qs")]);
    let slot = HiddenSlot::new(col(0), 0);
    FullState::new(
        visible,
        HiddenAssignments::new(vec![HiddenAssignment::new(slot, card("Ks"))]),
    )
}

fn safe_foundation_vs_shuffle_visible() -> VisibleState {
    let mut visible = VisibleState::default();
    visible.columns[0] = TableauColumn::new(0, vec![card("Ac")]);
    visible.columns[1] = TableauColumn::new(0, vec![card("6h"), card("5s")]);
    visible.columns[2] = TableauColumn::new(0, vec![card("6d")]);
    visible
}

fn reveal_vs_shuffle_visible() -> VisibleState {
    let mut visible = VisibleState::default();
    visible
        .foundations
        .set_top_rank(Suit::Spades, Some(Rank::Jack));
    visible.columns[0] = TableauColumn::new(1, vec![card("Qs")]);
    visible.columns[1] = TableauColumn::new(0, vec![card("6h"), card("5s")]);
    visible.columns[2] = TableauColumn::new(0, vec![card("6d")]);
    visible
}

fn stock_vs_churn_visible() -> VisibleState {
    let mut visible = VisibleState::default();
    visible.columns[0] = TableauColumn::new(0, vec![card("6h"), card("5s")]);
    visible.columns[1] = TableauColumn::new(0, vec![card("6d")]);
    visible.stock =
        CyclicStockState::from_parts(vec![card("7c"), card("8d"), card("9c")], 3, 0, 0, None, 3);
    visible
}

fn closure_invocation_state() -> FullState {
    let mut visible = VisibleState::default();
    visible.foundations = complete_foundations();
    visible.foundations.set_top_rank(Suit::Clubs, None);
    let clubs = [
        "2c", "3c", "4c", "5c", "6c", "7c", "8c", "9c", "Tc", "Jc", "Qc", "Kc", "Ac",
    ];
    visible.stock = CyclicStockState::from_parts(
        clubs.iter().map(|text| card(text)).collect(),
        0,
        1,
        0,
        Some(0),
        3,
    );
    full_from_visible(visible)
}

fn tt_config(enabled: bool) -> DeterministicSearchConfig {
    DeterministicSearchConfig {
        tt: DeterministicTtConfig {
            enabled,
            capacity: if enabled { 32 } else { 0 },
            store_approx: true,
        },
        ..DeterministicSearchConfig::default()
    }
}

fn flat_half_vnet() -> VNetEvaluator {
    let artifact = VNetInferenceArtifact {
        schema_version: "solitaire-vnet-mlp-json-v1".to_string(),
        model_role: "VNet".to_string(),
        model_type: "mlp".to_string(),
        input_dim: 114,
        hidden_sizes: Vec::new(),
        feature_normalization: "scale64".to_string(),
        label_mode: Some("TerminalOutcome".to_string()),
        dataset_metadata: None,
        layers: vec![VNetLayerArtifact {
            weights: vec![vec![0.0; 114]],
            biases: vec![0.0],
            activation: VNetActivation::Sigmoid,
        }],
    };
    VNetEvaluator::from_artifact(VNetBackend::RustMlpJson, artifact).unwrap()
}

#[test]
fn already_won_full_state_returns_exact_terminal_win() {
    let mut visible = VisibleState::default();
    visible.foundations = complete_foundations();
    let full = full_from_visible(visible);

    let result = DeterministicSolver::default().solve_exact(&full).unwrap();

    assert_eq!(result.outcome, SolveOutcome::ProvenWin);
    assert_eq!(result.value, 1.0);
    assert!(result.best_move.is_none());
}

#[test]
fn no_move_dead_end_returns_proven_loss() {
    let full = dead_end_state();

    let result = DeterministicSolver::default().solve_exact(&full).unwrap();

    assert_eq!(result.outcome, SolveOutcome::ProvenLoss);
    assert_eq!(result.value, 0.0);
}

#[test]
fn solver_identifies_trivial_one_move_win() {
    let full = one_move_to_win_state();

    let result = DeterministicSolver::default().solve_exact(&full).unwrap();

    assert_eq!(result.outcome, SolveOutcome::ProvenWin);
    assert!(matches!(
        result.best_move.map(|best| best.kind),
        Some(MacroMoveKind::MoveTopToFoundation { .. })
    ));
    assert_eq!(
        result.principal_line,
        vec![MacroMoveKind::MoveTopToFoundation { src: col(0) }]
    );
}

#[test]
fn full_state_reveal_apply_uses_and_removes_hidden_assignment() {
    let mut full = reveal_state();

    let transition =
        apply_atomic_move_full_state(&mut full, AtomicMove::TableauToFoundation { src: col(0) })
            .unwrap();

    assert!(full.hidden_assignments.is_empty());
    assert_eq!(full.visible.columns[0].face_up, vec![card("Ks")]);
    assert_eq!(transition.outcome.revealed.unwrap().card, card("Ks"));
}

#[test]
fn full_state_reveal_apply_undo_is_symmetric() {
    let mut full = reveal_state();
    let before = full.clone();

    let transition =
        apply_atomic_move_full_state(&mut full, AtomicMove::TableauToFoundation { src: col(0) })
            .unwrap();
    undo_atomic_move_full_state(&mut full, transition.undo).unwrap();

    assert_eq!(full, before);
}

#[test]
fn move_ordering_is_deterministic_and_prioritizes_reveals() {
    let full = reveal_state();

    let first = ordered_macro_moves(&full.visible, DeterministicSearchConfig::default());
    let second = ordered_macro_moves(&full.visible, DeterministicSearchConfig::default());

    assert_eq!(first, second);
    assert!(first[0].semantics.causes_reveal);
}

#[test]
fn move_ordering_prioritizes_safe_ace_foundation_over_tableau_shuffle() {
    let visible = safe_foundation_vs_shuffle_visible();
    let ordered = ordered_macro_moves(&visible, DeterministicSearchConfig::default());

    assert!(matches!(
        ordered.first().map(|macro_move| macro_move.kind),
        Some(MacroMoveKind::MoveTopToFoundation { src }) if src == col(0)
    ));
    assert!(strategic_move_score(&visible, &ordered[0]).safe_foundation_bonus > 0);
}

#[test]
fn move_ordering_keeps_reveal_before_no_information_shuffle() {
    let visible = reveal_vs_shuffle_visible();
    let ordered = ordered_macro_moves(&visible, DeterministicSearchConfig::default());

    assert!(ordered[0].semantics.causes_reveal);
    let shuffle = ordered
        .iter()
        .find(|macro_move| {
            matches!(
                macro_move.atomic,
                AtomicMove::TableauToTableau { src, dest, .. } if src == col(1) && dest == col(2)
            )
        })
        .expect("state should include a tableau shuffle");
    assert!(strategic_move_score(&visible, shuffle).churn_penalty < 0);
}

#[test]
fn stock_access_orders_before_low_information_tableau_churn() {
    let visible = stock_vs_churn_visible();
    let ordered = ordered_macro_moves(&visible, DeterministicSearchConfig::default());

    assert!(matches!(ordered[0].kind, MacroMoveKind::AdvanceStock));
    let shuffle = ordered
        .iter()
        .find(|macro_move| matches!(macro_move.kind, MacroMoveKind::MoveRun { .. }))
        .expect("state should include a tableau shuffle");
    let stock_score = strategic_move_score(&visible, &ordered[0]);
    let shuffle_score = strategic_move_score(&visible, shuffle);
    assert!(stock_score.total > shuffle_score.total);
    assert!(shuffle_score.churn_penalty < 0);
}

#[test]
fn approximate_evaluation_values_reveal_potential_over_churn_only_shape() {
    let reveal_visible = reveal_vs_shuffle_visible();
    let mut churn_visible = VisibleState::default();
    churn_visible.columns[0] = TableauColumn::new(0, vec![card("6h"), card("5s")]);
    churn_visible.columns[1] = TableauColumn::new(0, vec![card("6d")]);
    churn_visible.columns[2] = TableauColumn::new(1, Vec::new());

    assert!(
        evaluate_state(&reveal_visible, EvaluatorWeights::default())
            > evaluate_state(&churn_visible, EvaluatorWeights::default())
    );
}

#[test]
fn bounded_solve_respects_depth_budget() {
    let full = one_move_to_win_state();
    let solver = DeterministicSolver::new(DeterministicSearchConfig {
        budget: SolveBudget {
            node_budget: Some(100),
            depth_budget: Some(0),
            wall_clock_limit_ms: None,
        },
        ..DeterministicSearchConfig::default()
    });

    let result = solver.solve_bounded(&full).unwrap();

    assert_eq!(result.outcome, SolveOutcome::Unknown);
    assert!(result.stats.budget_cutoffs > 0);
}

#[test]
fn fast_evaluation_returns_stable_result_on_simple_state() {
    let full = one_move_to_win_state();
    let solver = DeterministicSolver::default();

    let first = solver.evaluate_fast(&full).unwrap();
    let second = solver.evaluate_fast(&full).unwrap();

    assert_eq!(first.value, second.value);
    assert!(first.value > 0.0);
}

#[test]
fn closure_is_invoked_during_search() {
    let full = closure_invocation_state();

    let result = DeterministicSolver::default().solve_bounded(&full).unwrap();

    assert!(result.stats.closure_steps_applied > 0);
}

#[test]
fn bounded_solver_returns_best_move_when_one_exists() {
    let full = one_move_to_win_state();

    let result = DeterministicSolver::default().solve_bounded(&full).unwrap();

    assert!(result.best_move.is_some());
}

#[test]
fn same_full_state_hashes_identically() {
    let full = reveal_state();

    let first = DeterministicHashKey::from_full_state(&full);
    let second = DeterministicHashKey::from_full_state(&full.clone());

    assert_eq!(first, second);
}

#[test]
fn equivalent_repeated_state_is_found_in_tt() {
    let full = one_move_to_win_state();
    let key = DeterministicHashKey::from_full_state(&full);
    let mut tt = DeterministicTt::with_capacity(16);
    let entry = DeterministicTtEntry {
        key,
        value: DeterministicTtValue {
            outcome: SolveOutcome::ProvenWin,
            value: 1.0,
            bound: DeterministicBound::Exact,
        },
        best_move: None,
        searched_depth: 4,
        age: 0,
    };

    assert!(tt.store(entry).stored);
    assert_eq!(
        tt.probe(key).unwrap().value.bound,
        DeterministicBound::Exact
    );
}

#[test]
fn exact_tt_entry_is_reused_on_repeated_solve() {
    let full = one_move_to_win_state();
    let solver = DeterministicSolver::new(tt_config(true));

    let first = solver.solve_exact(&full).unwrap();
    let second = solver.solve_exact(&full).unwrap();

    assert_eq!(first.outcome, SolveOutcome::ProvenWin);
    assert_eq!(second.outcome, SolveOutcome::ProvenWin);
    assert!(second.stats.tt_exact_hits > 0);
}

#[test]
fn approximate_tt_entry_does_not_masquerade_as_exact() {
    let full = one_move_to_win_state();
    let solver = DeterministicSolver::new(tt_config(true));
    let key = DeterministicHashKey::from_full_state(&full);

    solver.tt.borrow_mut().store(DeterministicTtEntry {
        key,
        value: DeterministicTtValue {
            outcome: SolveOutcome::Unknown,
            value: 0.25,
            bound: DeterministicBound::Approx,
        },
        best_move: None,
        searched_depth: u16::MAX,
        age: 0,
    });

    let result = solver.solve_exact(&full).unwrap();

    assert_eq!(result.outcome, SolveOutcome::ProvenWin);
    assert_eq!(result.value, 1.0);
    assert_eq!(result.stats.tt_exact_hits, 0);
}

#[test]
fn tt_best_move_hint_influences_ordering_deterministically() {
    let mut visible = VisibleState::default();
    visible.columns[0] = TableauColumn::new(0, vec![card("Kh")]);
    visible.columns[1] = TableauColumn::new(0, vec![card("Ks")]);

    let base = ordered_macro_moves(&visible, DeterministicSearchConfig::default());
    assert!(base.len() > 1);
    let hint = base.last().cloned().unwrap();
    let hinted =
        ordered_macro_moves_with_hint(&visible, DeterministicSearchConfig::default(), Some(&hint));

    assert!(same_macro_action(&hint, &hinted[0]));
}

#[test]
fn bounded_capacity_tt_remains_safe() {
    let first = one_move_to_win_state();
    let second = reveal_state();
    let mut tt = DeterministicTt::with_capacity(1);

    for full in [&first, &second] {
        let key = DeterministicHashKey::from_full_state(full);
        tt.store(DeterministicTtEntry {
            key,
            value: DeterministicTtValue {
                outcome: SolveOutcome::Unknown,
                value: 0.5,
                bound: DeterministicBound::Approx,
            },
            best_move: None,
            searched_depth: 0,
            age: 0,
        });
    }

    assert_eq!(tt.capacity(), 1);
    assert!(
        tt.probe(DeterministicHashKey::from_full_state(&first))
            .is_some()
            || tt
                .probe(DeterministicHashKey::from_full_state(&second))
                .is_some()
    );
}

#[test]
fn solver_with_tt_matches_solver_without_tt_on_simple_exact_case() {
    let full = one_move_to_win_state();

    let with_tt = DeterministicSolver::new(tt_config(true))
        .solve_exact(&full)
        .unwrap();
    let without_tt = DeterministicSolver::new(tt_config(false))
        .solve_exact(&full)
        .unwrap();

    assert_eq!(with_tt.outcome, without_tt.outcome);
    assert_eq!(with_tt.value, without_tt.value);
}

#[test]
fn reveal_position_with_tt_preserves_correctness() {
    let full = reveal_state();
    let solver = DeterministicSolver::new(tt_config(true));

    let result = solver.solve_exact(&full).unwrap();

    assert_eq!(result.outcome, SolveOutcome::ProvenWin);
    assert!(result.stats.tt_stores > 0);
}

#[test]
fn vnet_leaf_mode_uses_loaded_evaluator_for_fast_eval() {
    let full = one_move_to_win_state();
    let config = DeterministicSearchConfig {
        leaf_eval_mode: LeafEvaluationMode::VNet,
        ..tt_config(false)
    };
    let solver = DeterministicSolver::new_with_vnet_evaluator(config, Some(flat_half_vnet()));

    let result = solver.evaluate_fast(&full).unwrap();

    assert_eq!(result.value, 0.5);
    assert!(result.stats.vnet_inferences > 0);
    assert_eq!(result.stats.vnet_fallbacks, 0);
}

#[test]
fn vnet_leaf_mode_falls_back_when_model_is_unavailable() {
    let full = one_move_to_win_state();
    let config = DeterministicSearchConfig {
        leaf_eval_mode: LeafEvaluationMode::VNet,
        ..tt_config(false)
    };
    let solver = DeterministicSolver::new(config);

    let result = solver.evaluate_fast(&full).unwrap();

    assert!(result.value > 0.5);
    assert_eq!(result.stats.vnet_inferences, 0);
    assert!(result.stats.vnet_fallbacks > 0);
}
