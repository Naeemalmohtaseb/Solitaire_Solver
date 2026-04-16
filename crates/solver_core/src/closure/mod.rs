//! Conservative deterministic closure and corridor compression.
//!
//! Closure is not search. It applies only forced or explicitly conservative
//! macro moves, records every step, and stops before choosing among meaningful
//! alternatives. The engine is built on the move system so reveal handling and
//! undo-compatible state transitions stay centralized.

use std::fmt;

use serde::{Deserialize, Serialize};

use crate::{
    cards::{Card, Rank},
    core::{FullState, VisibleState},
    moves::{
        apply_atomic_move, apply_atomic_move_full_state, generate_legal_macro_moves_with_config,
        requires_reveal, AtomicMove, FullStateMoveUndo, MacroMove, MacroMoveKind,
        MoveGenerationConfig, MoveOutcome, MoveSemantics,
    },
    types::MoveId,
};

/// Configuration for deterministic corridor compression.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ClosureConfig {
    /// Maximum number of macro steps closure may apply in one run.
    pub max_corridor_steps: u8,
    /// Enables conservative Ace/Two foundation closure.
    pub enable_forced_foundation_closure: bool,
    /// Enables automatic application when exactly one legal macro move exists.
    pub enable_single_move_closure: bool,
    /// Enables automatic king placement when it is the only legal king placement.
    pub enable_single_king_placement_closure: bool,
    /// Stops immediately after a closure step reveals a hidden tableau card.
    pub stop_on_reveal: bool,
    /// Stops at stock pivots instead of advancing stock amid other choices.
    pub stop_on_stock_pivot: bool,
    /// Runs full state validation before and after each closure step in debug builds.
    pub debug_validate_each_step: bool,
}

impl Default for ClosureConfig {
    fn default() -> Self {
        Self {
            max_corridor_steps: 8,
            enable_forced_foundation_closure: true,
            enable_single_move_closure: true,
            enable_single_king_placement_closure: true,
            stop_on_reveal: true,
            stop_on_stock_pivot: true,
            debug_validate_each_step: true,
        }
    }
}

/// Deterministic closure runner.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ClosureEngine {
    /// Closure controls for this engine.
    pub config: ClosureConfig,
}

impl ClosureEngine {
    /// Creates a closure engine with explicit configuration.
    pub const fn new(config: ClosureConfig) -> Self {
        Self { config }
    }

    /// Runs closure without supplying reveal cards.
    ///
    /// If the only safe closure step would require a hidden-card identity, this
    /// method stops before applying it. Deterministic full-state callers and
    /// tests can use [`Self::run_with_reveals`] to provide reveal identities.
    pub fn run(&self, state: &mut VisibleState) -> ClosureResult {
        self.run_with_reveals(state, std::iter::empty())
    }

    /// Runs closure and consumes supplied reveal cards when an automatic step
    /// uncovers a hidden tableau card.
    pub fn run_with_reveals<I>(&self, state: &mut VisibleState, reveals: I) -> ClosureResult
    where
        I: IntoIterator<Item = Card>,
    {
        let mut transcript = ClosureTranscript::default();
        let mut reveal_iter = reveals.into_iter();
        let mut revealed_any = false;

        loop {
            if self.config.debug_validate_each_step && state.debug_validate().is_err() {
                return ClosureResult::new(
                    state,
                    transcript,
                    ClosureStopReason::NoSafeClosureMove,
                    revealed_any,
                );
            }

            if state.is_structural_win() {
                return ClosureResult::new(
                    state,
                    transcript,
                    ClosureStopReason::TerminalWin,
                    revealed_any,
                );
            }

            let macro_moves = generate_legal_macro_moves_with_config(
                state,
                MoveGenerationConfig {
                    allow_foundation_retreats: true,
                },
            );

            if macro_moves.is_empty() {
                return ClosureResult::new(
                    state,
                    transcript,
                    ClosureStopReason::TerminalNoMoves,
                    revealed_any,
                );
            }

            if transcript.len() >= usize::from(self.config.max_corridor_steps) {
                return ClosureResult::new(
                    state,
                    transcript,
                    ClosureStopReason::CorridorDepthLimit,
                    revealed_any,
                );
            }

            let candidate = match self.select_candidate(state, &macro_moves) {
                CandidateDecision::Apply(candidate) => candidate,
                CandidateDecision::Stop(stop_reason) => {
                    return ClosureResult::new(state, transcript, stop_reason, revealed_any);
                }
            };

            let reveal_card = if requires_reveal(state, candidate.macro_move.atomic) {
                match reveal_iter.next() {
                    Some(card) => Some(card),
                    None => {
                        return ClosureResult::new(
                            state,
                            transcript,
                            ClosureStopReason::NoSafeClosureMove,
                            revealed_any,
                        );
                    }
                }
            } else {
                None
            };

            let transition =
                match apply_atomic_move(state, candidate.macro_move.atomic, reveal_card) {
                    Ok(transition) => transition,
                    Err(_) => {
                        return ClosureResult::new(
                            state,
                            transcript,
                            ClosureStopReason::NoSafeClosureMove,
                            revealed_any,
                        );
                    }
                };

            let revealed_step = transition.outcome.revealed.is_some();
            revealed_any |= revealed_step;
            transcript.push(ClosureStep {
                move_id: candidate.macro_move.id,
                macro_move: candidate.macro_move,
                reason: candidate.reason,
                semantics: transition.outcome.semantics,
                outcome: transition.outcome,
            });

            if self.config.debug_validate_each_step && state.debug_validate().is_err() {
                return ClosureResult::new(
                    state,
                    transcript,
                    ClosureStopReason::NoSafeClosureMove,
                    revealed_any,
                );
            }

            if revealed_step && self.config.stop_on_reveal {
                return ClosureResult::new(
                    state,
                    transcript,
                    ClosureStopReason::RevealOccurred,
                    revealed_any,
                );
            }
        }
    }

    /// Runs closure on a full deterministic state and returns undo records for
    /// every automatic step.
    pub fn run_full_state_with_undos(&self, full_state: &mut FullState) -> FullClosureRun {
        let mut transcript = ClosureTranscript::default();
        let mut undos = Vec::new();
        let mut revealed_any = false;

        loop {
            if self.config.debug_validate_each_step && full_state.debug_validate().is_err() {
                return FullClosureRun::new(
                    &full_state.visible,
                    transcript,
                    undos,
                    ClosureStopReason::NoSafeClosureMove,
                    revealed_any,
                );
            }

            if full_state.visible.is_structural_win() {
                return FullClosureRun::new(
                    &full_state.visible,
                    transcript,
                    undos,
                    ClosureStopReason::TerminalWin,
                    revealed_any,
                );
            }

            let macro_moves = generate_legal_macro_moves_with_config(
                &full_state.visible,
                MoveGenerationConfig {
                    allow_foundation_retreats: true,
                },
            );

            if macro_moves.is_empty() {
                return FullClosureRun::new(
                    &full_state.visible,
                    transcript,
                    undos,
                    ClosureStopReason::TerminalNoMoves,
                    revealed_any,
                );
            }

            if transcript.len() >= usize::from(self.config.max_corridor_steps) {
                return FullClosureRun::new(
                    &full_state.visible,
                    transcript,
                    undos,
                    ClosureStopReason::CorridorDepthLimit,
                    revealed_any,
                );
            }

            let candidate = match self.select_candidate(&full_state.visible, &macro_moves) {
                CandidateDecision::Apply(candidate) => candidate,
                CandidateDecision::Stop(stop_reason) => {
                    return FullClosureRun::new(
                        &full_state.visible,
                        transcript,
                        undos,
                        stop_reason,
                        revealed_any,
                    );
                }
            };

            let transition =
                match apply_atomic_move_full_state(full_state, candidate.macro_move.atomic) {
                    Ok(transition) => transition,
                    Err(_) => {
                        return FullClosureRun::new(
                            &full_state.visible,
                            transcript,
                            undos,
                            ClosureStopReason::NoSafeClosureMove,
                            revealed_any,
                        );
                    }
                };

            let revealed_step = transition.outcome.revealed.is_some();
            revealed_any |= revealed_step;
            undos.push(transition.undo);
            transcript.push(ClosureStep {
                move_id: candidate.macro_move.id,
                macro_move: candidate.macro_move,
                reason: candidate.reason,
                semantics: transition.outcome.semantics,
                outcome: transition.outcome,
            });

            if self.config.debug_validate_each_step && full_state.debug_validate().is_err() {
                return FullClosureRun::new(
                    &full_state.visible,
                    transcript,
                    undos,
                    ClosureStopReason::NoSafeClosureMove,
                    revealed_any,
                );
            }

            if revealed_step && self.config.stop_on_reveal {
                return FullClosureRun::new(
                    &full_state.visible,
                    transcript,
                    undos,
                    ClosureStopReason::RevealOccurred,
                    revealed_any,
                );
            }
        }
    }

    fn select_candidate(
        &self,
        state: &VisibleState,
        macro_moves: &[MacroMove],
    ) -> CandidateDecision {
        let king_placements = legal_king_placements(macro_moves);
        if king_placements.len() > 1 {
            return CandidateDecision::Stop(ClosureStopReason::EmptyColumnDecision);
        }
        if self.config.enable_single_king_placement_closure && king_placements.len() == 1 {
            return CandidateDecision::Apply(ClosureCandidate {
                macro_move: king_placements[0].clone(),
                reason: ClosureReason::SingleKingPlacement,
            });
        }

        if self.config.enable_forced_foundation_closure {
            let foundation_candidates = foundation_closure_candidates(state, macro_moves);
            if foundation_candidates.len() == 1 {
                return CandidateDecision::Apply(ClosureCandidate {
                    macro_move: foundation_candidates[0].clone(),
                    reason: ClosureReason::ForcedFoundationAdvance,
                });
            }
        }

        if self.config.enable_single_move_closure && macro_moves.len() == 1 {
            return CandidateDecision::Apply(ClosureCandidate {
                macro_move: macro_moves[0].clone(),
                reason: ClosureReason::SingleLegalMove,
            });
        }

        if self.config.stop_on_stock_pivot
            && macro_moves.len() > 1
            && macro_moves
                .iter()
                .any(|macro_move| macro_move.semantics.affects_stock_cycle)
        {
            return CandidateDecision::Stop(ClosureStopReason::StockPivot);
        }

        if is_meaningful_branching(macro_moves) {
            CandidateDecision::Stop(ClosureStopReason::MeaningfulBranching)
        } else {
            CandidateDecision::Stop(ClosureStopReason::NoSafeClosureMove)
        }
    }
}

impl Default for ClosureEngine {
    fn default() -> Self {
        Self::new(ClosureConfig::default())
    }
}

/// Structured reason for an automatic closure step.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ClosureReason {
    /// Reveal happened as a side effect of a closure move.
    ForcedAutoFlip,
    /// Exactly one legal macro move existed.
    SingleLegalMove,
    /// Conservative Ace/Two foundation advance.
    ForcedFoundationAdvance,
    /// Exactly one legal king-headed run could fill an empty column.
    SingleKingPlacement,
    /// Stock-only deterministic normalization.
    StockNormalization,
    /// Deterministic corridor continuation.
    DeterministicCorridorStep,
    /// No further safe closure rule applied.
    NoFurtherSafeClosure,
}

/// Structured reason why closure stopped.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ClosureStopReason {
    /// All foundations are complete.
    TerminalWin,
    /// No legal macro moves remain.
    TerminalNoMoves,
    /// Multiple meaningful legal choices remain.
    MeaningfulBranching,
    /// A hidden tableau card was revealed and the config stops on reveal.
    RevealOccurred,
    /// Multiple meaningful empty-column king placements remain.
    EmptyColumnDecision,
    /// Stock accessibility is a live decision amid other moves.
    StockPivot,
    /// The configured corridor step limit was reached.
    CorridorDepthLimit,
    /// Legal moves exist, but none satisfies a safe closure rule.
    NoSafeClosureMove,
}

/// One automatic closure step.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ClosureStep {
    /// Stable move id from the macro list at the step's source state.
    pub move_id: MoveId,
    /// Macro move selected by closure.
    pub macro_move: MacroMove,
    /// Why closure was allowed to apply this move.
    pub reason: ClosureReason,
    /// Semantic tags computed from the pre-step state.
    pub semantics: MoveSemantics,
    /// Transition outcome returned by the move engine.
    pub outcome: MoveOutcome,
}

/// Ordered transcript of automatic closure steps.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ClosureTranscript {
    /// Applied closure steps in order.
    pub steps: Vec<ClosureStep>,
}

impl ClosureTranscript {
    /// Appends one closure step.
    pub fn push(&mut self, step: ClosureStep) {
        self.steps.push(step);
    }

    /// Returns the number of recorded steps.
    pub fn len(&self) -> usize {
        self.steps.len()
    }

    /// Returns true if no steps have been recorded.
    pub fn is_empty(&self) -> bool {
        self.steps.is_empty()
    }

    /// Iterates closure steps in order.
    pub fn iter(&self) -> impl Iterator<Item = &ClosureStep> {
        self.steps.iter()
    }
}

impl fmt::Display for ClosureTranscript {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (index, step) in self.steps.iter().enumerate() {
            if index > 0 {
                f.write_str(" -> ")?;
            }
            write!(f, "#{index}:{:?}:{:?}", step.reason, step.macro_move.kind)?;
        }
        Ok(())
    }
}

/// Result of running deterministic closure.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ClosureResult {
    /// Snapshot of the state after closure stopped.
    pub final_state: VisibleState,
    /// Ordered transcript of automatic steps.
    pub transcript: ClosureTranscript,
    /// Reason the closure loop stopped.
    pub stop_reason: ClosureStopReason,
    /// Number of automatic steps applied.
    pub steps: usize,
    /// Whether any step revealed a hidden tableau card.
    pub revealed: bool,
    /// Whether the stop reason is terminal.
    pub terminal: bool,
    /// Whether branching remains at the frontier.
    pub branching_remaining: bool,
    /// Whether the frontier is an empty-column decision.
    pub empty_column_pivot: bool,
    /// Whether the frontier is a stock-cycle pivot.
    pub stock_pivot: bool,
}

impl ClosureResult {
    fn new(
        state: &VisibleState,
        transcript: ClosureTranscript,
        stop_reason: ClosureStopReason,
        revealed: bool,
    ) -> Self {
        let steps = transcript.len();
        Self {
            final_state: state.clone(),
            transcript,
            stop_reason,
            steps,
            revealed,
            terminal: matches!(
                stop_reason,
                ClosureStopReason::TerminalWin | ClosureStopReason::TerminalNoMoves
            ),
            branching_remaining: matches!(stop_reason, ClosureStopReason::MeaningfulBranching),
            empty_column_pivot: matches!(stop_reason, ClosureStopReason::EmptyColumnDecision),
            stock_pivot: matches!(stop_reason, ClosureStopReason::StockPivot),
        }
    }
}

/// Full-state closure result plus undo records for search recursion.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FullClosureRun {
    /// Diagnostic closure result over the visible state.
    pub result: ClosureResult,
    /// Full-state undo records, one per transcript step.
    pub undos: Vec<FullStateMoveUndo>,
}

impl FullClosureRun {
    fn new(
        state: &VisibleState,
        transcript: ClosureTranscript,
        undos: Vec<FullStateMoveUndo>,
        stop_reason: ClosureStopReason,
        revealed: bool,
    ) -> Self {
        Self {
            result: ClosureResult::new(state, transcript, stop_reason, revealed),
            undos,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ClosureCandidate {
    macro_move: MacroMove,
    reason: ClosureReason,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum CandidateDecision {
    Apply(ClosureCandidate),
    Stop(ClosureStopReason),
}

/// Returns true if the state has at least one empty tableau column.
pub fn has_empty_column(state: &VisibleState) -> bool {
    count_empty_columns(state) > 0
}

/// Counts empty tableau columns.
pub fn count_empty_columns(state: &VisibleState) -> usize {
    state
        .columns
        .iter()
        .filter(|column| column.is_empty())
        .count()
}

/// Returns legal king-headed placements from an already-generated macro list.
pub fn legal_king_placements(macro_moves: &[MacroMove]) -> Vec<MacroMove> {
    macro_moves
        .iter()
        .filter(|macro_move| matches!(macro_move.kind, MacroMoveKind::PlaceKingRun { .. }))
        .cloned()
        .collect()
}

/// Returns conservative foundation closure candidates.
///
/// This first closure rule only auto-advances Aces and Twos. Those ranks cannot
/// be needed as tableau support for unrevealed lower ranks in a way that would
/// justify speculative retention, so the rule is intentionally narrow and
/// easy to audit.
pub fn foundation_closure_candidates(
    state: &VisibleState,
    macro_moves: &[MacroMove],
) -> Vec<MacroMove> {
    macro_moves
        .iter()
        .filter(|macro_move| is_conservative_foundation_candidate(state, macro_move))
        .cloned()
        .collect()
}

/// Returns true when the remaining move list represents meaningful branching.
pub fn is_meaningful_branching(macro_moves: &[MacroMove]) -> bool {
    macro_moves.len() > 1
}

fn is_conservative_foundation_candidate(state: &VisibleState, macro_move: &MacroMove) -> bool {
    let card = match macro_move.atomic {
        AtomicMove::WasteToFoundation => state.stock.accessible_card(),
        AtomicMove::TableauToFoundation { src } => {
            state.columns[usize::from(src.index())].top_face_up()
        }
        _ => None,
    };

    matches!(card.map(|card| card.rank()), Some(Rank::Ace | Rank::Two))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        cards::Suit, core::TableauColumn, moves::AtomicMove, stock::CyclicStockState,
        types::ColumnId,
    };

    fn col(index: u8) -> ColumnId {
        ColumnId::new(index).unwrap()
    }

    fn card(text: &str) -> Card {
        text.parse().unwrap()
    }

    fn stock_from_cards(cards: &[&str]) -> CyclicStockState {
        CyclicStockState::from_parts(
            cards.iter().map(|text| card(text)).collect(),
            cards.len(),
            0,
            0,
            None,
            3,
        )
    }

    fn one_step_engine() -> ClosureEngine {
        ClosureEngine::new(ClosureConfig {
            max_corridor_steps: 1,
            ..ClosureConfig::default()
        })
    }

    #[test]
    fn already_won_state_stops_with_terminal_win() {
        let mut state = VisibleState::default();
        for suit in [Suit::Clubs, Suit::Diamonds, Suit::Hearts, Suit::Spades] {
            state.foundations.set_top_rank(suit, Some(Rank::King));
        }

        let result = ClosureEngine::default().run(&mut state);

        assert_eq!(result.stop_reason, ClosureStopReason::TerminalWin);
        assert!(result.terminal);
        assert!(result.transcript.is_empty());
    }

    #[test]
    fn no_legal_move_state_stops_with_terminal_no_moves() {
        let mut state = VisibleState::default();

        let result = ClosureEngine::default().run(&mut state);

        assert_eq!(result.stop_reason, ClosureStopReason::TerminalNoMoves);
        assert!(result.terminal);
        assert!(result.transcript.is_empty());
    }

    #[test]
    fn exactly_one_legal_macro_move_gets_auto_applied() {
        let mut state = VisibleState::default();
        state.stock = stock_from_cards(&["Ac", "2c", "3c"]);

        let result = one_step_engine().run(&mut state);

        assert_eq!(result.steps, 1);
        assert_eq!(
            result.transcript.steps[0].reason,
            ClosureReason::SingleLegalMove
        );
        assert_eq!(state.stock.accessible_card(), Some(card("3c")));
    }

    #[test]
    fn safe_foundation_auto_closure_fires_when_enabled() {
        let mut state = VisibleState::default();
        state.stock = CyclicStockState::from_parts(vec![card("Ac")], 0, 1, 0, None, 3);
        let engine = ClosureEngine::new(ClosureConfig {
            enable_single_move_closure: false,
            max_corridor_steps: 1,
            ..ClosureConfig::default()
        });

        let result = engine.run(&mut state);

        assert_eq!(result.steps, 1);
        assert_eq!(
            result.transcript.steps[0].reason,
            ClosureReason::ForcedFoundationAdvance
        );
        assert_eq!(state.foundations.top_rank(Suit::Clubs), Some(Rank::Ace));
    }

    #[test]
    fn aggressive_foundation_move_does_not_auto_close() {
        let mut state = VisibleState::default();
        state
            .foundations
            .set_top_rank(Suit::Spades, Some(Rank::Six));
        state.columns[0] = TableauColumn::new(0, vec![card("7s")]);
        let engine = ClosureEngine::new(ClosureConfig {
            enable_single_move_closure: false,
            max_corridor_steps: 1,
            ..ClosureConfig::default()
        });

        let result = engine.run(&mut state);

        assert_eq!(result.steps, 0);
        assert_eq!(result.stop_reason, ClosureStopReason::NoSafeClosureMove);
        assert_eq!(state.columns[0].top_face_up(), Some(card("7s")));
    }

    #[test]
    fn single_king_placement_into_empty_column_auto_closes() {
        let mut state = VisibleState::default();
        state.columns[0] = TableauColumn::new(0, vec![card("Kh")]);
        state.columns[2] = TableauColumn::new(0, vec![card("Ac")]);
        state.columns[3] = TableauColumn::new(0, vec![card("2d")]);
        state.columns[4] = TableauColumn::new(0, vec![card("3h")]);
        state.columns[5] = TableauColumn::new(0, vec![card("4s")]);
        state.columns[6] = TableauColumn::new(0, vec![card("5c")]);
        let engine = ClosureEngine::new(ClosureConfig {
            enable_single_move_closure: false,
            max_corridor_steps: 1,
            ..ClosureConfig::default()
        });

        let result = engine.run(&mut state);

        assert_eq!(result.steps, 1);
        assert_eq!(
            result.transcript.steps[0].reason,
            ClosureReason::SingleKingPlacement
        );
        assert!(state.columns[0].is_empty());
        assert_eq!(state.columns[1].top_face_up(), Some(card("Kh")));
    }

    #[test]
    fn multiple_king_placements_stop_with_empty_column_decision() {
        let mut state = VisibleState::default();
        state.columns[0] = TableauColumn::new(0, vec![card("Kh")]);
        state.columns[1] = TableauColumn::new(0, vec![card("Ks")]);

        let result = ClosureEngine::default().run(&mut state);

        assert_eq!(result.stop_reason, ClosureStopReason::EmptyColumnDecision);
        assert!(result.empty_column_pivot);
        assert!(result.transcript.is_empty());
    }

    #[test]
    fn reveal_causing_closure_step_records_reveal() {
        let mut state = VisibleState::default();
        state.columns[0] = TableauColumn::new(1, vec![card("7s")]);
        state.columns[1] = TableauColumn::new(0, vec![card("8h")]);

        let result = ClosureEngine::default().run_with_reveals(&mut state, [card("Ac")]);

        assert_eq!(result.steps, 1);
        assert!(result.revealed);
        assert_eq!(result.stop_reason, ClosureStopReason::RevealOccurred);
        assert_eq!(
            result.transcript.steps[0].outcome.revealed.unwrap().card,
            card("Ac")
        );
        assert_eq!(state.columns[0].face_up, vec![card("Ac")]);
    }

    #[test]
    fn closure_stops_on_reveal_when_configured() {
        let mut state = VisibleState::default();
        state.columns[0] = TableauColumn::new(1, vec![card("7s")]);
        state.columns[1] = TableauColumn::new(0, vec![card("8h")]);

        let result = ClosureEngine::default().run_with_reveals(&mut state, [card("Ac")]);

        assert_eq!(result.stop_reason, ClosureStopReason::RevealOccurred);
        assert_eq!(result.steps, 1);
    }

    #[test]
    fn corridor_depth_limit_is_respected() {
        let mut state = VisibleState::default();
        state.stock = stock_from_cards(&["Ac"]);

        let result = one_step_engine().run(&mut state);

        assert_eq!(result.steps, 1);
        assert_eq!(result.stop_reason, ClosureStopReason::CorridorDepthLimit);
    }

    #[test]
    fn transcript_records_applied_steps_in_order() {
        let mut state = VisibleState::default();
        state.stock = stock_from_cards(&["Ac"]);
        let engine = ClosureEngine::new(ClosureConfig {
            max_corridor_steps: 2,
            ..ClosureConfig::default()
        });

        let result = engine.run(&mut state);

        assert_eq!(result.transcript.len(), 2);
        assert!(matches!(
            result.transcript.steps[0].macro_move.kind,
            MacroMoveKind::AdvanceStock
        ));
        assert!(matches!(
            result.transcript.steps[1].macro_move.kind,
            MacroMoveKind::PlayWasteToFoundation
        ));
        assert_eq!(state.foundations.top_rank(Suit::Clubs), Some(Rank::Ace));
    }

    #[test]
    fn apply_then_closure_remains_invariant_safe() {
        let mut state = VisibleState::default();
        state.columns[0] = TableauColumn::new(0, vec![card("Kh")]);
        state.columns[2] = TableauColumn::new(0, vec![card("Ac")]);
        state.columns[3] = TableauColumn::new(0, vec![card("2d")]);
        state.columns[4] = TableauColumn::new(0, vec![card("3h")]);
        state.columns[5] = TableauColumn::new(0, vec![card("4s")]);
        state.columns[6] = TableauColumn::new(0, vec![card("5c")]);

        let result = one_step_engine().run(&mut state);

        assert_eq!(result.steps, 1);
        state.validate_consistency().unwrap();
        result.final_state.validate_consistency().unwrap();
    }

    #[test]
    fn helper_classification_counts_empty_columns_and_king_moves() {
        let mut state = VisibleState::default();
        state.columns[0] = TableauColumn::new(0, vec![card("Kh")]);
        state.columns[1] = TableauColumn::new(0, vec![card("Qh")]);

        let moves = generate_legal_macro_moves_with_config(
            &state,
            MoveGenerationConfig {
                allow_foundation_retreats: true,
            },
        );

        assert!(has_empty_column(&state));
        assert_eq!(count_empty_columns(&state), 5);
        assert_eq!(legal_king_placements(&moves).len(), 5);
        assert!(moves.iter().any(|macro_move| {
            macro_move.atomic
                == AtomicMove::TableauToTableau {
                    src: col(0),
                    dest: col(2),
                    run_start: 0,
                }
        }));
    }
}
