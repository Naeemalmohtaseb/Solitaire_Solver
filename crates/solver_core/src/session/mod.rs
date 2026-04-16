//! Versioned session persistence and deterministic replay.
//!
//! Sessions are meant for paused real games and autoplay/debug runs. They store
//! the current public belief, optional true full state, and a replayable
//! move/reveal log. Replay always reuses the existing move and belief transition
//! engines; this module does not duplicate solver or move logic.

use std::{
    fs,
    path::Path,
    time::{SystemTime, UNIX_EPOCH},
};

use serde::{Deserialize, Serialize};

use crate::{
    belief::{
        apply_observed_belief_move, belief_from_full_state, validate_belief_against_full_state,
    },
    cards::Card,
    core::{BeliefState, FullState, VisibleState},
    error::{SolverError, SolverResult},
    experiments::{AutoplayPlannerSnapshot, AutoplayResult, PlannerBackend},
    moves::{apply_atomic_move_full_state, MacroMove, RevealRecord},
    planner::PlannerContinuation,
    types::{MoveId, SessionId},
};

/// Current session JSON schema version.
pub const SESSION_SCHEMA_VERSION: &str = "session-json-v1";

/// Metadata saved with a real or simulated game session.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SessionMetadata {
    /// Session identifier.
    pub id: SessionId,
    /// Persisted session schema version.
    pub schema_version: String,
    /// Solver crate version that wrote the session.
    pub engine_version: String,
    /// Creation timestamp in Unix seconds.
    pub created_unix_secs: u64,
    /// Human-readable label.
    pub label: Option<String>,
    /// Preset name used to drive the session, if known.
    pub preset_name: Option<String>,
    /// Backend label used to drive the session, if known.
    pub backend: Option<PlannerBackend>,
}

impl SessionMetadata {
    /// Creates metadata with an explicit id.
    pub fn new(id: SessionId, label: Option<String>) -> Self {
        Self {
            id,
            schema_version: SESSION_SCHEMA_VERSION.to_string(),
            engine_version: crate::VERSION.to_string(),
            created_unix_secs: current_unix_secs(),
            label,
            preset_name: None,
            backend: None,
        }
    }

    /// Creates metadata with a generated id.
    pub fn generated(label: Option<String>) -> Self {
        Self::new(generated_session_id(), label)
    }

    /// Sets preset/backend provenance.
    pub fn with_solver_provenance(
        mut self,
        preset_name: Option<String>,
        backend: Option<PlannerBackend>,
    ) -> Self {
        self.preset_name = preset_name;
        self.backend = backend;
        self
    }
}

/// Persisted state snapshot.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SessionSnapshot {
    /// Explicit visible state for user-facing persistence.
    pub visible: VisibleState,
    /// Current exact belief state.
    pub belief: BeliefState,
    /// Optional true full state for autoplay/debug sessions.
    pub full_state: Option<FullState>,
}

impl SessionSnapshot {
    /// Builds a public-only snapshot.
    pub fn from_belief(belief: BeliefState) -> SolverResult<Self> {
        belief.validate_consistency_against_visible()?;
        Ok(Self {
            visible: belief.visible.clone(),
            belief,
            full_state: None,
        })
    }

    /// Builds a full autoplay/debug snapshot.
    pub fn from_full_state(full_state: FullState) -> SolverResult<Self> {
        let belief = belief_from_full_state(&full_state)?;
        Ok(Self {
            visible: full_state.visible.clone(),
            belief,
            full_state: Some(full_state),
        })
    }

    /// Validates that duplicated persisted fields agree.
    pub fn validate_consistency(&self) -> SolverResult<()> {
        self.visible.validate_consistency()?;
        self.belief.validate_consistency_against_visible()?;
        if self.visible != self.belief.visible {
            return Err(SolverError::InvalidState(
                "session snapshot visible state differs from belief visible state".to_string(),
            ));
        }
        if let Some(full_state) = &self.full_state {
            validate_belief_against_full_state(&self.belief, full_state)?;
            if self.visible != full_state.visible {
                return Err(SolverError::InvalidState(
                    "session snapshot visible state differs from full-state visible state"
                        .to_string(),
                ));
            }
        }
        Ok(())
    }
}

/// Planner diagnostics persisted with a chosen move.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SessionPlannerSnapshot {
    /// Backend label used for the decision.
    pub backend: PlannerBackend,
    /// Best root value reported by the backend.
    pub best_value: f64,
    /// Planner elapsed time for this step in milliseconds.
    pub elapsed_ms: u64,
    /// Deterministic solver nodes reported by this step.
    pub deterministic_nodes: u64,
    /// Root visits/samples/simulations reported by the backend.
    pub root_visits: u64,
    /// Whether late-exact triggered during the decision.
    pub late_exact_triggered: bool,
}

impl From<AutoplayPlannerSnapshot> for SessionPlannerSnapshot {
    fn from(value: AutoplayPlannerSnapshot) -> Self {
        Self {
            backend: value.backend,
            best_value: value.best_value,
            elapsed_ms: value.elapsed_ms,
            deterministic_nodes: value.deterministic_nodes,
            root_visits: value.root_visits,
            late_exact_triggered: value.late_exact_triggered,
        }
    }
}

/// Persisted chosen move record.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SessionMoveRecord {
    /// Zero-based step index.
    pub step_index: usize,
    /// Stable move id assigned at the source node.
    pub move_id: MoveId,
    /// Full macro move, including its atomic action and semantic tags.
    pub macro_move: MacroMove,
    /// Optional planner diagnostics.
    pub planner: Option<SessionPlannerSnapshot>,
}

/// Persisted reveal observation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SessionRevealRecord {
    /// Step index where the reveal occurred.
    pub step_index: usize,
    /// Revealed card and column.
    pub reveal: RevealRecord,
}

/// One replayable session step.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SessionStep {
    /// Zero-based step index.
    pub step_index: usize,
    /// Chosen move.
    pub move_record: SessionMoveRecord,
    /// Observed reveal, if the move uncovered a hidden tableau card.
    pub reveal_record: Option<SessionRevealRecord>,
}

/// Persisted game session.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SessionRecord {
    /// Version/provenance metadata.
    pub metadata: SessionMetadata,
    /// Initial session snapshot.
    pub initial_snapshot: SessionSnapshot,
    /// Current session snapshot after all recorded steps.
    pub current_snapshot: SessionSnapshot,
    /// Replayable move history.
    pub steps: Vec<SessionStep>,
    /// Reveal history extracted from `steps` for quick inspection.
    pub reveal_history: Vec<SessionRevealRecord>,
    /// Optional lightweight root-recommendation cache for planner continuation.
    #[serde(default)]
    pub planner_continuation: Option<PlannerContinuation>,
}

impl SessionRecord {
    /// Creates a session from a public belief state.
    pub fn from_belief(
        metadata: SessionMetadata,
        belief: BeliefState,
    ) -> SolverResult<SessionRecord> {
        let snapshot = SessionSnapshot::from_belief(belief)?;
        let record = Self {
            metadata,
            initial_snapshot: snapshot.clone(),
            current_snapshot: snapshot,
            steps: Vec::new(),
            reveal_history: Vec::new(),
            planner_continuation: None,
        };
        record.validate_structure()?;
        Ok(record)
    }

    /// Creates a session from a true full state for autoplay/debug use.
    pub fn from_full_state(
        metadata: SessionMetadata,
        full_state: FullState,
    ) -> SolverResult<SessionRecord> {
        let snapshot = SessionSnapshot::from_full_state(full_state)?;
        let record = Self {
            metadata,
            initial_snapshot: snapshot.clone(),
            current_snapshot: snapshot,
            steps: Vec::new(),
            reveal_history: Vec::new(),
            planner_continuation: None,
        };
        record.validate_structure()?;
        Ok(record)
    }

    /// Reconstructs a replayable session from an autoplay result.
    pub fn from_autoplay_result(
        metadata: SessionMetadata,
        initial_full_state: FullState,
        result: &AutoplayResult,
    ) -> SolverResult<SessionRecord> {
        let mut record = Self::from_full_state(metadata, initial_full_state)?;
        for step in &result.trace.steps {
            let persisted_step = record.append_observed_move(
                step.chosen_move.clone(),
                step.revealed_card,
                Some(step.planner.clone().into()),
            )?;
            if persisted_step
                .reveal_record
                .as_ref()
                .map(|reveal| reveal.reveal.card)
                != step.revealed_card
            {
                return Err(SolverError::InvalidState(format!(
                    "autoplay reveal mismatch at step {}",
                    step.step_index
                )));
            }
        }
        if record.current_snapshot.belief != result.final_belief
            || record.current_snapshot.full_state.as_ref() != Some(&result.final_full_state)
        {
            return Err(SolverError::InvalidState(
                "autoplay session reconstruction did not match final autoplay state".to_string(),
            ));
        }
        Ok(record)
    }

    /// Returns the current snapshot.
    pub fn snapshot(&self) -> &SessionSnapshot {
        &self.current_snapshot
    }

    /// Returns the optional planner continuation metadata.
    pub fn planner_continuation(&self) -> Option<&PlannerContinuation> {
        self.planner_continuation.as_ref()
    }

    /// Stores lightweight planner continuation metadata for the next turn.
    pub fn set_planner_continuation(&mut self, continuation: PlannerContinuation) {
        self.planner_continuation = Some(continuation);
    }

    /// Clears any stored planner continuation metadata.
    pub fn clear_planner_continuation(&mut self) {
        self.planner_continuation = None;
    }

    /// Appends a chosen move with no explicit reveal observation.
    pub fn append_chosen_move(
        &mut self,
        macro_move: MacroMove,
        planner: Option<SessionPlannerSnapshot>,
    ) -> SolverResult<SessionStep> {
        self.append_observed_move(macro_move, None, planner)
    }

    /// Applies and records one observed move.
    pub fn append_observed_move(
        &mut self,
        macro_move: MacroMove,
        observed_reveal: Option<Card>,
        planner: Option<SessionPlannerSnapshot>,
    ) -> SolverResult<SessionStep> {
        self.current_snapshot.validate_consistency()?;

        let step_index = self.steps.len();
        let mut next_full = self.current_snapshot.full_state.clone();
        let replay_reveal = if let Some(full_state) = &mut next_full {
            let transition = apply_atomic_move_full_state(full_state, macro_move.atomic)?;
            let true_reveal = transition.outcome.revealed;
            if observed_reveal.is_some_and(|card| Some(card) != true_reveal.map(|r| r.card)) {
                return Err(SolverError::InvalidState(format!(
                    "observed reveal does not match true full-state reveal at step {step_index}"
                )));
            }
            true_reveal
        } else {
            match observed_reveal {
                Some(card) => {
                    let slot = crate::moves::next_hidden_slot_to_reveal(
                        &self.current_snapshot.belief.visible,
                        macro_move.atomic,
                    )
                    .ok_or(SolverError::UnexpectedRevealCard { card })?;
                    Some(RevealRecord {
                        column: slot.column,
                        card,
                    })
                }
                None => None,
            }
        };

        let (next_belief, move_outcome) = apply_observed_belief_move(
            &self.current_snapshot.belief,
            macro_move.atomic,
            replay_reveal.map(|reveal| reveal.card),
        )?;
        if move_outcome.revealed != replay_reveal {
            return Err(SolverError::InvalidState(format!(
                "belief transition reveal mismatch at step {step_index}"
            )));
        }

        let reveal_record = replay_reveal.map(|reveal| SessionRevealRecord { step_index, reveal });
        let move_record = SessionMoveRecord {
            step_index,
            move_id: macro_move.id,
            macro_move,
            planner,
        };
        let step = SessionStep {
            step_index,
            move_record,
            reveal_record: reveal_record.clone(),
        };

        self.current_snapshot = SessionSnapshot {
            visible: next_belief.visible.clone(),
            belief: next_belief,
            full_state: next_full,
        };
        self.current_snapshot.validate_consistency()?;
        if let Some(reveal) = reveal_record {
            self.reveal_history.push(reveal);
        }
        self.steps.push(step.clone());
        self.validate_structure()?;
        Ok(step)
    }

    /// Returns a compact summary.
    pub fn summary(&self) -> SessionSummary {
        SessionSummary {
            id: self.metadata.id,
            schema_version: self.metadata.schema_version.clone(),
            label: self.metadata.label.clone(),
            preset_name: self.metadata.preset_name.clone(),
            backend: self.metadata.backend,
            steps: self.steps.len(),
            reveals: self.reveal_history.len(),
            hidden_cards_remaining: self.current_snapshot.belief.hidden_card_count(),
            has_full_state: self.current_snapshot.full_state.is_some(),
            structural_win: self.current_snapshot.visible.is_structural_win(),
            has_planner_continuation: self.planner_continuation.is_some(),
        }
    }

    /// Validates persisted structure without replaying every move.
    pub fn validate_structure(&self) -> SolverResult<()> {
        if self.metadata.schema_version != SESSION_SCHEMA_VERSION {
            return Err(SolverError::InvalidState(format!(
                "unsupported session schema version {}",
                self.metadata.schema_version
            )));
        }
        self.initial_snapshot.validate_consistency()?;
        self.current_snapshot.validate_consistency()?;

        for (expected_index, step) in self.steps.iter().enumerate() {
            if step.step_index != expected_index || step.move_record.step_index != expected_index {
                return Err(SolverError::InvalidState(format!(
                    "session step index mismatch at position {expected_index}"
                )));
            }
            if step.move_record.move_id != step.move_record.macro_move.id {
                return Err(SolverError::InvalidState(format!(
                    "session move id mismatch at step {expected_index}"
                )));
            }
            if let Some(reveal) = &step.reveal_record {
                if reveal.step_index != expected_index {
                    return Err(SolverError::InvalidState(format!(
                        "session reveal index mismatch at step {expected_index}"
                    )));
                }
            }
        }

        let step_reveals = self
            .steps
            .iter()
            .filter_map(|step| step.reveal_record.clone())
            .collect::<Vec<_>>();
        if step_reveals != self.reveal_history {
            return Err(SolverError::InvalidState(
                "session reveal history does not match step reveal records".to_string(),
            ));
        }

        Ok(())
    }
}

/// Compact session summary for CLI inspection.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SessionSummary {
    /// Session id.
    pub id: SessionId,
    /// Schema version.
    pub schema_version: String,
    /// Human-readable label.
    pub label: Option<String>,
    /// Preset name, if known.
    pub preset_name: Option<String>,
    /// Backend, if known.
    pub backend: Option<PlannerBackend>,
    /// Number of moves recorded.
    pub steps: usize,
    /// Number of reveal observations recorded.
    pub reveals: usize,
    /// Number of hidden cards remaining.
    pub hidden_cards_remaining: usize,
    /// Whether true full state is persisted.
    pub has_full_state: bool,
    /// Whether the current visible state is structurally won.
    pub structural_win: bool,
    /// Whether lightweight planner continuation metadata is persisted.
    pub has_planner_continuation: bool,
}

/// Replay mismatch diagnostic.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReplayMismatch {
    /// Step index associated with the mismatch, if any.
    pub step_index: Option<usize>,
    /// Human-readable mismatch detail.
    pub message: String,
}

/// Replay result for a saved session.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReplayResult {
    /// Final reconstructed snapshot.
    pub final_snapshot: SessionSnapshot,
    /// Number of steps successfully replayed.
    pub replayed_steps: usize,
    /// Mismatch diagnostics. Empty means replay matched the saved current state.
    pub mismatches: Vec<ReplayMismatch>,
    /// True when no mismatches were found.
    pub matched: bool,
}

/// Replays a saved session from its initial snapshot.
pub fn replay_session(record: &SessionRecord) -> SolverResult<ReplayResult> {
    record.validate_structure()?;

    let mut belief = record.initial_snapshot.belief.clone();
    let mut full_state = record.initial_snapshot.full_state.clone();
    let mut replayed_steps = 0usize;
    let mut mismatches = Vec::new();
    let mut replayed_reveals = Vec::<SessionRevealRecord>::new();

    for step in &record.steps {
        let atomic = step.move_record.macro_move.atomic;
        let reveal_from_full = if let Some(full_state) = &mut full_state {
            match apply_atomic_move_full_state(full_state, atomic) {
                Ok(transition) => transition.outcome.revealed,
                Err(error) => {
                    mismatches.push(ReplayMismatch {
                        step_index: Some(step.step_index),
                        message: format!("full-state move apply failed: {error}"),
                    });
                    break;
                }
            }
        } else {
            step.reveal_record.as_ref().map(|record| record.reveal)
        };

        if step.reveal_record.as_ref().map(|record| record.reveal) != reveal_from_full {
            mismatches.push(ReplayMismatch {
                step_index: Some(step.step_index),
                message: "recorded reveal does not match replayed reveal".to_string(),
            });
        }

        match apply_observed_belief_move(
            &belief,
            atomic,
            reveal_from_full.map(|reveal| reveal.card),
        ) {
            Ok((next_belief, outcome)) => {
                if outcome.revealed != reveal_from_full {
                    mismatches.push(ReplayMismatch {
                        step_index: Some(step.step_index),
                        message: "belief replay reveal does not match recorded reveal".to_string(),
                    });
                }
                if let Some(reveal) = outcome.revealed {
                    replayed_reveals.push(SessionRevealRecord {
                        step_index: step.step_index,
                        reveal,
                    });
                }
                belief = next_belief;
                replayed_steps += 1;
            }
            Err(error) => {
                mismatches.push(ReplayMismatch {
                    step_index: Some(step.step_index),
                    message: format!("belief move apply failed: {error}"),
                });
                break;
            }
        }

        if let Some(full_state) = &full_state {
            if let Err(error) = validate_belief_against_full_state(&belief, full_state) {
                mismatches.push(ReplayMismatch {
                    step_index: Some(step.step_index),
                    message: format!("belief/full-state mismatch after replay: {error}"),
                });
                break;
            }
        }
    }

    if replayed_reveals != record.reveal_history {
        mismatches.push(ReplayMismatch {
            step_index: None,
            message: "replayed reveal history differs from saved reveal history".to_string(),
        });
    }

    let final_snapshot = SessionSnapshot {
        visible: belief.visible.clone(),
        belief,
        full_state,
    };

    compare_snapshot(
        "current",
        &record.current_snapshot,
        &final_snapshot,
        &mut mismatches,
    );

    Ok(ReplayResult {
        final_snapshot,
        replayed_steps,
        matched: mismatches.is_empty(),
        mismatches,
    })
}

/// Saves a session as deterministic pretty JSON.
pub fn save_session(path: impl AsRef<Path>, record: &SessionRecord) -> SolverResult<()> {
    record.validate_structure()?;
    let path = path.as_ref();
    if let Some(parent) = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        fs::create_dir_all(parent)?;
    }
    let json = serde_json::to_string_pretty(record)
        .map_err(|error| SolverError::Serialization(error.to_string()))?;
    fs::write(path, json)?;
    Ok(())
}

/// Loads a session from JSON.
pub fn load_session(path: impl AsRef<Path>) -> SolverResult<SessionRecord> {
    let contents = fs::read_to_string(path)?;
    let record: SessionRecord = serde_json::from_str(&contents)
        .map_err(|error| SolverError::Serialization(error.to_string()))?;
    record.validate_structure()?;
    Ok(record)
}

/// Compatibility alias for saving the current game session.
pub fn save_current_game_session(
    path: impl AsRef<Path>,
    record: &SessionRecord,
) -> SolverResult<()> {
    save_session(path, record)
}

/// Compatibility alias for loading the current game session.
pub fn load_current_game_session(path: impl AsRef<Path>) -> SolverResult<SessionRecord> {
    load_session(path)
}

/// Legacy saved-session name retained as a compatibility alias.
pub type SavedSession = SessionRecord;

/// Reuse hint for carrying search work across real user moves.
///
/// Detailed root recommendation reuse is stored in `PlannerContinuation`. This
/// compact hint remains as a caller-facing preference for direct subtree
/// alignment and fallback lookup behavior.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SubtreeReuseHint {
    /// Root move that may align with the next reusable subtree.
    pub root_move: Option<MoveId>,
    /// Whether a transposition lookup should be attempted if direct reuse fails.
    pub allow_transposition_lookup: bool,
}

fn compare_snapshot(
    label: &str,
    expected: &SessionSnapshot,
    actual: &SessionSnapshot,
    mismatches: &mut Vec<ReplayMismatch>,
) {
    if expected.visible != actual.visible {
        mismatches.push(ReplayMismatch {
            step_index: None,
            message: format!("{label} visible state differs from replayed state"),
        });
    }
    if expected.belief != actual.belief {
        mismatches.push(ReplayMismatch {
            step_index: None,
            message: format!("{label} belief state differs from replayed state"),
        });
    }
    if expected.full_state != actual.full_state {
        mismatches.push(ReplayMismatch {
            step_index: None,
            message: format!("{label} full state differs from replayed state"),
        });
    }
}

fn generated_session_id() -> SessionId {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_nanos())
        .unwrap_or(0);
    SessionId(nanos)
}

fn current_unix_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        cards::{Card, Rank, Suit},
        config::SolverConfig,
        core::{FoundationState, HiddenAssignments, TableauColumn, VisibleState},
        experiments::{play_game_with_planner, PimcConfig},
        moves::{generate_legal_macro_moves, MacroMoveKind},
        planner::{recommend_move_belief_uct_with_reuse, BeliefPlannerConfig, PlannerReuseContext},
        stock::CyclicStockState,
        types::ColumnId,
    };

    fn fixed_metadata() -> SessionMetadata {
        SessionMetadata {
            id: SessionId(7),
            schema_version: SESSION_SCHEMA_VERSION.to_string(),
            engine_version: "test".to_string(),
            created_unix_secs: 123,
            label: Some("unit".to_string()),
            preset_name: Some("test-preset".to_string()),
            backend: Some(PlannerBackend::BeliefUctLateExact),
        }
    }

    fn complete_foundations() -> FoundationState {
        let mut foundations = FoundationState::default();
        for suit in [Suit::Clubs, Suit::Diamonds, Suit::Hearts, Suit::Spades] {
            foundations.set_top_rank(suit, Some(Rank::King));
        }
        foundations
    }

    fn one_move_to_win_full_state() -> FullState {
        let mut visible = VisibleState::default();
        visible.foundations = complete_foundations();
        visible
            .foundations
            .set_top_rank(Suit::Spades, Some(Rank::Queen));
        visible.columns[0] = TableauColumn::new(0, vec!["Ks".parse().unwrap()]);
        FullState::new(visible, HiddenAssignments::empty())
    }

    fn forced_reveal_full_state() -> FullState {
        let mut visible = VisibleState::default();
        visible.foundations = complete_foundations();
        visible
            .foundations
            .set_top_rank(Suit::Spades, Some(Rank::Jack));
        visible.columns[0] = TableauColumn::new(1, vec!["Qs".parse().unwrap()]);
        FullState::new(
            visible,
            crate::core::HiddenAssignments::new(vec![crate::core::HiddenAssignment::new(
                crate::core::HiddenSlot::new(ColumnId::new(0).unwrap(), 0),
                "Ks".parse().unwrap(),
            )]),
        )
    }

    fn no_move_full_state() -> FullState {
        let cards = (0..Card::COUNT)
            .map(|index| Card::new(index as u8).unwrap())
            .collect();
        let mut visible = VisibleState::default();
        visible.stock = CyclicStockState::from_parts(cards, 0, 0, 0, Some(0), 3);
        FullState::new(visible, HiddenAssignments::empty())
    }

    fn first_macro(full_state: &FullState) -> MacroMove {
        generate_legal_macro_moves(&full_state.visible)
            .into_iter()
            .find(|macro_move| matches!(macro_move.kind, MacroMoveKind::MoveTopToFoundation { .. }))
            .unwrap()
    }

    #[test]
    fn save_load_round_trip_preserves_session() {
        let mut session =
            SessionRecord::from_full_state(fixed_metadata(), one_move_to_win_full_state()).unwrap();
        let macro_move = first_macro(session.current_snapshot.full_state.as_ref().unwrap());
        session.append_chosen_move(macro_move, None).unwrap();
        let path = std::env::temp_dir().join(format!(
            "solitaire-session-{}-{}.json",
            std::process::id(),
            crate::VERSION
        ));
        let _ = std::fs::remove_file(&path);

        save_session(&path, &session).unwrap();
        let loaded = load_session(&path).unwrap();

        assert_eq!(loaded, session);
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn replay_reconstructs_final_state() {
        let mut session =
            SessionRecord::from_full_state(fixed_metadata(), one_move_to_win_full_state()).unwrap();
        let macro_move = first_macro(session.current_snapshot.full_state.as_ref().unwrap());
        session.append_chosen_move(macro_move, None).unwrap();

        let replay = replay_session(&session).unwrap();

        assert!(replay.matched);
        assert_eq!(replay.replayed_steps, 1);
        assert_eq!(replay.final_snapshot, session.current_snapshot);
    }

    #[test]
    fn move_and_reveal_histories_serialize_deterministically() {
        let mut session =
            SessionRecord::from_full_state(fixed_metadata(), forced_reveal_full_state()).unwrap();
        let macro_move = first_macro(session.current_snapshot.full_state.as_ref().unwrap());
        session.append_chosen_move(macro_move, None).unwrap();

        let first = serde_json::to_string_pretty(&session).unwrap();
        let second = serde_json::to_string_pretty(&session).unwrap();

        assert_eq!(first, second);
        assert_eq!(session.reveal_history.len(), 1);
        assert!(first.contains("\"reveal_history\""));
    }

    #[test]
    fn session_metadata_survives_round_trip() {
        let session =
            SessionRecord::from_full_state(fixed_metadata(), one_move_to_win_full_state()).unwrap();
        let json = serde_json::to_string(&session).unwrap();
        let loaded: SessionRecord = serde_json::from_str(&json).unwrap();

        assert_eq!(loaded.metadata, session.metadata);
        assert_eq!(
            loaded.summary().preset_name,
            Some("test-preset".to_string())
        );
    }

    #[test]
    fn planner_continuation_survives_session_round_trip() {
        let mut session =
            SessionRecord::from_full_state(fixed_metadata(), one_move_to_win_full_state()).unwrap();
        let mut solver = SolverConfig::default();
        solver.late_exact.enabled = false;
        solver.deterministic.enable_tt = false;
        let planner = BeliefPlannerConfig {
            simulation_budget: 4,
            leaf_world_samples: 1,
            enable_early_stop: false,
            initial_screen_simulations: 0,
            max_active_root_actions: None,
            enable_second_reveal_refinement: false,
            ..BeliefPlannerConfig::default()
        };
        let result = recommend_move_belief_uct_with_reuse(
            &session.current_snapshot.belief,
            &solver,
            &planner,
            None,
            PlannerReuseContext {
                session_id: Some(session.metadata.id),
                backend_tag: Some("belief_uct".to_string()),
                preset_name: Some("unit".to_string()),
                ..PlannerReuseContext::default()
            },
        )
        .unwrap();

        session.set_planner_continuation(result.continuation.clone());
        assert!(session.summary().has_planner_continuation);

        let json = serde_json::to_string_pretty(&session).unwrap();
        let loaded: SessionRecord = serde_json::from_str(&json).unwrap();

        let loaded_continuation = loaded.planner_continuation().unwrap();
        assert_eq!(
            loaded_continuation.current_root_key,
            result.continuation.current_root_key
        );
        assert_eq!(
            loaded_continuation.config_fingerprint,
            result.continuation.config_fingerprint
        );
        assert_eq!(
            loaded_continuation.root_cache.candidate_actions,
            result.continuation.root_cache.candidate_actions
        );
        assert_eq!(
            loaded_continuation.root_cache.recommendation.best_move,
            result.continuation.root_cache.recommendation.best_move
        );
        assert!(loaded.summary().has_planner_continuation);
    }

    #[test]
    fn autoplay_produced_session_can_be_replayed() {
        let full = one_move_to_win_full_state();
        let autoplay = crate::experiments::AutoplayConfig {
            backend: PlannerBackend::BeliefUctLateExact,
            pimc: PimcConfig::default(),
            max_steps: 2,
            max_total_planner_time_ms: None,
            validate_each_step: true,
        };
        let result =
            play_game_with_planner(&full, &crate::config::SolverConfig::default(), &autoplay)
                .unwrap();
        let session = SessionRecord::from_autoplay_result(fixed_metadata(), full, &result).unwrap();

        let replay = replay_session(&session).unwrap();

        assert!(replay.matched);
        assert_eq!(replay.final_snapshot, session.current_snapshot);
    }

    #[test]
    fn malformed_session_file_is_rejected() {
        let path = std::env::temp_dir().join(format!(
            "solitaire-bad-session-{}-{}.json",
            std::process::id(),
            crate::VERSION
        ));
        std::fs::write(&path, "{ definitely not json").unwrap();

        assert!(matches!(
            load_session(&path),
            Err(SolverError::Serialization(_))
        ));
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn replay_reports_mismatch_for_tampered_current_state() {
        let mut session =
            SessionRecord::from_full_state(fixed_metadata(), no_move_full_state()).unwrap();
        session.current_snapshot =
            SessionSnapshot::from_full_state(one_move_to_win_full_state()).unwrap();

        let replay = replay_session(&session).unwrap();

        assert!(!replay.matched);
        assert!(!replay.mismatches.is_empty());
    }
}
