//! Late-game exact hidden-assignment evaluation.
//!
//! This subsystem is a small-hidden-count regime switch. When the belief state
//! has few face-down tableau cards, it enumerates the exact uniform posterior
//! over hidden tableau assignments instead of sampling worlds. It deliberately
//! does not introduce weighted posteriors, neural guidance, tree reuse, or a
//! belief-state transposition table.

use std::time::Instant;

use serde::{Deserialize, Serialize};

use crate::{
    cards::Card,
    core::{BeliefState, FullState, HiddenAssignment, HiddenAssignments, HiddenSlot},
    deterministic_solver::{DeterministicSearchConfig, DeterministicSolver, SolveOutcome},
    error::{SolverError, SolverResult},
    moves::{apply_atomic_move_full_state, MacroMove},
};

/// Deterministic continuation mode used after a root action is applied.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum LateExactEvaluationMode {
    /// Use proof-oriented deterministic solve attempts.
    Exact,
    /// Use bounded deterministic solve mode.
    Bounded,
    /// Use fast deterministic evaluation.
    Fast,
}

/// Configuration for late-game exact hidden-assignment evaluation.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LateExactConfig {
    /// Enables the exact-assignment regime switch.
    pub enabled: bool,
    /// Maximum number of hidden tableau cards for late-exact mode.
    pub hidden_card_threshold: u8,
    /// Maximum top root actions to evaluate exactly.
    pub max_root_actions: usize,
    /// Optional cap on enumerated hidden assignments.
    pub assignment_budget: Option<u64>,
    /// Deterministic continuation mode used per assignment.
    pub evaluation_mode: LateExactEvaluationMode,
}

impl Default for LateExactConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            hidden_card_threshold: 8,
            max_root_actions: 2,
            assignment_budget: None,
            evaluation_mode: LateExactEvaluationMode::Bounded,
        }
    }
}

/// Prefix state used by recursive deterministic assignment enumeration.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AssignmentPrefix {
    /// Hidden slots in deterministic column/depth order.
    pub slots: Vec<HiddenSlot>,
    /// Assignments fixed so far.
    pub assigned: Vec<HiddenAssignment>,
    /// Cards not yet assigned, kept in deterministic order.
    pub remaining_cards: Vec<Card>,
}

impl AssignmentPrefix {
    /// Creates a root prefix from a belief state.
    pub fn from_belief(belief: &BeliefState) -> SolverResult<Self> {
        belief.validate_consistency_against_visible()?;
        Ok(Self {
            slots: belief.visible.hidden_slots(),
            assigned: Vec::new(),
            remaining_cards: belief.unseen_cards.to_sorted_vec(),
        })
    }

    /// Returns the next slot that will be assigned, if any.
    pub fn next_slot(&self) -> Option<HiddenSlot> {
        self.slots.get(self.assigned.len()).copied()
    }
}

/// Aggregate exact value for one root action.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LateExactActionStats {
    /// Root action evaluated.
    pub action: MacroMove,
    /// Number of hidden assignments evaluated for this action.
    pub assignments_evaluated: u64,
    /// Assignments pruned by conservative hooks. Currently always zero in v1.
    pub assignments_pruned: u64,
    /// Expected value over evaluated assignments.
    pub mean_value: f64,
    /// Running second central moment.
    pub m2: f64,
    /// Sample variance when at least two assignments exist.
    pub variance: f64,
    /// Proven wins returned by the deterministic continuation solver.
    pub exact_wins: u64,
    /// Proven losses returned by the deterministic continuation solver.
    pub exact_losses: u64,
    /// Deterministic solver nodes consumed by this action.
    pub deterministic_nodes: u64,
}

impl LateExactActionStats {
    /// Creates empty stats for a root action.
    pub fn new(action: MacroMove) -> Self {
        Self {
            action,
            assignments_evaluated: 0,
            assignments_pruned: 0,
            mean_value: 0.0,
            m2: 0.0,
            variance: 0.0,
            exact_wins: 0,
            exact_losses: 0,
            deterministic_nodes: 0,
        }
    }

    fn record(&mut self, value: f32, outcome: SolveOutcome, deterministic_nodes: u64) {
        self.assignments_evaluated += 1;
        let value = f64::from(value);
        let delta = value - self.mean_value;
        self.mean_value += delta / self.assignments_evaluated as f64;
        let delta2 = value - self.mean_value;
        self.m2 += delta * delta2;
        self.variance = if self.assignments_evaluated > 1 {
            self.m2 / (self.assignments_evaluated - 1) as f64
        } else {
            0.0
        };
        if outcome == SolveOutcome::ProvenWin {
            self.exact_wins += 1;
        }
        if outcome == SolveOutcome::ProvenLoss {
            self.exact_losses += 1;
        }
        self.deterministic_nodes += deterministic_nodes;
    }
}

/// Result of a late-exact root-action evaluation pass.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LateExactResult {
    /// Whether the trigger condition was met and evaluation ran.
    pub triggered: bool,
    /// Hidden tableau card count at evaluation time.
    pub hidden_count: usize,
    /// Number of root actions evaluated.
    pub actions_evaluated: usize,
    /// Total assignments enumerated across evaluated actions.
    pub assignments_enumerated: u64,
    /// Assignments pruned by conservative hooks. Currently zero in v1.
    pub assignments_pruned: u64,
    /// Whether assignment enumeration was exhaustive.
    pub exhaustive: bool,
    /// Best action after exact assignment aggregation.
    pub best_move: Option<MacroMove>,
    /// Expected value for the best action.
    pub best_value: f64,
    /// Per-action exact assignment stats.
    pub action_stats: Vec<LateExactActionStats>,
    /// Deterministic solver nodes consumed.
    pub deterministic_nodes: u64,
    /// Elapsed evaluation time in milliseconds.
    pub elapsed_ms: u64,
}

impl LateExactResult {
    fn not_triggered(hidden_count: usize) -> Self {
        Self {
            triggered: false,
            hidden_count,
            actions_evaluated: 0,
            assignments_enumerated: 0,
            assignments_pruned: 0,
            exhaustive: false,
            best_move: None,
            best_value: 0.0,
            action_stats: Vec::new(),
            deterministic_nodes: 0,
            elapsed_ms: 0,
        }
    }
}

/// Late-game exact assignment evaluator.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LateExactEvaluator {
    /// Late-exact controls.
    pub config: LateExactConfig,
    /// Deterministic continuation solver controls.
    pub deterministic_config: DeterministicSearchConfig,
}

impl LateExactEvaluator {
    /// Creates an evaluator with explicit configs.
    pub const fn new(
        config: LateExactConfig,
        deterministic_config: DeterministicSearchConfig,
    ) -> Self {
        Self {
            config,
            deterministic_config,
        }
    }

    /// Returns true if the belief state is eligible for late-exact evaluation.
    pub fn should_trigger(&self, belief: &BeliefState) -> bool {
        self.config.enabled
            && belief.hidden_card_count() <= usize::from(self.config.hidden_card_threshold)
    }

    /// Evaluates the top candidate actions over exact hidden assignments.
    pub fn evaluate_actions(
        &self,
        belief: &BeliefState,
        actions: &[MacroMove],
    ) -> SolverResult<LateExactResult> {
        let started = Instant::now();
        belief.validate_consistency_against_visible()?;
        let hidden_count = belief.hidden_card_count();

        if !self.should_trigger(belief) || actions.is_empty() {
            return Ok(LateExactResult::not_triggered(hidden_count));
        }

        let action_limit = self.config.max_root_actions.max(1).min(actions.len());
        let actions = &actions[..action_limit];
        let assignments = enumerate_hidden_assignments(belief, self.config.assignment_budget)?;
        let exhaustive = self.config.assignment_budget.is_none_or(|budget| {
            assignments.len() < budget as usize
                || assignment_count_for_belief(belief).is_some_and(|total| total <= budget as u128)
        });

        let solver = DeterministicSolver::new(self.deterministic_config);
        let mut action_stats = actions
            .iter()
            .cloned()
            .map(LateExactActionStats::new)
            .collect::<Vec<_>>();

        for assignments in &assignments {
            let full = FullState::new(belief.visible.clone(), assignments.clone());
            full.validate_consistency()?;
            for stats in &mut action_stats {
                let value = self.evaluate_action_in_full_state(&full, &stats.action, &solver)?;
                stats.record(value.value, value.outcome, value.deterministic_nodes);
            }
        }

        action_stats.sort_by(|left, right| {
            right
                .mean_value
                .total_cmp(&left.mean_value)
                .then_with(|| left.action.kind.cmp(&right.action.kind))
                .then_with(|| left.action.id.cmp(&right.action.id))
        });

        let assignments_enumerated = action_stats
            .iter()
            .map(|stats| stats.assignments_evaluated)
            .sum::<u64>();
        let assignments_pruned = action_stats
            .iter()
            .map(|stats| stats.assignments_pruned)
            .sum::<u64>();
        let deterministic_nodes = action_stats
            .iter()
            .map(|stats| stats.deterministic_nodes)
            .sum::<u64>();
        let best_move = action_stats.first().map(|stats| stats.action.clone());
        let best_value = action_stats
            .first()
            .map(|stats| stats.mean_value)
            .unwrap_or_default();

        Ok(LateExactResult {
            triggered: true,
            hidden_count,
            actions_evaluated: action_stats.len(),
            assignments_enumerated,
            assignments_pruned,
            exhaustive,
            best_move,
            best_value,
            action_stats,
            deterministic_nodes,
            elapsed_ms: started.elapsed().as_millis() as u64,
        })
    }

    fn evaluate_action_in_full_state(
        &self,
        full_state: &FullState,
        action: &MacroMove,
        solver: &DeterministicSolver,
    ) -> SolverResult<LateExactActionValue> {
        let mut child = full_state.clone();
        apply_atomic_move_full_state(&mut child, action.atomic)?;

        match self.config.evaluation_mode {
            LateExactEvaluationMode::Exact => {
                let result = solver.solve_exact(&child)?;
                Ok(LateExactActionValue {
                    value: result.value,
                    outcome: result.outcome,
                    deterministic_nodes: result.stats.nodes_expanded,
                })
            }
            LateExactEvaluationMode::Bounded => {
                let result = solver.solve_bounded(&child)?;
                Ok(LateExactActionValue {
                    value: result.estimated_value,
                    outcome: result.outcome,
                    deterministic_nodes: result.stats.nodes_expanded,
                })
            }
            LateExactEvaluationMode::Fast => {
                let result = solver.evaluate_fast(&child)?;
                Ok(LateExactActionValue {
                    value: result.value,
                    outcome: SolveOutcome::Unknown,
                    deterministic_nodes: result.stats.nodes_expanded,
                })
            }
        }
    }
}

/// Enumerates hidden assignments in deterministic slot/card permutation order.
pub fn enumerate_hidden_assignments(
    belief: &BeliefState,
    assignment_budget: Option<u64>,
) -> SolverResult<Vec<HiddenAssignments>> {
    let prefix = AssignmentPrefix::from_belief(belief)?;
    if prefix.slots.len() != prefix.remaining_cards.len() {
        return Err(SolverError::UnseenCountMismatch {
            expected: prefix.slots.len(),
            actual: prefix.remaining_cards.len(),
        });
    }

    let budget = assignment_budget.unwrap_or(u64::MAX);
    if budget == 0 {
        return Ok(Vec::new());
    }

    let mut output = Vec::new();
    enumerate_prefix(prefix, budget, &mut output)?;
    Ok(output)
}

/// Returns the exact number of hidden assignments if it fits in `u128`.
pub fn assignment_count_for_belief(belief: &BeliefState) -> Option<u128> {
    if belief.hidden_card_count() != belief.unseen_card_count() {
        return None;
    }
    Some(factorial(belief.hidden_card_count() as u8))
}

fn enumerate_prefix(
    prefix: AssignmentPrefix,
    budget: u64,
    output: &mut Vec<HiddenAssignments>,
) -> SolverResult<()> {
    if output.len() as u64 >= budget {
        return Ok(());
    }

    let Some(slot) = prefix.next_slot() else {
        output.push(HiddenAssignments::new(prefix.assigned));
        return Ok(());
    };

    for index in 0..prefix.remaining_cards.len() {
        if output.len() as u64 >= budget {
            break;
        }
        let card = prefix.remaining_cards[index];
        let mut next_remaining = prefix.remaining_cards.clone();
        next_remaining.remove(index);

        let mut next_assigned = prefix.assigned.clone();
        next_assigned.push(HiddenAssignment::new(slot, card));

        enumerate_prefix(
            AssignmentPrefix {
                slots: prefix.slots.clone(),
                assigned: next_assigned,
                remaining_cards: next_remaining,
            },
            budget,
            output,
        )?;
    }

    Ok(())
}

const fn factorial(count: u8) -> u128 {
    let mut value = 1u128;
    let mut n = 2u8;
    while n <= count {
        value *= n as u128;
        n += 1;
    }
    value
}

#[derive(Debug, Copy, Clone, PartialEq)]
struct LateExactActionValue {
    value: f32,
    outcome: SolveOutcome,
    deterministic_nodes: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        cards::Card,
        core::{TableauColumn, UnseenCardSet, VisibleState},
        deterministic_solver::{DeterministicSearchConfig, DeterministicTtConfig, SolveBudget},
        moves::generate_legal_macro_moves,
        stock::CyclicStockState,
    };

    fn card(text: &str) -> Card {
        text.parse().unwrap()
    }

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

    fn small_belief() -> BeliefState {
        let hidden_cards = [card("Ac"), card("2c"), card("3c")];
        let seven = card("7s");
        let eight = card("8h");
        let mut visible = VisibleState::default();
        visible.columns[0] = TableauColumn::new(2, vec![seven]);
        visible.columns[1] = TableauColumn::new(1, vec![eight]);
        visible.stock = stock_with_all_except(&[
            hidden_cards[0],
            hidden_cards[1],
            hidden_cards[2],
            seven,
            eight,
        ]);
        BeliefState::new(
            visible,
            UnseenCardSet::from_cards(hidden_cards).expect("unique unseen cards"),
        )
    }

    fn deterministic_config() -> DeterministicSearchConfig {
        DeterministicSearchConfig {
            budget: SolveBudget {
                node_budget: Some(64),
                depth_budget: Some(2),
                wall_clock_limit_ms: None,
            },
            tt: DeterministicTtConfig {
                enabled: false,
                capacity: 0,
                store_approx: false,
            },
            ..DeterministicSearchConfig::default()
        }
    }

    #[test]
    fn trigger_respects_hidden_threshold() {
        let belief = small_belief();
        let evaluator = LateExactEvaluator::new(
            LateExactConfig {
                hidden_card_threshold: 2,
                ..LateExactConfig::default()
            },
            deterministic_config(),
        );
        assert!(!evaluator.should_trigger(&belief));

        let evaluator = LateExactEvaluator::new(
            LateExactConfig {
                hidden_card_threshold: 3,
                ..LateExactConfig::default()
            },
            deterministic_config(),
        );
        assert!(evaluator.should_trigger(&belief));
    }

    #[test]
    fn enumeration_order_is_deterministic() {
        let belief = small_belief();
        let first = enumerate_hidden_assignments(&belief, None).unwrap();
        let second = enumerate_hidden_assignments(&belief, None).unwrap();

        assert_eq!(first, second);
        assert_eq!(first.len(), 6);
        assert_eq!(first[0].entries[0].card, card("Ac"));
        assert_eq!(first[0].entries[1].card, card("2c"));
        assert_eq!(first[0].entries[2].card, card("3c"));
        assert_eq!(first[1].entries[0].card, card("Ac"));
        assert_eq!(first[1].entries[1].card, card("3c"));
        assert_eq!(first[1].entries[2].card, card("2c"));
    }

    #[test]
    fn all_assignments_are_unique_and_cover_hidden_slots() {
        let belief = small_belief();
        let assignments = enumerate_hidden_assignments(&belief, None).unwrap();
        let slots = belief.visible.hidden_slots();

        for assignment in &assignments {
            assignment
                .validate_against_visible(&belief.visible)
                .unwrap();
            assert_eq!(
                assignment
                    .iter()
                    .map(|entry| entry.slot)
                    .collect::<Vec<_>>(),
                slots
            );
        }

        let mut display_forms = assignments
            .iter()
            .map(|assignment| assignment.to_string())
            .collect::<Vec<_>>();
        display_forms.sort();
        display_forms.dedup();
        assert_eq!(display_forms.len(), assignments.len());
    }

    #[test]
    fn exact_assignment_count_matches_combinatorics() {
        let belief = small_belief();
        assert_eq!(assignment_count_for_belief(&belief), Some(6));
        assert_eq!(
            enumerate_hidden_assignments(&belief, Some(4))
                .unwrap()
                .len(),
            4
        );
    }

    #[test]
    fn late_exact_evaluation_does_not_mutate_belief() {
        let belief = small_belief();
        let before = belief.clone();
        let actions = generate_legal_macro_moves(&belief.visible);
        let evaluator = LateExactEvaluator::new(
            LateExactConfig {
                evaluation_mode: LateExactEvaluationMode::Fast,
                ..LateExactConfig::default()
            },
            deterministic_config(),
        );

        let result = evaluator.evaluate_actions(&belief, &actions[..1]).unwrap();

        assert!(result.triggered);
        assert_eq!(belief, before);
    }

    #[test]
    fn same_state_gives_same_exact_result() {
        let belief = small_belief();
        let actions = generate_legal_macro_moves(&belief.visible);
        let evaluator = LateExactEvaluator::new(
            LateExactConfig {
                evaluation_mode: LateExactEvaluationMode::Fast,
                max_root_actions: 1,
                ..LateExactConfig::default()
            },
            deterministic_config(),
        );

        let first = evaluator.evaluate_actions(&belief, &actions).unwrap();
        let second = evaluator.evaluate_actions(&belief, &actions).unwrap();

        assert_eq!(first.best_move, second.best_move);
        assert_eq!(first.best_value, second.best_value);
        assert_eq!(first.action_stats, second.action_stats);
    }
}
