//! Exact hidden-information transitions, reveal frontiers, and world sampling.
//!
//! This module preserves the exact uniform posterior over assignments of unseen
//! cards to face-down tableau slots. Stock/waste order is fully known and never
//! sampled here. Hidden uncertainty exists only in tableau slots that are still
//! face down.

use serde::{Deserialize, Serialize};

use crate::{
    cards::Card,
    core::{
        BeliefState, FullState, HiddenAssignment, HiddenAssignments, HiddenSlot, UnseenCardSet,
        VisibleState,
    },
    error::{SolverError, SolverResult},
    moves::{apply_atomic_move, requires_reveal, AtomicMove, MoveOutcome, MoveSemantics},
    types::{ColumnId, DealSeed},
};

/// Reveal observation produced when a hidden tableau card is exposed.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RevealEvent {
    /// Column where the card was revealed.
    pub column: ColumnId,
    /// Revealed card identity.
    pub card: Card,
}

/// Belief transition category.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum BeliefTransitionKind {
    /// A visible move with no reveal.
    Deterministic,
    /// A move that exposes one hidden tableau card and branches by observation.
    Reveal,
}

/// Lightweight action context for future planner screening.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BeliefActionEvaluationContext {
    /// Visible action being considered.
    pub action: AtomicMove,
    /// Whether this action reaches a reveal frontier.
    pub requires_reveal: bool,
    /// Number of possible reveal outcomes if `requires_reveal` is true.
    pub reveal_branch_count: usize,
}

impl BeliefActionEvaluationContext {
    /// Builds context for one action from the current belief state.
    pub fn new(belief: &BeliefState, action: AtomicMove) -> Self {
        let requires_reveal = requires_reveal_from_belief(belief, action);
        Self {
            action,
            requires_reveal,
            reveal_branch_count: if requires_reveal {
                belief.unseen_card_count()
            } else {
                0
            },
        }
    }
}

/// One exact reveal child under the uniform posterior.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RevealOutcome {
    /// Revealed card assigned to the exposed slot.
    pub revealed_card: Card,
    /// Resulting belief state after observing this card.
    pub belief: BeliefState,
    /// Equal branch probability.
    pub probability: f32,
    /// Visible move outcome for this branch.
    pub move_outcome: MoveOutcome,
}

/// Exact reveal frontier: one child per unseen card.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RevealFrontier {
    /// Reveal-causing visible action.
    pub action: AtomicMove,
    /// Equal-probability reveal outcomes in deterministic card order.
    pub outcomes: Vec<RevealOutcome>,
    /// Move semantics shared by all reveal outcomes.
    pub semantics: MoveSemantics,
}

impl RevealFrontier {
    /// Returns the number of reveal branches.
    pub fn len(&self) -> usize {
        self.outcomes.len()
    }

    /// Returns true if no reveal branches exist.
    pub fn is_empty(&self) -> bool {
        self.outcomes.is_empty()
    }

    /// Iterates reveal outcomes in deterministic card order.
    pub fn iter(&self) -> impl Iterator<Item = &RevealOutcome> {
        self.outcomes.iter()
    }

    /// Returns the sum of branch probabilities.
    pub fn total_probability(&self) -> f32 {
        self.outcomes
            .iter()
            .map(|outcome| outcome.probability)
            .sum()
    }
}

/// Result of applying a visible action at the belief layer.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum BeliefTransition {
    /// Deterministic non-reveal transition.
    Deterministic {
        /// Applied visible action.
        action: AtomicMove,
        /// Resulting belief state.
        belief: BeliefState,
        /// Visible move outcome.
        move_outcome: MoveOutcome,
    },
    /// Reveal transition represented as an exact frontier.
    Reveal {
        /// Exact reveal frontier.
        frontier: RevealFrontier,
    },
}

impl BeliefTransition {
    /// Returns the transition kind.
    pub const fn kind(&self) -> BeliefTransitionKind {
        match self {
            Self::Deterministic { .. } => BeliefTransitionKind::Deterministic,
            Self::Reveal { .. } => BeliefTransitionKind::Reveal,
        }
    }
}

/// One sampled deterministic world.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeterminizationSample {
    /// Sample index within a batch.
    pub sample_index: usize,
    /// Full deterministic state sampled from the uniform posterior.
    pub full_state: FullState,
}

/// Uniform sampler for hidden tableau assignments.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorldSampler {
    rng: SplitMix64,
}

impl WorldSampler {
    /// Creates a reproducible sampler.
    pub const fn new(seed: DealSeed) -> Self {
        Self {
            rng: SplitMix64::new(seed.0),
        }
    }

    /// Samples one full deterministic state uniformly from the belief state.
    pub fn sample_full_state(&mut self, belief: &BeliefState) -> SolverResult<FullState> {
        sample_full_state_with_rng(belief, &mut self.rng)
    }

    /// Samples a reproducible batch of full deterministic states.
    pub fn sample_full_states(
        &mut self,
        belief: &BeliefState,
        samples: usize,
    ) -> SolverResult<Vec<DeterminizationSample>> {
        if samples == 0 {
            return Ok(Vec::new());
        }

        let mut prepared = PreparedWorldSampler::new_with_rng(belief, self.rng)?;
        let mut worlds = Vec::with_capacity(samples);
        for sample_index in 0..samples {
            worlds.push(DeterminizationSample {
                sample_index,
                full_state: prepared.sample_full_state()?,
            });
        }
        self.rng = prepared.into_rng();
        Ok(worlds)
    }
}

/// Prepared uniform world sampler for repeated samples from one belief state.
///
/// It caches deterministic hidden-slot order and sorted unseen cards once, then
/// shuffles a local card vector for each sample. The posterior remains exactly
/// uniform over assignments of unseen cards to hidden tableau slots.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PreparedWorldSampler {
    visible: VisibleState,
    slots: Vec<HiddenSlot>,
    sorted_cards: Vec<Card>,
    rng: SplitMix64,
}

impl PreparedWorldSampler {
    /// Creates a prepared sampler from a belief state and seed.
    pub fn new(belief: &BeliefState, seed: DealSeed) -> SolverResult<Self> {
        Self::new_with_rng(belief, SplitMix64::new(seed.0))
    }

    fn new_with_rng(belief: &BeliefState, rng: SplitMix64) -> SolverResult<Self> {
        belief.validate_consistency_against_visible()?;
        let slots = belief.visible.hidden_slots();
        let sorted_cards = belief.unseen_cards.to_sorted_vec();
        if slots.len() != sorted_cards.len() {
            return Err(SolverError::UnseenCountMismatch {
                expected: slots.len(),
                actual: sorted_cards.len(),
            });
        }

        Ok(Self {
            visible: belief.visible.clone(),
            slots,
            sorted_cards,
            rng,
        })
    }

    /// Samples one full deterministic state from the prepared belief state.
    pub fn sample_full_state(&mut self) -> SolverResult<FullState> {
        sample_full_state_from_parts(
            &self.visible,
            &self.slots,
            &self.sorted_cards,
            &mut self.rng,
        )
    }

    fn into_rng(self) -> SplitMix64 {
        self.rng
    }
}

/// Returns true if applying `action` reaches a reveal frontier.
pub fn requires_reveal_from_belief(belief: &BeliefState, action: AtomicMove) -> bool {
    requires_reveal(&belief.visible, action)
}

/// Applies a non-reveal visible move exactly at the belief layer.
pub fn apply_atomic_move_belief_nonreveal(
    belief: &BeliefState,
    action: AtomicMove,
) -> SolverResult<(BeliefState, MoveOutcome)> {
    if requires_reveal_from_belief(belief, action) {
        return Err(SolverError::IllegalMove(
            "belief non-reveal transition was given a reveal-causing action".to_string(),
        ));
    }

    let mut visible = belief.visible.clone();
    let transition = apply_atomic_move(&mut visible, action, None)?;
    let child = BeliefState::new(visible, belief.unseen_cards);
    child.validate_consistency_against_visible()?;
    Ok((child, transition.outcome))
}

/// Expands an exact equal-probability reveal frontier.
pub fn expand_reveal_frontier(
    belief: &BeliefState,
    action: AtomicMove,
) -> SolverResult<RevealFrontier> {
    if !requires_reveal_from_belief(belief, action) {
        return Err(SolverError::IllegalMove(
            "cannot expand reveal frontier for a non-reveal action".to_string(),
        ));
    }

    belief.validate_consistency_against_visible()?;

    let branch_count = belief.unseen_card_count();
    if branch_count == 0 {
        return Err(SolverError::InvalidState(
            "reveal frontier requires at least one unseen card".to_string(),
        ));
    }

    let probability = 1.0 / branch_count as f32;
    let mut outcomes = Vec::with_capacity(branch_count);
    let mut semantics = None;

    for card in belief.unseen_cards.iter() {
        let mut visible = belief.visible.clone();
        let transition = apply_atomic_move(&mut visible, action, Some(card))?;
        semantics = Some(transition.outcome.semantics);

        let mut unseen = belief.unseen_cards;
        if !unseen.remove(card) {
            return Err(SolverError::InvalidState(format!(
                "revealed card {card} was not present in unseen set"
            )));
        }

        let child = BeliefState::new(visible, unseen);
        child.validate_consistency_against_visible()?;
        outcomes.push(RevealOutcome {
            revealed_card: card,
            belief: child,
            probability,
            move_outcome: transition.outcome,
        });
    }

    Ok(RevealFrontier {
        action,
        outcomes,
        semantics: semantics.unwrap_or_default(),
    })
}

/// Applies an action at the belief layer.
pub fn apply_belief_transition(
    belief: &BeliefState,
    action: AtomicMove,
) -> SolverResult<BeliefTransition> {
    if requires_reveal_from_belief(belief, action) {
        Ok(BeliefTransition::Reveal {
            frontier: expand_reveal_frontier(belief, action)?,
        })
    } else {
        let (belief, move_outcome) = apply_atomic_move_belief_nonreveal(belief, action)?;
        Ok(BeliefTransition::Deterministic {
            action,
            belief,
            move_outcome,
        })
    }
}

/// Constructs the public belief state corresponding to a true full state.
///
/// This is used by autoplay and benchmark harnesses: the visible state is known
/// publicly, while the hidden assignments are converted into the exact unseen
/// tableau card set.
pub fn belief_from_full_state(full_state: &FullState) -> SolverResult<BeliefState> {
    full_state.validate_consistency()?;
    let unseen_cards = UnseenCardSet::from_cards(
        full_state
            .hidden_assignments
            .iter()
            .map(|assignment| assignment.card),
    )?;
    let belief = BeliefState::new(full_state.visible.clone(), unseen_cards);
    belief.validate_consistency_against_visible()?;
    Ok(belief)
}

/// Validates that a public belief state matches a true full state.
pub fn validate_belief_against_full_state(
    belief: &BeliefState,
    full_state: &FullState,
) -> SolverResult<()> {
    belief.validate_consistency_against_visible()?;
    full_state.validate_consistency()?;

    if belief.visible != full_state.visible {
        return Err(SolverError::InvalidState(
            "belief visible state differs from true full-state visible state".to_string(),
        ));
    }

    let unseen_cards = UnseenCardSet::from_cards(
        full_state
            .hidden_assignments
            .iter()
            .map(|assignment| assignment.card),
    )?;
    if belief.unseen_cards != unseen_cards {
        return Err(SolverError::InvalidState(
            "belief unseen cards differ from true hidden assignments".to_string(),
        ));
    }

    Ok(())
}

/// Applies one real observed move to a belief state.
///
/// Non-reveal moves leave `unseen_cards` unchanged. Reveal moves must pass the
/// actual card observed from the true game; that card is removed from the exact
/// unseen tableau set.
pub fn apply_observed_belief_move(
    belief: &BeliefState,
    action: AtomicMove,
    revealed_card: Option<Card>,
) -> SolverResult<(BeliefState, MoveOutcome)> {
    belief.validate_consistency_against_visible()?;

    let mut visible = belief.visible.clone();
    let transition = apply_atomic_move(&mut visible, action, revealed_card)?;
    let mut unseen_cards = belief.unseen_cards;

    if let Some(reveal) = transition.outcome.revealed {
        if !unseen_cards.remove(reveal.card) {
            return Err(SolverError::InvalidState(format!(
                "observed revealed card {card} was not in the belief unseen set",
                card = reveal.card
            )));
        }
    }

    let child = BeliefState::new(visible, unseen_cards);
    child.validate_consistency_against_visible()?;
    Ok((child, transition.outcome))
}

/// Samples one full deterministic state using a reproducible seed.
pub fn sample_full_state(belief: &BeliefState, seed: DealSeed) -> SolverResult<FullState> {
    WorldSampler::new(seed).sample_full_state(belief)
}

/// Samples a reproducible batch of full deterministic states.
pub fn sample_full_states(
    belief: &BeliefState,
    samples: usize,
    seed: DealSeed,
) -> SolverResult<Vec<DeterminizationSample>> {
    WorldSampler::new(seed).sample_full_states(belief, samples)
}

/// Returns hidden slots in deterministic column/depth order.
pub fn hidden_slots_for_belief(belief: &BeliefState) -> Vec<HiddenSlot> {
    belief.visible.hidden_slots()
}

/// Validates that a sampled full state is consistent with its source belief.
pub fn validate_sample_against_belief(
    sample: &FullState,
    belief: &BeliefState,
) -> SolverResult<()> {
    sample.validate_consistency()?;
    belief.validate_consistency_against_visible()?;

    if sample.visible != belief.visible {
        return Err(SolverError::InvalidState(
            "sampled full state visible state differs from belief visible state".to_string(),
        ));
    }

    let sample_cards = sample
        .hidden_assignments
        .iter()
        .map(|assignment| assignment.card)
        .collect::<Vec<_>>();
    let sample_unseen = UnseenCardSet::from_cards(sample_cards)?;
    if sample_unseen != belief.unseen_cards {
        return Err(SolverError::InvalidState(
            "sampled hidden cards do not match belief unseen set".to_string(),
        ));
    }

    Ok(())
}

fn sample_full_state_with_rng(
    belief: &BeliefState,
    rng: &mut SplitMix64,
) -> SolverResult<FullState> {
    belief.validate_consistency_against_visible()?;

    let slots = belief.visible.hidden_slots();
    let sorted_cards = belief.unseen_cards.to_sorted_vec();
    if slots.len() != sorted_cards.len() {
        return Err(SolverError::UnseenCountMismatch {
            expected: slots.len(),
            actual: sorted_cards.len(),
        });
    }

    sample_full_state_from_parts(&belief.visible, &slots, &sorted_cards, rng)
}

fn sample_full_state_from_parts(
    visible: &VisibleState,
    slots: &[HiddenSlot],
    sorted_cards: &[Card],
    rng: &mut SplitMix64,
) -> SolverResult<FullState> {
    let mut cards = sorted_cards.to_vec();
    shuffle_cards(&mut cards, rng);
    let mut entries = Vec::with_capacity(slots.len());
    for (slot, card) in slots.iter().copied().zip(cards) {
        entries.push(HiddenAssignment::new(slot, card));
    }
    Ok(FullState::new(
        visible.clone(),
        HiddenAssignments::new(entries),
    ))
}

fn shuffle_cards(cards: &mut [Card], rng: &mut SplitMix64) {
    for index in (1..cards.len()).rev() {
        let swap_index = rng.next_bounded(index + 1);
        cards.swap(index, swap_index);
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    const fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut value = self.state;
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        value ^ (value >> 31)
    }

    fn next_bounded(&mut self, upper_exclusive: usize) -> usize {
        debug_assert!(upper_exclusive > 0);
        (self.next_u64() % upper_exclusive as u64) as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        core::TableauColumn,
        moves::AtomicMove,
        stock::CyclicStockState,
        types::{ColumnId, DealSeed},
    };

    fn col(index: u8) -> ColumnId {
        ColumnId::new(index).unwrap()
    }

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

    fn nonreveal_belief() -> BeliefState {
        let hidden = card("Ah");
        let seven = card("7s");
        let six = card("6h");
        let mut visible = crate::core::VisibleState::default();
        visible.columns[0] = TableauColumn::new(0, vec![seven]);
        visible.columns[1] = TableauColumn::new(0, vec![six]);
        visible.columns[2] = TableauColumn::new(1, Vec::new());
        visible.stock = stock_with_all_except(&[hidden, seven, six]);
        BeliefState::new(
            visible,
            UnseenCardSet::from_cards([hidden]).expect("unique unseen card"),
        )
    }

    fn reveal_belief() -> BeliefState {
        let hidden_cards = [card("Ac"), card("2c")];
        let seven = card("7s");
        let eight = card("8h");
        let mut visible = crate::core::VisibleState::default();
        visible.columns[0] = TableauColumn::new(2, vec![seven]);
        visible.columns[1] = TableauColumn::new(0, vec![eight]);
        visible.stock = stock_with_all_except(&[hidden_cards[0], hidden_cards[1], seven, eight]);
        BeliefState::new(
            visible,
            UnseenCardSet::from_cards(hidden_cards).expect("unique unseen cards"),
        )
    }

    fn sample_belief() -> BeliefState {
        let hidden_cards = [card("Ac"), card("2c"), card("3c")];
        let visible_card = card("7s");
        let mut visible = crate::core::VisibleState::default();
        visible.columns[0] = TableauColumn::new(2, vec![visible_card]);
        visible.columns[3] = TableauColumn::new(1, Vec::new());
        visible.stock = stock_with_all_except(&[
            hidden_cards[0],
            hidden_cards[1],
            hidden_cards[2],
            visible_card,
        ]);
        BeliefState::new(
            visible,
            UnseenCardSet::from_cards(hidden_cards).expect("unique unseen cards"),
        )
    }

    #[test]
    fn non_reveal_move_leaves_unseen_set_unchanged() {
        let belief = nonreveal_belief();
        let before_unseen = belief.unseen_cards;
        let action = AtomicMove::TableauToTableau {
            src: col(1),
            dest: col(0),
            run_start: 0,
        };

        let (child, outcome) = apply_atomic_move_belief_nonreveal(&belief, action).unwrap();

        assert_eq!(child.unseen_cards, before_unseen);
        assert!(outcome.revealed.is_none());
        child.validate_consistency_against_visible().unwrap();
    }

    #[test]
    fn reveal_move_expands_to_one_child_per_unseen_card() {
        let belief = reveal_belief();
        let action = AtomicMove::TableauToTableau {
            src: col(0),
            dest: col(1),
            run_start: 0,
        };

        let frontier = expand_reveal_frontier(&belief, action).unwrap();

        assert_eq!(frontier.len(), belief.unseen_card_count());
        assert_eq!(
            frontier
                .iter()
                .map(|outcome| outcome.revealed_card)
                .collect::<Vec<_>>(),
            belief.unseen_cards.to_sorted_vec()
        );
    }

    #[test]
    fn reveal_children_have_equal_probability_and_sum_to_one() {
        let frontier = expand_reveal_frontier(
            &reveal_belief(),
            AtomicMove::TableauToTableau {
                src: col(0),
                dest: col(1),
                run_start: 0,
            },
        )
        .unwrap();

        for outcome in frontier.iter() {
            assert!((outcome.probability - 0.5).abs() < f32::EPSILON);
        }
        assert!((frontier.total_probability() - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn each_reveal_child_removes_one_unseen_card_and_updates_visible_column() {
        let frontier = expand_reveal_frontier(
            &reveal_belief(),
            AtomicMove::TableauToTableau {
                src: col(0),
                dest: col(1),
                run_start: 0,
            },
        )
        .unwrap();

        for outcome in frontier.iter() {
            assert_eq!(outcome.belief.unseen_card_count(), 1);
            assert!(!outcome.belief.unseen_cards.contains(outcome.revealed_card));
            assert_eq!(
                outcome.belief.visible.columns[0].top_face_up(),
                Some(outcome.revealed_card)
            );
            outcome
                .belief
                .validate_consistency_against_visible()
                .unwrap();
        }
    }

    #[test]
    fn uniform_sampler_produces_valid_full_states() {
        let belief = sample_belief();
        let mut sampler = WorldSampler::new(DealSeed(123));

        let full = sampler.sample_full_state(&belief).unwrap();

        validate_sample_against_belief(&full, &belief).unwrap();
        assert_eq!(full.hidden_assignments.len(), belief.hidden_card_count());
    }

    #[test]
    fn fixed_seed_reproduces_sampled_full_states() {
        let belief = sample_belief();
        let first = sample_full_states(&belief, 4, DealSeed(77)).unwrap();
        let second = sample_full_states(&belief, 4, DealSeed(77)).unwrap();

        assert_eq!(first, second);
    }

    #[test]
    fn sampled_full_states_validate_against_source_belief() {
        let belief = sample_belief();
        let samples = sample_full_states(&belief, 5, DealSeed(999)).unwrap();

        for sample in samples {
            validate_sample_against_belief(&sample.full_state, &belief).unwrap();
        }
    }

    #[test]
    fn belief_transition_distinguishes_deterministic_vs_reveal() {
        let nonreveal = apply_belief_transition(
            &nonreveal_belief(),
            AtomicMove::TableauToTableau {
                src: col(1),
                dest: col(0),
                run_start: 0,
            },
        )
        .unwrap();
        assert_eq!(nonreveal.kind(), BeliefTransitionKind::Deterministic);

        let reveal = apply_belief_transition(
            &reveal_belief(),
            AtomicMove::TableauToTableau {
                src: col(0),
                dest: col(1),
                run_start: 0,
            },
        )
        .unwrap();
        assert_eq!(reveal.kind(), BeliefTransitionKind::Reveal);
    }

    #[test]
    fn action_context_reports_reveal_branch_count() {
        let belief = reveal_belief();
        let context = BeliefActionEvaluationContext::new(
            &belief,
            AtomicMove::TableauToTableau {
                src: col(0),
                dest: col(1),
                run_start: 0,
            },
        );

        assert!(context.requires_reveal);
        assert_eq!(context.reveal_branch_count, 2);
    }
}
