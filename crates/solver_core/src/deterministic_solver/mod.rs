//! First deterministic perfect-information solver layer.
//!
//! This module searches fully instantiated Klondike states. It does not model
//! hidden-information beliefs, sampling, UCT, or late-game assignment search.
//! Reveal identities come from `FullState::hidden_assignments`, while all
//! visible transitions still go through the move engine.

use std::{
    cell::RefCell,
    time::{Duration, Instant},
};

use serde::{Deserialize, Serialize};

use crate::{
    cards::{Card, Rank},
    closure::{ClosureConfig, ClosureEngine, ClosureStopReason, FullClosureRun},
    core::{FullState, VisibleState},
    error::SolverResult,
    ml::{LeafEvaluationMode, VNetEvaluator, VNetInferenceConfig},
    moves::{
        apply_atomic_move_full_state, generate_legal_macro_moves_with_config,
        undo_atomic_move_full_state, AtomicMove, FullStateMoveUndo, MacroMove, MacroMoveKind,
        MoveGenerationConfig,
    },
};

/// Deterministic solver operating mode.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SolveMode {
    /// Attempt to prove win/loss within budget.
    Exact,
    /// Search a bounded horizon and fall back to an evaluator.
    Bounded,
    /// Return a lightweight deterministic evaluation.
    FastEvaluate,
}

/// Proof or bounded-search status.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SolveOutcome {
    /// A winning continuation was proven.
    ProvenWin,
    /// All searched legal continuations were proven losing.
    ProvenLoss,
    /// The supplied budget was insufficient to prove the state.
    Unknown,
}

/// Compatibility alias for earlier scaffold naming.
pub type ProofStatus = SolveOutcome;

/// Search budget for one deterministic solve.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SolveBudget {
    /// Maximum recursive nodes to expand.
    pub node_budget: Option<u64>,
    /// Maximum explicit branch depth after closure.
    pub depth_budget: Option<u16>,
    /// Optional wall-clock cap.
    pub wall_clock_limit_ms: Option<u64>,
}

impl Default for SolveBudget {
    fn default() -> Self {
        Self {
            node_budget: Some(100_000),
            depth_budget: Some(64),
            wall_clock_limit_ms: None,
        }
    }
}

/// Lightweight evaluator weights used at bounded cutoffs.
#[derive(Debug, Copy, Clone, PartialEq, Serialize, Deserialize)]
pub struct EvaluatorWeights {
    /// Weight for foundation completion.
    pub foundation_progress: f32,
    /// Weight for having revealed hidden tableau cards.
    pub hidden_cards_revealed: f32,
    /// Weight for empty tableau columns.
    pub empty_columns: f32,
    /// Weight for legal macro-move mobility.
    pub mobility: f32,
    /// Weight for having an accessible waste card.
    pub waste_access: f32,
}

impl Default for EvaluatorWeights {
    fn default() -> Self {
        Self {
            foundation_progress: 0.60,
            hidden_cards_revealed: 0.15,
            empty_columns: 0.10,
            mobility: 0.10,
            waste_access: 0.05,
        }
    }
}

/// Deterministic transposition-table controls.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeterministicTtConfig {
    /// Enables deterministic-state TT lookup and storage.
    pub enabled: bool,
    /// Maximum number of direct-mapped TT slots.
    pub capacity: usize,
    /// Whether bounded heuristic results may be stored as approximate entries.
    pub store_approx: bool,
}

impl Default for DeterministicTtConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            capacity: 65_536,
            store_approx: true,
        }
    }
}

/// Configuration for deterministic open-card search.
#[derive(Debug, Copy, Clone, PartialEq, Serialize, Deserialize)]
pub struct DeterministicSearchConfig {
    /// Budget controls for this solve.
    pub budget: SolveBudget,
    /// Closure configuration used at every search node.
    pub closure: ClosureConfig,
    /// Whether foundation-to-tableau retreats are generated.
    pub allow_foundation_retreats: bool,
    /// Weights for bounded and fast evaluation.
    pub evaluator_weights: EvaluatorWeights,
    /// Deterministic transposition-table controls.
    pub tt: DeterministicTtConfig,
    /// Approximate evaluator used at budget/depth cutoffs and fast leaves.
    pub leaf_eval_mode: LeafEvaluationMode,
}

impl Default for DeterministicSearchConfig {
    fn default() -> Self {
        Self {
            budget: SolveBudget::default(),
            closure: ClosureConfig::default(),
            allow_foundation_retreats: true,
            evaluator_weights: EvaluatorWeights::default(),
            tt: DeterministicTtConfig::default(),
            leaf_eval_mode: LeafEvaluationMode::Heuristic,
        }
    }
}

/// Deterministic search diagnostics.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeterministicSearchStats {
    /// Recursive nodes entered.
    pub nodes_expanded: u64,
    /// Closure steps applied across all nodes.
    pub closure_steps_applied: u64,
    /// Maximum explicit search depth reached.
    pub max_depth_reached: u16,
    /// Number of child wins found during search.
    pub exact_wins_found: u64,
    /// Number of cutoffs caused by node/depth/time budget.
    pub budget_cutoffs: u64,
    /// Number of child branches considered.
    pub branches_considered: u64,
    /// Deterministic TT probes.
    pub tt_probes: u64,
    /// Deterministic TT hits that matched the full-state key.
    pub tt_hits: u64,
    /// TT hits reused as exact proof results.
    pub tt_exact_hits: u64,
    /// TT hits reused as approximate bounded values or ordering hints.
    pub tt_approx_hits: u64,
    /// TT entries stored.
    pub tt_stores: u64,
    /// TT stores that replaced an existing slot entry.
    pub tt_replacements: u64,
    /// TT stores that replaced a different key in the same direct-mapped slot.
    pub tt_collisions: u64,
    /// V-Net inference calls used for approximate leaf evaluation.
    pub vnet_inferences: u64,
    /// V-Net inference wall-clock time in microseconds.
    pub vnet_inference_elapsed_us: u64,
    /// Times a requested V-Net leaf evaluation fell back to the heuristic.
    pub vnet_fallbacks: u64,
    /// Elapsed wall-clock time in milliseconds.
    pub elapsed_ms: u64,
}

/// Stable structural hash key for a full deterministic state.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct DeterministicHashKey {
    /// Primary 64-bit structural hash.
    pub primary: u64,
    /// Secondary 64-bit structural hash to reduce accidental collision risk.
    pub secondary: u64,
}

impl DeterministicHashKey {
    /// Computes a deterministic key from all full-state components.
    pub fn from_full_state(full_state: &FullState) -> Self {
        let mut builder = StableHashBuilder::new();
        builder.add_tag(0x10);
        for top_rank in full_state.visible.foundations.top_ranks {
            builder.add_u8(top_rank.map_or(0, |rank| rank.value()));
        }

        builder.add_tag(0x20);
        for column in &full_state.visible.columns {
            builder.add_u8(column.hidden_count);
            builder.add_usize(column.face_up.len());
            for card in &column.face_up {
                builder.add_card(*card);
            }
        }

        builder.add_tag(0x30);
        let stock = &full_state.visible.stock;
        builder.add_usize(stock.ring_cards.len());
        for card in &stock.ring_cards {
            builder.add_card(*card);
        }
        builder.add_usize(stock.stock_len);
        builder.add_option_usize(stock.cursor);
        builder.add_u8(stock.accessible_depth);
        builder.add_u64(u64::from(stock.pass_index));
        builder.add_option_u32(stock.max_passes);
        builder.add_u8(stock.draw_count);

        builder.add_tag(0x40);
        builder.add_usize(full_state.hidden_assignments.entries.len());
        for assignment in &full_state.hidden_assignments.entries {
            builder.add_u8(assignment.slot.column.index());
            builder.add_u8(assignment.slot.depth);
            builder.add_card(assignment.card);
        }

        builder.finish()
    }

    fn table_index(self, capacity: usize) -> usize {
        debug_assert!(capacity > 0);
        (self.primary as usize) % capacity
    }
}

/// TT bound/value kind.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum DeterministicBound {
    /// Proven exact win/loss.
    Exact,
    /// Lower-bound value, reserved for future proof search refinements.
    Lower,
    /// Upper-bound value, reserved for future proof search refinements.
    Upper,
    /// Heuristic or budget-limited estimate.
    Approx,
}

/// Stored deterministic TT value.
#[derive(Debug, Copy, Clone, PartialEq, Serialize, Deserialize)]
pub struct DeterministicTtValue {
    /// Outcome associated with the entry.
    pub outcome: SolveOutcome,
    /// Numeric value in 0..=1.
    pub value: f32,
    /// Exact/bound/approximate semantics.
    pub bound: DeterministicBound,
}

/// One deterministic transposition-table entry.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DeterministicTtEntry {
    /// Full deterministic state key.
    pub key: DeterministicHashKey,
    /// Stored value and proof semantics.
    pub value: DeterministicTtValue,
    /// Best macro move observed from this state, if known.
    pub best_move: Option<MacroMove>,
    /// Remaining search depth/horizon backing this entry.
    pub searched_depth: u16,
    /// Monotonic generation assigned by the table on store.
    pub age: u64,
}

impl DeterministicTtEntry {
    fn exact_result(&self) -> Option<NodeSearchResult> {
        if self.value.bound != DeterministicBound::Exact {
            return None;
        }
        Some(NodeSearchResult {
            outcome: self.value.outcome,
            value: self.value.value,
            best_move: self.best_move.clone(),
            principal_line: self
                .best_move
                .as_ref()
                .map(|best| vec![best.kind])
                .unwrap_or_default(),
        })
    }

    fn approximate_result(&self, depth_remaining: u16) -> Option<NodeSearchResult> {
        if self.value.bound == DeterministicBound::Exact {
            return self.exact_result();
        }
        if self.value.bound == DeterministicBound::Approx && self.searched_depth >= depth_remaining
        {
            return Some(NodeSearchResult {
                outcome: SolveOutcome::Unknown,
                value: self.value.value,
                best_move: self.best_move.clone(),
                principal_line: self
                    .best_move
                    .as_ref()
                    .map(|best| vec![best.kind])
                    .unwrap_or_default(),
            });
        }
        None
    }

    fn replacement_priority(&self) -> (u8, u16, u64) {
        let bound_priority = match self.value.bound {
            DeterministicBound::Exact => 3,
            DeterministicBound::Lower | DeterministicBound::Upper => 2,
            DeterministicBound::Approx => 1,
        };
        (bound_priority, self.searched_depth, self.age)
    }
}

/// Result of storing a TT entry.
#[derive(Debug, Copy, Clone, Default, PartialEq, Eq)]
pub struct DeterministicTtStoreReport {
    /// Whether the new entry was stored.
    pub stored: bool,
    /// Whether an existing entry was replaced.
    pub replaced: bool,
    /// Whether the replaced entry had a different key.
    pub collision: bool,
}

/// Simple direct-mapped deterministic transposition table.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DeterministicTt {
    entries: Vec<Option<DeterministicTtEntry>>,
    generation: u64,
}

impl DeterministicTt {
    /// Creates a bounded deterministic TT.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            entries: vec![None; capacity],
            generation: 0,
        }
    }

    /// Returns the number of available slots.
    pub fn capacity(&self) -> usize {
        self.entries.len()
    }

    /// Returns true when the table has no slots.
    pub fn is_disabled(&self) -> bool {
        self.entries.is_empty()
    }

    /// Probes a full deterministic key.
    pub fn probe(&self, key: DeterministicHashKey) -> Option<&DeterministicTtEntry> {
        if self.entries.is_empty() {
            return None;
        }
        self.entries[key.table_index(self.entries.len())]
            .as_ref()
            .filter(|entry| entry.key == key)
    }

    /// Stores an entry using a simple deterministic direct-mapped policy.
    pub fn store(&mut self, mut entry: DeterministicTtEntry) -> DeterministicTtStoreReport {
        if self.entries.is_empty() {
            return DeterministicTtStoreReport::default();
        }

        self.generation = self.generation.saturating_add(1);
        entry.age = self.generation;
        let index = entry.key.table_index(self.entries.len());
        let Some(existing) = &self.entries[index] else {
            self.entries[index] = Some(entry);
            return DeterministicTtStoreReport {
                stored: true,
                replaced: false,
                collision: false,
            };
        };

        let collision = existing.key != entry.key;
        if should_replace_tt_entry(existing, &entry) {
            self.entries[index] = Some(entry);
            DeterministicTtStoreReport {
                stored: true,
                replaced: true,
                collision,
            }
        } else {
            DeterministicTtStoreReport {
                stored: false,
                replaced: false,
                collision,
            }
        }
    }
}

/// Result of an exact/proof-oriented solve attempt.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExactSolveResult {
    /// Proof status reached within budget.
    pub outcome: SolveOutcome,
    /// Best root macro move if one is known.
    pub best_move: Option<MacroMove>,
    /// Principal line as macro kinds.
    pub principal_line: Vec<MacroMoveKind>,
    /// Estimated win probability for this result.
    pub value: f32,
    /// Search diagnostics.
    pub stats: DeterministicSearchStats,
}

/// Result of bounded deterministic search.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BoundedSolveResult {
    /// Proof status if the bounded search proved one.
    pub outcome: SolveOutcome,
    /// Best root macro move found under budget.
    pub best_move: Option<MacroMove>,
    /// Principal line as macro kinds.
    pub principal_line: Vec<MacroMoveKind>,
    /// Bounded value estimate in 0..=1.
    pub estimated_value: f32,
    /// Search diagnostics.
    pub stats: DeterministicSearchStats,
}

/// Result of fast deterministic evaluation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FastEvalResult {
    /// Stable approximate value in 0..=1.
    pub value: f32,
    /// Best move candidate under one-ply ordering/evaluation.
    pub best_move: Option<MacroMove>,
    /// Lightweight diagnostics.
    pub stats: DeterministicSearchStats,
}

/// Recommendation from the open-card solver.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OpenCardRecommendation {
    /// Suggested macro move for the fully instantiated state.
    pub best_move: Option<MacroMove>,
    /// Estimated value after choosing the move.
    pub value: f32,
    /// Search diagnostics.
    pub stats: DeterministicSearchStats,
}

/// Deterministic perfect-information solver.
#[derive(Debug)]
pub struct DeterministicSolver {
    /// Solver configuration.
    pub config: DeterministicSearchConfig,
    tt: RefCell<DeterministicTt>,
    vnet: Option<VNetEvaluator>,
}

impl DeterministicSolver {
    /// Creates a deterministic solver.
    pub fn new(config: DeterministicSearchConfig) -> Self {
        let tt_capacity = if config.tt.enabled {
            config.tt.capacity
        } else {
            0
        };
        Self {
            config,
            tt: RefCell::new(DeterministicTt::with_capacity(tt_capacity)),
            vnet: None,
        }
    }

    /// Creates a deterministic solver with an explicitly supplied V-Net evaluator.
    pub fn new_with_vnet_evaluator(
        config: DeterministicSearchConfig,
        vnet: Option<VNetEvaluator>,
    ) -> Self {
        let tt_capacity = if config.tt.enabled {
            config.tt.capacity
        } else {
            0
        };
        Self {
            config,
            tt: RefCell::new(DeterministicTt::with_capacity(tt_capacity)),
            vnet,
        }
    }

    /// Creates a deterministic solver and attempts to load a configured V-Net.
    ///
    /// Loading failures are recorded and leaf evaluation falls back to the
    /// existing heuristic. Call [`VNetEvaluator::load`](crate::ml::VNetEvaluator::load)
    /// directly when a hard load error is desired.
    pub fn new_with_vnet_config(
        config: DeterministicSearchConfig,
        vnet_config: &VNetInferenceConfig,
    ) -> Self {
        let requested =
            config.leaf_eval_mode == LeafEvaluationMode::VNet && vnet_config.enable_vnet;
        let loaded = if requested {
            VNetEvaluator::load(vnet_config).ok()
        } else {
            None
        };
        Self::new_with_vnet_evaluator(config, loaded)
    }

    /// Clears all deterministic TT entries held by this solver.
    pub fn clear_tt(&self) {
        let capacity = self.tt.borrow().capacity();
        self.tt.replace(DeterministicTt::with_capacity(capacity));
    }

    /// Attempts to prove a full deterministic state as win/loss.
    pub fn solve_exact(&self, full_state: &FullState) -> SolverResult<ExactSolveResult> {
        let mut state = full_state.clone();
        state.validate_consistency()?;
        let mut context = SearchContext::new(self.config, SolveMode::Exact);
        let depth_budget = self.config.budget.depth_budget.unwrap_or(u16::MAX);
        let result = self.search(&mut state, depth_budget, 0, &mut context)?;
        context.finish();

        Ok(ExactSolveResult {
            outcome: result.outcome,
            best_move: result.best_move,
            principal_line: result.principal_line,
            value: result.value,
            stats: context.stats,
        })
    }

    /// Searches a bounded horizon and returns a best effort value.
    pub fn solve_bounded(&self, full_state: &FullState) -> SolverResult<BoundedSolveResult> {
        let mut state = full_state.clone();
        state.validate_consistency()?;
        let mut context = SearchContext::new(self.config, SolveMode::Bounded);
        let depth_budget = self.config.budget.depth_budget.unwrap_or(8);
        let result = self.search(&mut state, depth_budget, 0, &mut context)?;
        context.finish();

        Ok(BoundedSolveResult {
            outcome: result.outcome,
            best_move: result.best_move,
            principal_line: result.principal_line,
            estimated_value: result.value,
            stats: context.stats,
        })
    }

    /// Returns a fast deterministic value estimate.
    pub fn evaluate_fast(&self, full_state: &FullState) -> SolverResult<FastEvalResult> {
        full_state.validate_consistency()?;
        let start = Instant::now();
        let mut stats = DeterministicSearchStats {
            nodes_expanded: 1,
            ..DeterministicSearchStats::default()
        };
        let key = DeterministicHashKey::from_full_state(full_state);
        if let Some(entry) = self.probe_tt(key, 0, SolveMode::FastEvaluate, &mut stats) {
            stats.elapsed_ms = start.elapsed().as_millis() as u64;
            return Ok(FastEvalResult {
                value: entry.value.value,
                best_move: entry.best_move,
                stats,
            });
        }

        let value = self.evaluate_leaf_value(full_state, &mut stats);

        let mut best_move = None;
        let mut best_value = f32::NEG_INFINITY;
        let mut child = full_state.clone();
        for macro_move in ordered_macro_moves_with_hint(&full_state.visible, self.config, None) {
            if let Ok(transition) = apply_atomic_move_full_state(&mut child, macro_move.atomic) {
                let child_value = self.evaluate_leaf_value(&child, &mut stats);
                undo_atomic_move_full_state(&mut child, transition.undo)?;
                if child_value > best_value {
                    best_value = child_value;
                    best_move = Some(macro_move);
                }
            } else {
                child = full_state.clone();
            }
        }
        self.store_tt(
            DeterministicTtEntry {
                key,
                value: DeterministicTtValue {
                    outcome: SolveOutcome::Unknown,
                    value,
                    bound: DeterministicBound::Approx,
                },
                best_move: best_move.clone(),
                searched_depth: 0,
                age: 0,
            },
            &mut stats,
        );
        stats.elapsed_ms = start.elapsed().as_millis() as u64;

        Ok(FastEvalResult {
            value,
            best_move,
            stats,
        })
    }

    /// Returns the bounded-search recommendation surface for open-card play.
    pub fn best_move_open(&self, full_state: &FullState) -> SolverResult<OpenCardRecommendation> {
        let result = self.solve_bounded(full_state)?;
        Ok(OpenCardRecommendation {
            best_move: result.best_move,
            value: result.estimated_value,
            stats: result.stats,
        })
    }

    fn search(
        &self,
        state: &mut FullState,
        depth_remaining: u16,
        depth: u16,
        context: &mut SearchContext,
    ) -> SolverResult<NodeSearchResult> {
        if !context.enter_node(depth) {
            return Ok(NodeSearchResult::unknown(
                self.evaluate_leaf_value(state, &mut context.stats),
            ));
        }

        let key = DeterministicHashKey::from_full_state(state);
        if let Some(entry) = self.probe_tt(key, depth_remaining, context.mode, &mut context.stats) {
            if let Some(result) = entry_to_node_result(&entry, depth_remaining, context.mode) {
                return Ok(result);
            }
        }

        let closure_engine = ClosureEngine::new(self.config.closure);
        let closure_run = closure_engine.run_full_state_with_undos(state);
        context.stats.closure_steps_applied += closure_run.result.steps as u64;

        let mut result = match closure_run.result.stop_reason {
            ClosureStopReason::TerminalWin => NodeSearchResult::win(),
            ClosureStopReason::TerminalNoMoves => NodeSearchResult::loss(),
            _ if depth_remaining == 0 => {
                context.stats.budget_cutoffs += 1;
                NodeSearchResult::unknown(self.evaluate_leaf_value(state, &mut context.stats))
            }
            _ => self.search_after_closure(state, depth_remaining, depth, context)?,
        };

        prefix_closure_line(&closure_run, &mut result);
        undo_full_closure(state, closure_run.undos)?;
        self.store_search_result(key, depth_remaining, &result, &mut context.stats);
        Ok(result)
    }

    fn search_after_closure(
        &self,
        state: &mut FullState,
        depth_remaining: u16,
        depth: u16,
        context: &mut SearchContext,
    ) -> SolverResult<NodeSearchResult> {
        let tt_hint = self.tt_best_move_hint(state, context);
        let moves = ordered_macro_moves_with_hint(&state.visible, self.config, tt_hint.as_ref());
        if moves.is_empty() {
            return Ok(NodeSearchResult::loss());
        }

        let mut best =
            NodeSearchResult::unknown(self.evaluate_leaf_value(state, &mut context.stats));
        let mut all_children_proven_loss = true;
        let mut searched_all_children = true;

        for macro_move in moves {
            if context.is_budget_exhausted() {
                context.stats.budget_cutoffs += 1;
                searched_all_children = false;
                break;
            }

            let transition = apply_atomic_move_full_state(state, macro_move.atomic)?;
            context.stats.branches_considered += 1;
            let mut child = self.search(state, depth_remaining - 1, depth + 1, context)?;
            undo_atomic_move_full_state(state, transition.undo)?;

            if child.outcome == SolveOutcome::ProvenWin {
                context.stats.exact_wins_found += 1;
                child.best_move = Some(macro_move.clone());
                child.principal_line.insert(0, macro_move.kind);
                return Ok(child);
            }

            if child.outcome != SolveOutcome::ProvenLoss {
                all_children_proven_loss = false;
            }

            if child.value > best.value || best.best_move.is_none() {
                child.best_move = Some(macro_move.clone());
                child.principal_line.insert(0, macro_move.kind);
                best = child;
            }
        }

        if searched_all_children && all_children_proven_loss {
            Ok(NodeSearchResult::loss())
        } else {
            Ok(best)
        }
    }

    fn probe_tt(
        &self,
        key: DeterministicHashKey,
        depth_remaining: u16,
        mode: SolveMode,
        stats: &mut DeterministicSearchStats,
    ) -> Option<DeterministicTtEntry> {
        if !self.config.tt.enabled {
            return None;
        }

        stats.tt_probes += 1;
        let entry = self.tt.borrow().probe(key).cloned()?;
        stats.tt_hits += 1;
        match entry.value.bound {
            DeterministicBound::Exact => stats.tt_exact_hits += 1,
            DeterministicBound::Approx => {
                if mode != SolveMode::Exact && entry.searched_depth >= depth_remaining {
                    stats.tt_approx_hits += 1;
                }
            }
            DeterministicBound::Lower | DeterministicBound::Upper => {}
        }
        Some(entry)
    }

    fn tt_best_move_hint(
        &self,
        state: &FullState,
        context: &mut SearchContext,
    ) -> Option<MacroMove> {
        if !self.config.tt.enabled {
            return None;
        }

        let key = DeterministicHashKey::from_full_state(state);
        context.stats.tt_probes += 1;
        let entry = self.tt.borrow().probe(key).cloned()?;
        context.stats.tt_hits += 1;
        if entry.value.bound == DeterministicBound::Approx {
            context.stats.tt_approx_hits += 1;
        }
        entry.best_move
    }

    fn store_search_result(
        &self,
        key: DeterministicHashKey,
        depth_remaining: u16,
        result: &NodeSearchResult,
        stats: &mut DeterministicSearchStats,
    ) {
        let bound = match result.outcome {
            SolveOutcome::ProvenWin | SolveOutcome::ProvenLoss => DeterministicBound::Exact,
            SolveOutcome::Unknown => DeterministicBound::Approx,
        };
        if bound == DeterministicBound::Approx && !self.config.tt.store_approx {
            return;
        }

        self.store_tt(
            DeterministicTtEntry {
                key,
                value: DeterministicTtValue {
                    outcome: result.outcome,
                    value: result.value,
                    bound,
                },
                best_move: result.best_move.clone(),
                searched_depth: depth_remaining,
                age: 0,
            },
            stats,
        );
    }

    fn store_tt(&self, entry: DeterministicTtEntry, stats: &mut DeterministicSearchStats) {
        if !self.config.tt.enabled {
            return;
        }

        let report = self.tt.borrow_mut().store(entry);
        if report.stored {
            stats.tt_stores += 1;
        }
        if report.replaced {
            stats.tt_replacements += 1;
        }
        if report.collision {
            stats.tt_collisions += 1;
        }
    }

    fn evaluate_leaf_value(
        &self,
        full_state: &FullState,
        stats: &mut DeterministicSearchStats,
    ) -> f32 {
        if full_state.visible.is_structural_win() {
            return 1.0;
        }

        if self.config.leaf_eval_mode == LeafEvaluationMode::VNet {
            if let Some(vnet) = &self.vnet {
                let started = Instant::now();
                match vnet.evaluate_full_state(full_state) {
                    Ok(value) => {
                        stats.vnet_inferences = stats.vnet_inferences.saturating_add(1);
                        stats.vnet_inference_elapsed_us = stats
                            .vnet_inference_elapsed_us
                            .saturating_add(started.elapsed().as_micros() as u64);
                        return value;
                    }
                    Err(_) => {
                        stats.vnet_fallbacks = stats.vnet_fallbacks.saturating_add(1);
                    }
                }
            } else {
                stats.vnet_fallbacks = stats.vnet_fallbacks.saturating_add(1);
            }
        }

        evaluate_state(&full_state.visible, self.config.evaluator_weights)
    }
}

impl Default for DeterministicSolver {
    fn default() -> Self {
        Self::new(DeterministicSearchConfig::default())
    }
}

/// Attempts an exact deterministic solve with the supplied config.
pub fn solve_exact(
    full_state: &FullState,
    config: DeterministicSearchConfig,
) -> SolverResult<ExactSolveResult> {
    DeterministicSolver::new(config).solve_exact(full_state)
}

/// Runs bounded deterministic search with the supplied config.
pub fn solve_bounded(
    full_state: &FullState,
    config: DeterministicSearchConfig,
) -> SolverResult<BoundedSolveResult> {
    DeterministicSolver::new(config).solve_bounded(full_state)
}

/// Runs fast deterministic evaluation with the supplied config.
pub fn evaluate_fast(
    full_state: &FullState,
    config: DeterministicSearchConfig,
) -> SolverResult<FastEvalResult> {
    DeterministicSolver::new(config).evaluate_fast(full_state)
}

/// Returns ordered macro moves using the deterministic solver's current policy.
pub fn ordered_macro_moves(
    state: &VisibleState,
    config: DeterministicSearchConfig,
) -> Vec<MacroMove> {
    ordered_macro_moves_with_hint(state, config, None)
}

/// Returns ordered macro moves with an optional TT best-move hint first.
pub fn ordered_macro_moves_with_hint(
    state: &VisibleState,
    config: DeterministicSearchConfig,
    tt_best_move: Option<&MacroMove>,
) -> Vec<MacroMove> {
    let mut moves = generate_legal_macro_moves_with_config(
        state,
        MoveGenerationConfig {
            allow_foundation_retreats: config.allow_foundation_retreats,
        },
    );
    moves.sort_by_key(|macro_move| {
        (
            tt_best_move.is_none_or(|hint| !same_macro_action(hint, macro_move)) as u8,
            move_order_bucket(state, macro_move),
            macro_move.kind,
            macro_move.id,
        )
    });
    moves
}

fn same_macro_action(left: &MacroMove, right: &MacroMove) -> bool {
    left.atomic == right.atomic && left.kind == right.kind
}

fn entry_to_node_result(
    entry: &DeterministicTtEntry,
    depth_remaining: u16,
    mode: SolveMode,
) -> Option<NodeSearchResult> {
    match mode {
        SolveMode::Exact => entry.exact_result(),
        SolveMode::Bounded | SolveMode::FastEvaluate => entry.approximate_result(depth_remaining),
    }
}

fn should_replace_tt_entry(
    existing: &DeterministicTtEntry,
    incoming: &DeterministicTtEntry,
) -> bool {
    if existing.key == incoming.key {
        return incoming.replacement_priority() >= existing.replacement_priority();
    }
    incoming.replacement_priority() >= existing.replacement_priority()
}

fn move_order_bucket(state: &VisibleState, macro_move: &MacroMove) -> u8 {
    if macro_move.semantics.causes_reveal {
        return 0;
    }
    if macro_move.semantics.moves_to_foundation
        && foundation_move_card(state, macro_move)
            .is_some_and(|card| matches!(card.rank(), Rank::Ace | Rank::Two))
    {
        return 1;
    }
    if macro_move.semantics.creates_empty_column || macro_move.semantics.fills_empty_column {
        return 2;
    }
    if matches!(macro_move.kind, MacroMoveKind::PlayWasteToTableau { .. }) {
        return 3;
    }
    if matches!(
        macro_move.kind,
        MacroMoveKind::MoveRun { .. } | MacroMoveKind::PlaceKingRun { .. }
    ) {
        return 4;
    }
    if macro_move.semantics.affects_stock_cycle {
        return 5;
    }
    if macro_move.semantics.moves_from_foundation {
        return 6;
    }
    7
}

fn foundation_move_card(state: &VisibleState, macro_move: &MacroMove) -> Option<Card> {
    match macro_move.atomic {
        AtomicMove::WasteToFoundation => state.stock.accessible_card(),
        AtomicMove::TableauToFoundation { src } => {
            state.columns[usize::from(src.index())].top_face_up()
        }
        _ => None,
    }
}

fn evaluate_state(state: &VisibleState, weights: EvaluatorWeights) -> f32 {
    if state.is_structural_win() {
        return 1.0;
    }

    let foundation = state.foundations.card_count() as f32 / Card::COUNT as f32;
    let hidden_revealed = 1.0 - (state.hidden_slot_count().min(21) as f32 / 21.0);
    let empty_columns = state
        .columns
        .iter()
        .filter(|column| column.is_empty())
        .count() as f32
        / crate::types::TABLEAU_COLUMN_COUNT as f32;
    let mobility = (generate_legal_macro_moves_with_config(
        state,
        MoveGenerationConfig {
            allow_foundation_retreats: true,
        },
    )
    .len()
    .min(20) as f32)
        / 20.0;
    let waste_access = f32::from(state.stock.accessible_card().is_some());

    (foundation * weights.foundation_progress
        + hidden_revealed * weights.hidden_cards_revealed
        + empty_columns * weights.empty_columns
        + mobility * weights.mobility
        + waste_access * weights.waste_access)
        .clamp(0.0, 1.0)
}

fn undo_full_closure(state: &mut FullState, undos: Vec<FullStateMoveUndo>) -> SolverResult<()> {
    for undo in undos.into_iter().rev() {
        undo_atomic_move_full_state(state, undo)?;
    }
    Ok(())
}

fn prefix_closure_line(closure_run: &FullClosureRun, result: &mut NodeSearchResult) {
    if closure_run.result.transcript.is_empty() {
        return;
    }

    let mut prefix: Vec<MacroMoveKind> = closure_run
        .result
        .transcript
        .iter()
        .map(|step| step.macro_move.kind)
        .collect();
    prefix.extend(result.principal_line.iter().copied());
    result.principal_line = prefix;
    result.best_move = closure_run
        .result
        .transcript
        .steps
        .first()
        .map(|step| step.macro_move.clone());
}

#[derive(Debug, Clone, PartialEq)]
struct NodeSearchResult {
    outcome: SolveOutcome,
    value: f32,
    best_move: Option<MacroMove>,
    principal_line: Vec<MacroMoveKind>,
}

impl NodeSearchResult {
    fn win() -> Self {
        Self {
            outcome: SolveOutcome::ProvenWin,
            value: 1.0,
            best_move: None,
            principal_line: Vec::new(),
        }
    }

    fn loss() -> Self {
        Self {
            outcome: SolveOutcome::ProvenLoss,
            value: 0.0,
            best_move: None,
            principal_line: Vec::new(),
        }
    }

    fn unknown(value: f32) -> Self {
        Self {
            outcome: SolveOutcome::Unknown,
            value,
            best_move: None,
            principal_line: Vec::new(),
        }
    }
}

#[derive(Debug)]
struct SearchContext {
    config: DeterministicSearchConfig,
    mode: SolveMode,
    stats: DeterministicSearchStats,
    start: Instant,
    deadline: Option<Instant>,
}

impl SearchContext {
    fn new(config: DeterministicSearchConfig, mode: SolveMode) -> Self {
        let start = Instant::now();
        let deadline = config
            .budget
            .wall_clock_limit_ms
            .map(|millis| start + Duration::from_millis(millis));
        Self {
            config,
            mode,
            stats: DeterministicSearchStats::default(),
            start,
            deadline,
        }
    }

    fn enter_node(&mut self, depth: u16) -> bool {
        if self.is_budget_exhausted() {
            self.stats.budget_cutoffs += 1;
            return false;
        }

        self.stats.nodes_expanded += 1;
        self.stats.max_depth_reached = self.stats.max_depth_reached.max(depth);
        true
    }

    fn is_budget_exhausted(&self) -> bool {
        if self
            .config
            .budget
            .node_budget
            .is_some_and(|limit| self.stats.nodes_expanded >= limit)
        {
            return true;
        }
        if self
            .deadline
            .is_some_and(|deadline| Instant::now() >= deadline)
        {
            return true;
        }
        false
    }

    fn finish(&mut self) {
        self.stats.elapsed_ms = self.start.elapsed().as_millis() as u64;
    }
}

#[derive(Debug, Copy, Clone)]
struct StableHashBuilder {
    primary: u64,
    secondary: u64,
}

impl StableHashBuilder {
    fn new() -> Self {
        Self {
            primary: 0xcbf2_9ce4_8422_2325,
            secondary: 0x9e37_79b9_7f4a_7c15,
        }
    }

    fn add_tag(&mut self, tag: u8) {
        self.add_u64(u64::from(tag));
    }

    fn add_card(&mut self, card: Card) {
        self.add_u8(card.index());
    }

    fn add_u8(&mut self, value: u8) {
        self.add_u64(u64::from(value));
    }

    fn add_usize(&mut self, value: usize) {
        self.add_u64(value as u64);
    }

    fn add_option_usize(&mut self, value: Option<usize>) {
        match value {
            Some(value) => {
                self.add_tag(1);
                self.add_usize(value);
            }
            None => self.add_tag(0),
        }
    }

    fn add_option_u32(&mut self, value: Option<u32>) {
        match value {
            Some(value) => {
                self.add_tag(1);
                self.add_u64(u64::from(value));
            }
            None => self.add_tag(0),
        }
    }

    fn add_u64(&mut self, value: u64) {
        self.primary ^= value.wrapping_add(0x9e37_79b9_7f4a_7c15);
        self.primary = self.primary.wrapping_mul(0x1000_0000_01b3).rotate_left(13);

        self.secondary = self
            .secondary
            .wrapping_add(value ^ 0xa076_1d64_78bd_642f)
            .rotate_left(27)
            .wrapping_mul(0xe703_7ed1_a0b4_28db);
    }

    fn finish(self) -> DeterministicHashKey {
        DeterministicHashKey {
            primary: self.primary,
            secondary: self.secondary,
        }
    }
}

#[cfg(test)]
mod tests;
