//! PIMC baseline over uniform determinizations.

use std::time::Instant;

use serde::{Deserialize, Serialize};

use crate::{
    belief::{sample_full_states, validate_sample_against_belief, DeterminizationSample},
    core::{BeliefState, FullState},
    deterministic_solver::{
        ordered_macro_moves, DeterministicSearchConfig, DeterministicSolver, FastEvalResult,
        SolveOutcome,
    },
    error::{SolverError, SolverResult},
    ml::VNetInferenceConfig,
    moves::{apply_atomic_move_full_state, MacroMove},
    types::DealSeed,
};

use super::{action_seed, deterministic_config_with_override};
/// Deterministic continuation mode used by the PIMC baseline.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PimcEvaluationMode {
    /// Use proof-oriented deterministic solve attempts.
    Exact,
    /// Use bounded deterministic solve mode.
    Bounded,
    /// Use fast deterministic evaluation.
    Fast,
}

/// Configuration for the PIMC hidden-information baseline.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PimcConfig {
    /// Number of uniform determinizations to sample for non-trivial root decisions.
    pub sample_count: usize,
    /// Deterministic solver mode used after applying a candidate root action.
    pub deterministic_mode: PimcEvaluationMode,
    /// Optional per-world node-budget override for deterministic continuation.
    pub per_world_node_budget_override: Option<u64>,
    /// Whether all root actions are evaluated on the same sampled worlds.
    pub shared_world_batch: bool,
    /// Reproducible sampler seed.
    pub rng_seed: DealSeed,
    /// Optional cap on root candidate actions after deterministic move ordering.
    pub max_candidate_actions: Option<usize>,
    /// Whether standard error should be populated in action stats.
    pub report_standard_error: bool,
}

impl Default for PimcConfig {
    fn default() -> Self {
        Self {
            sample_count: 32,
            deterministic_mode: PimcEvaluationMode::Bounded,
            per_world_node_budget_override: None,
            shared_world_batch: true,
            rng_seed: DealSeed(0),
            max_candidate_actions: None,
            report_standard_error: true,
        }
    }
}

/// One batch of sampled worlds for a root PIMC decision.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PimcWorldBatch {
    /// Seed used to produce the batch.
    pub seed: DealSeed,
    /// Sampled full worlds in deterministic sample order.
    pub samples: Vec<DeterminizationSample>,
}

impl PimcWorldBatch {
    /// Samples a uniform batch from a belief state.
    pub fn sample(belief: &BeliefState, sample_count: usize, seed: DealSeed) -> SolverResult<Self> {
        Ok(Self {
            seed,
            samples: sample_full_states(belief, sample_count, seed)?,
        })
    }
}

/// Running statistics for one root action under PIMC.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PimcActionStats {
    /// Root action evaluated.
    pub action: MacroMove,
    /// Number of sampled worlds evaluated.
    pub visits: usize,
    /// Mean action value over sampled worlds.
    pub mean_value: f64,
    /// Running second central moment for variance.
    pub m2: f64,
    /// Sample variance, if at least two visits exist.
    pub variance: f64,
    /// Standard error of the mean.
    pub standard_error: f64,
    /// Number of deterministic continuations that proved win.
    pub exact_wins: usize,
    /// Number of deterministic continuations that proved loss.
    pub exact_losses: usize,
    /// Total deterministic solver nodes used for this action.
    pub deterministic_nodes: u64,
    /// V-Net inference calls used for this action.
    pub vnet_inferences: u64,
    /// V-Net fallback count for this action.
    pub vnet_fallbacks: u64,
    /// V-Net inference time in microseconds for this action.
    pub vnet_inference_elapsed_us: u64,
}

impl PimcActionStats {
    /// Creates empty stats for an action.
    pub fn new(action: MacroMove) -> Self {
        Self {
            action,
            visits: 0,
            mean_value: 0.0,
            m2: 0.0,
            variance: 0.0,
            standard_error: 0.0,
            exact_wins: 0,
            exact_losses: 0,
            deterministic_nodes: 0,
            vnet_inferences: 0,
            vnet_fallbacks: 0,
            vnet_inference_elapsed_us: 0,
        }
    }

    pub(crate) fn record(&mut self, value: PimcActionValue) {
        self.visits += 1;
        let scalar = f64::from(value.value);
        let delta = scalar - self.mean_value;
        self.mean_value += delta / self.visits as f64;
        let delta2 = scalar - self.mean_value;
        self.m2 += delta * delta2;
        self.variance = if self.visits > 1 {
            self.m2 / (self.visits - 1) as f64
        } else {
            0.0
        };
        self.standard_error = if self.visits > 1 {
            (self.variance / self.visits as f64).sqrt()
        } else {
            0.0
        };
        if value.outcome == SolveOutcome::ProvenWin {
            self.exact_wins += 1;
        }
        if value.outcome == SolveOutcome::ProvenLoss {
            self.exact_losses += 1;
        }
        self.deterministic_nodes += value.deterministic_nodes;
        self.vnet_inferences += value.vnet_inferences;
        self.vnet_fallbacks += value.vnet_fallbacks;
        self.vnet_inference_elapsed_us += value.vnet_inference_elapsed_us;
    }
}

/// Value returned by evaluating one action in one sampled world.
#[derive(Debug, Copy, Clone, PartialEq, Serialize, Deserialize)]
pub struct PimcActionValue {
    /// Scalar action value in 0..=1.
    pub value: f32,
    /// Deterministic continuation status.
    pub outcome: SolveOutcome,
    /// Deterministic solver nodes used.
    pub deterministic_nodes: u64,
    /// V-Net inference calls used.
    pub vnet_inferences: u64,
    /// V-Net fallback count.
    pub vnet_fallbacks: u64,
    /// V-Net inference time in microseconds.
    pub vnet_inference_elapsed_us: u64,
}

/// PIMC baseline recommendation for one belief root.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PimcRecommendation {
    /// Best root action by mean sampled value.
    pub best_move: Option<MacroMove>,
    /// Best root action mean value.
    pub best_value: f64,
    /// Per-action sampled statistics.
    pub action_stats: Vec<PimcActionStats>,
    /// Shared sampled world batch, when one was used.
    pub world_batch: Option<PimcWorldBatch>,
    /// Number of sampled worlds per action.
    pub sample_count: usize,
    /// Whether the recommendation used one shared batch for all actions.
    pub shared_world_batch: bool,
    /// Whether the root had zero legal moves.
    pub no_legal_moves: bool,
    /// Elapsed recommendation time in milliseconds.
    pub elapsed_ms: u64,
    /// Total deterministic continuation nodes used.
    pub deterministic_nodes: u64,
    /// Total V-Net inference calls used.
    pub vnet_inferences: u64,
    /// Total V-Net fallback count.
    pub vnet_fallbacks: u64,
    /// Total V-Net inference time in microseconds.
    pub vnet_inference_elapsed_us: u64,
}

/// PIMC baseline evaluator.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PimcEvaluator {
    /// PIMC controls.
    pub pimc_config: PimcConfig,
    /// Deterministic continuation solver controls.
    pub deterministic_config: DeterministicSearchConfig,
    /// Optional V-Net inference controls.
    pub vnet_inference: VNetInferenceConfig,
}

impl PimcEvaluator {
    /// Creates a PIMC evaluator.
    pub fn new(pimc_config: PimcConfig, deterministic_config: DeterministicSearchConfig) -> Self {
        Self::new_with_vnet(
            pimc_config,
            deterministic_config,
            VNetInferenceConfig::default(),
        )
    }

    /// Creates a PIMC evaluator with optional V-Net inference controls.
    pub fn new_with_vnet(
        pimc_config: PimcConfig,
        deterministic_config: DeterministicSearchConfig,
        vnet_inference: VNetInferenceConfig,
    ) -> Self {
        Self {
            pimc_config,
            deterministic_config,
            vnet_inference,
        }
    }

    /// Recommends one move from a belief state using uniform determinizations.
    pub fn recommend(&self, belief: &BeliefState) -> SolverResult<PimcRecommendation> {
        recommend_move_pimc(belief, self.deterministic_config, self.pimc_config)
    }

    /// Evaluates one root action in one sampled full world.
    pub fn evaluate_action_in_world(
        &self,
        world: &FullState,
        action: &MacroMove,
        solver: &DeterministicSolver,
    ) -> SolverResult<PimcActionValue> {
        let mut child = world.clone();
        apply_atomic_move_full_state(&mut child, action.atomic)?;

        match self.pimc_config.deterministic_mode {
            PimcEvaluationMode::Exact => {
                let result = solver.solve_exact(&child)?;
                Ok(PimcActionValue {
                    value: result.value,
                    outcome: result.outcome,
                    deterministic_nodes: result.stats.nodes_expanded,
                    vnet_inferences: result.stats.vnet_inferences,
                    vnet_fallbacks: result.stats.vnet_fallbacks,
                    vnet_inference_elapsed_us: result.stats.vnet_inference_elapsed_us,
                })
            }
            PimcEvaluationMode::Bounded => {
                let result = solver.solve_bounded(&child)?;
                Ok(PimcActionValue {
                    value: result.estimated_value,
                    outcome: result.outcome,
                    deterministic_nodes: result.stats.nodes_expanded,
                    vnet_inferences: result.stats.vnet_inferences,
                    vnet_fallbacks: result.stats.vnet_fallbacks,
                    vnet_inference_elapsed_us: result.stats.vnet_inference_elapsed_us,
                })
            }
            PimcEvaluationMode::Fast => {
                let FastEvalResult { value, stats, .. } = solver.evaluate_fast(&child)?;
                Ok(PimcActionValue {
                    value,
                    outcome: SolveOutcome::Unknown,
                    deterministic_nodes: stats.nodes_expanded,
                    vnet_inferences: stats.vnet_inferences,
                    vnet_fallbacks: stats.vnet_fallbacks,
                    vnet_inference_elapsed_us: stats.vnet_inference_elapsed_us,
                })
            }
        }
    }
}

/// Recommends a root move with the PIMC baseline.
pub fn recommend_move_pimc(
    belief: &BeliefState,
    deterministic_config: DeterministicSearchConfig,
    pimc_config: PimcConfig,
) -> SolverResult<PimcRecommendation> {
    recommend_move_pimc_with_vnet(
        belief,
        deterministic_config,
        pimc_config,
        VNetInferenceConfig::default(),
    )
}

/// Recommends a root move with the PIMC baseline and optional V-Net leaf evaluator.
pub fn recommend_move_pimc_with_vnet(
    belief: &BeliefState,
    deterministic_config: DeterministicSearchConfig,
    pimc_config: PimcConfig,
    vnet_inference: VNetInferenceConfig,
) -> SolverResult<PimcRecommendation> {
    let started = Instant::now();
    belief.validate_consistency_against_visible()?;
    let evaluator = PimcEvaluator::new_with_vnet(
        pimc_config,
        deterministic_config_with_override(deterministic_config, pimc_config),
        vnet_inference,
    );
    let mut candidates = ordered_macro_moves(&belief.visible, evaluator.deterministic_config);
    if let Some(limit) = pimc_config.max_candidate_actions {
        candidates.truncate(limit);
    }

    if candidates.is_empty() {
        return Ok(PimcRecommendation {
            best_move: None,
            best_value: 0.0,
            action_stats: Vec::new(),
            world_batch: None,
            sample_count: 0,
            shared_world_batch: pimc_config.shared_world_batch,
            no_legal_moves: true,
            elapsed_ms: started.elapsed().as_millis() as u64,
            deterministic_nodes: 0,
            vnet_inferences: 0,
            vnet_fallbacks: 0,
            vnet_inference_elapsed_us: 0,
        });
    }

    if candidates.len() == 1 {
        return Ok(PimcRecommendation {
            best_move: candidates.first().cloned(),
            best_value: 1.0,
            action_stats: vec![PimcActionStats::new(candidates[0].clone())],
            world_batch: None,
            sample_count: 0,
            shared_world_batch: pimc_config.shared_world_batch,
            no_legal_moves: false,
            elapsed_ms: started.elapsed().as_millis() as u64,
            deterministic_nodes: 0,
            vnet_inferences: 0,
            vnet_fallbacks: 0,
            vnet_inference_elapsed_us: 0,
        });
    }

    if pimc_config.sample_count == 0 {
        return Err(SolverError::InvalidState(
            "PIMC sample_count must be greater than zero when multiple actions exist".to_string(),
        ));
    }

    let shared_batch = if pimc_config.shared_world_batch {
        Some(PimcWorldBatch::sample(
            belief,
            pimc_config.sample_count,
            pimc_config.rng_seed,
        )?)
    } else {
        None
    };

    let solver = DeterministicSolver::new_with_vnet_config(
        evaluator.deterministic_config,
        &evaluator.vnet_inference,
    );
    let mut action_stats = Vec::with_capacity(candidates.len());
    let mut deterministic_nodes = 0u64;
    let mut vnet_inferences = 0u64;
    let mut vnet_fallbacks = 0u64;
    let mut vnet_inference_elapsed_us = 0u64;

    for (action_index, action) in candidates.into_iter().enumerate() {
        let owned_batch;
        let batch = match &shared_batch {
            Some(batch) => batch,
            None => {
                owned_batch = PimcWorldBatch::sample(
                    belief,
                    pimc_config.sample_count,
                    action_seed(pimc_config.rng_seed, action_index),
                )?;
                &owned_batch
            }
        };

        let mut stats = PimcActionStats::new(action.clone());
        for sample in &batch.samples {
            validate_sample_against_belief(&sample.full_state, belief)?;
            let value = evaluator.evaluate_action_in_world(&sample.full_state, &action, &solver)?;
            deterministic_nodes += value.deterministic_nodes;
            vnet_inferences += value.vnet_inferences;
            vnet_fallbacks += value.vnet_fallbacks;
            vnet_inference_elapsed_us += value.vnet_inference_elapsed_us;
            stats.record(value);
        }
        if !pimc_config.report_standard_error {
            stats.standard_error = 0.0;
        }
        action_stats.push(stats);
    }

    action_stats.sort_by(|left, right| {
        right
            .mean_value
            .total_cmp(&left.mean_value)
            .then_with(|| left.action.kind.cmp(&right.action.kind))
            .then_with(|| left.action.id.cmp(&right.action.id))
    });

    let best_move = action_stats.first().map(|stats| stats.action.clone());
    let best_value = action_stats
        .first()
        .map(|stats| stats.mean_value)
        .unwrap_or_default();

    Ok(PimcRecommendation {
        best_move,
        best_value,
        action_stats,
        world_batch: shared_batch,
        sample_count: pimc_config.sample_count,
        shared_world_batch: pimc_config.shared_world_batch,
        no_legal_moves: false,
        elapsed_ms: started.elapsed().as_millis() as u64,
        deterministic_nodes,
        vnet_inferences,
        vnet_fallbacks,
        vnet_inference_elapsed_us,
    })
}
