//! Machine-learning dataset export surfaces.
//!
//! This module prepares deterministic full-state examples for a future V-Net.
//! It does not contain neural inference, model training, PyTorch bindings, or
//! policy-network logic.

use std::{
    fs::{self, File},
    io::{BufWriter, Write},
    path::Path,
};

use serde::{Deserialize, Serialize};

use crate::{
    belief::{
        apply_observed_belief_move, belief_from_full_state, validate_belief_against_full_state,
    },
    cards::Card,
    core::FullState,
    deterministic_solver::{DeterministicSearchConfig, DeterministicSolver},
    error::{SolverError, SolverResult},
    experiments::{
        deterministic_search_config_from_solver, recommend_autoplay_move, BenchmarkSuite,
        BenchmarkSuiteDescription, ExperimentPreset, ExperimentRunner,
    },
    moves::{apply_atomic_move_full_state, MacroMoveKind},
    types::{DealSeed, TABLEAU_COLUMN_COUNT},
};

/// Future model role.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModelRole {
    /// Full-state deterministic value network.
    VNet,
    /// Visible belief-state policy/value prior network.
    PNet,
}

/// Dataset split marker for exported training examples.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum DatasetSplit {
    /// Training data.
    Train,
    /// Validation data.
    Validation,
    /// Frozen test data.
    Test,
}

/// High-level exported example kind.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum TrainingExampleKind {
    /// Full deterministic state labeled by eventual win/loss or value.
    DeterministicValue,
    /// Belief root state labeled by planner statistics.
    BeliefPolicy,
}

/// Shape metadata for encoded state tensors.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EncodedStateShape {
    /// Number of scalar features.
    pub feature_count: usize,
    /// Number of planes for future spatial encodings.
    pub plane_count: usize,
}

/// Explicit label source for V-Net examples.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum VNetLabelMode {
    /// Label every exported state by the final win/loss of the configured run.
    TerminalOutcome,
    /// Label each state with the deterministic open-card solver's bounded value.
    DeterministicSolverValue,
    /// Label each decision state with the hidden-information planner's root value.
    PlannerBackedApproximateValue,
}

/// Dataset file format supported by the v1 exporter.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum DatasetFormat {
    /// One JSON object per line. The first record is metadata, followed by examples.
    Jsonl,
}

/// Deterministic split strategy for exported examples.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum DatasetSplitStrategy {
    /// Put every example into the training split.
    AllTrain,
    /// Split by `seed % modulo`.
    SeedModulo {
        /// Modulus used for deterministic split assignment.
        modulo: u64,
        /// Remainder assigned to validation.
        validation_remainder: u64,
        /// Remainder assigned to test. Test takes precedence if remainders overlap.
        test_remainder: u64,
    },
}

impl DatasetSplitStrategy {
    /// Returns the split for a deal seed.
    pub fn split_for_seed(self, seed: DealSeed) -> SolverResult<DatasetSplit> {
        match self {
            Self::AllTrain => Ok(DatasetSplit::Train),
            Self::SeedModulo {
                modulo,
                validation_remainder,
                test_remainder,
            } => {
                if modulo == 0 {
                    return Err(SolverError::InvalidState(
                        "dataset split modulo must be greater than zero".to_string(),
                    ));
                }
                let remainder = seed.0 % modulo;
                if remainder == test_remainder % modulo {
                    Ok(DatasetSplit::Test)
                } else if remainder == validation_remainder % modulo {
                    Ok(DatasetSplit::Validation)
                } else {
                    Ok(DatasetSplit::Train)
                }
            }
        }
    }
}

impl Default for DatasetSplitStrategy {
    fn default() -> Self {
        Self::AllTrain
    }
}

/// Source subsystem that produced a dataset example.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum VNetDataSource {
    /// Full-game autoplay trace driven by a configured hidden-information backend.
    AutoplayTrace,
    /// Direct deterministic open-card solver labeling hook.
    DeterministicSolve,
    /// Uniform sampled full world from a belief state.
    PimcSampledWorld,
}

/// Dataset-level metadata written as the first JSONL record.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DatasetMetadata {
    /// Stable schema version for this exporter.
    pub dataset_version: String,
    /// Model role this dataset is intended for.
    pub model_role: ModelRole,
    /// Example kind.
    pub example_kind: TrainingExampleKind,
    /// Label mode used for every example.
    pub label_mode: VNetLabelMode,
    /// Source subsystem.
    pub source: VNetDataSource,
    /// Output format.
    pub format: DatasetFormat,
    /// Preset name used for collection.
    pub preset_name: String,
    /// Suite name used for collection.
    pub suite_name: String,
    /// Compact suite description.
    pub suite: BenchmarkSuiteDescription,
    /// Number of games/deals requested.
    pub games: usize,
    /// Number of examples written.
    pub example_count: usize,
}

/// Full deterministic state encoding for V-Net training.
///
/// The `card_locations` and `card_positions` vectors are fixed at 52 elements
/// and form the stable v1 numeric representation. Structured fields are included
/// to keep JSONL examples inspectable while the later tensor layout evolves.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VNetStateEncoding {
    /// Foundation top ranks by suit, encoded as 0 for empty and 1..=13 otherwise.
    pub foundation_tops: [u8; 4],
    /// Hidden tableau cards by column in hidden-slot order, encoded as 0..=51.
    pub tableau_hidden_cards: Vec<Vec<u8>>,
    /// Face-up tableau cards by column in visible order, encoded as 0..=51.
    pub tableau_face_up_cards: Vec<Vec<u8>>,
    /// Known stock/waste ring cards in exact order, encoded as 0..=51.
    pub stock_ring_cards: Vec<u8>,
    /// Number of cards currently in the stock prefix.
    pub stock_len: usize,
    /// Current accessible waste cursor index.
    pub stock_cursor: Option<usize>,
    /// Current draw-window accessibility depth.
    pub stock_accessible_depth: u8,
    /// Number of completed stock passes.
    pub stock_pass_index: u32,
    /// Optional pass cap.
    pub stock_max_passes: Option<u32>,
    /// Draw count, normally 3.
    pub stock_draw_count: u8,
    /// Per-card location code, indexed by card id.
    pub card_locations: Vec<u8>,
    /// Per-card local position/depth/order code, indexed by card id.
    pub card_positions: Vec<u8>,
    /// Stable numeric features for v1 training experiments.
    pub flat_features: Vec<i16>,
    /// Shape metadata for `flat_features`.
    pub shape: EncodedStateShape,
}

impl VNetStateEncoding {
    /// Encodes one full deterministic state.
    pub fn from_full_state(full_state: &FullState) -> SolverResult<Self> {
        full_state.validate_consistency()?;

        let mut card_locations = vec![LOCATION_UNSET; Card::COUNT];
        let mut card_positions = vec![0u8; Card::COUNT];

        for card in full_state.visible.foundations.iter_cards() {
            mark_card(
                &mut card_locations,
                &mut card_positions,
                card,
                LOCATION_FOUNDATION,
                card.rank().value(),
            )?;
        }

        for (ring_index, card) in full_state
            .visible
            .stock
            .ring_cards
            .iter()
            .copied()
            .enumerate()
        {
            mark_card(
                &mut card_locations,
                &mut card_positions,
                card,
                LOCATION_STOCK_RING,
                checked_u8(ring_index + 1, "stock ring position")?,
            )?;
        }

        let mut tableau_hidden_cards = vec![Vec::new(); TABLEAU_COLUMN_COUNT];
        for slot in full_state.visible.hidden_slots() {
            let card = full_state
                .hidden_assignments
                .card_for_slot(slot)
                .ok_or_else(|| {
                    SolverError::InvalidState(format!(
                        "missing hidden assignment for encoding slot {slot}"
                    ))
                })?;
            let column = usize::from(slot.column.index());
            tableau_hidden_cards[column].push(card.index());
            mark_card(
                &mut card_locations,
                &mut card_positions,
                card,
                LOCATION_TABLEAU_HIDDEN_BASE + slot.column.index(),
                slot.depth + 1,
            )?;
        }

        let mut tableau_face_up_cards = Vec::with_capacity(TABLEAU_COLUMN_COUNT);
        for (column_index, column) in full_state.visible.columns.iter().enumerate() {
            let mut encoded_column = Vec::with_capacity(column.face_up.len());
            for (face_up_index, card) in column.face_up.iter().copied().enumerate() {
                encoded_column.push(card.index());
                mark_card(
                    &mut card_locations,
                    &mut card_positions,
                    card,
                    LOCATION_TABLEAU_FACE_UP_BASE + column_index as u8,
                    checked_u8(face_up_index + 1, "tableau face-up position")?,
                )?;
            }
            tableau_face_up_cards.push(encoded_column);
        }

        if let Some(missing_card) = card_locations
            .iter()
            .position(|code| *code == LOCATION_UNSET)
        {
            return Err(SolverError::InvalidState(format!(
                "full-state encoding did not assign card index {missing_card}"
            )));
        }

        let foundation_tops = std::array::from_fn(|index| {
            full_state.visible.foundations.top_ranks[index].map_or(0, |rank| rank.value())
        });
        let stock = &full_state.visible.stock;
        let stock_ring_cards = stock.ring_cards.iter().map(|card| card.index()).collect();

        let mut flat_features = Vec::with_capacity(4 + 6 + Card::COUNT * 2);
        flat_features.extend(foundation_tops.iter().map(|value| i16::from(*value)));
        flat_features.push(clamp_i16(stock.stock_len));
        flat_features.push(clamp_i16(stock.cursor.map_or(0, |cursor| cursor + 1)));
        flat_features.push(i16::from(stock.accessible_depth));
        flat_features.push(clamp_i16(stock.pass_index as usize));
        flat_features.push(clamp_i16(
            stock.max_passes.map_or(0, |passes| passes as usize + 1),
        ));
        flat_features.push(i16::from(stock.draw_count));
        flat_features.extend(card_locations.iter().map(|value| i16::from(*value)));
        flat_features.extend(card_positions.iter().map(|value| i16::from(*value)));

        Ok(Self {
            foundation_tops,
            tableau_hidden_cards,
            tableau_face_up_cards,
            stock_ring_cards,
            stock_len: stock.stock_len,
            stock_cursor: stock.cursor,
            stock_accessible_depth: stock.accessible_depth,
            stock_pass_index: stock.pass_index,
            stock_max_passes: stock.max_passes,
            stock_draw_count: stock.draw_count,
            card_locations,
            card_positions,
            shape: EncodedStateShape {
                feature_count: flat_features.len(),
                plane_count: 0,
            },
            flat_features,
        })
    }
}

/// Provenance for one V-Net example.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VNetProvenance {
    /// Source subsystem.
    pub source: VNetDataSource,
    /// Preset used to produce this example.
    pub preset_name: String,
    /// Deal seed.
    pub deal_seed: DealSeed,
    /// Step index within autoplay, if applicable.
    pub step_index: Option<usize>,
    /// Root move selected at this state, if a decision was made.
    pub chosen_move: Option<MacroMoveKind>,
    /// Planner root value reported for this state, if available.
    pub planner_value: Option<f64>,
    /// Final win/loss of the configured run that produced this example.
    pub terminal_won: Option<bool>,
}

/// One supervised V-Net training example.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VNetExample {
    /// Stable deterministic example id.
    pub example_id: String,
    /// Dataset split.
    pub split: DatasetSplit,
    /// Label mode used for this example.
    pub label_mode: VNetLabelMode,
    /// Scalar target in 0..=1.
    pub label: f32,
    /// Encoded full deterministic state.
    pub encoded_state: VNetStateEncoding,
    /// Source/provenance metadata.
    pub provenance: VNetProvenance,
}

/// In-memory V-Net dataset ready for export.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VNetDataset {
    /// Dataset metadata.
    pub metadata: DatasetMetadata,
    /// Examples in deterministic suite/step order.
    pub examples: Vec<VNetExample>,
}

/// Export configuration for V-Net datasets.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VNetExportConfig {
    /// Label mode to use.
    pub label_mode: VNetLabelMode,
    /// Deterministic split assignment.
    pub split_strategy: DatasetSplitStrategy,
    /// Optional autoplay step cap override.
    pub max_steps: Option<usize>,
    /// Whether to include the final state reached by each autoplay run.
    pub include_terminal_state: bool,
    /// Output format.
    pub format: DatasetFormat,
}

impl Default for VNetExportConfig {
    fn default() -> Self {
        Self {
            label_mode: VNetLabelMode::TerminalOutcome,
            split_strategy: DatasetSplitStrategy::AllTrain,
            max_steps: None,
            include_terminal_state: true,
            format: DatasetFormat::Jsonl,
        }
    }
}

/// JSONL record variants emitted by [`VNetDatasetWriter`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "record_type", rename_all = "snake_case")]
pub enum VNetDatasetRecord {
    /// Dataset metadata record.
    Metadata {
        /// Metadata payload.
        metadata: DatasetMetadata,
    },
    /// Training example record.
    Example {
        /// Example payload.
        example: VNetExample,
    },
}

/// Dataset writer for V-Net exports.
#[derive(Debug, Copy, Clone, Default, PartialEq, Eq)]
pub struct VNetDatasetWriter;

impl VNetDatasetWriter {
    /// Writes a V-Net dataset as JSONL.
    pub fn write_jsonl(path: impl AsRef<Path>, dataset: &VNetDataset) -> SolverResult<()> {
        let path = path.as_ref();
        if let Some(parent) = path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
        {
            fs::create_dir_all(parent)?;
        }
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);
        write_jsonl_record(
            &mut writer,
            &VNetDatasetRecord::Metadata {
                metadata: dataset.metadata.clone(),
            },
        )?;
        for example in &dataset.examples {
            write_jsonl_record(
                &mut writer,
                &VNetDatasetRecord::Example {
                    example: example.clone(),
                },
            )?;
        }
        writer.flush()?;
        Ok(())
    }
}

/// Builds a V-Net example from a labeled full deterministic state.
pub fn vnet_example_from_full_state(
    full_state: &FullState,
    label_mode: VNetLabelMode,
    label: f32,
    provenance: VNetProvenance,
    split: DatasetSplit,
) -> SolverResult<VNetExample> {
    Ok(VNetExample {
        example_id: example_id(&provenance),
        split,
        label_mode,
        label: label.clamp(0.0, 1.0),
        encoded_state: VNetStateEncoding::from_full_state(full_state)?,
        provenance,
    })
}

/// Hook for deterministic open-card solver value labeling.
pub fn vnet_example_from_deterministic_solve(
    full_state: &FullState,
    deterministic_config: DeterministicSearchConfig,
    provenance: VNetProvenance,
    split: DatasetSplit,
) -> SolverResult<VNetExample> {
    let result = DeterministicSolver::new(deterministic_config).solve_bounded(full_state)?;
    vnet_example_from_full_state(
        full_state,
        VNetLabelMode::DeterministicSolverValue,
        result.estimated_value,
        provenance,
        split,
    )
}

/// Collects V-Net examples from full-game autoplay over a seeded suite.
pub fn collect_vnet_examples_from_autoplay_suite(
    suite: &BenchmarkSuite,
    preset: &ExperimentPreset,
    export_config: &VNetExportConfig,
) -> SolverResult<VNetDataset> {
    let mut autoplay_config = preset.autoplay;
    if let Some(max_steps) = export_config.max_steps {
        autoplay_config.max_steps = max_steps;
    }

    let deterministic_solver =
        DeterministicSolver::new(deterministic_search_config_from_solver(&preset.solver));
    let runner = ExperimentRunner;
    let mut pending = Vec::new();

    for seed in &suite.seeds {
        let deal = runner.generate_deal(*seed)?;
        let split = export_config.split_strategy.split_for_seed(*seed)?;
        pending.extend(collect_pending_autoplay_examples_for_deal(
            *seed,
            deal.full_state,
            &preset.name,
            &preset.solver,
            &autoplay_config,
            export_config.include_terminal_state,
            split,
        )?);
    }

    let mut examples = Vec::with_capacity(pending.len());
    for pending_example in pending {
        let label = label_pending_example(
            &pending_example,
            export_config.label_mode,
            &deterministic_solver,
        )?;
        examples.push(vnet_example_from_full_state(
            &pending_example.full_state,
            export_config.label_mode,
            label,
            pending_example.provenance,
            pending_example.split,
        )?);
    }

    let metadata = DatasetMetadata {
        dataset_version: "vnet-jsonl-v1".to_string(),
        model_role: ModelRole::VNet,
        example_kind: TrainingExampleKind::DeterministicValue,
        label_mode: export_config.label_mode,
        source: VNetDataSource::AutoplayTrace,
        format: export_config.format,
        preset_name: preset.name.clone(),
        suite_name: suite.name.clone(),
        suite: suite.description(),
        games: suite.seeds.len(),
        example_count: examples.len(),
    };

    Ok(VNetDataset { metadata, examples })
}

/// Collects and writes a V-Net autoplay dataset.
pub fn export_vnet_dataset_from_autoplay_suite(
    path: impl AsRef<Path>,
    suite: &BenchmarkSuite,
    preset: &ExperimentPreset,
    export_config: &VNetExportConfig,
) -> SolverResult<VNetDataset> {
    let dataset = collect_vnet_examples_from_autoplay_suite(suite, preset, export_config)?;
    match export_config.format {
        DatasetFormat::Jsonl => VNetDatasetWriter::write_jsonl(path, &dataset)?,
    }
    Ok(dataset)
}

#[derive(Debug, Clone, PartialEq)]
struct PendingVNetExample {
    full_state: FullState,
    provenance: VNetProvenance,
    split: DatasetSplit,
}

fn collect_pending_autoplay_examples_for_deal(
    seed: DealSeed,
    full_state: FullState,
    preset_name: &str,
    solver_config: &crate::config::SolverConfig,
    autoplay_config: &crate::experiments::AutoplayConfig,
    include_terminal_state: bool,
    split: DatasetSplit,
) -> SolverResult<Vec<PendingVNetExample>> {
    let mut true_state = full_state;
    let mut belief = belief_from_full_state(&true_state)?;
    let mut pending = Vec::new();
    let mut total_planner_time_ms = 0u64;

    for step_index in 0..autoplay_config.max_steps {
        if true_state.visible.is_structural_win() {
            break;
        }

        if autoplay_config
            .max_total_planner_time_ms
            .is_some_and(|limit| total_planner_time_ms >= limit)
        {
            break;
        }

        let decision = recommend_autoplay_move(
            &belief,
            solver_config,
            autoplay_config.backend,
            autoplay_config.pimc,
            step_index,
        )?;
        total_planner_time_ms = total_planner_time_ms.saturating_add(decision.snapshot.elapsed_ms);

        pending.push(PendingVNetExample {
            full_state: true_state.clone(),
            provenance: VNetProvenance {
                source: VNetDataSource::AutoplayTrace,
                preset_name: preset_name.to_string(),
                deal_seed: seed,
                step_index: Some(step_index),
                chosen_move: decision
                    .best_move
                    .as_ref()
                    .map(|macro_move| macro_move.kind),
                planner_value: Some(decision.snapshot.best_value),
                terminal_won: None,
            },
            split,
        });

        let Some(chosen_move) = decision.best_move else {
            break;
        };

        let true_transition = apply_atomic_move_full_state(&mut true_state, chosen_move.atomic)?;
        let revealed_card = true_transition.outcome.revealed.map(|reveal| reveal.card);
        let (next_belief, _observed_outcome) =
            apply_observed_belief_move(&belief, chosen_move.atomic, revealed_card)?;
        belief = next_belief;

        if autoplay_config.validate_each_step {
            validate_belief_against_full_state(&belief, &true_state)?;
        }
    }

    let terminal_won = true_state.visible.is_structural_win();
    if include_terminal_state {
        pending.push(PendingVNetExample {
            full_state: true_state,
            provenance: VNetProvenance {
                source: VNetDataSource::AutoplayTrace,
                preset_name: preset_name.to_string(),
                deal_seed: seed,
                step_index: None,
                chosen_move: None,
                planner_value: None,
                terminal_won: Some(terminal_won),
            },
            split,
        });
    }

    for example in &mut pending {
        example.provenance.terminal_won = Some(terminal_won);
    }

    Ok(pending)
}

fn label_pending_example(
    pending: &PendingVNetExample,
    label_mode: VNetLabelMode,
    deterministic_solver: &DeterministicSolver,
) -> SolverResult<f32> {
    match label_mode {
        VNetLabelMode::TerminalOutcome => Ok(if pending.provenance.terminal_won.unwrap_or(false) {
            1.0
        } else {
            0.0
        }),
        VNetLabelMode::DeterministicSolverValue => Ok(deterministic_solver
            .solve_bounded(&pending.full_state)?
            .estimated_value),
        VNetLabelMode::PlannerBackedApproximateValue => Ok(pending
            .provenance
            .planner_value
            .map(|value| value as f32)
            .unwrap_or_else(|| {
                if pending.provenance.terminal_won.unwrap_or(false) {
                    1.0
                } else {
                    0.0
                }
            })
            .clamp(0.0, 1.0)),
    }
}

fn write_jsonl_record(
    writer: &mut BufWriter<File>,
    record: &VNetDatasetRecord,
) -> SolverResult<()> {
    serde_json::to_writer(&mut *writer, record)
        .map_err(|error| SolverError::Serialization(error.to_string()))?;
    writer.write_all(b"\n")?;
    Ok(())
}

fn example_id(provenance: &VNetProvenance) -> String {
    let step = provenance
        .step_index
        .map(|step| step.to_string())
        .unwrap_or_else(|| "final".to_string());
    format!(
        "{}:{}:{}",
        provenance.preset_name, provenance.deal_seed.0, step
    )
}

fn mark_card(
    locations: &mut [u8],
    positions: &mut [u8],
    card: Card,
    location: u8,
    position: u8,
) -> SolverResult<()> {
    let index = usize::from(card.index());
    if locations[index] != LOCATION_UNSET {
        return Err(SolverError::DuplicateCard(card));
    }
    locations[index] = location;
    positions[index] = position;
    Ok(())
}

fn checked_u8(value: usize, context: &str) -> SolverResult<u8> {
    u8::try_from(value)
        .map_err(|_| SolverError::InvalidState(format!("{context} {value} exceeds u8 range")))
}

fn clamp_i16(value: usize) -> i16 {
    value.min(i16::MAX as usize) as i16
}

const LOCATION_UNSET: u8 = u8::MAX;
const LOCATION_FOUNDATION: u8 = 1;
const LOCATION_STOCK_RING: u8 = 2;
const LOCATION_TABLEAU_HIDDEN_BASE: u8 = 10;
const LOCATION_TABLEAU_FACE_UP_BASE: u8 = 20;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        cards::{Rank, Suit},
        core::{FoundationState, HiddenAssignments, VisibleState},
        experiments::fast_benchmark,
    };

    fn won_full_state() -> FullState {
        let mut visible = VisibleState::default();
        let mut foundations = FoundationState::default();
        for suit in [Suit::Clubs, Suit::Diamonds, Suit::Hearts, Suit::Spades] {
            foundations.set_top_rank(suit, Some(Rank::King));
        }
        visible.foundations = foundations;
        FullState::new(visible, HiddenAssignments::empty())
    }

    fn provenance(seed: DealSeed) -> VNetProvenance {
        VNetProvenance {
            source: VNetDataSource::DeterministicSolve,
            preset_name: "unit".to_string(),
            deal_seed: seed,
            step_index: Some(0),
            chosen_move: None,
            planner_value: None,
            terminal_won: Some(true),
        }
    }

    #[test]
    fn full_state_encoding_is_stable_for_same_deal() {
        let runner = ExperimentRunner;
        let first = runner.generate_deal(DealSeed(123)).unwrap();
        let second = runner.generate_deal(DealSeed(123)).unwrap();

        let first_encoding = VNetStateEncoding::from_full_state(&first.full_state).unwrap();
        let second_encoding = VNetStateEncoding::from_full_state(&second.full_state).unwrap();

        assert_eq!(first_encoding, second_encoding);
        assert_eq!(first_encoding.card_locations.len(), Card::COUNT);
        assert_eq!(first_encoding.card_positions.len(), Card::COUNT);
        assert_eq!(first_encoding.shape.feature_count, 114);
    }

    #[test]
    fn exported_example_shape_contains_label_and_encoding() {
        let full = won_full_state();
        let example = vnet_example_from_full_state(
            &full,
            VNetLabelMode::TerminalOutcome,
            1.0,
            provenance(DealSeed(1)),
            DatasetSplit::Train,
        )
        .unwrap();

        assert_eq!(example.label, 1.0);
        assert_eq!(example.label_mode, VNetLabelMode::TerminalOutcome);
        assert_eq!(example.encoded_state.foundation_tops, [13, 13, 13, 13]);
        assert_eq!(example.encoded_state.flat_features.len(), 114);
    }

    #[test]
    fn deterministic_solver_label_hook_uses_deterministic_mode() {
        let full = won_full_state();
        let example = vnet_example_from_deterministic_solve(
            &full,
            DeterministicSearchConfig::default(),
            provenance(DealSeed(2)),
            DatasetSplit::Validation,
        )
        .unwrap();

        assert_eq!(example.split, DatasetSplit::Validation);
        assert_eq!(example.label_mode, VNetLabelMode::DeterministicSolverValue);
        assert_eq!(example.label, 1.0);
    }

    #[test]
    fn split_strategy_is_reproducible() {
        let strategy = DatasetSplitStrategy::SeedModulo {
            modulo: 10,
            validation_remainder: 1,
            test_remainder: 2,
        };

        assert_eq!(
            strategy.split_for_seed(DealSeed(21)).unwrap(),
            DatasetSplit::Validation
        );
        assert_eq!(
            strategy.split_for_seed(DealSeed(22)).unwrap(),
            DatasetSplit::Test
        );
        assert_eq!(
            strategy.split_for_seed(DealSeed(23)).unwrap(),
            DatasetSplit::Train
        );
    }

    #[test]
    fn autoplay_dataset_export_is_reproducible_under_fixed_seed() {
        let suite = BenchmarkSuite::from_base_seed("vnet-repro", 50, 1);
        let preset = fast_benchmark();
        let config = VNetExportConfig {
            max_steps: Some(0),
            ..VNetExportConfig::default()
        };

        let first = collect_vnet_examples_from_autoplay_suite(&suite, &preset, &config).unwrap();
        let second = collect_vnet_examples_from_autoplay_suite(&suite, &preset, &config).unwrap();

        assert_eq!(first, second);
        assert_eq!(first.metadata.example_count, 1);
        assert_eq!(first.examples.len(), 1);
        assert_eq!(first.examples[0].provenance.deal_seed, DealSeed(50));
    }

    #[test]
    fn jsonl_writer_emits_metadata_and_examples() {
        let suite = BenchmarkSuite::from_base_seed("vnet-jsonl", 60, 1);
        let preset = fast_benchmark();
        let config = VNetExportConfig {
            max_steps: Some(0),
            ..VNetExportConfig::default()
        };
        let dataset = collect_vnet_examples_from_autoplay_suite(&suite, &preset, &config).unwrap();
        let path = std::env::temp_dir().join(format!(
            "solitaire-vnet-{}-{}.jsonl",
            std::process::id(),
            crate::VERSION
        ));
        let _ = std::fs::remove_file(&path);

        VNetDatasetWriter::write_jsonl(&path, &dataset).unwrap();
        let contents = std::fs::read_to_string(&path).unwrap();

        assert!(contents.lines().next().unwrap().contains("\"metadata\""));
        assert!(contents.contains("\"example\""));
        let _ = std::fs::remove_file(path);
    }
}
