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
fn autoplay_export_respects_decision_stride() {
    let suite = BenchmarkSuite::from_base_seed("vnet-stride", 55, 1);
    let preset = fast_benchmark();
    let config = VNetExportConfig {
        max_steps: Some(4),
        decision_stride: 2,
        ..VNetExportConfig::default()
    };

    let dataset = collect_vnet_examples_from_autoplay_suite(&suite, &preset, &config).unwrap();

    assert_eq!(dataset.metadata.decision_stride, 2);
    assert!(dataset.examples.iter().all(|example| example
        .provenance
        .step_index
        .is_none_or(|step| step % 2 == 0)));
}

#[test]
fn export_rejects_zero_decision_stride() {
    let suite = BenchmarkSuite::from_base_seed("vnet-bad-stride", 56, 1);
    let preset = fast_benchmark();
    let config = VNetExportConfig {
        decision_stride: 0,
        ..VNetExportConfig::default()
    };

    assert!(collect_vnet_examples_from_autoplay_suite(&suite, &preset, &config).is_err());
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

#[test]
fn streaming_jsonl_writer_appends_examples() {
    let suite = BenchmarkSuite::from_base_seed("vnet-streaming", 61, 1);
    let preset = fast_benchmark();
    let config = VNetExportConfig {
        max_steps: Some(0),
        ..VNetExportConfig::default()
    };
    let dataset = collect_vnet_examples_from_autoplay_suite(&suite, &preset, &config).unwrap();
    let path = std::env::temp_dir().join(format!(
        "solitaire-vnet-streaming-{}-{}.jsonl",
        std::process::id(),
        crate::VERSION
    ));
    let _ = std::fs::remove_file(&path);

    let mut writer = VNetDatasetWriter::create_jsonl(&path, &dataset.metadata).unwrap();
    assert_eq!(writer.path(), path.as_path());
    for example in &dataset.examples {
        writer.append_example(example).unwrap();
    }
    writer.finish().unwrap();

    let line_count = std::fs::read_to_string(&path).unwrap().lines().count();
    assert_eq!(line_count, dataset.examples.len() + 1);
    let _ = std::fs::remove_file(path);
}

#[test]
fn rust_mlp_vnet_evaluator_scores_flat_features() {
    let artifact = VNetInferenceArtifact {
        schema_version: "solitaire-vnet-mlp-json-v1".to_string(),
        model_role: "VNet".to_string(),
        model_type: "mlp".to_string(),
        input_dim: 2,
        hidden_sizes: Vec::new(),
        feature_normalization: "scale64".to_string(),
        label_mode: Some("TerminalOutcome".to_string()),
        dataset_metadata: None,
        layers: vec![VNetLayerArtifact {
            weights: vec![vec![1.0, 1.0]],
            biases: vec![0.0],
            activation: VNetActivation::Sigmoid,
        }],
    };
    let evaluator = VNetEvaluator::from_artifact(VNetBackend::RustMlpJson, artifact).unwrap();

    let value = evaluator.evaluate_flat_features(&[32, 32]).unwrap();

    assert!((value - 0.731_058_6).abs() < 1e-5);
}

#[test]
fn vnet_evaluator_loads_json_artifact() {
    let artifact = VNetInferenceArtifact {
        schema_version: "solitaire-vnet-mlp-json-v1".to_string(),
        model_role: "VNet".to_string(),
        model_type: "mlp".to_string(),
        input_dim: 1,
        hidden_sizes: Vec::new(),
        feature_normalization: "none".to_string(),
        label_mode: Some("TerminalOutcome".to_string()),
        dataset_metadata: None,
        layers: vec![VNetLayerArtifact {
            weights: vec![vec![0.0]],
            biases: vec![0.0],
            activation: VNetActivation::Sigmoid,
        }],
    };
    let path = std::env::temp_dir().join(format!(
        "solitaire-vnet-artifact-{}-{}.json",
        std::process::id(),
        crate::VERSION
    ));
    let _ = std::fs::remove_file(&path);
    std::fs::write(&path, serde_json::to_string(&artifact).unwrap()).unwrap();

    let evaluator = VNetEvaluator::load(&VNetInferenceConfig {
        enable_vnet: true,
        model_path: Some(path.clone()),
        ..VNetInferenceConfig::default()
    })
    .unwrap();

    assert_eq!(evaluator.evaluate_flat_features(&[0]).unwrap(), 0.5);
    let _ = std::fs::remove_file(path);
}
