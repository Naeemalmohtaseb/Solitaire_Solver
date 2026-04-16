# ml

Owns machine-learning data boundaries and the optional V-Net inference hook.

The Rust ML module intentionally stays library-first: dataset records,
full-state encoding, JSONL writing, and the optional Rust MLP evaluator live in
one production module, while tests live in `ml/tests.rs`. Training code remains
under `python/solitaire_ml/`.

The current implementation exports V-Net supervised examples from fully known
`FullState` positions. Each example contains a deterministic structured encoding,
a fixed per-card numeric feature vector, an explicit scalar label, and provenance
metadata linking it back to a preset, deal seed, and autoplay step.

Public naming is model-specific: `VNetExample`, `VNetDatasetMetadata`,
`EncodedFullState`, `VNetDatasetWriter`, and `VNetExportConfig` are the primary
library surfaces.

## V-Net Labels

Supported label modes are explicit and must not be mixed silently:

- `TerminalOutcome`: labels every exported state by the final win/loss of the configured autoplay run.
- `DeterministicSolverValue`: labels a full state with the deterministic open-card solver's bounded value.
- `PlannerBackedApproximateValue`: labels decision states with the planner root value when available.

## Export Format

The v1 writer emits JSONL:

1. one metadata record
2. one example record per exported full state

`VNetDatasetWriter` supports both whole-dataset writes and streaming writes:
create the JSONL file with metadata, append examples, then flush/finish.

The CLI entry point is:

```text
solitaire-cli dataset export-vnet --preset fast_benchmark --games 10 --seed 1 --max-steps 20 --decision-stride 2 --out data/vnet.jsonl
```

CLI label aliases are accepted for convenience:

- `--label-mode terminal`
- `--label-mode deterministic`
- `--label-mode planner`

## Offline Training Scaffold

The Python-side training scaffold lives under `python/solitaire_ml/`. It loads
the Rust JSONL export, filters examples by explicit label mode, builds
deterministic train/validation/test splits, and trains a small MLP V-Net with
PyTorch.

From the repository root:

```powershell
py -m pip install -r python/requirements.txt
$env:PYTHONPATH = "python"
py -m solitaire_ml.train --data data/vnet.jsonl --out python/runs/vnet-smoke --epochs 5 --batch-size 128 --learning-rate 0.001 --hidden-sizes 256,128,64 --split-seed 123
```

Training writes:

- `best.pt`: best checkpoint by validation loss
- `last.pt`: final checkpoint
- `best_vnet_inference.json`: Rust-native MLP inference artifact
- `last_vnet_inference.json`: Rust-native MLP artifact for the final epoch
- `training_config.json`: exact training arguments
- `metrics_history.json`: per-epoch loss/MAE rows
- `dataset_metadata.json`: snapshot of the Rust dataset metadata
- `summary.json`: compact run summary

The v1 model is intentionally simple: flat full-state features into a
configurable ReLU MLP, with a sigmoid scalar output interpreted as a win
probability/value target.

## Rust Inference

Rust inference uses `VNetEvaluator` with `VNetBackend::RustMlpJson`. This avoids
embedding Python in the solver and keeps the runtime dependency surface small.
The evaluator loads `*_vnet_inference.json`, reuses `EncodedFullState` /
`VNetStateEncoding::from_full_state`, applies the recorded feature
normalization, and runs the dense MLP directly in Rust.

V-Net evaluation is approximate only. Exact deterministic solve paths still
distinguish proof from cutoff value, and V-Net outputs are never stored or
reported as proven wins/losses.

Benchmark example:

```powershell
cargo run -p solitaire-cli -- benchmark autoplay --preset fast_benchmark --games 25 --seed 100 --leaf-eval-mode vnet --vnet-model python/runs/vnet-smoke/best_vnet_inference.json
cargo run -p solitaire-cli -- benchmark compare --baseline balanced_benchmark --candidate balanced_vnet_benchmark --games 25 --seed 100 --candidate-vnet-model python/runs/vnet-smoke/best_vnet_inference.json
```

If the model path is missing or loading fails through the optional solver path,
the deterministic solver falls back to the existing heuristic leaf evaluator and
increments V-Net fallback diagnostics.

V-Net-aware benchmark presets exist for fast, balanced, and quality runs:

- `fast_vnet_benchmark`
- `balanced_vnet_benchmark`
- `quality_vnet_benchmark`

They only switch approximate cutoff evaluation to V-Net. Exact proof-oriented
paths still use the deterministic solver and preserve exact/unknown semantics.

## Deferred

P-Net exports, policy training, model serving, ONNX/accelerated runtimes, and
any new search architecture remain deferred. The current inference path is a
minimal benchmarkable V-Net leaf evaluator.
