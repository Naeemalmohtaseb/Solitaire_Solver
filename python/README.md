# Solitaire ML

Offline Python training utilities for datasets exported by the Rust solver.

This package currently supports only a full-state value network, V-Net. It does
not integrate inference into the Rust solver and does not implement P-Net.

## Install

Use a Python environment with PyTorch installed:

```powershell
py -m pip install -r python/requirements.txt
```

## Export Data

```powershell
cargo run -p solitaire-cli -- dataset export-vnet --preset fast_benchmark --games 25 --seed 100 --max-steps 20 --decision-stride 2 --out data/vnet.jsonl
```

## Train

Run from the repository root:

```powershell
$env:PYTHONPATH = "python"
py -m solitaire_ml.train --data data/vnet.jsonl --out artifacts/vnet-smoke --epochs 5 --batch-size 128 --learning-rate 0.001 --hidden-sizes 256,128,64 --split-seed 123
```

`--device auto` is the default and selects CUDA when PyTorch can see it,
otherwise CPU. The package does not load checkpoints back into the Rust solver;
this is offline training only.

Artifacts written:

- `best.pt`
- `last.pt`
- `best_vnet_inference.json`
- `last_vnet_inference.json`
- `training_config.json`
- `metrics_history.json`
- `dataset_metadata.json`
- `summary.json`

The `*_vnet_inference.json` files are Rust-native MLP artifacts. Point the Rust
solver at one of those files when enabling V-Net leaf evaluation.

## Benchmark With V-Net Leaves

```powershell
cargo run -p solitaire-cli -- benchmark autoplay --preset fast_benchmark --games 25 --seed 100 --leaf-eval-mode vnet --vnet-model artifacts/vnet-smoke/best_vnet_inference.json
```

V-Net leaf evaluation is approximate and optional. Exact solver proofs remain
separate from neural values.
