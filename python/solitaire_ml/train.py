"""Command-line training entry point for the offline V-Net scaffold."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from .dataset import TorchVNetDataset, canonical_label_mode, load_vnet_jsonl, split_examples
from .model import VNetMlp, count_parameters


@dataclass(frozen=True)
class TrainingConfig:
    data: str
    out: str
    label_mode: str | None
    max_examples: int | None
    epochs: int
    batch_size: int
    learning_rate: float
    hidden_sizes: list[int]
    split_seed: int
    train_fraction: float
    val_fraction: float
    test_fraction: float
    device: str
    normalize_features: bool


def parse_hidden_sizes(text: str) -> list[int]:
    sizes = [int(part.strip()) for part in text.split(",") if part.strip()]
    if not sizes:
        raise argparse.ArgumentTypeError("hidden sizes must not be empty")
    if any(size <= 0 for size in sizes):
        raise argparse.ArgumentTypeError("hidden sizes must be positive")
    return sizes


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a simple offline V-Net MLP")
    parser.add_argument("--data", required=True, help="Rust-exported V-Net JSONL path")
    parser.add_argument("--out", required=True, help="Output artifact directory")
    parser.add_argument("--label-mode", default=None, help="Optional label mode filter")
    parser.add_argument("--max-examples", type=int, default=None, help="Optional example cap")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-sizes", type=parse_hidden_sizes, default=[256, 128, 64])
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument(
        "--device",
        default="auto",
        help="Training device: auto, cpu, cuda, cuda:0, or another torch device string",
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Use raw exported flat_features instead of scale64 normalization",
    )
    return parser


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = build_parser().parse_args(argv)
    config = TrainingConfig(
        data=args.data,
        out=args.out,
        label_mode=canonical_label_mode(args.label_mode),
        max_examples=args.max_examples,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        hidden_sizes=list(args.hidden_sizes),
        split_seed=args.split_seed,
        train_fraction=args.train_fraction,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        device=args.device,
        normalize_features=not args.no_normalize,
    )
    return train(config)


def train(config: TrainingConfig) -> dict[str, Any]:
    if config.epochs <= 0:
        raise ValueError("epochs must be positive")
    if config.batch_size <= 0:
        raise ValueError("batch size must be positive")

    random.seed(config.split_seed)
    torch.manual_seed(config.split_seed)

    dataset = load_vnet_jsonl(
        config.data,
        label_mode=config.label_mode,
        max_examples=config.max_examples,
    )
    if not dataset.examples:
        raise ValueError("no examples loaded")

    splits = split_examples(
        dataset.examples,
        train_fraction=config.train_fraction,
        val_fraction=config.val_fraction,
        test_fraction=config.test_fraction,
        seed=config.split_seed,
    )
    train_examples = splits["Train"] or dataset.examples
    val_examples = splits["Validation"] or train_examples
    test_examples = splits["Test"]

    train_ds = TorchVNetDataset(train_examples, normalize=config.normalize_features)
    val_ds = TorchVNetDataset(val_examples, normalize=config.normalize_features)
    test_ds = TorchVNetDataset(test_examples, normalize=config.normalize_features)

    generator = torch.Generator()
    generator.manual_seed(config.split_seed)
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, generator=generator)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=config.batch_size, shuffle=False)

    device = resolve_device(config.device)
    model = VNetMlp(dataset.feature_dim, config.hidden_sizes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.BCELoss()

    out_dir = Path(config.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, Any]] = []
    best_val_loss = float("inf")

    for epoch in range(1, config.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = evaluate(model, val_loader, criterion, device)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_mae": val_metrics["mae"],
            "val_count": val_metrics["count"],
        }
        history.append(row)
        print(
            f"epoch {epoch:03d} "
            f"train_loss={train_loss:.6f} "
            f"val_loss={val_metrics['loss']:.6f} "
            f"val_mae={val_metrics['mae']:.6f}"
        )
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            save_checkpoint(out_dir / "best.pt", model, config, dataset.metadata, best_val_loss)
            export_inference_artifact(
                out_dir / "best_vnet_inference.json",
                model,
                config,
                dataset.metadata,
                best_val_loss,
            )

    test_metrics = evaluate(model, test_loader, criterion, device) if len(test_ds) else {"loss": None, "mae": None, "count": 0}
    save_checkpoint(out_dir / "last.pt", model, config, dataset.metadata, history[-1]["val_loss"])
    export_inference_artifact(
        out_dir / "last_vnet_inference.json",
        model,
        config,
        dataset.metadata,
        history[-1]["val_loss"],
    )

    artifacts = {
        "config": asdict(config),
        "dataset_metadata": dataset.metadata,
        "feature_dim": dataset.feature_dim,
        "parameter_count": count_parameters(model),
        "resolved_device": str(device),
        "split_counts": {
            "train": len(train_examples),
            "validation": len(val_examples),
            "test": len(test_examples),
        },
        "best_val_loss": best_val_loss,
        "test_metrics": test_metrics,
        "history": history,
    }
    write_json(out_dir / "training_config.json", asdict(config))
    write_json(out_dir / "metrics_history.json", history)
    write_json(out_dir / "dataset_metadata.json", dataset.metadata)
    write_json(out_dir / "summary.json", artifacts)
    return artifacts


def resolve_device(device: str) -> torch.device:
    """Resolve a CLI device string into a concrete torch device."""

    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def train_one_epoch(
    model: VNetMlp,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    total_count = 0
    for features, labels in loader:
        features = features.to(device)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        predictions = model(features)
        loss = criterion(predictions, labels)
        loss.backward()
        optimizer.step()
        batch_count = int(labels.shape[0])
        total_loss += float(loss.item()) * batch_count
        total_count += batch_count
    return total_loss / max(total_count, 1)


@torch.no_grad()
def evaluate(
    model: VNetMlp,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, float | int]:
    model.eval()
    total_loss = 0.0
    total_abs = 0.0
    total_count = 0
    for features, labels in loader:
        features = features.to(device)
        labels = labels.to(device)
        predictions = model(features)
        loss = criterion(predictions, labels)
        batch_count = int(labels.shape[0])
        total_loss += float(loss.item()) * batch_count
        total_abs += float(torch.abs(predictions - labels).sum().item())
        total_count += batch_count
    return {
        "loss": total_loss / max(total_count, 1),
        "mae": total_abs / max(total_count, 1),
        "count": total_count,
    }


def save_checkpoint(
    path: Path,
    model: VNetMlp,
    config: TrainingConfig,
    dataset_metadata: dict[str, Any],
    val_loss: float,
) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "training_config": asdict(config),
            "dataset_metadata": dataset_metadata,
            "val_loss": val_loss,
            "model_class": "VNetMlp",
            "input_dim": model.input_dim,
            "hidden_sizes": model.hidden_sizes,
        },
        path,
    )


def export_inference_artifact(
    path: Path,
    model: VNetMlp,
    config: TrainingConfig,
    dataset_metadata: dict[str, Any],
    val_loss: float,
) -> None:
    """Write a small Rust-native MLP artifact for solver-side inference."""

    layers: list[dict[str, Any]] = []
    modules = list(model.network)
    for index, module in enumerate(modules):
        if not isinstance(module, nn.Linear):
            continue
        activation = "linear"
        if index + 1 < len(modules):
            next_module = modules[index + 1]
            if isinstance(next_module, nn.ReLU):
                activation = "relu"
            elif isinstance(next_module, nn.Sigmoid):
                activation = "sigmoid"
        layers.append(
            {
                "weights": module.weight.detach().cpu().tolist(),
                "biases": module.bias.detach().cpu().tolist(),
                "activation": activation,
            }
        )

    artifact = {
        "schema_version": "solitaire-vnet-mlp-json-v1",
        "model_role": "VNet",
        "model_type": "mlp",
        "input_dim": model.input_dim,
        "hidden_sizes": model.hidden_sizes,
        "feature_normalization": "scale64" if config.normalize_features else "none",
        "label_mode": config.label_mode or dataset_metadata.get("label_mode"),
        "dataset_metadata": dataset_metadata,
        "val_loss": val_loss,
        "layers": layers,
    }
    write_json(path, artifact)


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
