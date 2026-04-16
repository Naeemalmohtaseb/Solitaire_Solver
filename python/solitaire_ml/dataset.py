"""Dataset loading and split utilities for Rust-exported V-Net JSONL files."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal

DatasetSplit = Literal["Train", "Validation", "Test"]

LABEL_MODE_ALIASES = {
    "terminal": "TerminalOutcome",
    "terminal_outcome": "TerminalOutcome",
    "TerminalOutcome": "TerminalOutcome",
    "deterministic": "DeterministicSolverValue",
    "deterministic_solver_value": "DeterministicSolverValue",
    "DeterministicSolverValue": "DeterministicSolverValue",
    "planner": "PlannerBackedApproximateValue",
    "planner_approximate_value": "PlannerBackedApproximateValue",
    "planner_backed_approximate_value": "PlannerBackedApproximateValue",
    "PlannerBackedApproximateValue": "PlannerBackedApproximateValue",
}


def canonical_label_mode(label_mode: str | None) -> str | None:
    """Normalize user-facing label mode aliases to Rust enum names."""

    if label_mode is None:
        return None
    key = label_mode.strip()
    if key in LABEL_MODE_ALIASES:
        return LABEL_MODE_ALIASES[key]
    lowered = key.lower().replace("-", "_")
    if lowered in LABEL_MODE_ALIASES:
        return LABEL_MODE_ALIASES[lowered]
    raise ValueError(f"unknown V-Net label mode: {label_mode}")


@dataclass(frozen=True)
class VNetJsonlDataset:
    """In-memory view of a Rust-exported V-Net JSONL dataset."""

    metadata: dict[str, Any]
    examples: list[dict[str, Any]]
    source_path: Path

    @property
    def feature_dim(self) -> int:
        """Return the stable flat feature length."""

        if not self.examples:
            shape = self.metadata.get("shape")
            if isinstance(shape, dict) and "feature_count" in shape:
                return int(shape["feature_count"])
            return 0
        return len(flat_features(self.examples[0]))

    def split_counts(self) -> dict[str, int]:
        """Count examples by exported split marker."""

        counts = {"Train": 0, "Validation": 0, "Test": 0}
        for example in self.examples:
            split = example.get("split", "Train")
            counts[split] = counts.get(split, 0) + 1
        return counts


def load_vnet_jsonl(
    path: str | Path,
    *,
    label_mode: str | None = None,
    max_examples: int | None = None,
) -> VNetJsonlDataset:
    """Load a Rust V-Net JSONL export in deterministic file order."""

    path = Path(path)
    wanted_label_mode = canonical_label_mode(label_mode)
    metadata: dict[str, Any] | None = None
    examples: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            record_type = record.get("record_type")
            if record_type == "metadata":
                metadata = record["metadata"]
                continue
            if record_type != "example":
                raise ValueError(f"unsupported JSONL record_type at line {line_number}: {record_type!r}")

            example = record["example"]
            example_label_mode = canonical_label_mode(example.get("label_mode"))
            if wanted_label_mode is not None and example_label_mode != wanted_label_mode:
                continue
            examples.append(example)
            if max_examples is not None and len(examples) >= max_examples:
                break

    if metadata is None:
        raise ValueError(f"dataset is missing metadata record: {path}")
    return VNetJsonlDataset(metadata=metadata, examples=examples, source_path=path)


def flat_features(example: dict[str, Any]) -> list[int]:
    """Return the exported flat feature vector from one example."""

    return list(example["encoded_state"]["flat_features"])


def feature_vector(example: dict[str, Any], *, normalize: bool = True) -> list[float]:
    """Convert one exported example into a tensor-ready feature vector."""

    values = [float(value) for value in flat_features(example)]
    if normalize:
        return [value / 64.0 for value in values]
    return values


def label_value(example: dict[str, Any]) -> float:
    """Return the scalar V-Net target."""

    value = float(example["label"])
    if value < 0.0 or value > 1.0:
        raise ValueError(f"V-Net label must be in [0, 1], got {value}")
    return value


def split_examples(
    examples: Iterable[dict[str, Any]],
    *,
    train_fraction: float = 0.8,
    val_fraction: float = 0.1,
    test_fraction: float = 0.1,
    seed: int = 0,
) -> dict[DatasetSplit, list[dict[str, Any]]]:
    """Split examples deterministically by hashing example id plus seed.

    The function preserves original ordering inside each split.
    """

    examples = list(examples)
    _validate_fractions(train_fraction, val_fraction, test_fraction)
    if not examples:
        return {"Train": [], "Validation": [], "Test": []}

    ranked = sorted(
        ((hash_fraction(example, seed), index) for index, example in enumerate(examples)),
        key=lambda item: (item[0], item[1]),
    )
    count = len(examples)
    train_count = int(round(count * train_fraction))
    val_count = int(round(count * val_fraction))
    if train_count + val_count > count:
        val_count = max(0, count - train_count)
    test_count = count - train_count - val_count
    if test_fraction > 0.0 and test_count == 0 and count >= 3:
        test_count = 1
        if val_count > 0:
            val_count -= 1
        else:
            train_count -= 1
    if val_fraction > 0.0 and val_count == 0 and count >= 3:
        val_count = 1
        train_count = max(0, train_count - 1)

    train_ids = {index for _, index in ranked[:train_count]}
    val_ids = {index for _, index in ranked[train_count : train_count + val_count]}
    test_ids = {index for _, index in ranked[train_count + val_count :]}

    splits: dict[DatasetSplit, list[dict[str, Any]]] = {
        "Train": [],
        "Validation": [],
        "Test": [],
    }
    for index, example in enumerate(examples):
        if index in train_ids:
            splits["Train"].append(example)
        elif index in val_ids:
            splits["Validation"].append(example)
        elif index in test_ids:
            splits["Test"].append(example)
    return splits


def hash_fraction(example: dict[str, Any], seed: int) -> float:
    """Map an example deterministically into [0, 1)."""

    example_id = str(example.get("example_id", ""))
    digest = hashlib.blake2b(f"{seed}:{example_id}".encode("utf-8"), digest_size=8).digest()
    value = int.from_bytes(digest, byteorder="big", signed=False)
    return value / float(1 << 64)


def _validate_fractions(train_fraction: float, val_fraction: float, test_fraction: float) -> None:
    for name, value in {
        "train_fraction": train_fraction,
        "val_fraction": val_fraction,
        "test_fraction": test_fraction,
    }.items():
        if value < 0.0:
            raise ValueError(f"{name} must be non-negative")
    total = train_fraction + val_fraction + test_fraction
    if abs(total - 1.0) > 1e-6:
        raise ValueError("train/val/test fractions must sum to 1.0")


class TorchVNetDataset:
    """Torch Dataset wrapper for V-Net examples.

    PyTorch is imported lazily so metadata-only tests can run in lean
    environments.
    """

    def __init__(self, examples: Iterable[dict[str, Any]], *, normalize: bool = True) -> None:
        import torch

        self.examples = list(examples)
        self.normalize = normalize
        self._torch = torch

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int):
        example = self.examples[index]
        features = self._torch.tensor(feature_vector(example, normalize=self.normalize), dtype=self._torch.float32)
        label = self._torch.tensor([label_value(example)], dtype=self._torch.float32)
        return features, label
