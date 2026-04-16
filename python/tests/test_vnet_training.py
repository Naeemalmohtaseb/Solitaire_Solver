from __future__ import annotations

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from solitaire_ml.dataset import TorchVNetDataset, load_vnet_jsonl, split_examples
from solitaire_ml.train import main as train_main


def write_tiny_dataset(path: Path, feature_count: int = 6, examples: int = 6) -> None:
    metadata = {
        "dataset_version": "vnet-jsonl-v1",
        "model_role": "VNet",
        "example_kind": "DeterministicValue",
        "label_mode": "TerminalOutcome",
        "source": "AutoplayTrace",
        "format": "Jsonl",
        "preset_name": "unit",
        "suite_name": "tiny",
        "suite": {"name": "tiny", "base_seed": 1, "count": examples},
        "games": examples,
        "example_count": examples,
        "decision_stride": 1,
    }
    with path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps({"record_type": "metadata", "metadata": metadata}) + "\n")
        for index in range(examples):
            features = [index, index + 1, index + 2, 0, 1, 2][:feature_count]
            example = {
                "example_id": f"tiny-{index}",
                "split": "Train",
                "label_mode": "TerminalOutcome",
                "label": 1.0 if index % 2 == 0 else 0.0,
                "encoded_state": {
                    "flat_features": features,
                    "shape": {"feature_count": len(features), "plane_count": 0},
                },
                "provenance": {
                    "source": "AutoplayTrace",
                    "preset_name": "unit",
                    "deal_seed": index,
                    "step_index": index,
                    "chosen_move": None,
                    "planner_value": None,
                    "terminal_won": index % 2 == 0,
                },
            }
            handle.write(json.dumps({"record_type": "example", "example": example}) + "\n")


class VNetTrainingTests(unittest.TestCase):
    def test_dataset_loading_and_label_filter(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tiny.jsonl"
            write_tiny_dataset(path)

            dataset = load_vnet_jsonl(path, label_mode="terminal")

            self.assertEqual(dataset.metadata["preset_name"], "unit")
            self.assertEqual(len(dataset.examples), 6)
            self.assertEqual(dataset.feature_dim, 6)

    def test_split_reproducibility(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tiny.jsonl"
            write_tiny_dataset(path, examples=10)
            dataset = load_vnet_jsonl(path)

            first = split_examples(dataset.examples, seed=99)
            second = split_examples(dataset.examples, seed=99)

            self.assertEqual(
                [example["example_id"] for example in first["Train"]],
                [example["example_id"] for example in second["Train"]],
            )
            self.assertEqual(sum(len(split) for split in first.values()), 10)

    def test_torch_dataset_shapes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tiny.jsonl"
            write_tiny_dataset(path)
            dataset = load_vnet_jsonl(path)
            torch_dataset = TorchVNetDataset(dataset.examples)

            features, label = torch_dataset[0]

            self.assertEqual(tuple(features.shape), (6,))
            self.assertEqual(tuple(label.shape), (1,))

    def test_training_smoke_run_writes_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            data_path = tmp_path / "tiny.jsonl"
            out_path = tmp_path / "out"
            write_tiny_dataset(data_path, examples=8)

            summary = train_main(
                [
                    "--data",
                    str(data_path),
                    "--out",
                    str(out_path),
                    "--epochs",
                    "1",
                    "--batch-size",
                    "4",
                    "--hidden-sizes",
                    "8,4",
                    "--split-seed",
                    "5",
                ]
            )

            self.assertEqual(summary["feature_dim"], 6)
            self.assertTrue((out_path / "best.pt").exists())
            self.assertTrue((out_path / "last.pt").exists())
            self.assertTrue((out_path / "best_vnet_inference.json").exists())
            self.assertTrue((out_path / "last_vnet_inference.json").exists())
            self.assertTrue((out_path / "metrics_history.json").exists())
            self.assertTrue((out_path / "dataset_metadata.json").exists())


if __name__ == "__main__":
    unittest.main()
