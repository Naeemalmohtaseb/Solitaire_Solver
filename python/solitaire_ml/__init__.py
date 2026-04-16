"""Offline ML utilities for the Draw-3 Klondike solver.

This package trains models from Rust-exported datasets only. It does not
integrate inference into the solver and does not change solver behavior.
"""

from .dataset import (
    LABEL_MODE_ALIASES,
    TorchVNetDataset,
    VNetJsonlDataset,
    canonical_label_mode,
    feature_vector,
    load_vnet_jsonl,
    split_examples,
)
from .model import VNetMlp, count_parameters

__all__ = [
    "LABEL_MODE_ALIASES",
    "TorchVNetDataset",
    "VNetJsonlDataset",
    "VNetMlp",
    "canonical_label_mode",
    "count_parameters",
    "feature_vector",
    "load_vnet_jsonl",
    "split_examples",
]
