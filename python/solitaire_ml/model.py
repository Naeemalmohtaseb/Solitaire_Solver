"""Simple V-Net model definitions."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


class VNetMlp(nn.Module):
    """Small full-state value network.

    Architecture: input -> hidden ReLU stack -> sigmoid scalar win probability.
    """

    def __init__(self, input_dim: int, hidden_sizes: Sequence[int] = (256, 128, 64)) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        self.input_dim = int(input_dim)
        self.hidden_sizes = [int(size) for size in hidden_sizes]
        layers: list[nn.Module] = []
        previous = self.input_dim
        for hidden_size in self.hidden_sizes:
            if hidden_size <= 0:
                raise ValueError("hidden sizes must be positive")
            layers.append(nn.Linear(previous, hidden_size))
            layers.append(nn.ReLU())
            previous = hidden_size
        layers.append(nn.Linear(previous, 1))
        layers.append(nn.Sigmoid())
        self.network = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""

    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
