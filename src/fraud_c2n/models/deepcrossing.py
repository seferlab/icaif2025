from __future__ import annotations

import torch
import torch.nn as nn


class DeepCrossing(nn.Module):
    """A simple DeepCrossing-style residual MLP for dense inputs.

    Note: In the original literature, DeepCrossing uses an embedding layer for sparse features.
    In this repo we feed a pre-encoded dense design matrix (e.g., one-hot + numeric),
    which is consistent with a practical reproduction pipeline.
    """

    def __init__(self, input_dim: int, hidden_dims: list[int], dropout: float = 0.0):
        super().__init__()
        layers: list[nn.Module] = []
        d = input_dim
        for h in hidden_dims:
            block = nn.Sequential(
                nn.Linear(d, h),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(h, d),
                nn.Dropout(dropout),
            )
            layers.append(block)
        self.blocks = nn.ModuleList(layers)
        self.out = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for blk in self.blocks:
            h = h + blk(h)
        return self.out(h).squeeze(-1)
