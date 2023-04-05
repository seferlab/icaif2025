from __future__ import annotations

import math
import torch
import torch.nn as nn


class FTTransformer(nn.Module):
    """A lightweight FT-Transformer style model for dense tabular inputs.

    This implementation assumes inputs are already a dense float matrix (e.g., one-hot + numeric).
    We treat each feature as a "token" via a linear projection and apply Transformer encoder blocks.
    """

    def __init__(self, input_dim: int, d_model: int = 128, n_heads: int = 8, n_layers: int = 3, dropout: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model

        # Project each scalar feature to a token embedding
        self.feature_proj = nn.Linear(1, d_model)
        self.pos = nn.Parameter(torch.randn(input_dim, d_model) * 0.02)

        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        self.norm = nn.LayerNorm(d_model)
        self.out = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, D)
        B, D = x.shape
        assert D == self.input_dim
        tokens = self.feature_proj(x.unsqueeze(-1))  # (B, D, d_model)
        tokens = tokens + self.pos.unsqueeze(0)
        h = self.encoder(tokens)
        # pool by mean
        h = self.norm(h.mean(dim=1))
        return self.out(h).squeeze(-1)
