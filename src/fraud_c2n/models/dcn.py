from __future__ import annotations

import torch
import torch.nn as nn


class CrossLayer(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(d))
        self.b = nn.Parameter(torch.zeros(d))

    def forward(self, x0: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # x_{l+1} = x0 * (w^T x_l) + b + x_l
        # where (w^T x_l) produces a scalar per row
        xlw = torch.sum(x * self.w, dim=1, keepdim=True)
        return x0 * xlw + self.b + x


class DCN(nn.Module):
    """Deep & Cross Network (DCN) for dense pre-encoded inputs."""

    def __init__(self, input_dim: int, cross_layers: int, deep_dims: list[int], dropout: float = 0.0):
        super().__init__()
        self.cross = nn.ModuleList([CrossLayer(input_dim) for _ in range(cross_layers)])
        deep: list[nn.Module] = []
        d = input_dim
        for h in deep_dims:
            deep += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        self.deep = nn.Sequential(*deep) if deep else None
        out_in = input_dim + (deep_dims[-1] if deep else 0)
        self.out = nn.Linear(out_in, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = x
        xc = x
        for layer in self.cross:
            xc = layer(x0, xc)
        if self.deep is None:
            h = xc
        else:
            xd = self.deep(x0)
            h = torch.cat([xc, xd], dim=1)
        return self.out(h).squeeze(-1)


class CrossLayerV2(nn.Module):
    """Low-rank mixture-of-experts cross layer (DCN-V2 style)."""

    def __init__(self, d: int, low_rank: int, num_experts: int):
        super().__init__()
        self.num_experts = num_experts
        self.U = nn.Parameter(torch.randn(num_experts, d, low_rank) * 0.02)
        self.V = nn.Parameter(torch.randn(num_experts, low_rank, d) * 0.02)
        self.C = nn.Parameter(torch.randn(num_experts, low_rank, low_rank) * 0.02)
        self.b = nn.Parameter(torch.zeros(num_experts, d))
        self.gate = nn.Linear(d, num_experts)

    def forward(self, x0: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # For each expert e:
        #   x_{l+1}^e = x0 * ( (x V_e) C_e U_e^T ) + b_e + x
        # We combine experts using a soft gate over x.
        g = torch.softmax(self.gate(x), dim=1)  # (B, E)
        B, d = x.shape
        out = torch.zeros_like(x)
        for e in range(self.num_experts):
            xv = x @ self.V[e].T  # (B, r)
            xvc = xv @ self.C[e]  # (B, r)
            proj = xvc @ self.U[e].T  # (B, d)
            out_e = x0 * proj + self.b[e] + x
            out = out + out_e * g[:, e:e+1]
        return out


class DCNV2(nn.Module):
    """DCN-V2 style: low-rank MoE cross + deep tower."""

    def __init__(
        self,
        input_dim: int,
        cross_layers: int,
        low_rank: int,
        num_experts: int,
        deep_dims: list[int],
        dropout: float = 0.0,
    ):
        super().__init__()
        self.cross = nn.ModuleList([
            CrossLayerV2(input_dim, low_rank=low_rank, num_experts=num_experts)
            for _ in range(cross_layers)
        ])
        deep: list[nn.Module] = []
        d = input_dim
        for h in deep_dims:
            deep += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        self.deep = nn.Sequential(*deep) if deep else None
        out_in = input_dim + (deep_dims[-1] if deep else 0)
        self.out = nn.Linear(out_in, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = x
        xc = x
        for layer in self.cross:
            xc = layer(x0, xc)
        if self.deep is None:
            h = xc
        else:
            xd = self.deep(x0)
            h = torch.cat([xc, xd], dim=1)
        return self.out(h).squeeze(-1)
