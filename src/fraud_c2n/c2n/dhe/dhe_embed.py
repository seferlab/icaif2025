
from __future__ import annotations
from dataclasses import dataclass
from typing import List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def _hash_int(x: int, seed: int) -> int:
    # simple mix
    x = (x ^ seed) * 0x45d9f3b
    x = (x ^ (x >> 16)) & 0xFFFFFFFF
    return x

class DHEModel(nn.Module):
    def __init__(self, n_buckets: int, emb_dim: int, n_hashes: int, hidden_dim: int):
        super().__init__()
        self.n_buckets=n_buckets
        self.emb_dim=emb_dim
        self.n_hashes=n_hashes
        self.table = nn.Embedding(n_buckets, emb_dim)
        nn.init.normal_(self.table.weight, std=0.02)
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, bucket_ids: torch.Tensor) -> torch.Tensor:
        # bucket_ids: (B, n_hashes)
        e = self.table(bucket_ids)  # (B, H, D)
        e = e.mean(dim=1)           # (B, D)
        logit = self.mlp(e).squeeze(-1)
        return logit

def compute_buckets(value_ids: np.ndarray, n_hashes: int, n_buckets: int, seed: int) -> np.ndarray:
    buckets=np.zeros((len(value_ids), n_hashes), dtype=np.int64)
    for i,vid in enumerate(value_ids):
        for h in range(n_hashes):
            buckets[i,h] = _hash_int(int(vid), seed + 9973*h) % n_buckets
    return buckets

def train_dhe(value_ids: np.ndarray, y: np.ndarray, n_hashes: int, n_buckets: int, emb_dim: int, hidden_dim: int,
              epochs: int, lr: float, seed: int=42, device: str="cpu") -> np.ndarray:
    torch.manual_seed(seed)
    buckets = compute_buckets(value_ids, n_hashes, n_buckets, seed)
    X = torch.tensor(buckets, dtype=torch.long, device=device)
    Y = torch.tensor(y.astype(np.float32), dtype=torch.float32, device=device)
    model = DHEModel(n_buckets, emb_dim, n_hashes, hidden_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(int(epochs)):
        model.train()
        logit = model(X)
        loss = F.binary_cross_entropy_with_logits(logit, Y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    # produce per-value embeddings by averaging its hash bucket vectors
    model.eval()
    with torch.no_grad():
        table = model.table.weight.detach().cpu().numpy()  # (n_buckets, emb_dim)
    emb = np.zeros((int(value_ids.max())+1, emb_dim), dtype=np.float32)
    for vid in np.unique(value_ids):
        b = compute_buckets(np.array([vid]), n_hashes, n_buckets, seed)[0]
        emb[int(vid)] = table[b].mean(axis=0)
    return emb
