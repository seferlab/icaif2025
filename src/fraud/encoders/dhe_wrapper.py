from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer

from fraud.encoders.base import BaseEncoder
from fraud.encoders.utils import build_global_value_index, split_cols, stable_hash_to_int


class _DHE(nn.Module):
    """Deep Hash Embedding module (hash -> projection -> MLP)."""

    def __init__(self, K: int, B: int, proj_dim: int, emb_dim: int, hidden: int):
        super().__init__()
        self.K = int(K)
        self.B = int(B)

        # Equivalent to one-hot concatenation + linear projection:
        # using K embedding tables of size B x proj_dim and summing them.
        self.hash_embeds = nn.ModuleList([nn.Embedding(self.B, proj_dim) for _ in range(self.K)])
        self.mlp = nn.Sequential(
            nn.Linear(proj_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, emb_dim),
        )

    def forward(self, buckets: torch.Tensor) -> torch.Tensor:
        # buckets: (N,K) int64 in [0,B)
        z = 0
        for k in range(self.K):
            z = z + self.hash_embeds[k](buckets[:, k])
        return self.mlp(z)


class DHEWrapper(BaseEncoder):
    """Deep Hash Embedding (DHE) categorical-to-numerical encoder.

    We follow the paper summary: deterministic multi-hash, projection, and
    a small MLP to produce embeddings. For use with downstream non-neural
    classifiers (e.g., XGBoost), we train a light supervised head on the
    training split and then export per-category embeddings as numeric codes.
    See Sec. 4.4. 
    """

    def __init__(
        self,
        emb_dim: int = 16,
        hash_K: int = 4,
        hash_B: int = 2**15,
        proj_dim: int = 64,
        hidden: int = 64,
        epochs: int = 10,
        lr: float = 5e-3,
        batch_size: int = 2048,
        per_feature_scalar: bool = True,
        random_state: int = 0,
        device: str = "cpu",
    ):
        self.emb_dim = int(emb_dim)
        self.hash_K = int(hash_K)
        self.hash_B = int(hash_B)
        self.proj_dim = int(proj_dim)
        self.hidden = int(hidden)
        self.epochs = int(epochs)
        self.lr = float(lr)
        self.batch_size = int(batch_size)
        self.per_feature_scalar = bool(per_feature_scalar)
        self.random_state = int(random_state)
        self.device = str(device)

        self._cat_cols: List[str] = []
        self._num_cols: List[str] = []
        self._num_imputer = SimpleImputer(strategy="median")

        self._global_values: List[str] = []
        self._val_index: Dict[Tuple[str, str], int] = {}
        self._value_emb: Optional[np.ndarray] = None
        self._feature_value_scalar: Dict[str, Dict[str, float]] = {}

    def _buckets_for_tokens(self, tokens: List[str]) -> np.ndarray:
        out = np.zeros((len(tokens), self.hash_K), dtype=np.int64)
        for i, t in enumerate(tokens):
            for k in range(self.hash_K):
                out[i, k] = stable_hash_to_int(t, seed=1337 + k, modulo=self.hash_B)
        return out

    def fit(self, X: pd.DataFrame, y: Optional[np.ndarray] = None) -> "DHEWrapper":
        if y is None:
            raise ValueError("DHE requires y for supervised fitting")

        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        self._cat_cols, self._num_cols = split_cols(X)
        self._num_imputer.fit(X[self._num_cols])

        if not self._cat_cols:
            self._value_emb = np.zeros((0, self.emb_dim), dtype=np.float32)
            self._feature_value_scalar = {}
            return self

        X_cat = X[self._cat_cols].astype(str).fillna("__NA__")
        self._global_values, self._val_index = build_global_value_index(X_cat)
        L = len(self._global_values)

        # Build per-record token list for each categorical column
        # and the corresponding hashed buckets.
        # Token format: "col=value" to keep namespaces distinct.
        tokens_per_col: Dict[str, List[str]] = {}
        for c in self._cat_cols:
            ser = X_cat[c]
            tokens_per_col[c] = [f"{c}={v}" for v in ser.tolist()]

        # Model: DHE embedding for each categorical *token*, then concat across categorical columns + numeric,
        # then a linear head for binary classification.
        dev = torch.device(self.device)
        dhe = _DHE(K=self.hash_K, B=self.hash_B, proj_dim=self.proj_dim, emb_dim=self.emb_dim, hidden=self.hidden).to(dev)
        head = nn.Linear(self.emb_dim * len(self._cat_cols) + len(self._num_cols), 1).to(dev)
        opt = torch.optim.Adam(list(dhe.parameters()) + list(head.parameters()), lr=self.lr)

        y_t = torch.from_numpy(np.asarray(y).astype(np.float32)).to(dev)
        num_np = self._num_imputer.transform(X[self._num_cols]) if self._num_cols else np.zeros((len(X), 0), dtype=np.float32)
        num_t = torch.from_numpy(num_np.astype(np.float32)).to(dev)

        # Precompute bucket tensors per col for speed
        buckets_cols = []
        for c in self._cat_cols:
            buckets = self._buckets_for_tokens(tokens_per_col[c])
            buckets_cols.append(torch.from_numpy(buckets).to(dev))

        n = len(X)
        bs = max(64, self.batch_size)
        idx = np.arange(n)
        for _ in range(self.epochs):
            np.random.shuffle(idx)
            for s in range(0, n, bs):
                batch = idx[s : s + bs]
                opt.zero_grad(set_to_none=True)

                emb_parts = []
                for bcol in buckets_cols:
                    emb_parts.append(dhe(bcol[batch]))
                emb_cat = torch.cat(emb_parts, dim=1)  # (B, C*emb_dim)
                feats = torch.cat([num_t[batch], emb_cat], dim=1)
                logits = head(feats).squeeze(1)
                loss = F.binary_cross_entropy_with_logits(logits, y_t[batch])
                loss.backward()
                opt.step()

        # Export node embeddings e(v) for each global categorical value token.
        # We compute DHE embedding for each unique token using the trained DHE network.
        with torch.no_grad():
            token_buckets = self._buckets_for_tokens(self._global_values)
            emb = dhe(torch.from_numpy(token_buckets).to(dev)).cpu().numpy().astype(np.float32)
        self._value_emb = emb

        # Per-feature scalarization
        self._feature_value_scalar = {}
        if self.per_feature_scalar:
            for c in self._cat_cols:
                ser = X_cat[c]
                uniq = pd.unique(ser).tolist()
                idxs = np.array([self._val_index[(c, v)] for v in uniq], dtype=int)
                if len(idxs) == 1:
                    scalars = np.array([0.0], dtype=np.float32)
                else:
                    p = PCA(n_components=1, random_state=self.random_state)
                    scalars = p.fit_transform(self._value_emb[idxs]).reshape(-1).astype(np.float32)
                self._feature_value_scalar[c] = {v: float(sv) for v, sv in zip(uniq, scalars)}

        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        num = self._num_imputer.transform(X[self._num_cols]) if self._num_cols else np.zeros((len(X), 0))
        if not self._cat_cols:
            return num.astype(np.float32)

        X_cat = X[self._cat_cols].astype(str).fillna("__NA__")
        if self.per_feature_scalar:
            parts = []
            for c in self._cat_cols:
                m = self._feature_value_scalar.get(c, {})
                parts.append(X_cat[c].map(lambda v: float(m.get(v, 0.0))).to_numpy().reshape(-1, 1))
            cat_arr = np.concatenate(parts, axis=1) if parts else np.zeros((len(X), 0))
        else:
            assert self._value_emb is not None
            parts = []
            for c in self._cat_cols:
                ids = X_cat[c].map(lambda v: self._val_index.get((c, v), 0)).to_numpy()
                parts.append(self._value_emb[ids])
            cat_arr = np.concatenate(parts, axis=1)

        return np.concatenate([num, cat_arr], axis=1).astype(np.float32)

    def save_artifacts(self, out_dir: Path, X_train: pd.DataFrame) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "encoder": "dhe",
            "cat_cols": self._cat_cols,
            "num_cols": self._num_cols,
            "per_feature_scalar": self.per_feature_scalar,
            "value_to_scalar": self._feature_value_scalar,
        }
        (out_dir / "dhe_value_to_scalar.json").write_text(json.dumps(payload, indent=2))
