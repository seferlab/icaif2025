from __future__ import annotations

import json
from dataclasses import dataclass
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
from fraud.encoders.utils import build_global_value_index, split_cols


class _GCNLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor, adj_norm: torch.Tensor) -> torch.Tensor:
        # x: (N,F), adj_norm: (N,N)
        return torch.matmul(adj_norm, self.lin(x))


class _GCEDiffPoolAE(nn.Module):
    """A small GCN + DiffPool-style autoencoder.

    This is a pragmatic implementation aligned with the paper summary: build
    co-occurrence adjacency, apply GCN, learn soft assignment S, pool, and
    reconstruct adjacency. 
    """

    def __init__(self, n_nodes: int, in_dim: int, hid: int, emb_dim: int, n_clusters: int):
        super().__init__()
        self.gcn1 = _GCNLayer(in_dim, hid)
        self.gcn2 = _GCNLayer(hid, emb_dim)
        self.gcn_s = _GCNLayer(hid, n_clusters)

    def forward(self, x: torch.Tensor, adj_norm: torch.Tensor, adj_target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h1 = F.relu(self.gcn1(x, adj_norm))
        z = self.gcn2(h1, adj_norm)  # (N,emb_dim)
        s_logits = self.gcn_s(h1, adj_norm)
        s = F.softmax(s_logits, dim=-1)  # (N,C)

        # pooled embeddings and adjacency (DiffPool equations) fileciteturn5file1L352-L363
        z_pool = torch.matmul(s.t(), z)  # (C,emb_dim)
        a_pool = torch.matmul(s.t(), torch.matmul(adj_target, s))  # (C,C)

        # Decode: reconstruct adjacency from pooled representations
        # Use a simple inner-product decoder on pooled node embeddings.
        a_pool_hat = torch.sigmoid(torch.matmul(z_pool, z_pool.t()))
        # lift back to original graph with S
        a_hat = torch.sigmoid(torch.matmul(s, torch.matmul(a_pool_hat, s.t())))

        return z, s, a_hat


class GCEWrapper(BaseEncoder):
    def __init__(
        self,
        emb_dim: int = 16,
        hid_dim: int = 32,
        n_clusters: int = 64,
        epochs: int = 200,
        lr: float = 1e-2,
        weight_decay: float = 1e-4,
        per_feature_scalar: bool = True,
        random_state: int = 0,
        device: str = "cpu",
    ):
        self.emb_dim = int(emb_dim)
        self.hid_dim = int(hid_dim)
        self.n_clusters = int(n_clusters)
        self.epochs = int(epochs)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
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
        self._S: Optional[np.ndarray] = None

    def _build_cooc_adj(self, X_cat: pd.DataFrame) -> np.ndarray:
        L = len(self._global_values)
        A = np.zeros((L, L), dtype=np.float32)

        # For each record, add +1 to co-occurrence pairs.
        # This is O(n * D^2), which is reasonable for moderate D.
        # Use int indices per column for speed.
        col_ids = []
        for c in self._cat_cols:
            ser = X_cat[c].astype(str).fillna("__NA__")
            col_ids.append(ser.map(lambda v: self._val_index[(c, v)]).to_numpy())
        col_ids = np.stack(col_ids, axis=1)  # (N,D)

        for row in col_ids:
            # unique within row, though typically already unique across columns
            ids = row.tolist()
            for i in range(len(ids)):
                A[ids[i], ids[i]] += 1.0
                for j in range(i + 1, len(ids)):
                    A[ids[i], ids[j]] += 1.0
                    A[ids[j], ids[i]] += 1.0

        # Normalize counts -> weighted adjacency
        # We do a simple min-max normalization into [0,1].
        if A.max() > 0:
            A = A / A.max()
        return A

    @staticmethod
    def _normalize_adj(A: np.ndarray) -> np.ndarray:
        # GCN normalization: D^{-1/2} (A + I) D^{-1/2} fileciteturn5file1L341-L351
        I = np.eye(A.shape[0], dtype=np.float32)
        Ab = A + I
        deg = Ab.sum(axis=1)
        deg_inv_sqrt = np.power(deg, -0.5, where=deg > 0)
        D = np.diag(deg_inv_sqrt.astype(np.float32))
        return (D @ Ab @ D).astype(np.float32)

    def fit(self, X: pd.DataFrame, y: Optional[np.ndarray] = None) -> "GCEWrapper":
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

        A = self._build_cooc_adj(X_cat)  # counts -> [0,1]
        A_norm = self._normalize_adj(A)

        # Node attributes: one-hot identity (as in paper summary) fileciteturn5file1L323-L337
        X0 = np.eye(L, dtype=np.float32)

        dev = torch.device(self.device)
        x = torch.from_numpy(X0).to(dev)
        adj_norm = torch.from_numpy(A_norm).to(dev)
        adj_target = torch.from_numpy(A).to(dev)

        n_clusters = min(self.n_clusters, max(2, L // 2))
        model = _GCEDiffPoolAE(n_nodes=L, in_dim=L, hid=self.hid_dim, emb_dim=self.emb_dim, n_clusters=n_clusters).to(dev)
        opt = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        # Reconstruction loss: BCE between A_hat and A (normalized) fileciteturn5file1L368-L376
        A_target = adj_target
        for _ in range(self.epochs):
            model.train()
            opt.zero_grad(set_to_none=True)
            z, s, A_hat = model(x, adj_norm, A_target)
            loss = F.binary_cross_entropy(A_hat, A_target)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            z, s, _ = model(x, adj_norm, A_target)

        self._value_emb = z.detach().cpu().numpy().astype(np.float32)
        self._S = s.detach().cpu().numpy().astype(np.float32)

        # Per-feature scalarization (optional)
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
            parts: List[np.ndarray] = []
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
            "encoder": "gce",
            "cat_cols": self._cat_cols,
            "num_cols": self._num_cols,
            "per_feature_scalar": self.per_feature_scalar,
            "value_to_scalar": self._feature_value_scalar,
        }
        (out_dir / "gce_value_to_scalar.json").write_text(json.dumps(payload, indent=2))
        if self._S is not None:
            np.save(out_dir / "gce_assignment_S.npy", self._S)
