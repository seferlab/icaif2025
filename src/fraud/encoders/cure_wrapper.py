from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer

from fraud.encoders.base import BaseEncoder, EncoderArtifacts
from fraud.encoders.utils import build_global_value_index, compute_feature_nmi, split_cols


class CUREWrapper(BaseEncoder):
    """Practical implementation of the CURE-style hierarchical coupling encoder.

    This follows the paper's high-level stages (value coupling -> multi-granularity
    clustering -> PCA fusion -> per-object representation). See Sec. 4.1. 
    """

    def __init__(
        self,
        embed_dim: int = 16,
        granularities: tuple[int, ...] = (8, 16, 32),
        pca_dim: int = 8,
        per_feature_scalar: bool = True,
        random_state: int = 0,
    ):
        self.embed_dim = int(embed_dim)
        self.granularities = tuple(int(g) for g in granularities)
        self.pca_dim = int(pca_dim)
        self.per_feature_scalar = bool(per_feature_scalar)
        self.random_state = int(random_state)

        self._cat_cols: List[str] = []
        self._num_cols: List[str] = []
        self._num_imputer = SimpleImputer(strategy="median")

        self._global_values: List[str] = []
        self._val_index: Dict[Tuple[str, str], int] = {}
        self._value_embeddings: Optional[np.ndarray] = None  # (L, pca_dim)
        self._feature_value_scalar: Dict[str, Dict[str, float]] = {}

    def _build_coupling_matrices(self, X_cat: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Build two coupling matrices: occurrence-based and co-occurrence-based."""

        # Global value frequency p(v)
        L = len(self._global_values)
        counts_v = np.zeros(L, dtype=np.float64)
        n = len(X_cat)
        for c in self._cat_cols:
            ser = X_cat[c].astype(str).fillna("__NA__")
            for v, cnt in ser.value_counts().items():
                counts_v[self._val_index[(c, v)]] += float(cnt)
        p_v = counts_v / max(counts_v.sum(), 1.0)

        # NMI between feature pairs
        nmi = compute_feature_nmi(X_cat)

        M_occ = np.zeros((L, L), dtype=np.float32)
        M_cooc = np.zeros((L, L), dtype=np.float32)

        # precompute value lists per col
        col_values: Dict[str, List[str]] = {}
        for c in self._cat_cols:
            col_values[c] = pd.unique(X_cat[c].astype(str).fillna("__NA__")).tolist()

        # Iterate feature pairs; fill blocks in the global matrix
        for i, ci in enumerate(self._cat_cols):
            vi_list = col_values[ci]
            ser_i = X_cat[ci].astype(str).fillna("__NA__")
            cnt_i = ser_i.value_counts().to_dict()
            for cj in self._cat_cols[i + 1 :]:
                vj_list = col_values[cj]
                ser_j = X_cat[cj].astype(str).fillna("__NA__")

                # occurrence-based coupling uses NMI(ci,cj) * p(vj)/p(vi) fileciteturn5file1L261-L275
                w_nmi = float(nmi.get((ci, cj), nmi.get((cj, ci), 0.0)))
                for vi in vi_list:
                    idx_i = self._val_index[(ci, vi)]
                    p_vi = p_v[idx_i] if p_v[idx_i] > 0 else 1e-9
                    for vj in vj_list:
                        idx_j = self._val_index[(cj, vj)]
                        M_occ[idx_i, idx_j] = w_nmi * float(p_v[idx_j] / p_vi)
                        M_occ[idx_j, idx_i] = w_nmi * float(p_v[idx_i] / (p_v[idx_j] if p_v[idx_j] > 0 else 1e-9))

                # co-occurrence-based coupling p(vi,vj)/p(vi) fileciteturn5file1L276-L287
                ct = pd.crosstab(ser_i, ser_j)
                for vi in ct.index:
                    idx_i = self._val_index[(ci, str(vi))]
                    denom = float(cnt_i.get(str(vi), 0.0))
                    if denom <= 0:
                        continue
                    row = ct.loc[vi]
                    for vj, c_ij in row.items():
                        if c_ij <= 0:
                            continue
                        idx_j = self._val_index[(cj, str(vj))]
                        M_cooc[idx_i, idx_j] = float(c_ij / denom)
                        # symmetric counterpart uses p(vj,vi)/p(vj)
        # Fill reverse direction for M_cooc to make it symmetric-ish
        # (When iterating pairs (ci,cj), we set only i->j; here set j->i similarly)
        for i, ci in enumerate(self._cat_cols):
            ser_i = X_cat[ci].astype(str).fillna("__NA__")
            cnt_i = ser_i.value_counts().to_dict()
            for cj in self._cat_cols[i + 1 :]:
                ser_j = X_cat[cj].astype(str).fillna("__NA__")
                cnt_j = ser_j.value_counts().to_dict()
                ct = pd.crosstab(ser_j, ser_i)
                for vj in ct.index:
                    idx_j = self._val_index[(cj, str(vj))]
                    denom = float(cnt_j.get(str(vj), 0.0))
                    if denom <= 0:
                        continue
                    row = ct.loc[vj]
                    for vi, c_ji in row.items():
                        if c_ji <= 0:
                            continue
                        idx_i = self._val_index[(ci, str(vi))]
                        M_cooc[idx_j, idx_i] = float(c_ji / denom)

        return M_occ, M_cooc

    @staticmethod
    def _cluster_rows(M: np.ndarray, q: int, random_state: int) -> np.ndarray:
        # cluster the L rows of the coupling matrix -> labels in [0,q)
        km = KMeans(n_clusters=q, n_init=10, random_state=random_state)
        return km.fit_predict(M)

    def fit(self, X: pd.DataFrame, y: Optional[np.ndarray] = None) -> "CUREWrapper":
        self._cat_cols, self._num_cols = split_cols(X)
        self._num_imputer.fit(X[self._num_cols])

        X_cat = X[self._cat_cols].copy() if self._cat_cols else pd.DataFrame(index=X.index)
        self._global_values, self._val_index = build_global_value_index(X_cat)
        L = len(self._global_values)
        if L == 0:
            self._value_embeddings = np.zeros((0, self.pca_dim), dtype=np.float32)
            self._feature_value_scalar = {}
            return self

        M_occ, M_cooc = self._build_coupling_matrices(X_cat)

        # Multi-granularity clustering on each coupling matrix. fileciteturn5file1L288-L306
        assignment_parts: List[np.ndarray] = []
        for M in (M_occ, M_cooc):
            for q in self.granularities:
                q_eff = min(q, max(2, L))
                labels = self._cluster_rows(M, q_eff, self.random_state)
                C = np.eye(q_eff, dtype=np.float32)[labels]  # (L,q)
                assignment_parts.append(C)

        C_all = np.concatenate(assignment_parts, axis=1)

        # Dimensionality reduction (PCA) to obtain compact embeddings V. fileciteturn5file1L303-L308
        pca_dim = min(self.pca_dim, C_all.shape[1])
        pca = PCA(n_components=pca_dim, random_state=self.random_state)
        V = pca.fit_transform(C_all).astype(np.float32)
        self._value_embeddings = V

        # Convert value embeddings into per-feature scalar codes if requested.
        self._feature_value_scalar = {}
        if self.per_feature_scalar:
            for c in self._cat_cols:
                ser = X_cat[c].astype(str).fillna("__NA__")
                uniq = pd.unique(ser).tolist()
                idxs = np.array([self._val_index[(c, v)] for v in uniq], dtype=int)
                # Project value vectors for this feature to 1D
                if len(idxs) == 1:
                    scalars = np.array([0.0], dtype=np.float32)
                else:
                    p = PCA(n_components=1, random_state=self.random_state)
                    scalars = p.fit_transform(V[idxs]).reshape(-1).astype(np.float32)
                self._feature_value_scalar[c] = {v: float(s) for v, s in zip(uniq, scalars)}

        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        num = self._num_imputer.transform(X[self._num_cols]) if self._num_cols else np.zeros((len(X), 0))
        if not self._cat_cols:
            return num.astype(np.float32)

        X_cat = X[self._cat_cols].copy()
        X_cat = X_cat.astype(str).fillna("__NA__")

        if self.per_feature_scalar:
            cat_parts: List[np.ndarray] = []
            for c in self._cat_cols:
                m = self._feature_value_scalar.get(c, {})
                col = X_cat[c].map(lambda v: float(m.get(v, 0.0))).to_numpy().reshape(-1, 1)
                cat_parts.append(col)
            cat_arr = np.concatenate(cat_parts, axis=1) if cat_parts else np.zeros((len(X), 0))
        else:
            # Concatenate embeddings for each feature value (D * pca_dim)
            assert self._value_embeddings is not None
            parts: List[np.ndarray] = []
            for c in self._cat_cols:
                idxs = X_cat[c].map(lambda v: self._val_index.get((c, v), -1)).to_numpy()
                idxs = np.where(idxs < 0, 0, idxs)
                parts.append(self._value_embeddings[idxs])
            cat_arr = np.concatenate(parts, axis=1)

        return np.concatenate([num, cat_arr], axis=1).astype(np.float32)

    def save_artifacts(self, out_dir: Path, X_train: pd.DataFrame) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "encoder": "cure",
            "cat_cols": self._cat_cols,
            "num_cols": self._num_cols,
            "per_feature_scalar": self.per_feature_scalar,
            "value_to_scalar": self._feature_value_scalar,
        }
        (out_dir / "cure_value_to_scalar.json").write_text(json.dumps(payload, indent=2))
