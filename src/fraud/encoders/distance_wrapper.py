from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import dijkstra
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer

from fraud.encoders.base import BaseEncoder
from fraud.encoders.utils import build_global_value_index, split_cols


class DistanceEmbeddingWrapper(BaseEncoder):
    """Transitive distance learning + embedding encoder.

    Implementation mirrors the paper summary in Sec. 4.3: build a d-partite
    co-occurrence graph with Jaccard-like weights, define multiple base
    distances, compute transitive (shortest-path) distances, then embed with
    classical MDS. 
    """

    def __init__(
        self,
        emb_dim: int = 8,
        per_feature_scalar: bool = True,
        random_state: int = 0,
        max_nodes_for_dense: int = 6000,
    ):
        self.emb_dim = int(emb_dim)
        self.per_feature_scalar = bool(per_feature_scalar)
        self.random_state = int(random_state)
        self.max_nodes_for_dense = int(max_nodes_for_dense)

        self._cat_cols: List[str] = []
        self._num_cols: List[str] = []
        self._num_imputer = SimpleImputer(strategy="median")

        self._global_values: List[str] = []
        self._val_index: Dict[Tuple[str, str], int] = {}
        self._value_emb: Optional[np.ndarray] = None
        self._feature_value_scalar: Dict[str, Dict[str, float]] = {}

    def _build_counts(self, X_cat: pd.DataFrame) -> tuple[Dict[int, int], Dict[tuple[int, int], int]]:
        """Return node counts and pair counts for co-occurring values."""

        counts: Dict[int, int] = {}
        pair_counts: Dict[tuple[int, int], int] = {}

        col_ids = []
        for c in self._cat_cols:
            ser = X_cat[c].astype(str).fillna("__NA__")
            col_ids.append(ser.map(lambda v: self._val_index[(c, v)]).to_numpy())
        ids = np.stack(col_ids, axis=1)

        for row in ids:
            # update node counts
            for a in row:
                counts[int(a)] = counts.get(int(a), 0) + 1
            # update pair counts for cross-feature co-occurrence
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    a = int(row[i]); b = int(row[j])
                    if a == b:
                        continue
                    if a > b:
                        a, b = b, a
                    pair_counts[(a, b)] = pair_counts.get((a, b), 0) + 1

        return counts, pair_counts

    @staticmethod
    def _classical_mds(D: np.ndarray, d: int) -> np.ndarray:
        """Classical MDS on a full distance matrix."""

        n = D.shape[0]
        # double-center squared distances
        D2 = D ** 2
        J = np.eye(n) - np.ones((n, n)) / n
        B = -0.5 * J @ D2 @ J
        # eigen-decomposition
        vals, vecs = np.linalg.eigh(B)
        idx = np.argsort(vals)[::-1]
        vals = vals[idx]
        vecs = vecs[:, idx]
        vals = np.maximum(vals, 0)
        r = min(d, n)
        L = np.diag(np.sqrt(vals[:r]))
        X = vecs[:, :r] @ L
        return X.astype(np.float32)

    def fit(self, X: pd.DataFrame, y: Optional[np.ndarray] = None) -> "DistanceEmbeddingWrapper":
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

        counts, pair_counts = self._build_counts(X_cat)

        # Build sparse similarity weights w(a,b) = count(a,b)/(count(a)+count(b)-count(a,b)) fileciteturn5file1L403-L412
        rows = []
        cols = []
        data_sim = []
        data_cond = []

        for (a, b), c_ab in pair_counts.items():
            ca = counts.get(a, 1)
            cb = counts.get(b, 1)
            denom = (ca + cb - c_ab)
            sim = float(c_ab / denom) if denom > 0 else 0.0
            # base distance #1: co-occurrence distance
            d1 = 1.0 - sim
            # base distance #2: conditional distance (both directions, as directed edge weights)
            p_b_a = float(c_ab / ca) if ca > 0 else 0.0
            p_a_b = float(c_ab / cb) if cb > 0 else 0.0
            d2_ab = 1.0 - p_b_a
            d2_ba = 1.0 - p_a_b

            # undirected for d1
            rows += [a, b]
            cols += [b, a]
            data_sim += [d1, d1]

            # directed conditional for d2
            # store as two directed entries in same COO list
            data_cond += [d2_ab, d2_ba]

        # sparse matrices
        # co-occurrence distance: symmetric
        G1 = coo_matrix((np.array(data_sim, dtype=np.float32), (np.array(rows), np.array(cols))), shape=(L, L)).tocsr()
        # conditional distance: directed; rebuild rows/cols accordingly
        rows2 = []
        cols2 = []
        data2 = []
        for (a, b), c_ab in pair_counts.items():
            ca = counts.get(a, 1)
            cb = counts.get(b, 1)
            p_b_a = float(c_ab / ca) if ca > 0 else 0.0
            p_a_b = float(c_ab / cb) if cb > 0 else 0.0
            rows2 += [a, b]
            cols2 += [b, a]
            data2 += [1.0 - p_b_a, 1.0 - p_a_b]
        G2 = coo_matrix((np.array(data2, dtype=np.float32), (np.array(rows2), np.array(cols2))), shape=(L, L)).tocsr()

        # Transitive distance = shortest-path distance in each base metric, then elementwise min across metrics. fileciteturn5file1L413-L453
        # For speed, compute all-pairs shortest paths with Dijkstra on sparse graphs.
        D1 = dijkstra(G1, directed=False, unweighted=False)
        D2 = dijkstra(G2, directed=True, unweighted=False)
        D = np.minimum(D1, D2)
        # Replace inf with a large value (max finite * 1.1) to keep MDS stable.
        finite = D[np.isfinite(D)]
        fill = float(np.max(finite) * 1.1) if finite.size else 1.0
        D = np.where(np.isfinite(D), D, fill).astype(np.float32)

        # MDS embedding fileciteturn5file1L477-L497
        if L > self.max_nodes_for_dense:
            # For very large L, subsample anchor points for approximate embedding.
            # (Keeps the pipeline from exploding; users can increase the threshold if desired.)
            rng = np.random.default_rng(self.random_state)
            anchor = rng.choice(L, size=self.max_nodes_for_dense, replace=False)
            D_anchor = D[np.ix_(anchor, anchor)]
            X_anchor = self._classical_mds(D_anchor, self.emb_dim)
            # out-of-sample: place each point by nearest anchor distances (simple heuristic)
            # This is an approximation, but works reasonably for visualization.
            X_full = np.zeros((L, X_anchor.shape[1]), dtype=np.float32)
            X_full[anchor] = X_anchor
            # assign non-anchors to nearest anchor
            for i in range(L):
                if i in set(anchor):
                    continue
                j = int(anchor[np.argmin(D[i, anchor])])
                X_full[i] = X_full[j]
            self._value_emb = X_full
        else:
            self._value_emb = self._classical_mds(D, self.emb_dim)

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
            "encoder": "distance",
            "cat_cols": self._cat_cols,
            "num_cols": self._num_cols,
            "per_feature_scalar": self.per_feature_scalar,
            "value_to_scalar": self._feature_value_scalar,
        }
        (out_dir / "distance_value_to_scalar.json").write_text(json.dumps(payload, indent=2))
