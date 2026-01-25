from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction import FeatureHasher
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from fraud.encoders.base import BaseEncoder


def _split_cols(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    cat_cols = [c for c in X.columns if X[c].dtype == object or str(X[c].dtype).startswith("category")]
    num_cols = [c for c in X.columns if c not in cat_cols]
    return cat_cols, num_cols


class OneHotEncoderWrapper(BaseEncoder):
    def __init__(self):
        self._ct: Optional[ColumnTransformer] = None

    def fit(self, X: pd.DataFrame, y: Optional[np.ndarray] = None) -> "OneHotEncoderWrapper":
        cat_cols, num_cols = _split_cols(X)
        self._ct = ColumnTransformer(
            transformers=[
                ("num", Pipeline([("imp", SimpleImputer(strategy="median"))]), num_cols),
                (
                    "cat",
                    Pipeline(
                        [
                            ("imp", SimpleImputer(strategy="most_frequent")),
                            ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                        ]
                    ),
                    cat_cols,
                ),
            ],
            remainder="drop",
            verbose_feature_names_out=False,
        )
        self._ct.fit(X)
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        if self._ct is None:
            raise RuntimeError("Encoder not fit")
        return self._ct.transform(X)


class LabelEncoderWrapper(BaseEncoder):
    """Simple integer encoding (per column), with unknown handling."""

    def __init__(self):
        self._maps: Dict[str, Dict[str, int]] = {}
        self._cat_cols: List[str] = []
        self._num_cols: List[str] = []
        self._num_imputer = SimpleImputer(strategy="median")

    def fit(self, X: pd.DataFrame, y: Optional[np.ndarray] = None) -> "LabelEncoderWrapper":
        self._cat_cols, self._num_cols = _split_cols(X)
        # numeric
        self._num_imputer.fit(X[self._num_cols])
        # categorical
        self._maps = {}
        for c in self._cat_cols:
            ser = X[c].astype(str).fillna("__NA__")
            uniq = pd.unique(ser)
            # reserve 0 for unknown
            mapping = {v: i + 1 for i, v in enumerate(uniq.tolist())}
            self._maps[c] = mapping
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        num = self._num_imputer.transform(X[self._num_cols]) if self._num_cols else np.zeros((len(X), 0))
        cats = []
        for c in self._cat_cols:
            mapping = self._maps[c]
            ser = X[c].astype(str).fillna("__NA__")
            cats.append(ser.map(lambda v: mapping.get(v, 0)).to_numpy().reshape(-1, 1))
        cat_arr = np.concatenate(cats, axis=1) if cats else np.zeros((len(X), 0))
        return np.concatenate([num, cat_arr], axis=1).astype(np.float32)


class HelmertEncoderWrapper(BaseEncoder):
    """Helmert contrast coding for each categorical column.

    For a K-level category, produces (K-1) numeric columns. This is commonly
    used for linear models; we include it for completeness.
    """

    def __init__(self):
        self._levels: Dict[str, List[str]] = {}
        self._cat_cols: List[str] = []
        self._num_cols: List[str] = []
        self._num_imputer = SimpleImputer(strategy="median")

    @staticmethod
    def _helmert_matrix(k: int) -> np.ndarray:
        # Standard Helmert coding: columns j=1..k-1
        # For level i (1-indexed):
        #   if i <= j: 1
        #   if i == j+1: -j
        #   else: 0
        H = np.zeros((k, k - 1), dtype=np.float32)
        for j in range(1, k):
            H[:j, j - 1] = 1.0
            H[j, j - 1] = -float(j)
            # rows > j are already 0
        # Optional scaling to make contrasts orthonormal (common in stats).
        # We leave unscaled because many ML models do not require orthonormality.
        return H

    def fit(self, X: pd.DataFrame, y: Optional[np.ndarray] = None) -> "HelmertEncoderWrapper":
        self._cat_cols, self._num_cols = _split_cols(X)
        self._num_imputer.fit(X[self._num_cols])
        self._levels = {}
        for c in self._cat_cols:
            ser = X[c].astype(str).fillna("__NA__")
            self._levels[c] = pd.unique(ser).tolist()
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        num = self._num_imputer.transform(X[self._num_cols]) if self._num_cols else np.zeros((len(X), 0))
        cat_parts: List[np.ndarray] = []
        for c in self._cat_cols:
            levels = self._levels[c]
            k = len(levels)
            if k <= 1:
                continue
            H = self._helmert_matrix(k)
            idx = {v: i for i, v in enumerate(levels)}
            ser = X[c].astype(str).fillna("__NA__")
            ids = ser.map(lambda v: idx.get(v, 0)).to_numpy()
            cat_parts.append(H[ids])
        cat_arr = np.concatenate(cat_parts, axis=1) if cat_parts else np.zeros((len(X), 0))
        return np.concatenate([num, cat_arr], axis=1).astype(np.float32)


class HashingEncoderWrapper(BaseEncoder):
    """Feature hashing for categorical columns + numeric passthrough."""

    def __init__(self, n_features: int = 2**18):
        self.n_features = int(n_features)
        self._cat_cols: List[str] = []
        self._num_cols: List[str] = []
        self._num_imputer = SimpleImputer(strategy="median")
        self._hasher = FeatureHasher(n_features=self.n_features, input_type="string")

    def fit(self, X: pd.DataFrame, y: Optional[np.ndarray] = None) -> "HashingEncoderWrapper":
        self._cat_cols, self._num_cols = _split_cols(X)
        self._num_imputer.fit(X[self._num_cols])
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        num = self._num_imputer.transform(X[self._num_cols]) if self._num_cols else np.zeros((len(X), 0))
        if not self._cat_cols:
            return num.astype(np.float32)
        # build list of token strings per row
        tokens: List[List[str]] = []
        Xc = X[self._cat_cols].astype(str).fillna("__NA__")
        for _, row in Xc.iterrows():
            tokens.append([f"{c}={row[c]}" for c in self._cat_cols])
        hashed = self._hasher.transform(tokens).toarray().astype(np.float32)
        return np.concatenate([num, hashed], axis=1).astype(np.float32)


class TargetEncoderWrapper(BaseEncoder):
    """Mean target encoding with simple smoothing.

    This is a *supervised* encoder. To avoid leakage, call fit only on the
    training split.
    """

    def __init__(self, smoothing: float = 20.0):
        self.smoothing = float(smoothing)
        self._global_mean: float = 0.0
        self._stats: Dict[str, pd.DataFrame] = {}
        self._cat_cols: List[str] = []
        self._num_cols: List[str] = []
        self._num_imputer = SimpleImputer(strategy="median")

    def fit(self, X: pd.DataFrame, y: Optional[np.ndarray] = None) -> "TargetEncoderWrapper":
        if y is None:
            raise ValueError("TargetEncoder requires y")
        self._cat_cols, self._num_cols = _split_cols(X)
        self._num_imputer.fit(X[self._num_cols])
        y = np.asarray(y).astype(float)
        self._global_mean = float(np.mean(y))
        self._stats = {}
        for c in self._cat_cols:
            ser = X[c].astype(str).fillna("__NA__")
            df = pd.DataFrame({"cat": ser, "y": y})
            g = df.groupby("cat").agg(cnt=("y", "size"), mean=("y", "mean")).reset_index()
            # smoothing: blended mean
            g["te"] = (g["cnt"] * g["mean"] + self.smoothing * self._global_mean) / (g["cnt"] + self.smoothing)
            self._stats[c] = g.set_index("cat")[["te"]]
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        num = self._num_imputer.transform(X[self._num_cols]) if self._num_cols else np.zeros((len(X), 0))
        cat_parts: List[np.ndarray] = []
        for c in self._cat_cols:
            ser = X[c].astype(str).fillna("__NA__")
            lookup = self._stats[c]["te"]
            vals = ser.map(lambda v: float(lookup.get(v, self._global_mean))).to_numpy().reshape(-1, 1)
            cat_parts.append(vals)
        cat_arr = np.concatenate(cat_parts, axis=1) if cat_parts else np.zeros((len(X), 0))
        return np.concatenate([num, cat_arr], axis=1).astype(np.float32)
