from __future__ import annotations

import hashlib
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import normalized_mutual_info_score


def split_cols(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Return (categorical_cols, numerical_cols)."""

    cat_cols = [c for c in X.columns if X[c].dtype == object or str(X[c].dtype).startswith("category")]
    num_cols = [c for c in X.columns if c not in cat_cols]
    return cat_cols, num_cols


def stable_hash_to_int(s: str, seed: int, modulo: int) -> int:
    """Stable hash -> [0, modulo)."""

    h = hashlib.md5(f"{seed}|{s}".encode("utf-8")).hexdigest()
    return int(h[:8], 16) % modulo


def compute_feature_nmi(df_cat: pd.DataFrame) -> Dict[Tuple[str, str], float]:
    """Compute NMI between every pair of categorical columns."""

    cols = list(df_cat.columns)
    out: Dict[Tuple[str, str], float] = {}
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            a = df_cat[cols[i]].astype(str).fillna("__NA__")
            b = df_cat[cols[j]].astype(str).fillna("__NA__")
            out[(cols[i], cols[j])] = float(normalized_mutual_info_score(a, b))
    return out


def build_global_value_index(df_cat: pd.DataFrame) -> tuple[list[str], dict[tuple[str, str], int]]:
    """Return list of global value ids and (col,val)->global_idx.

    Each categorical value is namespaced by its column:
        token = f"{col}={val}".
    """

    values: List[str] = []
    mapping: Dict[tuple[str, str], int] = {}
    for c in df_cat.columns:
        ser = df_cat[c].astype(str).fillna("__NA__")
        for v in pd.unique(ser).tolist():
            mapping[(c, v)] = len(values)
            values.append(f"{c}={v}")
    return values, mapping
