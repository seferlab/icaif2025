
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np
import pandas as pd

@dataclass
class DatasetBundle:
    df: pd.DataFrame
    year_col: str
    label_col: str
    cat_cols: List[str]
    num_cols: List[str]

def infer_columns(df: pd.DataFrame, year_col: str, label_col: str, explicit_cat: Optional[List[str]]=None) -> Tuple[List[str], List[str]]:
    if explicit_cat is not None and len(explicit_cat)>0:
        cat_cols = [c for c in explicit_cat if c in df.columns]
    else:
        # heuristic: object/category columns + low-cardinality integer columns excluding year/label
        cat_cols = []
        for c in df.columns:
            if c in [year_col, label_col]:
                continue
            if df[c].dtype == "object" or str(df[c].dtype).startswith("category"):
                cat_cols.append(c)
            elif np.issubdtype(df[c].dtype, np.integer):
                nunq = df[c].nunique(dropna=True)
                if nunq <= 50:
                    cat_cols.append(c)
        cat_cols = sorted(set(cat_cols))
    num_cols = [c for c in df.columns if c not in [year_col, label_col] and c not in cat_cols]
    return cat_cols, num_cols

def basic_clean(df: pd.DataFrame, cat_cols: List[str], num_cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cat_cols:
        out[c] = out[c].astype("string").fillna("__NA__")
    for c in num_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce").astype("float32")
        out[c] = out[c].fillna(out[c].median())
    return out
