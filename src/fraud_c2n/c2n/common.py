
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

@dataclass
class ValueVocab:
    # mapping from (col,value) -> idx
    key_to_idx: Dict[str, int]
    idx_to_key: List[str]

def build_vocab(df: pd.DataFrame, cat_cols: List[str], max_vocab: int | None=None) -> ValueVocab:
    keys=[]
    for c in cat_cols:
        vc = df[c].astype("string").value_counts()
        for v in vc.index.tolist():
            keys.append(f"{c}={v}")
    if max_vocab is not None and len(keys) > max_vocab:
        keys = keys[:max_vocab]
    key_to_idx={k:i for i,k in enumerate(keys)}
    return ValueVocab(key_to_idx, keys)

def row_value_indices(df: pd.DataFrame, cat_cols: List[str], vocab: ValueVocab) -> List[List[int]]:
    rows=[]
    for _,r in df[cat_cols].iterrows():
        idxs=[]
        for c in cat_cols:
            k=f"{c}={r[c]}"
            if k in vocab.key_to_idx:
                idxs.append(vocab.key_to_idx[k])
        rows.append(idxs)
    return rows

def pool_embeddings(rows: List[List[int]], emb: np.ndarray, mode: str="mean") -> np.ndarray:
    d=emb.shape[1]
    out=np.zeros((len(rows), d), dtype=np.float32)
    for i,idxs in enumerate(rows):
        if not idxs:
            continue
        vec=emb[idxs]
        if mode=="sum":
            out[i]=vec.sum(axis=0)
        else:
            out[i]=vec.mean(axis=0)
    return out
