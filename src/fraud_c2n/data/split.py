
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, List
import numpy as np
import pandas as pd

@dataclass
class Split:
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray

def _year_mask(years: pd.Series, yr_range: List[int]) -> np.ndarray:
    y0,y1 = int(yr_range[0]), int(yr_range[1])
    return (years>=y0) & (years<=y1)

def make_splits(df: pd.DataFrame, year_col: str, splits_cfg: Dict[str, Dict]) -> Dict[int, Split]:
    years = df[year_col].astype(int)
    splits: Dict[int, Split] = {}
    for k_str, cfg in splits_cfg.items():
        k=int(k_str)
        tr = _year_mask(years, cfg["train_years"])
        va = _year_mask(years, cfg["val_years"])
        te = _year_mask(years, cfg["test_years"])
        splits[k]=Split(np.where(tr)[0], np.where(va)[0], np.where(te)[0])
    return splits
