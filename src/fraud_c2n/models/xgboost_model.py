
from __future__ import annotations
from typing import Dict, Optional
import numpy as np
from xgboost import XGBClassifier

def build_xgb(params: Dict, seed: int) -> XGBClassifier:
    p = dict(params)
    p.setdefault("random_state", seed)
    p.setdefault("verbosity", 0)
    p.setdefault("objective", "binary:logistic")
    # deterministic-ish
    p.setdefault("nthread", p.get("n_jobs", -1))
    return XGBClassifier(**p)

def fit_xgb(model: XGBClassifier, X_tr: np.ndarray, y_tr: np.ndarray, X_va: Optional[np.ndarray]=None, y_va: Optional[np.ndarray]=None):
    if X_va is not None and y_va is not None and len(y_va)>0:
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    else:
        model.fit(X_tr, y_tr, verbose=False)
    return model
