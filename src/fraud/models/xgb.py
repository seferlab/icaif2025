from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import sparse

import xgboost as xgb

class XGBoostClassifier:
    def __init__(self, n_estimators: int = 300, max_depth: int = 5, learning_rate: float = 0.05, subsample: float = 0.9, colsample_bytree: float = 0.9):
        self.params = dict(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
        )
        self.model = None

    def fit(self, X, y, X_val_num=None, y_val=None, seed: int = 0):
        self.params["random_state"] = seed
        self.model = xgb.XGBClassifier(**self.params)
        if X_val_num is not None and y_val is not None:
            self.model.fit(X, y, eval_set=[(X_val_num, y_val)], verbose=False)
        else:
            self.model.fit(X, y, verbose=False)
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(X)
