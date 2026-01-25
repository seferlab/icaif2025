from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import sparse

from .baseline_encoders import OneHotEncoderWrapper

class SpecialEncoderBase:
    """
    Placeholder base for specialized C2N encoders.
    It provides a simple numeric transform (one-hot) so the codebase runs end-to-end,
    and optionally saves:
      - attention_matrix.npz (A, feature_names)
      - category_projection.csv (feature, category, value)
    Replace internals with your paper-accurate methods.
    """

    name: str = "special"

    def __init__(self):
        self._ohe = OneHotEncoderWrapper()
        self.feature_names_: List[str] = []

    def fit_transform(self, X: pd.DataFrame, y):
        Xnum = self._ohe.fit_transform(X, y)
        # Produce feature names from numeric + one-hot feature names
        self.feature_names_ = list(X.columns)
        return Xnum

    def transform(self, X: pd.DataFrame):
        return self._ohe.transform(X)

    def export_attention(self) -> Tuple[np.ndarray, List[str]]:
        # Demo: random symmetric interaction matrix
        d = max(5, min(30, len(self.feature_names_)))
        rng = np.random.default_rng(0)
        A = rng.normal(size=(d, d))
        A = (A + A.T) / 2.0
        names = self.feature_names_[:d]
        return A, names

    def export_category_projection(self, X: pd.DataFrame) -> pd.DataFrame:
        # Demo: pick up to 6 binary/low-card categorical cols and map categories to numbers
        rows = []
        for c in X.columns:
            if X[c].dtype == "object" or str(X[c].dtype).startswith("category"):
                vals = pd.Series(X[c].astype(str).fillna("__NA__")).unique().tolist()
                if 2 <= len(vals) <= 6:
                    for i, v in enumerate(vals):
                        rows.append({"feature": c, "category": v, "value": float(i)})
        return pd.DataFrame(rows)

    def save_artifacts(self, out_dir: Path, X_train: pd.DataFrame):
        out_dir.mkdir(parents=True, exist_ok=True)
        A, names = self.export_attention()
        np.savez(out_dir / "attention_matrix.npz", A=A, feature_names=np.array(names, dtype=object))
        proj = self.export_category_projection(X_train)
        if len(proj):
            proj.to_csv(out_dir / "category_projection.csv", index=False)
