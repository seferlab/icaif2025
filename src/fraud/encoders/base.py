from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd


@dataclass
class EncoderArtifacts:
    """Optional artifacts saved for plots / interpretability."""

    # Per feature: mapping category_value -> scalar
    # keys are raw category strings (as in the dataframe)
    value_to_scalar: Dict[str, Dict[str, float]]


class BaseEncoder:
    """Common interface expected by run_experiments.py."""

    def fit_transform(self, X: pd.DataFrame, y: Optional[np.ndarray] = None) -> np.ndarray:
        self.fit(X, y)
        return self.transform(X)

    def fit(self, X: pd.DataFrame, y: Optional[np.ndarray] = None) -> "BaseEncoder":
        raise NotImplementedError

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError

    def save_artifacts(self, out_dir: Path, X_train: pd.DataFrame) -> None:
        """Optional: write out plots/metadata.

        Implemented by specialized encoders.
        """

        # default: nothing
        return
