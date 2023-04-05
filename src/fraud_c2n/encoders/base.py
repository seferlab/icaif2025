
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple
import numpy as np
import pandas as pd

class Encoder(ABC):
    def __init__(self, cat_cols: List[str]):
        self.cat_cols = cat_cols

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: Optional[np.ndarray]=None) -> "Encoder":
        ...

    @abstractmethod
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        ...

    def fit_transform(self, X: pd.DataFrame, y: Optional[np.ndarray]=None) -> np.ndarray:
        self.fit(X,y)
        return self.transform(X)
