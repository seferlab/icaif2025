
from __future__ import annotations
from typing import List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.feature_extraction import FeatureHasher
import category_encoders as ce

from abc import ABC, abstractmethod
#import sys
#sys.path.append(".")
#from base import Encoder

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


class OneHotEncoderCE(Encoder):
    def __init__(self, cat_cols: List[str]):
        super().__init__(cat_cols)
        self.enc = ce.OneHotEncoder(cols=cat_cols, handle_unknown="value", handle_missing="value", use_cat_names=True)

    def fit(self, X: pd.DataFrame, y=None):
        self.enc.fit(X[self.cat_cols], y)
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        return self.enc.transform(X[self.cat_cols]).to_numpy(dtype=np.float32)

class OrdinalEncoderCE(Encoder):
    def __init__(self, cat_cols: List[str]):
        super().__init__(cat_cols)
        self.enc = ce.OrdinalEncoder(cols=cat_cols, handle_unknown="value", handle_missing="value")

    def fit(self, X: pd.DataFrame, y=None):
        self.enc.fit(X[self.cat_cols], y)
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        return self.enc.transform(X[self.cat_cols]).to_numpy(dtype=np.float32)

class HelmertEncoderCE(Encoder):
    def __init__(self, cat_cols: List[str]):
        super().__init__(cat_cols)
        self.enc = ce.HelmertEncoder(cols=cat_cols, handle_unknown="value", handle_missing="value")

    def fit(self, X: pd.DataFrame, y=None):
        self.enc.fit(X[self.cat_cols], y)
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        return self.enc.transform(X[self.cat_cols]).to_numpy(dtype=np.float32)

class HashingEncoder(Encoder):
    def __init__(self, cat_cols: List[str], n_features: int = 2000):
        super().__init__(cat_cols)
        self.n_features = int(n_features)
        self.hasher = FeatureHasher(n_features=self.n_features, input_type="string")

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        if len(self.cat_cols)==0:
            return np.zeros((len(X), 0), dtype=np.float32)
        rows = []
        for _, r in X[self.cat_cols].iterrows():
            # represent row as list of "col=value" tokens
            tokens = [f"{c}={r[c]}" for c in self.cat_cols]
            rows.append(tokens)
        mat = self.hasher.transform(rows)
        return mat.toarray().astype(np.float32)

def build_encoder(encoder_type: str, cat_cols: List[str], **kwargs) -> Encoder:
    t = encoder_type.lower()
    if t == "onehot":
        return OneHotEncoderCE(cat_cols)
    if t == "label":
        return OrdinalEncoderCE(cat_cols)
    if t == "helmert" or t == "helmet":
        return HelmertEncoderCE(cat_cols)
    if t == "hashing" or t == "feature":
        return HashingEncoder(cat_cols, n_features=kwargs.get("n_features", 2000))
    raise ValueError(f"Unknown encoder_type: {encoder_type}")
