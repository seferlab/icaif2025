
from __future__ import annotations
from typing import Dict
import numpy as np
from sklearn.neural_network import MLPClassifier

def build_mlp(params: Dict, seed: int) -> MLPClassifier:
    p = dict(params)
    p.setdefault("random_state", seed)
    p.setdefault("early_stopping", True)
    p.setdefault("n_iter_no_change", 5)
    p.setdefault("validation_fraction", 0.1)
    p.setdefault("verbose", False)
    return MLPClassifier(**p)
