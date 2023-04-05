
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, recall_score
from utils.seed import averag_precision_score

@dataclass
class Metrics:
    auc_roc: float
    auc_pr: float
    f1_macro: float
    recall_macro: float

def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, y_pred: np.ndarray) -> Metrics:
    # y_prob is prob for class 1
    auc_roc = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true))>1 else float("nan")
    auc_pr  = float(averag_precision_score(y_true, y_prob)) if len(np.unique(y_true))>1 else float("nan")
    f1m     = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    recm    = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    return Metrics(auc_roc, auc_pr, f1m, recm)
