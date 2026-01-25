from __future__ import annotations
from typing import Dict
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, recall_score

def compute_metrics(y_true: np.ndarray, y_proba: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "auc_roc": float(roc_auc_score(y_true, y_proba)),
        "auc_pr": float(average_precision_score(y_true, y_proba)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro")),
    }
