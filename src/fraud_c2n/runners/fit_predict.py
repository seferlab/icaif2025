
from __future__ import annotations
from typing import Dict, Tuple, Optional, Any
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import sys
sys.path.append("..")
from utils.metrics import compute_metrics
from models.xgboost_model import build_xgb, fit_xgb
from models.mlp import build_mlp
from models.deepcrossing import DeepCrossing
from models.dcn import DCN, DCNV2
from models.fttransformer import FTTransformer

def train_eval_xgb(X_tr: np.ndarray, y_tr: np.ndarray, X_va: np.ndarray, y_va: np.ndarray, X_te: np.ndarray, y_te: np.ndarray,
                   xgb_params: Dict, seed: int) -> Tuple[Dict, np.ndarray]:
    model = build_xgb(xgb_params, seed)
    model = fit_xgb(model, X_tr, y_tr, X_va, y_va)
    prob = model.predict_proba(X_te)[:,1]
    pred = (prob>=0.5).astype(int)
    m = compute_metrics(y_te, prob, pred)
    return {"auc_roc":m.auc_roc,"auc_pr":m.auc_pr,"f1_macro":m.f1_macro,"recall_macro":m.recall_macro}, prob

def train_eval_mlp(X_tr: np.ndarray, y_tr: np.ndarray, X_va: np.ndarray, y_va: np.ndarray, X_te: np.ndarray, y_te: np.ndarray,
                   mlp_params: Dict, seed: int) -> Tuple[Dict, np.ndarray]:
    model = build_mlp(mlp_params, seed)
    model.fit(np.vstack([X_tr, X_va]), np.concatenate([y_tr, y_va]))
    prob = model.predict_proba(X_te)[:,1]
    pred = (prob>=0.5).astype(int)
    m = compute_metrics(y_te, prob, pred)
    return {"auc_roc":m.auc_roc,"auc_pr":m.auc_pr,"f1_macro":m.f1_macro,"recall_macro":m.recall_macro}, prob


def _torch_train_binary(
    model: nn.Module,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_va: np.ndarray,
    y_va: np.ndarray,
    X_te: np.ndarray,
    y_te: np.ndarray,
    lr: float,
    weight_decay: float,
    batch_size: int,
    max_epochs: int,
    patience: int,
    device: str = "cpu",
    seed: int = 42,
) -> Tuple[Dict, np.ndarray]:
    torch.manual_seed(seed)
    device_t = torch.device(device)
    model = model.to(device_t)

    Xtr = torch.tensor(X_tr, dtype=torch.float32)
    ytr = torch.tensor(y_tr, dtype=torch.float32)
    Xva = torch.tensor(X_va, dtype=torch.float32)
    yva = torch.tensor(y_va, dtype=torch.float32)
    Xte = torch.tensor(X_te, dtype=torch.float32)

    tr_loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=batch_size, shuffle=True)
    va_loader = DataLoader(TensorDataset(Xva, yva), batch_size=batch_size, shuffle=False)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()

    best_state: Optional[dict[str, Any]] = None
    best_score = -1e18
    bad = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        for xb, yb in tr_loader:
            xb = xb.to(device_t)
            yb = yb.to(device_t)
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()

        # validation
        model.eval()
        all_prob = []
        all_y = []
        with torch.no_grad():
            for xb, yb in va_loader:
                xb = xb.to(device_t)
                logits = model(xb)
                prob = torch.sigmoid(logits).cpu().numpy()
                all_prob.append(prob)
                all_y.append(yb.numpy())
        vprob = np.concatenate(all_prob)
        vy = np.concatenate(all_y).astype(int)
        vpred = (vprob >= 0.5).astype(int)
        vm = compute_metrics(vy, vprob, vpred)
        score = vm.auc_pr  # match fraud-recall oriented objective

        if score > best_score + 1e-6:
            best_score = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        prob = torch.sigmoid(model(Xte.to(device_t))).cpu().numpy()
    pred = (prob >= 0.5).astype(int)
    m = compute_metrics(y_te, prob, pred)
    return {"auc_roc": m.auc_roc, "auc_pr": m.auc_pr, "f1_macro": m.f1_macro, "recall_macro": m.recall_macro}, prob


def train_eval_catboost(
    df_tr: pd.DataFrame,
    y_tr: np.ndarray,
    df_va: pd.DataFrame,
    y_va: np.ndarray,
    df_te: pd.DataFrame,
    y_te: np.ndarray,
    cat_cols: list[str],
    num_cols: list[str],
    params: Dict,
    seed: int,
) -> Tuple[Dict, np.ndarray]:
    from catboost import CatBoostClassifier

    Xtr = df_tr[cat_cols + num_cols].copy()
    Xva = df_va[cat_cols + num_cols].copy()
    Xte = df_te[cat_cols + num_cols].copy()

    # CatBoost expects categorical columns as strings
    for c in cat_cols:
        Xtr[c] = Xtr[c].astype(str)
        Xva[c] = Xva[c].astype(str)
        Xte[c] = Xte[c].astype(str)

    # numeric cast
    for c in num_cols:
        Xtr[c] = Xtr[c].astype(float)
        Xva[c] = Xva[c].astype(float)
        Xte[c] = Xte[c].astype(float)


    #The problematic part solution
    AdaBoost_accuracy = 0

    while AdaBoost_accuracy == 0:
        try:
            cat_idx = list(range(len(cat_cols)))
            p = dict(params)
            p["random_seed"] = seed
            model = CatBoostClassifier(**p)
            model.fit(Xtr, y_tr, eval_set=(Xva, y_va), cat_features=cat_idx, use_best_model=True)
            prob = model.predict_proba(Xte)[:, 1]
            pred = (prob >= 0.5).astype(int)
            m = compute_metrics(y_te, prob, pred)
            AdaBoost_accuracy = m.auc_roc
        #classifier = AdaBoostClassifier(base_estimator=Perceptron(), n_estimators=15, algorithm='SAMME')
        #classifier = classifier.fit(x_train, y_train)
        #y_pred = classifier.predict(x_test)
        #AdaBoost_accuracy = metrics.accuracy_score(y_test, y_pred)
        except:
            print("Let me reclassify AdaBoost again")

     #print("Accuracy of AdaBoost:", AdaBoost_accuracy)

    #cat_idx = list(range(len(cat_cols)))
    #p = dict(params)
    #p["random_seed"] = seed
    #model = CatBoostClassifier(**p)
    #model.fit(Xtr, y_tr, eval_set=(Xva, y_va), cat_features=cat_idx, use_best_model=True)
    #prob = model.predict_proba(Xte)[:, 1]
    #pred = (prob >= 0.5).astype(int)
    #m = compute_metrics(y_te, prob, pred)
    return {"auc_roc": m.auc_roc, "auc_pr": m.auc_pr, "f1_macro": m.f1_macro, "recall_macro": m.recall_macro}, prob


def train_eval_rusboost(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_va: np.ndarray,
    y_va: np.ndarray,
    X_te: np.ndarray,
    y_te: np.ndarray,
    params: Dict,
    seed: int,
) -> Tuple[Dict, np.ndarray]:
    from imblearn.ensemble import RUSBoostClassifier

    AdaBoost_accuracy = 0
    while AdaBoost_accuracy == 0:
        try:
            p = dict(params)
            p["random_state"] = seed
            model = RUSBoostClassifier(**p)
            model.fit(np.vstack([X_tr, X_va]), np.concatenate([y_tr, y_va]))
            prob = model.predict_proba(X_te)[:, 1]
            pred = (prob >= 0.5).astype(int)
            m = compute_metrics(y_te, prob, pred)
        #classifier = AdaBoostClassifier(base_estimator=Perceptron(), n_estimators=15, algorithm='SAMME')
        #classifier = classifier.fit(x_train, y_train)
        #y_pred = classifier.predict(x_test)
        #AdaBoost_accuracy = metrics.accuracy_score(y_test, y_pred)
        except:
            print("Let me reclassify RusBoost again")
            
    #p = dict(params)
    #p["random_state"] = seed
    #model = RUSBoostClassifier(**p)
    #model.fit(np.vstack([X_tr, X_va]), np.concatenate([y_tr, y_va]))
    #prob = model.predict_proba(X_te)[:, 1]
    #pred = (prob >= 0.5).astype(int)
    #m = compute_metrics(y_te, prob, pred)
    return {"auc_roc": m.auc_roc, "auc_pr": m.auc_pr, "f1_macro": m.f1_macro, "recall_macro": m.recall_macro}, prob


def train_eval_xgbod(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_va: np.ndarray,
    y_va: np.ndarray,
    X_te: np.ndarray,
    y_te: np.ndarray,
    params: Dict,
    seed: int,
) -> Tuple[Dict, np.ndarray]:
    # XGBOD in pyod uses labeled data (semi-supervised). We train on train+val.
    from pyod.models.xgbod import XGBOD

    p = dict(params)
    p["random_state"] = seed
    model = XGBOD(**p)
    model.fit(np.vstack([X_tr, X_va]), np.concatenate([y_tr, y_va]))
    # decision_function gives outlier scores; convert to pseudo-prob via sigmoid
    scores = model.decision_function(X_te)
    prob = 1.0 / (1.0 + np.exp(-scores))
    pred = (prob >= 0.5).astype(int)
    m = compute_metrics(y_te, prob, pred)
    return {"auc_roc": m.auc_roc, "auc_pr": m.auc_pr, "f1_macro": m.f1_macro, "recall_macro": m.recall_macro}, prob


def train_eval_tabnet(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_va: np.ndarray,
    y_va: np.ndarray,
    X_te: np.ndarray,
    y_te: np.ndarray,
    params: Dict,
    seed: int,
    device: str = "cpu",
) -> Tuple[Dict, np.ndarray]:
    from pytorch_tabnet.tab_model import TabNetClassifier

    p = dict(params)
    max_epochs = int(p.pop("max_epochs"))
    patience = int(p.pop("patience"))
    batch_size = int(p.pop("batch_size"))
    virtual_batch_size = int(p.pop("virtual_batch_size"))
    num_workers = int(p.pop("num_workers"))

    opt_params = p.pop("optimizer_params", {"lr": 0.02})
    optimizer_fn_name = p.pop("optimizer_fn", "adam").lower()
    optimizer_fn = torch.optim.Adam
    if optimizer_fn_name == "adamw":
        optimizer_fn = torch.optim.AdamW

    model = TabNetClassifier(
        **p,
        optimizer_fn=optimizer_fn,
        optimizer_params=opt_params,
        #seed=seed,
        device_name="cuda" if (device.startswith("cuda") and torch.cuda.is_available()) else "cpu",
    )
    model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        eval_name=["val"],
        eval_metric=["auc"],
        max_epochs=max_epochs,
        patience=patience,
        batch_size=batch_size,
        virtual_batch_size=virtual_batch_size,
        num_workers=num_workers,
        drop_last=False,
    )
    prob = model.predict_proba(X_te)[:, 1]
    pred = (prob >= 0.5).astype(int)
    m = compute_metrics(y_te, prob, pred)
    return {"auc_roc": m.auc_roc, "auc_pr": m.auc_pr, "f1_macro": m.f1_macro, "recall_macro": m.recall_macro}, prob


def train_eval_deepcrossing(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_va: np.ndarray,
    y_va: np.ndarray,
    X_te: np.ndarray,
    y_te: np.ndarray,
    params: Dict,
    seed: int,
    device: str = "cpu",
) -> Tuple[Dict, np.ndarray]:
    model = DeepCrossing(int(X_tr.shape[1]), hidden_dims=list(params["hidden_dims"]), dropout=float(params.get("dropout", 0.0)))
    return _torch_train_binary(
        model,
        X_tr, y_tr, X_va, y_va, X_te, y_te,
        lr=float(params.get("lr", 1e-3)),
        weight_decay=float(params.get("weight_decay", 0.0)),
        batch_size=int(params.get("batch_size", 1024)),
        max_epochs=int(params.get("max_epochs", 100)),
        patience=int(params.get("patience", 10)),
        device=device,
        seed=seed,
    )


def train_eval_dcn(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_va: np.ndarray,
    y_va: np.ndarray,
    X_te: np.ndarray,
    y_te: np.ndarray,
    params: Dict,
    seed: int,
    device: str = "cpu",
) -> Tuple[Dict, np.ndarray]:
    model = DCN(
        input_dim=int(X_tr.shape[1]),
        cross_layers=int(params.get("cross_layers", 3)),
        deep_dims=list(params.get("deep_dims", [])),
        dropout=float(params.get("dropout", 0.0)),
    )
    return _torch_train_binary(
        model,
        X_tr, y_tr, X_va, y_va, X_te, y_te,
        lr=float(params.get("lr", 1e-3)),
        weight_decay=float(params.get("weight_decay", 0.0)),
        batch_size=int(params.get("batch_size", 1024)),
        max_epochs=int(params.get("max_epochs", 100)),
        patience=int(params.get("patience", 10)),
        device=device,
        seed=seed,
    )


def train_eval_dcnv2(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_va: np.ndarray,
    y_va: np.ndarray,
    X_te: np.ndarray,
    y_te: np.ndarray,
    params: Dict,
    seed: int,
    device: str = "cpu",
) -> Tuple[Dict, np.ndarray]:
    model = DCNV2(
        input_dim=int(X_tr.shape[1]),
        cross_layers=int(params.get("cross_layers", 3)),
        low_rank=int(params.get("low_rank", 32)),
        num_experts=int(params.get("num_experts", 4)),
        deep_dims=list(params.get("deep_dims", [])),
        dropout=float(params.get("dropout", 0.0)),
    )
    return _torch_train_binary(
        model,
        X_tr, y_tr, X_va, y_va, X_te, y_te,
        lr=float(params.get("lr", 1e-3)),
        weight_decay=float(params.get("weight_decay", 0.0)),
        batch_size=int(params.get("batch_size", 1024)),
        max_epochs=int(params.get("max_epochs", 100)),
        patience=int(params.get("patience", 10)),
        device=device,
        seed=seed,
    )


def train_eval_fttransformer(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_va: np.ndarray,
    y_va: np.ndarray,
    X_te: np.ndarray,
    y_te: np.ndarray,
    params: Dict,
    seed: int,
    device: str = "cpu",
) -> Tuple[Dict, np.ndarray]:
    model = FTTransformer(
        input_dim=int(X_tr.shape[1]),
        d_model=int(params.get("d_model", 128)),
        n_heads=int(params.get("n_heads", 8)),
        n_layers=int(params.get("n_layers", 3)),
        dropout=float(params.get("dropout", 0.1)),
    )
    return _torch_train_binary(
        model,
        X_tr, y_tr, X_va, y_va, X_te, y_te,
        lr=float(params.get("lr", 1e-3)),
        weight_decay=float(params.get("weight_decay", 0.0)),
        batch_size=int(params.get("batch_size", 1024)),
        max_epochs=int(params.get("max_epochs", 100)),
        patience=int(params.get("patience", 10)),
        device=device,
        seed=seed,
    )
