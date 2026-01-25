from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml

from fraud.metrics import compute_metrics
from fraud.datasets import load_dataset_csv, apply_year_split

def available_encoders() -> Dict[str, object]:
    from fraud.encoders.baseline_encoders import (
        OneHotEncoderWrapper, LabelEncoderWrapper, HelmertEncoderWrapper, HashingEncoderWrapper, TargetEncoderWrapper
    )
    from fraud.encoders.cure_wrapper import CUREWrapper
    from fraud.encoders.gce_wrapper import GCEWrapper
    from fraud.encoders.distance_wrapper import DistanceEmbeddingWrapper
    from fraud.encoders.dhe_wrapper import DHEWrapper

    return {
        "onehot": OneHotEncoderWrapper(),
        "label": LabelEncoderWrapper(),
        "helmert": HelmertEncoderWrapper(),
        "hashing": HashingEncoderWrapper(),
        "target": TargetEncoderWrapper(),
        "cure": CUREWrapper(),
        "gce": GCEWrapper(),
        "distance": DistanceEmbeddingWrapper(),
        "dhe": DHEWrapper(),
    }

def available_classifiers() -> Dict[str, object]:
    from fraud.models.xgb import XGBoostClassifier
    from fraud.models.mlp import MLPClassifier
    return {"xgb": XGBoostClassifier(), "mlp": MLPClassifier()}

@dataclass
class ResultRow:
    dataset: str
    subset: int
    encoder: str
    classifier: str
    seed: int
    auc_roc: float
    auc_pr: float
    f1_macro: float
    recall_macro: float
    n_train: int
    n_val: int
    n_test: int

def run_one(
    df: pd.DataFrame,
    cfg: dict,
    subset_cfg: dict,
    dataset_name: str,
    encoder_name: str,
    clf_name: str,
    seed: int,
    models_dir: Path,
) -> ResultRow:
    np.random.seed(seed)

    time_col = cfg["time_column"]
    label_col = cfg["label_column"]

    train_df, val_df, test_df = apply_year_split(
        df, time_col, subset_cfg["train_years"], subset_cfg["val_years"], subset_cfg["test_years"]
    )

    y_train = train_df[label_col].astype(int).to_numpy()
    y_val   = val_df[label_col].astype(int).to_numpy()
    y_test  = test_df[label_col].astype(int).to_numpy()

    X_train = train_df.drop(columns=[label_col])
    X_val   = val_df.drop(columns=[label_col])
    X_test  = test_df.drop(columns=[label_col])

    enc = available_encoders()[encoder_name]
    clf = available_classifiers()[clf_name]

    X_train_num = enc.fit_transform(X_train, y_train)
    X_val_num   = enc.transform(X_val)
    X_test_num  = enc.transform(X_test)

    clf.fit(X_train_num, y_train, X_val_num=X_val_num, y_val=y_val, seed=seed)

    proba_test = clf.predict_proba(X_test_num)[:, 1]
    preds_test = (proba_test >= 0.5).astype(int)

    m = compute_metrics(y_test, proba_test, preds_test)

    # Save model-side artifacts for plots (only for specialized encoders by default)
    subset_id = int(subset_cfg["id"])
    out_subset_dir = models_dir / dataset_name / f"subset_{subset_id}"
    if hasattr(enc, "save_artifacts") and encoder_name in {"cure", "gce", "distance", "dhe"}:
        enc.save_artifacts(out_subset_dir, X_train)
        # Also save the test split for fallback heatmaps
        test_df.to_csv(out_subset_dir / "subset_test.csv", index=False)

    return ResultRow(
        dataset=dataset_name,
        subset=subset_id,
        encoder=encoder_name,
        classifier=clf_name,
        seed=seed,
        auc_roc=m["auc_roc"],
        auc_pr=m["auc_pr"],
        f1_macro=m["f1_macro"],
        recall_macro=m["recall_macro"],
        n_train=len(train_df),
        n_val=len(val_df),
        n_test=len(test_df),
    )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["usfsd", "figraph", "all"])
    ap.add_argument("--usfsd-path", default="data/usfsd.csv")
    ap.add_argument("--figraph-path", default="data/figraph.csv")
    ap.add_argument("--splits-config-dir", default="configs")
    ap.add_argument("--out-dir", default="artifacts/results")
    ap.add_argument("--models-dir", default="artifacts/models")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seeds", default="0,1,2")
    args = ap.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    models_dir = Path(args.models_dir); models_dir.mkdir(parents=True, exist_ok=True)

    if args.quick:
        encoders = ["cure", "gce"]
        classifiers = ["xgb"]
        seeds = seeds[:1]
    else:
        encoders = ["onehot", "label", "helmert", "hashing", "target", "cure", "gce", "distance", "dhe"]
        classifiers = ["xgb", "mlp"]

    jobs: List[Tuple[str, str, str]] = []
    if args.dataset in ("usfsd", "all"):
        jobs.append(("usfsd", args.usfsd_path, str(Path(args.splits_config_dir) / "usfsd_splits.yaml")))
    if args.dataset in ("figraph", "all"):
        jobs.append(("figraph", args.figraph_path, str(Path(args.splits_config_dir) / "figraph_splits.yaml")))

    all_rows: List[ResultRow] = []

    for dataset_name, csv_path, split_path in jobs:
        df = load_dataset_csv(csv_path)
        cfg = yaml.safe_load(Path(split_path).read_text())

        for subset_cfg in cfg["subsets"]:
            for enc in encoders:
                for clf in classifiers:
                    for seed in seeds:
                        row = run_one(df, cfg, subset_cfg, dataset_name, enc, clf, seed, models_dir=models_dir)
                        all_rows.append(row)

    metrics_df = pd.DataFrame([asdict(r) for r in all_rows])
    metrics_df.to_parquet(out_dir / "metrics.parquet", index=False)
    (out_dir / "run_manifest.json").write_text(json.dumps({
        "dataset": args.dataset,
        "quick": bool(args.quick),
        "encoders": encoders,
        "classifiers": classifiers,
        "seeds": seeds,
    }, indent=2))

    print(f"[OK] Wrote {len(metrics_df)} rows to {out_dir/'metrics.parquet'}")

if __name__ == "__main__":
    main()
