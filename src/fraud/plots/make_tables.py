from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

METRICS = ["auc_roc", "auc_pr", "f1_macro", "recall_macro"]

def _fmt_mean_std(series: pd.Series) -> str:
    mu = series.mean()
    sd = series.std(ddof=1) if len(series) > 1 else 0.0
    return f"{mu:.4f} ± {sd:.4f}"

def _agg_metrics(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    rows = []
    g = df.groupby(group_cols, dropna=False)
    for key, sub in g:
        if not isinstance(key, tuple):
            key = (key,)
        out = {col: val for col, val in zip(group_cols, key)}
        for m in METRICS:
            out[m] = _fmt_mean_std(sub[m])
            out[f"{m}__mean"] = float(sub[m].mean())
        rows.append(out)
    return pd.DataFrame(rows)

def _rank_best(df: pd.DataFrame, group_cols: List[str], metric_mean_col: str) -> pd.DataFrame:
    df = df.copy()
    df["rank"] = df.groupby(group_cols)[metric_mean_col].rank(ascending=False, method="min")
    return df

def _to_latex(df: pd.DataFrame, out_path: Path, caption: str, label: str) -> None:
    latex = df.to_latex(index=False, escape=True, longtable=False, caption=caption, label=label)
    out_path.write_text(latex)

def make_table3_traditional_encoders(metrics_df: pd.DataFrame) -> pd.DataFrame:
    df = metrics_df.copy()
    df = df[df["encoder"].isin(["onehot", "label", "helmert", "hashing", "target"])]
    agg = _agg_metrics(df, group_cols=["dataset", "subset", "encoder", "classifier"])
    agg = _rank_best(agg, group_cols=["dataset", "subset", "classifier"], metric_mean_col="auc_pr__mean")
    out = agg[["dataset", "subset", "classifier", "encoder", "auc_roc", "auc_pr", "f1_macro", "recall_macro", "rank"]]
    return out.sort_values(["dataset", "subset", "classifier", "rank", "encoder"])

def make_table2_specialized_and_sota(metrics_df: pd.DataFrame) -> pd.DataFrame:
    df = metrics_df.copy()
    specialized_enc = {"cure", "gce", "distance", "dhe"}
    direct_baselines = {"catboost", "tabnet", "fttransformer", "saint", "autoint"}

    a = df[df["encoder"].isin(list(specialized_enc)) & df["classifier"].isin(["xgb", "mlp"])]
    b1 = df[(df["encoder"] == "direct") & (df["classifier"].isin(list(direct_baselines)))]
    b2 = df[(df["classifier"] == "direct") & (df["encoder"].isin(list(direct_baselines)))]

    df2 = pd.concat([a, b1, b2], ignore_index=True)

    df2["method"] = np.where(df2["encoder"].isin(list(direct_baselines)),
                             df2["encoder"],
                             np.where(df2["classifier"].isin(list(direct_baselines)),
                                      df2["classifier"],
                                      df2["encoder"].str.upper() + "+" + df2["classifier"].str.upper()))
    agg = _agg_metrics(df2, group_cols=["dataset", "subset", "method"])
    agg = _rank_best(agg, group_cols=["dataset", "subset"], metric_mean_col="auc_pr__mean")
    out = agg[["dataset", "subset", "method", "auc_roc", "auc_pr", "f1_macro", "recall_macro", "rank"]]
    return out.sort_values(["dataset", "subset", "rank", "method"])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    results_dir = Path(args.results)
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(results_dir / "metrics.parquet")

    t3 = make_table3_traditional_encoders(df)
    t3.to_csv(out_dir / "table3_traditional_encoders.csv", index=False)
    _to_latex(t3, out_dir / "table3_traditional_encoders.tex",
              "Traditional categorical encoders with XGBoost/MLP across chronological subsets.",
              "tab:traditional-encoders")

    t2 = make_table2_specialized_and_sota(df)
    t2.to_csv(out_dir / "table2_specialized_and_sota.csv", index=False)
    _to_latex(t2, out_dir / "table2_specialized_and_sota.tex",
              "Specialized C2N methods and direct tabular baselines across chronological subsets.",
              "tab:specialized-sota")

    print(f"[OK] Wrote tables to {out_dir}")

if __name__ == "__main__":
    main()
