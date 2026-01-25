from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def _load_attention(models_dir: Path, dataset: str, subset: int) -> Tuple[np.ndarray, List[str]]:
    base = models_dir / dataset / f"subset_{subset}"
    npz = base / "attention_matrix.npz"
    if npz.exists():
        data = np.load(npz, allow_pickle=True)
        A = data["A"]
        names = list(data["feature_names"].tolist())
        return A, names
    csv = base / "attention_matrix.csv"
    if csv.exists():
        mat = pd.read_csv(csv, index_col=0)
        return mat.to_numpy(), list(mat.columns)
    raise FileNotFoundError

def _fallback_interactions_from_data(df: pd.DataFrame, label_col: str) -> Tuple[np.ndarray, List[str]]:
    X = df.drop(columns=[label_col])
    Xn = X.select_dtypes(include=[np.number]).copy()
    if Xn.shape[1] > 30:
        vars_ = Xn.var().sort_values(ascending=False)
        Xn = Xn[vars_.head(30).index]
    corr = Xn.corr().fillna(0.0).to_numpy()
    return corr, list(Xn.columns)

def _save_heatmap(A: np.ndarray, names: List[str], out_path: Path, title: str, top_k: Optional[int] = 25) -> None:
    if top_k is not None and len(names) > top_k:
        score = np.sum(np.abs(A), axis=1)
        idx = np.argsort(-score)[:top_k]
        A = A[np.ix_(idx, idx)]
        names = [names[i] for i in idx]

    fig = plt.figure(figsize=(10, 8))
    ax = plt.gca()
    im = ax.imshow(A, aspect="auto")
    ax.set_title(title)
    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=90, fontsize=8)
    ax.set_yticklabels(names, fontsize=8)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

def _load_category_projection(models_dir: Path, dataset: str, subset: int) -> Optional[pd.DataFrame]:
    p = models_dir / dataset / f"subset_{subset}" / "category_projection.csv"
    if not p.exists():
        return None
    return pd.read_csv(p)

def _plot_c2n_projection(proj: pd.DataFrame, out_path: Path, title: str) -> None:
    counts = proj.groupby("feature")["category"].nunique().sort_values()
    binary_feats = counts[counts == 2].index.tolist()
    feats = binary_feats[:6] if binary_feats else counts.index[:6].tolist()
    sub = proj[proj["feature"].isin(feats)].copy()

    fig = plt.figure(figsize=(10, 5))
    ax = plt.gca()
    feat_to_x = {f: i for i, f in enumerate(feats)}
    xs, ys, labels = [], [], []
    for _, r in sub.iterrows():
        xs.append(feat_to_x[r["feature"]])
        ys.append(r["value"])
        labels.append(str(r["category"]))

    ax.scatter(xs, ys)
    ax.set_title(title)
    ax.set_xticks(range(len(feats)))
    ax.set_xticklabels(feats, rotation=20, ha="right")
    ax.set_ylabel("Numeric mapping / projection")

    for x, y, lab in zip(xs, ys, labels):
        ax.annotate(lab, (x, y), textcoords="offset points", xytext=(5, 5), fontsize=8)

    plt.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--models", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--subset", default=None, type=int)
    ap.add_argument("--label-col", default="label")
    args = ap.parse_args()

    models_dir = Path(args.models)
    out_dir = Path(args.out)
    _ensure_dir(out_dir)

    datasets = [args.dataset] if args.dataset else [p.name for p in models_dir.iterdir() if p.is_dir()]
    for ds in datasets:
        ds_dir = models_dir / ds
        if not ds_dir.exists():
            continue
        subsets = [args.subset] if args.subset is not None else sorted(
            [int(p.name.replace("subset_", "")) for p in ds_dir.iterdir() if p.is_dir() and p.name.startswith("subset_")]
        )
        for subset in subsets:
            try:
                A, names = _load_attention(models_dir, ds, subset)
                out_path = out_dir / f"fig1b_{ds}_subset{subset}_attention.png"
                _save_heatmap(A, names, out_path, title=f"{ds.upper()} subset {subset}: interaction/attention heatmap")
            except FileNotFoundError:
                data_path = models_dir / ds / f"subset_{subset}" / "subset_test.csv"
                if data_path.exists():
                    df = pd.read_csv(data_path)
                    A, names = _fallback_interactions_from_data(df, label_col=args.label_col)
                    out_path = out_dir / f"fig1b_{ds}_subset{subset}_fallback_corr.png"
                    _save_heatmap(A, names, out_path, title=f"{ds.upper()} subset {subset}: correlation heatmap (fallback)")

            proj = _load_category_projection(models_dir, ds, subset)
            if proj is not None and {"feature", "category", "value"}.issubset(set(proj.columns)):
                out_path = out_dir / f"fig1a_{ds}_subset{subset}_c2n.png"
                _plot_c2n_projection(proj, out_path, title=f"{ds.upper()} subset {subset}: C2N category-to-number mapping")

    print(f"[OK] Wrote figures to: {out_dir}")

if __name__ == "__main__":
    main()
