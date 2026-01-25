from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

def _make_dataset(years, n_per_year, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for y in years:
        n = n_per_year
        # numeric features
        x1 = rng.normal(0, 1, size=n)
        x2 = rng.normal(0, 1, size=n)
        # categorical features (binary + multi)
        c_bin = rng.choice(["Yes", "No"], size=n, p=[0.4, 0.6])
        c_tri = rng.choice(["A", "B", "C"], size=n)
        # label depends on a mix, with mild drift
        logit = 0.8*x1 - 0.5*x2 + (c_bin=="Yes")*0.7 + (c_tri=="C")*0.4 + (y - min(years))*0.02
        p = 1/(1+np.exp(-logit))
        label = rng.binomial(1, p)
        for i in range(n):
            rows.append({"year": int(y), "x1": float(x1[i]), "x2": float(x2[i]), "bin_feat": c_bin[i], "tri_feat": c_tri[i], "label": int(label[i])})
    return pd.DataFrame(rows)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    usfsd_years = list(range(1991, 2009))
    figraph_years = list(range(2014, 2023))

    usfsd = _make_dataset(usfsd_years, n_per_year=500, seed=1)
    figraph = _make_dataset(figraph_years, n_per_year=400, seed=2)

    (out_dir / "usfsd.csv").write_text(usfsd.to_csv(index=False))
    (out_dir / "figraph.csv").write_text(figraph.to_csv(index=False))
    print(f"[OK] wrote {out_dir/'usfsd.csv'} and {out_dir/'figraph.csv'}")

if __name__ == "__main__":
    main()
