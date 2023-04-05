
from __future__ import annotations
import argparse, os
import numpy as np
import pandas as pd

from utils.io import load_yaml, save_json, ensure_dir
from utils.seed import set_seed
from data.load import load_csv
from data.preprocess import infer_columns, basic_clean
from data.split import make_splits
from runners.pipelines import (
    make_features_traditional, make_features_cure, make_features_gce, make_features_distance, make_features_dhe
)
from runners.fit_predict import (
    train_eval_xgb,
    train_eval_mlp,
    train_eval_catboost,
    train_eval_rusboost,
    train_eval_xgbod,
    train_eval_tabnet,
    train_eval_deepcrossing,
    train_eval_dcn,
    train_eval_dcnv2,
    train_eval_fttransformer,
)

def _dataset_cfg_path(name: str) -> str:
    name=name.lower()
    if name=="usfsd":
        return "configs/usfsd.yaml"
    if name=="figraph":
        return "configs/figraph.yaml"
    raise ValueError(f"Unknown dataset: {name}")

def cmd_preprocess(args):
    cfg = load_yaml(_dataset_cfg_path(args.dataset))
    df = load_csv(cfg["raw_path"])
    cat_cols, num_cols = infer_columns(df, cfg["year_col"], cfg["label_col"], cfg.get("categorical_cols"))
    if cfg.get("categorical_cols") is not None and len(cfg.get("categorical_cols"))>0:
        cat_cols = cfg["categorical_cols"]
    if cfg.get("numerical_cols") is not None and len(cfg.get("numerical_cols"))>0:
        num_cols = cfg["numerical_cols"]
    df = basic_clean(df, cat_cols, num_cols)
    out_dir = f"data/processed/{args.dataset}"
    ensure_dir(out_dir)
    df.to_parquet(os.path.join(out_dir, "data.parquet"), index=False)
    meta={"year_col":cfg["year_col"],"label_col":cfg["label_col"],"cat_cols":cat_cols,"num_cols":num_cols,"splits":cfg["splits"]}
    save_json(meta, os.path.join(out_dir, "meta.json"))
    print(f"[OK] Saved processed data to {out_dir}")

def _load_processed(dataset: str):
    df = pd.read_parquet(f"data/processed/{dataset}/data.parquet")
    import json
    meta=json.load(open(f"data/processed/{dataset}/meta.json"))
    return df, meta

def _subset_frames(df, meta, subset_id: int):
    splits = make_splits(df, meta["year_col"], meta["splits"])
    sp = splits[int(subset_id)]
    df_tr = df.iloc[sp.train_idx].reset_index(drop=True)
    df_va = df.iloc[sp.val_idx].reset_index(drop=True)
    df_te = df.iloc[sp.test_idx].reset_index(drop=True)
    y_tr = df_tr[meta["label_col"]].to_numpy(dtype=int)
    y_va = df_va[meta["label_col"]].to_numpy(dtype=int)
    y_te = df_te[meta["label_col"]].to_numpy(dtype=int)
    return df_tr, y_tr, df_va, y_va, df_te, y_te

def cmd_run_table3(args):
    set_seed(args.seed)
    df, meta = _load_processed(args.dataset)
    out_root = f"outputs/{args.dataset}"
    ensure_dir(out_root)
    ensure_dir(os.path.join(out_root,"runs"))

    # configs
    xgb_params = load_yaml("configs/models/xgb.yaml")
    mlp_params = load_yaml("configs/models/mlp.yaml")
    enc_cfgs = {
        "One-hot": load_yaml("configs/encoders/onehot.yaml"),
        "Label": load_yaml("configs/encoders/label.yaml"),
        "Helmert": load_yaml("configs/encoders/helmert.yaml"),
        "Feature": load_yaml("configs/encoders/hashing.yaml"),
    }
    c2n_cfgs = {
        "CURE": load_yaml("configs/c2n/cure.yaml"),
        "GCE": load_yaml("configs/c2n/gce.yaml"),
        "Distance": load_yaml("configs/c2n/distance.yaml"),
        "DHE": load_yaml("configs/c2n/dhe.yaml"),
    }

    methods = []
    # traditional + XGB
    #for enc_name, ecfg in enc_cfgs.items():
    #    methods.append((f"{enc_name}+XGBoost","xgb",("enc",enc_name,ecfg)))
    # traditional + MLP
    #for enc_name, ecfg in enc_cfgs.items():
    #    methods.append((f"{enc_name}+MLP","mlp",("enc",enc_name,ecfg)))
    # c2n + XGB
    for c2n_name, ccfg in c2n_cfgs.items():
        methods.append((f"{c2n_name}+XGBoost","xgb",("c2n",c2n_name,ccfg)))
        #methods.append((f"{c2n_name}+MLP","mlp",("c2n",c2n_name,ccfg)))

    subset_ids = sorted([int(k) for k in meta["splits"].keys()])
    rows=[]
    for method_name, model_kind, spec in methods:
        per_subset=[]
        for k in subset_ids:
            df_tr,y_tr,df_va,y_va,df_te,y_te = _subset_frames(df, meta, k)
            if spec[0]=="enc":
                _, enc_name, ecfg = spec
                X_tr,X_va,X_te = make_features_traditional(df_tr, df_va, df_te, meta["cat_cols"], meta["num_cols"], ecfg)
            else:
                _, c2n_name, ccfg = spec
                if c2n_name=="CURE":
                    X_tr,X_va,X_te = make_features_cure(df, df_tr, df_va, df_te, meta["cat_cols"], meta["num_cols"], ccfg, args.seed)
                elif c2n_name=="GCE":
                    X_tr,X_va,X_te = make_features_gce(df, df_tr, df_va, df_te, meta["cat_cols"], meta["num_cols"], ccfg, args.seed, device=args.device)
                elif c2n_name=="Distance":
                    X_tr,X_va,X_te = make_features_distance(df, df_tr, df_va, df_te, meta["cat_cols"], meta["num_cols"], ccfg, args.seed)
                elif c2n_name=="DHE":
                    X_tr,X_va,X_te = make_features_dhe(df, df_tr, df_va, df_te, y_tr, meta["cat_cols"], meta["num_cols"], ccfg, args.seed, device=args.device)
                else:
                    raise ValueError(c2n_name)

            print(method_name,c2n_name)
            #print(X_tr,X_va,X_te)
            if model_kind=="xgb":
                metrics, _ = train_eval_xgb(X_tr,y_tr,X_va,y_va,X_te,y_te,xgb_params,args.seed)
            else:
                metrics, _ = train_eval_mlp(X_tr,y_tr,X_va,y_va,X_te,y_te,mlp_params,args.seed)

            print(metrics)
            rec={"dataset":args.dataset,"subset":k,"method":method_name,**metrics}
            save_json(rec, os.path.join(out_root,"runs",f"{method_name.replace('/','_').replace(' ','')}_subset{k}.json"))
            per_subset.append(metrics)
        # average over subsets
        avg={m: float(np.nanmean([ps[m] for ps in per_subset])) for m in ["auc_roc","auc_pr","f1_macro","recall_macro"]}
        rows.append({"Method":method_name,"AUC-ROC":avg["auc_roc"],"AUC-PR":avg["auc_pr"],"F1-macro":avg["f1_macro"],"Recall-macro":avg["recall_macro"]})

    out_df = pd.DataFrame(rows)
    out_df.to_csv(os.path.join(out_root,"table3.csv"), index=False)
    print(f"[OK] Wrote {os.path.join(out_root,'table3.csv')}")


def cmd_run_table2(args):
    """Reproduce Table 2 style baselines (direct tabular models).

    For comparability and reproducibility, most deep baselines operate on the same
    dense design matrix (one-hot + numeric) produced by the One-hot encoder config.
    CatBoost is run on raw (cat+num) columns.
    """
    set_seed(args.seed)
    df, meta = _load_processed(args.dataset)
    out_root = f"outputs/{args.dataset}"
    ensure_dir(out_root)
    ensure_dir(os.path.join(out_root, "runs"))

    # shared one-hot design matrix for most baselines
    onehot_cfg = load_yaml("configs/encoders/onehot.yaml")
    xgb_params = load_yaml("configs/models/xgb.yaml")

    # baseline model configs
    cat_params = load_yaml("configs/models/catboost.yaml")
    rus_params = load_yaml("configs/models/rusboost.yaml")
    xgbod_params = load_yaml("configs/models/xgbod.yaml")
    tabnet_params = load_yaml("configs/models/tabnet.yaml")
    ft_params = load_yaml("configs/models/fttransformer.yaml")
    dc_params = load_yaml("configs/models/deepcrossing.yaml")
    dcn_params = load_yaml("configs/models/dcn.yaml")
    dcnv2_params = load_yaml("configs/models/dcnv2.yaml")

    methods = [
        #("CatBoost", "catboost", cat_params),
        #("RUSBoost", "rusboost", rus_params),
        #("XGBOD", "xgbod", xgbod_params),

        ("TabNet", "tabnet", tabnet_params),
        #("FTTransformer", "fttransformer", ft_params),
        #("DeepCrossing", "deepcrossing", dc_params),
        ("DCN", "dcn", dcn_params),
        #("DCN-V2", "dcnv2", dcnv2_params),
        #("XGBoost", "xgb", xgb_params),
    ]

    subset_ids = sorted([int(k) for k in meta["splits"].keys()])
    rows = []
    for method_name, kind, params in methods:
        per_subset = []
        for k in subset_ids:
            df_tr, y_tr, df_va, y_va, df_te, y_te = _subset_frames(df, meta, k)

            if kind == "catboost":
                metrics, _ = train_eval_catboost(
                    df_tr, y_tr, df_va, y_va, df_te, y_te,
                    cat_cols=meta["cat_cols"], num_cols=meta["num_cols"],
                    params=params, seed=args.seed,
                )
            else:
                # dense matrix (one-hot + numeric)
                X_tr, X_va, X_te = make_features_traditional(
                    df_tr, df_va, df_te,
                    meta["cat_cols"], meta["num_cols"],
                    onehot_cfg,
                )

                print(method_name)
                if kind == "xgb":
                    metrics, _ = train_eval_xgb(X_tr, y_tr, X_va, y_va, X_te, y_te, params, args.seed)
                elif kind == "rusboost":
                    metrics, _ = train_eval_rusboost(X_tr, y_tr, X_va, y_va, X_te, y_te, params, args.seed)
                elif kind == "xgbod":
                    metrics, _ = train_eval_xgbod(X_tr, y_tr, X_va, y_va, X_te, y_te, params, args.seed)
                elif kind == "tabnet":
                    metrics, _ = train_eval_tabnet(X_tr, y_tr, X_va, y_va, X_te, y_te, params, args.seed, device=args.device)
                elif kind == "fttransformer":
                    metrics, _ = train_eval_fttransformer(X_tr, y_tr, X_va, y_va, X_te, y_te, params, args.seed, device=args.device)
                elif kind == "deepcrossing":
                    metrics, _ = train_eval_deepcrossing(X_tr, y_tr, X_va, y_va, X_te, y_te, params, args.seed, device=args.device)
                elif kind == "dcn":
                    metrics, _ = train_eval_dcn(X_tr, y_tr, X_va, y_va, X_te, y_te, params, args.seed, device=args.device)
                elif kind == "dcnv2":
                    metrics, _ = train_eval_dcnv2(X_tr, y_tr, X_va, y_va, X_te, y_te, params, args.seed, device=args.device)
                else:
                    raise ValueError(kind)

            rec = {"dataset": args.dataset, "subset": k, "method": method_name, **metrics}
            save_json(rec, os.path.join(out_root, "runs", f"table2_{method_name.replace(' ','').replace('/','_')}_subset{k}.json"))
            per_subset.append(metrics)

        avg = {m: float(np.nanmean([ps[m] for ps in per_subset])) for m in ["auc_roc", "auc_pr", "f1_macro", "recall_macro"]}
        rows.append({
            "Method": method_name,
            "AUC-ROC": avg["auc_roc"],
            "AUC-PR": avg["auc_pr"],
            "F1-macro": avg["f1_macro"],
            "Recall-macro": avg["recall_macro"],
        })

    out_df = pd.DataFrame(rows)
    out_df.to_csv(os.path.join(out_root, "table2.csv"), index=False)
    print(f"[OK] Wrote {os.path.join(out_root, 'table2.csv')}")

def main():
    p=argparse.ArgumentParser()
    sub=p.add_subparsers(dest="cmd", required=True)
    p1=sub.add_parser("preprocess")
    p1.add_argument("--dataset", required=True, choices=["usfsd","figraph"])
    p1.set_defaults(func=cmd_preprocess)

    p2=sub.add_parser("run_table3")
    p2.add_argument("--dataset", required=True, choices=["usfsd","figraph"])
    p2.add_argument("--seed", type=int, default=42)
    p2.add_argument("--device", type=str, default="cpu")
    p2.set_defaults(func=cmd_run_table3)

    p3=sub.add_parser("run_table2")
    p3.add_argument("--dataset", required=True, choices=["usfsd","figraph"])
    p3.add_argument("--seed", type=int, default=42)
    p3.add_argument("--device", type=str, default="cpu")
    p3.set_defaults(func=cmd_run_table2)

    args=p.parse_args()
    args.func(args)

if __name__=="__main__":
    main()
