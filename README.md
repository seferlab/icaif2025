# fraud-c2n-repro
 
 **"Financial Statement Fraud Detection with a Categorical-to-Numerical Data Representation".

This repo implements:
- Chronological splits (Table 1)
- Traditional encoders + XGBoost/MLP (Table 3)
- Four categorical-to-numerical (C2N) methods: **CURE**, **GCE**, **Distance**, **DHE** + XGBoost (Table 3)
- Direct tabular baselines (Table 2): **CatBoost**, **TabNet**, **FTTransformer**, **XGBOD**, **RUSBoost**, **DeepCrossing**, **DCN**, **DCN-V2**.

> Note: USFSD in the provided package is purely numerical (besides `Year`), so only FiGraph exercises categorical pipelines by default.
> If you have a variant of USFSD with categorical columns, set them explicitly in `configs/usfsd.yaml`.

## Quickstart

### 1) Create environment
```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

### 2) Put datasets
Place the CSVs in:
- `data/raw/usfsd/USFSD.csv`
- `data/raw/figraph/ListedCompanyFeatures.csv`

(If you used the zip provided in the chat, you can copy them from `aaaipaper/USFSD/USFSD.csv` and `aaaipaper/FiGraph/ListedCompanyFeatures.csv`.)

Change your directory to src/fraud_c2n

### 3) Preprocess + run Table 3
```bash
python cli.py preprocess --dataset usfsd
python cli.py preprocess --dataset figraph
```

### 4) Run Table 2 and 3 baselines
```bash
python -m fraud_c2n.cli run_table2 --dataset usfsd
python -m fraud_c2n.cli run_table2 --dataset figraph
```

Outputs:
- `outputs/<dataset>/table2.csv`
- `outputs/<dataset>/runs/table2_<method>_subset<k>.json`

Outputs:
- `outputs/<dataset>/table3.csv`
- `outputs/<dataset>/runs/<method>_subset<k>.json`

## Info
- XGBoost uses deterministic settings where possible.
- To tune to match paper tables, see `configs/models/xgb.yaml` and `configs/c2n/*.yaml`.

