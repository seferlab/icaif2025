# ICAIF 2025 Experiment Runner (GitHub Actions-ready)

This repository is a **reproducible experiment harness** to run the paper's **chronological-split** evaluation and
generate **tables/plots** from the results.

## Quick start (local)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Create toy data (for smoke tests)
python -m fraud.tools.make_toy_data --out-dir data

# Run a quick experiment sweep (few methods/seeds)
python -m fraud.run_experiments --dataset all --quick \
  --usfsd-path data/usfsd.csv --figraph-path data/figraph.csv \
  --splits-config-dir configs --out-dir artifacts/results --models-dir artifacts/models

# Make tables & figures
python -m fraud.plots.make_tables --results artifacts/results --out artifacts/paper_tables
python -m fraud.plots.make_figures --results artifacts/results --models artifacts/models --out artifacts/paper_figures
```

## Plug in real datasets
Replace `data/usfsd.csv` and `data/figraph.csv` with real files. Expected columns:
- `year` (int)
- `label` (0/1)
- mixture of numeric and categorical features

If your dataset uses a date column instead of `year`, adapt `fraud.datasets.load_dataset_csv()`.
