from __future__ import annotations
from typing import List, Tuple
import pandas as pd

def load_dataset_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Expect a 'year' column. If you have 'date', derive year here.
    if "year" not in df.columns and "date" in df.columns:
        df["year"] = pd.to_datetime(df["date"]).dt.year.astype(int)
    return df

def apply_year_split(
    df: pd.DataFrame,
    time_col: str,
    train_years: List[int],
    val_years: List[int],
    test_years: List[int],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df = df[df[time_col].isin(train_years)].copy()
    val_df   = df[df[time_col].isin(val_years)].copy()
    test_df  = df[df[time_col].isin(test_years)].copy()
    return train_df, val_df, test_df
