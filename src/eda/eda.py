"""
EDA utilities: data loading, cleaning, and exploratory analysis.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_fraud_data(path: str | Path) -> pd.DataFrame:
    """Load Fraud_Data.csv with correct dtypes."""
    df = pd.read_csv(path, parse_dates=["signup_time", "purchase_time"])
    logger.info(f"Loaded fraud data: {df.shape}")
    return df


def load_ip_country(path: str | Path) -> pd.DataFrame:
    """Load IpAddress_to_Country.csv."""
    df = pd.read_csv(path)
    logger.info(f"Loaded IP→country map: {df.shape}")
    return df


def load_creditcard(path: str | Path) -> pd.DataFrame:
    """Load creditcard.csv."""
    df = pd.read_csv(path)
    logger.info(f"Loaded credit-card data: {df.shape}")
    return df


# ---------------------------------------------------------------------------
# Data cleaning
# ---------------------------------------------------------------------------


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute or drop missing values.

    Strategy:
    - Numeric columns: fill with median.
    - Categorical columns: fill with mode.
    - Rows still missing after imputation: dropped.
    """
    before = df.isnull().sum().sum()
    num_cols = df.select_dtypes(include="number").columns
    cat_cols = df.select_dtypes(include="object").columns

    for col in num_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

    for col in cat_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].mode()[0])

    df = df.dropna()
    after = df.isnull().sum().sum()
    logger.info(f"Missing values: {before} → {after}")
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate rows and report count."""
    before = len(df)
    df = df.drop_duplicates()
    after = len(df)
    logger.info(f"Removed {before - after} duplicate rows ({before} → {after})")
    return df


def fix_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure datetime columns are parsed and strip any lingering whitespace
    from string columns.
    """
    datetime_cols = [c for c in df.columns if "time" in c.lower() or "date" in c.lower()]
    for col in datetime_cols:
        if df[col].dtype == object:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    str_cols = df.select_dtypes(include="object").columns
    for col in str_cols:
        df[col] = df[col].str.strip()

    return df


# ---------------------------------------------------------------------------
# Exploratory analysis
# ---------------------------------------------------------------------------


def analyze_class_distribution(
    df: pd.DataFrame, target_col: str = "class"
) -> dict:
    """Return class counts and imbalance ratio."""
    counts = df[target_col].value_counts().to_dict()
    majority = max(counts.values())
    minority = min(counts.values())
    ratio = majority / minority if minority > 0 else float("inf")
    result = {
        "counts": counts,
        "imbalance_ratio": round(ratio, 2),
        "fraud_pct": round(100 * counts.get(1, 0) / len(df), 3),
    }
    logger.info(f"Class distribution: {result}")
    return result


# ---------------------------------------------------------------------------
# Country fraud stats  (kept from original eda.py)
# ---------------------------------------------------------------------------


def get_country_fraud_stats(
    df: pd.DataFrame, target_col: str = "class"
) -> pd.DataFrame:
    """Compute fraud statistics per country."""
    stats = (
        df.groupby("country", as_index=False)
        .agg(
            total_transactions=("country", "size"),
            fraud_count=(target_col, "sum"),
            fraud_rate=(target_col, "mean"),
        )
        .sort_values("fraud_count", ascending=False)
    )
    return stats


def get_top_countries_by_fraud_count(
    country_stats: pd.DataFrame,
    top_n: int = 15,
) -> pd.DataFrame:
    """Return top N countries by fraud count."""
    return country_stats.head(top_n)


def get_top_countries_by_fraud_rate(
    country_stats: pd.DataFrame,
    min_transactions: int = 100,
    top_n: int = 15,
) -> pd.DataFrame:
    """Return top N countries by fraud rate (filtered by min transaction count)."""
    filtered = country_stats[
        country_stats["total_transactions"] >= min_transactions
    ]
    return filtered.sort_values("fraud_rate", ascending=False).head(top_n)