"""
Domain-specific feature engineering for Fraud_Data.csv.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Time-based features
# ---------------------------------------------------------------------------


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add hour_of_day and day_of_week derived from purchase_time.

    Both columns are integers (0-23 and 0-6 respectively).
    """
    df = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df["purchase_time"]):
        df["purchase_time"] = pd.to_datetime(df["purchase_time"])

    df["hour_of_day"] = df["purchase_time"].dt.hour.astype(int)
    df["day_of_week"] = df["purchase_time"].dt.dayofweek.astype(int)
    logger.info("Added hour_of_day and day_of_week features.")
    return df


def add_time_since_signup(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add time_since_signup: seconds elapsed between signup_time and purchase_time.

    Negative values (purchase before signup — data anomalies) are clipped to 0.
    """
    df = df.copy()
    for col in ("signup_time", "purchase_time"):
        if not pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = pd.to_datetime(df[col])

    df["time_since_signup"] = (
        (df["purchase_time"] - df["signup_time"]).dt.total_seconds().clip(lower=0)
    )
    logger.info("Added time_since_signup (seconds).")
    return df


# ---------------------------------------------------------------------------
# Transaction velocity
# ---------------------------------------------------------------------------


def add_transaction_velocity(
    df: pd.DataFrame,
    window_hours: int = 24,
    user_col: str = "user_id",
    time_col: str = "purchase_time",
) -> pd.DataFrame:
    """
    Add transaction count per user within a rolling time window.

    For each row, counts how many prior transactions the same user made
    within the last *window_hours* hours (exclusive of the current one).

    Parameters
    ----------
    df : pd.DataFrame
        Must contain *user_col* and *time_col* (datetime).
    window_hours : int
        Size of the rolling window in hours.

    Returns
    -------
    pd.DataFrame with added column ``txn_count_{window_hours}h``.
    """
    df = df.copy().sort_values(time_col).reset_index(drop=True)
    col_name = f"txn_count_{window_hours}h"
    window_delta = pd.Timedelta(hours=window_hours)

    counts: list[int] = []
    for idx, row in df.iterrows():
        user = row[user_col]
        t = row[time_col]
        mask = (
            (df[user_col] == user)
            & (df[time_col] >= t - window_delta)
            & (df[time_col] < t)
        )
        counts.append(int(mask.sum()))

    df[col_name] = counts
    logger.info(f"Added {col_name} (rolling {window_hours}h velocity).")
    return df


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def build_feature_matrix(
    df: pd.DataFrame,
    categorical_cols: list[str] | None = None,
    target_col: str = "class",
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Apply all feature engineering steps and return (X, y).

    Steps applied in order:
    1. time_since_signup
    2. hour_of_day / day_of_week
    3. transaction velocity (24 h window)
    4. Drop raw datetime and identifier columns
    5. One-hot encode categorical columns

    Parameters
    ----------
    df : pd.DataFrame
        Raw or geo-merged Fraud_Data DataFrame.
    categorical_cols : list[str] | None
        Columns to one-hot encode. Defaults to ['source', 'browser', 'sex', 'country'].
    target_col : str
        Name of the target column.

    Returns
    -------
    X : pd.DataFrame (feature matrix)
    y : pd.Series (target)
    """
    if categorical_cols is None:
        categorical_cols = ["source", "browser", "sex", "country"]
        # keep only those actually present
        categorical_cols = [c for c in categorical_cols if c in df.columns]

    df = add_time_since_signup(df)
    df = add_time_features(df)
    df = add_transaction_velocity(df, window_hours=24)

    # Drop columns not useful for modeling
    drop_cols = [
        "signup_time",
        "purchase_time",
        "user_id",
        "device_id",
        "ip_address",
        target_col,
    ]
    drop_cols = [c for c in drop_cols if c in df.columns]

    y = df[target_col].copy()
    X = df.drop(columns=drop_cols)

    # One-hot encode
    X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

    logger.info(f"Feature matrix built: {X.shape}, target: {y.shape}")
    return X, y
