"""
Preprocessing utilities:
  - IP address ↔ integer conversion
  - Geolocation merge (range-based)
  - Feature scaling
  - Categorical encoding
  - SMOTE oversampling
"""
from __future__ import annotations

import logging
from typing import Tuple

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# IP / geolocation
# ---------------------------------------------------------------------------


def ip_to_int(ip_str: str) -> int:
    """
    Convert a dotted-decimal IP string (or a float repr) to an integer.

    Examples
    --------
    >>> ip_to_int("192.168.1.1")
    3232235777
    """
    try:
        # The dataset stores IPs as floats like 732758368.79972
        val = float(ip_str)
        return int(val)
    except (ValueError, TypeError):
        pass

    parts = str(ip_str).split(".")
    if len(parts) == 4:
        return (
            int(parts[0]) * 16_777_216
            + int(parts[1]) * 65_536
            + int(parts[2]) * 256
            + int(parts[3])
        )
    raise ValueError(f"Cannot convert IP: {ip_str!r}")


def merge_with_geolocation(
    fraud_df: pd.DataFrame,
    ip_df: pd.DataFrame,
    ip_col: str = "ip_address",
) -> pd.DataFrame:
    """
    Range-based merge of fraud transactions with the IP→country table.

    Uses `pd.merge_asof` on the lower-bound IP integer, then masks rows
    where the IP falls outside the upper bound.
    """
    fraud = fraud_df.copy()
    ip_lookup = ip_df.copy()

    # Convert IPs in fraud data to integers
    fraud["ip_int"] = fraud[ip_col].apply(ip_to_int)

    # Ensure bounds are int64 so merge_asof dtype matches ip_int (int64)
    ip_lookup["lower_bound_ip_address"] = ip_lookup["lower_bound_ip_address"].astype("int64")
    ip_lookup["upper_bound_ip_address"] = ip_lookup["upper_bound_ip_address"].astype("int64")

    # Sort both tables by the key used in merge_asof
    fraud_sorted = fraud.sort_values("ip_int").reset_index(drop=True)
    ip_lookup_sorted = ip_lookup.sort_values("lower_bound_ip_address").reset_index(
        drop=True
    )

    merged = pd.merge_asof(
        fraud_sorted,
        ip_lookup_sorted[["lower_bound_ip_address", "upper_bound_ip_address", "country"]],
        left_on="ip_int",
        right_on="lower_bound_ip_address",
        direction="backward",
    )

    # Invalidate rows where IP exceeds upper bound
    mask = merged["ip_int"] > merged["upper_bound_ip_address"]
    merged.loc[mask, "country"] = "Unknown"
    merged["country"] = merged["country"].fillna("Unknown")

    # Drop helper columns
    merged.drop(
        columns=["ip_int", "lower_bound_ip_address", "upper_bound_ip_address"],
        inplace=True,
        errors="ignore",
    )

    logger.info(
        f"Geolocation merge complete. Unique countries: {merged['country'].nunique()}"
    )
    return merged


# ---------------------------------------------------------------------------
# Encoding / scaling
# ---------------------------------------------------------------------------


def encode_categoricals(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """One-hot encode the specified categorical columns (drop first to avoid multicollinearity)."""
    return pd.get_dummies(df, columns=cols, drop_first=True)


def scale_features(
    X_train: pd.DataFrame | np.ndarray,
    X_test: pd.DataFrame | np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    """
    Fit a StandardScaler on *training* data only, then transform both splits.

    Returns
    -------
    X_train_scaled, X_test_scaled, fitted_scaler
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    logger.info("Features scaled (StandardScaler, fit on train only).")
    return X_train_scaled, X_test_scaled, scaler


# ---------------------------------------------------------------------------
# Class imbalance
# ---------------------------------------------------------------------------


def apply_smote(
    X_train: np.ndarray,
    y_train: pd.Series | np.ndarray,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply SMOTE to the *training* set only.

    Strategy: SMOTE is preferred over undersampling here because the minority
    class (fraud) is small and undersampling would discard legitimate signal.

    Returns
    -------
    X_resampled, y_resampled
    """
    before = dict(zip(*np.unique(y_train, return_counts=True)))
    sm = SMOTE(random_state=random_state)
    X_res, y_res = sm.fit_resample(X_train, y_train)
    after = dict(zip(*np.unique(y_res, return_counts=True)))
    logger.info(f"SMOTE: class counts {before} → {after}")
    return X_res, y_res
