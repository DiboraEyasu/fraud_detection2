"""
Unit tests for data loading and cleaning functions.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.eda.eda import (
    analyze_class_distribution,
    handle_missing_values,
    remove_duplicates,
)
from src.eda.featuring.preprocess import ip_to_int, merge_with_geolocation


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_fraud_df() -> pd.DataFrame:
    """Minimal synthetic Fraud_Data-like DataFrame."""
    return pd.DataFrame(
        {
            "user_id": [1, 2, 3, 4, 5],
            "signup_time": pd.to_datetime(
                ["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04", "2020-01-05"]
            ),
            "purchase_time": pd.to_datetime(
                ["2020-01-10", "2020-01-11", "2020-01-12", "2020-01-13", "2020-01-14"]
            ),
            "purchase_value": [100, 200, 150, np.nan, 300],
            "ip_address": [16777217.0, 16777472.0, 16778000.0, 16780000.0, 50331648.0],
            "class": [0, 0, 1, 0, 1],
        }
    )


@pytest.fixture
def sample_ip_df() -> pd.DataFrame:
    """Minimal IP→country DataFrame."""
    return pd.DataFrame(
        {
            "lower_bound_ip_address": [16777216, 16777472, 16778240, 50331648],
            "upper_bound_ip_address": [16777471, 16778239, 16779263, 50397183],
            "country": ["Australia", "China", "Australia", "USA"],
        }
    )


# ---------------------------------------------------------------------------
# handle_missing_values
# ---------------------------------------------------------------------------


def test_handle_missing_values_fills_numeric(sample_fraud_df):
    result = handle_missing_values(sample_fraud_df)
    assert result.isnull().sum().sum() == 0, "All NaNs should be resolved"


def test_handle_missing_values_preserves_rows(sample_fraud_df):
    result = handle_missing_values(sample_fraud_df)
    # Should have same or fewer rows (no extra dropped)
    assert len(result) <= len(sample_fraud_df)
    assert len(result) > 0


# ---------------------------------------------------------------------------
# remove_duplicates
# ---------------------------------------------------------------------------


def test_remove_duplicates():
    df = pd.DataFrame({"a": [1, 1, 2], "b": [3, 3, 4]})
    result = remove_duplicates(df)
    assert len(result) == 2, "Duplicate row should be removed"


def test_remove_duplicates_no_change():
    df = pd.DataFrame({"a": [1, 2, 3]})
    result = remove_duplicates(df)
    assert len(result) == 3


# ---------------------------------------------------------------------------
# analyze_class_distribution
# ---------------------------------------------------------------------------


def test_analyze_class_distribution_keys(sample_fraud_df):
    dist = analyze_class_distribution(sample_fraud_df, target_col="class")
    assert "counts" in dist
    assert "imbalance_ratio" in dist
    assert "fraud_pct" in dist


def test_analyze_class_distribution_values(sample_fraud_df):
    dist = analyze_class_distribution(sample_fraud_df, target_col="class")
    assert dist["counts"][0] == 3   # 3 negatives
    assert dist["counts"][1] == 2   # 2 positives


# ---------------------------------------------------------------------------
# ip_to_int
# ---------------------------------------------------------------------------


def test_ip_to_int_dotted():
    assert ip_to_int("192.168.1.1") == 3_232_235_777


def test_ip_to_int_float():
    # Dataset stores IPs as floats; should convert cleanly
    result = ip_to_int(16_777_217.0)
    assert result == 16_777_217


def test_ip_to_int_string_float():
    result = ip_to_int("16777217.5")
    assert result == 16_777_217


# ---------------------------------------------------------------------------
# merge_with_geolocation
# ---------------------------------------------------------------------------


def test_merge_geolocation_adds_country(sample_fraud_df, sample_ip_df):
    result = merge_with_geolocation(sample_fraud_df, sample_ip_df)
    assert "country" in result.columns


def test_merge_geolocation_known_country(sample_ip_df):
    df = pd.DataFrame(
        {
            "user_id": [1],
            "signup_time": pd.to_datetime(["2020-01-01"]),
            "purchase_time": pd.to_datetime(["2020-01-10"]),
            "purchase_value": [100],
            "ip_address": [16_777_216.0],  # lower bound of Australia range
            "class": [0],
        }
    )
    result = merge_with_geolocation(df, sample_ip_df)
    assert result.iloc[0]["country"] == "Australia"


def test_merge_geolocation_unknown_ip(sample_ip_df):
    """IP outside all known ranges should map to Unknown."""
    df = pd.DataFrame(
        {
            "user_id": [99],
            "signup_time": pd.to_datetime(["2020-01-01"]),
            "purchase_time": pd.to_datetime(["2020-01-10"]),
            "purchase_value": [50],
            "ip_address": [999_999_999.0],
            "class": [0],
        }
    )
    result = merge_with_geolocation(df, sample_ip_df)
    assert result.iloc[0]["country"] == "Unknown"
