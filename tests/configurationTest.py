"""
Unit tests for feature engineering and model evaluation utilities.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.eda.featuring.custom_feature import (
    add_time_features,
    add_time_since_signup,
    add_transaction_velocity,
)
from src.eda.featuring.preprocess import scale_features
from src.modeling.evaluate import evaluate_model
from src.modeling.train import train_logistic_regression


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def time_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "user_id": [1, 1, 2, 2],
            "signup_time": pd.to_datetime(
                ["2020-01-01 00:00:00", "2020-01-01 00:00:00",
                 "2020-02-01 00:00:00", "2020-02-01 00:00:00"]
            ),
            "purchase_time": pd.to_datetime(
                ["2020-01-05 10:30:00", "2020-01-05 22:45:00",
                 "2020-02-03 08:00:00", "2020-02-03 09:00:00"]
            ),
            "purchase_value": [50, 80, 120, 40],
            "class": [0, 1, 0, 0],
        }
    )


# ---------------------------------------------------------------------------
# add_time_features
# ---------------------------------------------------------------------------


def test_add_time_features_creates_columns(time_df):
    result = add_time_features(time_df)
    assert "hour_of_day" in result.columns
    assert "day_of_week" in result.columns


def test_add_time_features_correct_dtype(time_df):
    result = add_time_features(time_df)
    assert result["hour_of_day"].dtype == int
    assert result["day_of_week"].dtype == int


def test_add_time_features_values(time_df):
    result = add_time_features(time_df)
    # Row 0: 2020-01-05 10:30 → hour=10
    assert result.iloc[0]["hour_of_day"] == 10
    # Row 1: 2020-01-05 22:45 → hour=22
    assert result.iloc[1]["hour_of_day"] == 22


def test_add_time_features_day_of_week_range(time_df):
    result = add_time_features(time_df)
    assert result["day_of_week"].between(0, 6).all()


# ---------------------------------------------------------------------------
# add_time_since_signup
# ---------------------------------------------------------------------------


def test_add_time_since_signup_creates_column(time_df):
    result = add_time_since_signup(time_df)
    assert "time_since_signup" in result.columns


def test_add_time_since_signup_non_negative(time_df):
    result = add_time_since_signup(time_df)
    assert (result["time_since_signup"] >= 0).all()


def test_add_time_since_signup_correct_value(time_df):
    result = add_time_since_signup(time_df)
    # Row 0: 2020-01-01 00:00 → 2020-01-05 10:30 = 4 days 10.5 hours
    expected = (4 * 24 + 10.5) * 3600
    assert abs(result.iloc[0]["time_since_signup"] - expected) < 60  # within 1 minute


# ---------------------------------------------------------------------------
# add_transaction_velocity
# ---------------------------------------------------------------------------


def test_add_transaction_velocity_creates_column(time_df):
    result = add_transaction_velocity(time_df, window_hours=24)
    assert "txn_count_24h" in result.columns


def test_add_transaction_velocity_non_negative(time_df):
    result = add_transaction_velocity(time_df, window_hours=24)
    assert (result["txn_count_24h"] >= 0).all()


def test_add_transaction_velocity_same_user(time_df):
    # Users 1 has 2 purchases on the same day → second should see 1 in window
    result = add_transaction_velocity(time_df, window_hours=24)
    user2_rows = result[result["user_id"] == 2].sort_values("purchase_time")
    # Second row of user 2 should have count ≥ 1 (previous purchase within 1h)
    assert user2_rows.iloc[1]["txn_count_24h"] >= 1


# ---------------------------------------------------------------------------
# scale_features
# ---------------------------------------------------------------------------


def test_scale_features_shape():
    X_tr = np.random.randn(100, 5)
    X_te = np.random.randn(20, 5)
    X_tr_sc, X_te_sc, _ = scale_features(X_tr, X_te)
    assert X_tr_sc.shape == X_tr.shape
    assert X_te_sc.shape == X_te.shape


def test_scale_features_train_normalized():
    X_tr = np.random.randn(200, 4) * 10 + 50
    X_te = np.random.randn(40, 4) * 10 + 50
    X_tr_sc, _, _ = scale_features(X_tr, X_te)
    # Train mean should be ~0, std ~1
    assert abs(X_tr_sc.mean()) < 0.1
    assert abs(X_tr_sc.std() - 1.0) < 0.1


def test_scale_features_returns_scaler():
    X_tr = np.random.randn(50, 3)
    X_te = np.random.randn(10, 3)
    _, _, scaler = scale_features(X_tr, X_te)
    from sklearn.preprocessing import StandardScaler
    assert isinstance(scaler, StandardScaler)


# ---------------------------------------------------------------------------
# evaluate_model (integration)
# ---------------------------------------------------------------------------


def test_evaluate_model_keys():
    """Small synthetic binary problem to verify metric keys are returned."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((200, 5))
    y = (X[:, 0] + rng.standard_normal(200) > 0).astype(int)

    X_train, X_test = X[:160], X[160:]
    y_train, y_test = y[:160], y[160:]

    model = train_logistic_regression(X_train, y_train)
    metrics = evaluate_model(model, X_test, y_test)

    for key in ("auc_roc", "auc_pr", "f1", "confusion_matrix", "classification_report"):
        assert key in metrics, f"Missing key: {key}"


def test_evaluate_model_values_range():
    """Metrics should be in [0, 1]."""
    rng = np.random.default_rng(42)
    X = rng.standard_normal((300, 6))
    y = (X[:, 0] > 0).astype(int)

    model = train_logistic_regression(X[:240], y[:240])
    metrics = evaluate_model(model, X[240:], y[240:])

    assert 0.0 <= metrics["auc_roc"] <= 1.0
    assert 0.0 <= metrics["auc_pr"] <= 1.0
    assert 0.0 <= metrics["f1"] <= 1.0
