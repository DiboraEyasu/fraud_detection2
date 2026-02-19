"""
Model training utilities:
  - Logistic Regression baseline
  - LightGBM ensemble
  - Stratified K-Fold cross-validation
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    make_scorer,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model factories
# ---------------------------------------------------------------------------


def train_logistic_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    max_iter: int = 1000,
    random_state: int = 42,
) -> LogisticRegression:
    """
    Train a Logistic Regression baseline.

    class_weight='balanced' handles any residual imbalance after SMOTE.
    """
    model = LogisticRegression(
        max_iter=max_iter,
        class_weight="balanced",
        solver="lbfgs",
        random_state=random_state,
    )
    model.fit(X_train, y_train)
    logger.info("LogisticRegression trained.")
    return model


def train_lightgbm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    random_state: int = 42,
    **kwargs: Any,
) -> LGBMClassifier:
    """
    Train a LightGBM classifier with sensible fraud-detection defaults.

    Hyperparameters can be overridden via **kwargs.
    is_unbalance=True provides an additional push even on SMOTE-resampled data.
    """
    defaults = dict(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        num_leaves=63,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        is_unbalance=True,
        random_state=random_state,
        verbose=-1,
    )
    defaults.update(kwargs)

    model = LGBMClassifier(**defaults)
    model.fit(X_train, y_train)
    logger.info("LightGBM trained.")
    return model


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------


def cross_validate_model(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    random_state: int = 42,
) -> dict[str, float]:
    """
    Run Stratified K-Fold cross-validation and return mean ± std of key metrics.

    Returns
    -------
    dict with keys:
      - ap_mean / ap_std  (Average Precision / AUC-PR)
      - f1_mean / f1_std
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    scoring = {
        "ap": make_scorer(average_precision_score, needs_proba=True),
        "f1": make_scorer(f1_score, zero_division=0),
    }

    cv_results = cross_validate(
        model, X, y, cv=skf, scoring=scoring, n_jobs=-1
    )

    results = {
        "ap_mean": float(np.mean(cv_results["test_ap"])),
        "ap_std": float(np.std(cv_results["test_ap"])),
        "f1_mean": float(np.mean(cv_results["test_f1"])),
        "f1_std": float(np.std(cv_results["test_f1"])),
    }
    logger.info(
        f"Cross-validation ({n_splits}-fold): "
        f"AUC-PR={results['ap_mean']:.4f}±{results['ap_std']:.4f}  "
        f"F1={results['f1_mean']:.4f}±{results['f1_std']:.4f}"
    )
    return results
