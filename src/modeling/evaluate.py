"""
Model evaluation utilities:
  - Evaluate a trained model (AUC-PR, F1, confusion matrix)
  - Side-by-side model comparison
  - Save / load models with joblib
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Single-model evaluation
# ---------------------------------------------------------------------------


def evaluate_model(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """
    Evaluate a binary classifier and return a metrics dictionary.

    Metrics
    -------
    - auc_roc        (ROC-AUC)
    - auc_pr         (Average Precision / AUC-PR)
    - f1             (macro F1 at given threshold)
    - confusion_matrix
    - classification_report (str)
    """
    proba = model.predict_proba(X_test)[:, 1]
    preds = (proba >= threshold).astype(int)

    metrics: dict[str, Any] = {
        "auc_roc": float(roc_auc_score(y_test, proba)),
        "auc_pr": float(average_precision_score(y_test, proba)),
        "f1": float(f1_score(y_test, preds, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test, preds).tolist(),
        "classification_report": classification_report(y_test, preds, zero_division=0),
    }

    logger.info(
        f"Evaluation → AUC-ROC={metrics['auc_roc']:.4f}  "
        f"AUC-PR={metrics['auc_pr']:.4f}  F1={metrics['f1']:.4f}"
    )
    return metrics


# ---------------------------------------------------------------------------
# Model comparison
# ---------------------------------------------------------------------------


def compare_models(results: dict[str, dict[str, Any]]) -> pd.DataFrame:
    """
    Build a side-by-side comparison table from a dict of evaluation results.

    Parameters
    ----------
    results : dict[str, dict]
        Keys are model names; values are dicts returned by ``evaluate_model``.

    Returns
    -------
    pd.DataFrame with models as rows, metrics as columns.
    """
    rows = []
    for name, m in results.items():
        rows.append(
            {
                "model": name,
                "AUC-ROC": round(m.get("auc_roc", float("nan")), 4),
                "AUC-PR": round(m.get("auc_pr", float("nan")), 4),
                "F1": round(m.get("f1", float("nan")), 4),
            }
        )
    df = pd.DataFrame(rows).set_index("model")
    logger.info(f"Model comparison:\n{df}")
    return df


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def save_model(model: Any, path: str | Path) -> None:
    """Serialize model to disk using joblib."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    logger.info(f"Model saved → {path}")


def load_model(path: str | Path) -> Any:
    """Deserialize model from disk."""
    model = joblib.load(path)
    logger.info(f"Model loaded ← {path}")
    return model
