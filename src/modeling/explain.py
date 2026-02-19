"""
SHAP explainability utilities for LightGBM (and tree-based) models.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SHAP value computation
# ---------------------------------------------------------------------------


def compute_shap_values(
    model: Any,
    X: pd.DataFrame | np.ndarray,
) -> tuple[shap.TreeExplainer, np.ndarray]:
    """
    Compute SHAP values using TreeExplainer (fast, exact for tree models).

    Returns
    -------
    explainer : shap.TreeExplainer
    shap_values : np.ndarray  shape (n_samples, n_features)
        For binary classification, returns the positive-class (fraud) values.
    """
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X)

    # LightGBM returns a list [neg_class, pos_class]; take pos_class
    if isinstance(sv, list) and len(sv) == 2:
        shap_values = sv[1]
    else:
        shap_values = sv

    logger.info(f"SHAP values computed: {shap_values.shape}")
    return explainer, shap_values


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def plot_shap_summary(
    shap_values: np.ndarray,
    X: pd.DataFrame | np.ndarray,
    save_path: str | Path | None = None,
    max_display: int = 20,
) -> None:
    """
    Generate and optionally save a SHAP beeswarm summary plot.
    Shows global feature importance and direction of effect.
    """
    plt.figure()
    shap.summary_plot(shap_values, X, max_display=max_display, show=False)
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        logger.info(f"SHAP summary plot saved → {save_path}")
    plt.close()


def plot_shap_force(
    explainer: shap.TreeExplainer,
    shap_values: np.ndarray,
    X: pd.DataFrame | np.ndarray,
    idx: int,
    label: str = "",
    save_path: str | Path | None = None,
) -> None:
    """
    Generate a SHAP force plot for a single prediction.

    Parameters
    ----------
    idx : int
        Row index within X to explain.
    label : str
        Descriptive label (e.g. 'True Positive', 'False Positive').
    """
    shap.initjs()
    force = shap.force_plot(
        explainer.expected_value if not isinstance(explainer.expected_value, list)
        else explainer.expected_value[1],
        shap_values[idx],
        X.iloc[idx] if hasattr(X, "iloc") else X[idx],
        matplotlib=True,
        show=False,
    )
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        logger.info(f"SHAP force plot [{label}] saved → {save_path}")
    plt.close()


def plot_feature_importance(
    model: Any,
    feature_names: list[str],
    top_n: int = 10,
    save_path: str | Path | None = None,
) -> None:
    """Plot built-in LightGBM feature importances (gain)."""
    importances = model.booster_.feature_importance(importance_type="gain")
    fi = pd.Series(importances, index=feature_names).nlargest(top_n)

    fig, ax = plt.subplots(figsize=(8, 5))
    fi.sort_values().plot(kind="barh", ax=ax, color="steelblue")
    ax.set_title(f"Top {top_n} Feature Importances (Gain)")
    ax.set_xlabel("Importance (Gain)")
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
        logger.info(f"Feature importance plot saved → {save_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Feature ranking
# ---------------------------------------------------------------------------


def get_top_features(
    shap_values: np.ndarray,
    X: pd.DataFrame | np.ndarray,
    n: int = 10,
) -> list[str]:
    """
    Return the names of the top-N features by mean |SHAP value|.
    """
    mean_abs = np.abs(shap_values).mean(axis=0)
    if hasattr(X, "columns"):
        names = list(X.columns)
    else:
        names = [f"feature_{i}" for i in range(mean_abs.shape[0])]

    ranked = sorted(zip(names, mean_abs), key=lambda x: x[1], reverse=True)
    top = [name for name, _ in ranked[:n]]
    logger.info(f"Top {n} SHAP features: {top}")
    return top
