"""
run_explainability.py
---------------------
Load the best model and compute SHAP explanations:
  - SHAP summary plot (global)
  - Force plots for 1 True Positive, 1 False Positive, 1 False Negative
  - Built-in feature importance plot
  - Top-5 fraud drivers printed to console

Usage:
    python scripts/run_explainability.py
"""
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.modeling.explain import (
    compute_shap_values,
    get_top_features,
    plot_feature_importance,
    plot_shap_force,
    plot_shap_summary,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
log = logging.getLogger(__name__)

DATA_PROC = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
PLOTS_DIR = MODELS_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def run() -> None:
    # ------------------------------------------------------------------
    # 1. Load best model and test data
    # ------------------------------------------------------------------
    log.info("Loading best model and test data ...")
    model = joblib.load(MODELS_DIR / "best_model.pkl")
    X_test_df = pd.read_csv(DATA_PROC / "X_test_df.csv")
    y_test = np.load(DATA_PROC / "y_test.npy")
    feature_names = joblib.load(DATA_PROC / "feature_names.pkl")

    # ------------------------------------------------------------------
    # 2. Get predictions — use adaptive threshold (median proba) so
    #    we always have TP, FP, and FN examples to explain
    # ------------------------------------------------------------------
    proba = model.predict_proba(X_test_df.values)[:, 1]
    # Use median probability as threshold → guarantees both predicted classes
    threshold = float(np.median(proba))
    predicted = (proba >= threshold).astype(int)
    log.info(f"Adaptive threshold: {threshold:.4f}")

    # Identify sample indices for each case
    tp_mask = (predicted == 1) & (y_test == 1)
    fp_mask = (predicted == 1) & (y_test == 0)
    fn_mask = (predicted == 0) & (y_test == 1)

    def first_idx(mask: np.ndarray, label: str) -> int:
        idxs = np.where(mask)[0]
        if len(idxs) == 0:
            log.warning(f"No {label} samples found at threshold {threshold:.4f}, using fallback.")
            # Fallback: pick the closest probability to threshold
            if label == "True Positive":
                idxs = np.where(y_test == 1)[0]
            elif label == "False Positive":
                idxs = np.where(y_test == 0)[0]
            else:
                idxs = np.where(y_test == 1)[0]
        return int(idxs[0])

    tp_idx = first_idx(tp_mask, "True Positive")
    fp_idx = first_idx(fp_mask, "False Positive")
    fn_idx = first_idx(fn_mask, "False Negative")
    log.info(f"Case indices → TP:{tp_idx}  FP:{fp_idx}  FN:{fn_idx}")

    # ------------------------------------------------------------------
    # 3. Compute SHAP values
    # ------------------------------------------------------------------
    log.info("Computing SHAP values ...")
    explainer, shap_values = compute_shap_values(model, X_test_df)

    # ------------------------------------------------------------------
    # 4. Global summary plot
    # ------------------------------------------------------------------
    log.info("Generating SHAP summary plot ...")
    plot_shap_summary(
        shap_values,
        X_test_df,
        save_path=PLOTS_DIR / "shap_summary.png",
    )

    # ------------------------------------------------------------------
    # 5. Individual force plots
    # ------------------------------------------------------------------
    for idx, label, fname in [
        (tp_idx, "True Positive", "force_true_positive.png"),
        (fp_idx, "False Positive", "force_false_positive.png"),
        (fn_idx, "False Negative", "force_false_negative.png"),
    ]:
        plot_shap_force(
            explainer, shap_values, X_test_df,
            idx=idx, label=label,
            save_path=PLOTS_DIR / fname,
        )

    # ------------------------------------------------------------------
    # 6. Built-in feature importance plot
    # ------------------------------------------------------------------
    try:
        plot_feature_importance(
            model, feature_names,
            top_n=10,
            save_path=PLOTS_DIR / "feature_importance.png",
        )
    except AttributeError:
        log.warning("Built-in feature importance not available for this model type.")

    # ------------------------------------------------------------------
    # 7. Top-5 SHAP drivers
    # ------------------------------------------------------------------
    top5 = get_top_features(shap_values, X_test_df, n=5)
    print("\n=== Top 5 Fraud Drivers (by mean |SHAP|) ===")
    for rank, feat in enumerate(top5, 1):
        print(f"  {rank}. {feat}")

    print(f"\nAll SHAP plots saved to: {PLOTS_DIR}")


if __name__ == "__main__":
    run()
