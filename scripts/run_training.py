"""
run_training.py
---------------
Train Logistic Regression and LightGBM models, evaluate, compare, and
save the best model to models/.

Usage:
    python scripts/run_training.py
"""
import logging
import sys
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.modeling.evaluate import compare_models, evaluate_model, save_model
from src.modeling.train import (
    cross_validate_model,
    train_lightgbm,
    train_logistic_regression,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
log = logging.getLogger(__name__)

DATA_PROC = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42


def run() -> None:
    # ------------------------------------------------------------------
    # 1. Load processed data
    # ------------------------------------------------------------------
    log.info("Loading processed data ...")
    X_train = np.load(DATA_PROC / "X_train.npy")
    y_train = np.load(DATA_PROC / "y_train.npy")
    X_test = np.load(DATA_PROC / "X_test.npy")
    y_test = np.load(DATA_PROC / "y_test.npy")
    feature_names = joblib.load(DATA_PROC / "feature_names.pkl")

    log.info(f"Train: {X_train.shape}  Test: {X_test.shape}")

    # ------------------------------------------------------------------
    # 2. Train models
    # ------------------------------------------------------------------
    log.info("Training Logistic Regression ...")
    lr = train_logistic_regression(X_train, y_train, random_state=RANDOM_STATE)

    log.info("Training LightGBM ...")
    lgbm = train_lightgbm(X_train, y_train, random_state=RANDOM_STATE)

    # ------------------------------------------------------------------
    # 3. Cross-validation
    # ------------------------------------------------------------------
    log.info("Cross-validating Logistic Regression (5-fold) ...")
    lr_cv = cross_validate_model(lr, X_train, y_train, n_splits=5)

    log.info("Cross-validating LightGBM (5-fold) ...")
    lgbm_cv = cross_validate_model(lgbm, X_train, y_train, n_splits=5)

    # ------------------------------------------------------------------
    # 4. Hold-out evaluation
    # ------------------------------------------------------------------
    log.info("Evaluating on test set ...")
    lr_metrics = evaluate_model(lr, X_test, y_test)
    lgbm_metrics = evaluate_model(lgbm, X_test, y_test)

    results = {
        "LogisticRegression": lr_metrics,
        "LightGBM": lgbm_metrics,
    }
    comparison = compare_models(results)
    print("\n=== Model Comparison (Hold-out Test Set) ===")
    print(comparison.to_string())
    print()

    print("=== Cross-Validation Summary ===")
    for name, cv in [("LogisticRegression", lr_cv), ("LightGBM", lgbm_cv)]:
        print(
            f"{name}: AUC-PR={cv['ap_mean']:.4f}±{cv['ap_std']:.4f}  "
            f"F1={cv['f1_mean']:.4f}±{cv['f1_std']:.4f}"
        )
    print()

    print("=== Classification Reports ===")
    for name, m in results.items():
        print(f"--- {name} ---")
        print(m["classification_report"])

    # ------------------------------------------------------------------
    # 5. Select best model (by AUC-PR on test set)
    # ------------------------------------------------------------------
    best_name = max(results, key=lambda k: results[k]["auc_pr"])
    best_model = lr if best_name == "LogisticRegression" else lgbm
    log.info(f"Best model: {best_name} (AUC-PR={results[best_name]['auc_pr']:.4f})")

    # ------------------------------------------------------------------
    # 6. Save models
    # ------------------------------------------------------------------
    save_model(lr, MODELS_DIR / "logistic_regression.pkl")
    save_model(lgbm, MODELS_DIR / "lightgbm.pkl")
    save_model(best_model, MODELS_DIR / "best_model.pkl")
    joblib.dump({"best_model_name": best_name, "results": results, "cv": {"LR": lr_cv, "LightGBM": lgbm_cv}},
                MODELS_DIR / "training_summary.pkl")
    log.info(f"Models saved to {MODELS_DIR}")


if __name__ == "__main__":
    run()
