"""
run_preprocessing.py
--------------------
End-to-end preprocessing pipeline:
  1. Load raw data files from data/
  2. Clean, merge, feature-engineer
  3. Scale and apply SMOTE
  4. Save processed artefacts to data/processed/

Usage:
    python scripts/run_preprocessing.py
"""
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Ensure project root is on path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.eda.eda import (
    analyze_class_distribution,
    fix_dtypes,
    handle_missing_values,
    load_fraud_data,
    load_ip_country,
    remove_duplicates,
)
from src.eda.featuring.custom_feature import build_feature_matrix
from src.eda.featuring.preprocess import (
    apply_smote,
    merge_with_geolocation,
    scale_features,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
log = logging.getLogger(__name__)

DATA_RAW = ROOT / "data"
DATA_PROC = ROOT / "data" / "processed"
DATA_PROC.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42


def run() -> None:
    # ------------------------------------------------------------------
    # 1. Load
    # ------------------------------------------------------------------
    log.info("Loading raw data ...")
    fraud_df = load_fraud_data(DATA_RAW / "Fraud_Data.csv")
    ip_df = load_ip_country(DATA_RAW / "IpAddress_to_Country.csv")

    # ------------------------------------------------------------------
    # 2. Clean
    # ------------------------------------------------------------------
    log.info("Cleaning fraud data ...")
    fraud_df = fix_dtypes(fraud_df)
    fraud_df = handle_missing_values(fraud_df)
    fraud_df = remove_duplicates(fraud_df)

    # ------------------------------------------------------------------
    # 3. Class distribution (before)
    # ------------------------------------------------------------------
    dist = analyze_class_distribution(fraud_df, target_col="class")
    log.info(f"Class distribution (before SMOTE): {dist}")

    # ------------------------------------------------------------------
    # 4. Geolocation merge
    # ------------------------------------------------------------------
    log.info("Merging with geolocation ...")
    fraud_geo = merge_with_geolocation(fraud_df, ip_df)

    # ------------------------------------------------------------------
    # 5. Feature engineering → X, y
    # ------------------------------------------------------------------
    log.info("Building feature matrix ...")
    X, y = build_feature_matrix(fraud_geo)

    # ------------------------------------------------------------------
    # 6. Train / test split (stratified)
    # ------------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    log.info(f"Train size: {X_train.shape}, Test size: {X_test.shape}")

    # ------------------------------------------------------------------
    # 7. Scale
    # ------------------------------------------------------------------
    X_train_sc, X_test_sc, scaler = scale_features(X_train, X_test)

    # ------------------------------------------------------------------
    # 8. SMOTE (training set only)
    # ------------------------------------------------------------------
    X_res, y_res = apply_smote(X_train_sc, y_train.values)
    dist_after = dict(zip(*np.unique(y_res, return_counts=True)))
    log.info(f"Class distribution (after SMOTE): {dist_after}")

    # ------------------------------------------------------------------
    # 9. Persist processed splits
    # ------------------------------------------------------------------
    log.info("Saving processed artefacts ...")
    feature_names = list(X.columns)

    np.save(DATA_PROC / "X_train.npy", X_res)
    np.save(DATA_PROC / "y_train.npy", y_res)
    np.save(DATA_PROC / "X_test.npy", X_test_sc)
    np.save(DATA_PROC / "y_test.npy", y_test.values)

    # Save feature names and test DataFrame for SHAP
    pd.DataFrame(X_test_sc, columns=feature_names).to_csv(
        DATA_PROC / "X_test_df.csv", index=False
    )
    import joblib
    joblib.dump(scaler, DATA_PROC / "scaler.pkl")
    joblib.dump(feature_names, DATA_PROC / "feature_names.pkl")

    # Save full geo-merged dataset for notebooks
    fraud_geo.to_csv(DATA_PROC / "fraud_geo.csv", index=False)

    log.info(f"All artefacts saved to {DATA_PROC}")


if __name__ == "__main__":
    run()
