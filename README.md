# Fraud Detection System — Comprehensive Project Report

## Project Overview
This project implements a complete Fraud Detection pipeline designed to identify fraudulent e-commerce and bank transactions. It covers data preprocessing, advanced feature engineering, machine learning modeling (Logistic Regression and LightGBM), and model interpretability using SHAP.

---

## 📁 Repository Structure

```
dibo/
├── src/
│   ├── eda/
│   │   ├── eda.py              # Data cleaning and visual analysis
│   │   └── featuring/
│   │       ├── preprocess.py   # Transformation, scaling, SMOTE
│   │       └── custom_feature.py  # Domain features & velocity
│   └── modeling/
│       ├── train.py            # Training logic (LR, LightGBM, CV)
│       ├── evaluate.py         # Metrics & comparisons
│       └── explain.py          # SHAP interpretability
├── scripts/
│   ├── run_preprocessing.py    # Pipeline: Data → Features
│   ├── run_training.py         # Pipeline: Features → Models
│   └── run_explainability.py   # Pipeline: Model → Insights
├── notebooks/
│   ├── eda-fraud-data.ipynb    # Visual data discovery
│   ├── feature-engineering.ipynb # Engineering walkthrough
│   ├── modeling.ipynb          # Model performance report
│   └── shap-explainability.ipynb # Driver analysis report
├── tests/
│   ├── dataHandlerTest.py      # Cleaning & Geo-merge tests
│   └── configurationTest.py    # Feature & Scaling tests
├── requirements.txt            # Pinned environment
└── pyproject.toml              # Tooling config
```

---

## ⚙️ Task 1: Data Engineering

### Geolocation Integration
Transactions were enriched with country information by mapping `ip_address` to ranges in `IpAddress_to_Country.csv` using specialized `pd.merge_asof` logic for high-performance numeric merging.

### Feature Engineering
- **Time Dynamics**: `time_since_signup` (seconds from signup to txn), `hour_of_day`, and `day_of_week`.
- **Transaction Velocity**: Rolling 24-hour transaction frequency per user (`txn_count_24h`).
- **Standardization**: All features scaled via `StandardScaler` fitted exclusively on training data to prevent leakage.
- **Handling Imbalance**: Applied **SMOTE** (Synthetic Minority Over-sampling Technique) to balance the fraud cases in the training set, increasing the fraud minority from ~9% to 50%.

---

## 🤖 Task 2: Machine Learning Modeling

### Models Evaluated
1. **Logistic Regression**: Used as a baseline for interpretability and linear relationship capture.
2. **LightGBM**: Advanced GBDT chosen for its ability to handle non-linear interactions and built-in support for imbalanced data.

### Performance Summary (Test Set)
| Metric | Logistic Regression | LightGBM |
| :--- | :--- | :--- |
| **AUC-PR** | 0.4145 | **0.6150** |
| **F1-Score** | 0.2745 | **0.6856** |
| **ROC AUC** | 0.7507 | **0.7587** |

**Conclusion**: **LightGBM** outperformed the baseline significantly in AUC-PR and F1-score, making it the superior choice for detecting fraudulent transactions without excessive false positives.

---

## 🔍 Task 3: Model Explainability (SHAP)

Using SHAP (SHapley Additive exPlanations), we identified the primary drivers behind the model's decisions:

### Top Fraud Drivers
1. **time_since_signup**: The most powerful predictor. Fraudsters typically transaction immediately after creating accounts.
2. **day_of_week / hour_of_day**: Fraud clusters around high-velocity time windows and night hours.
3. **age**: Certain age demographics showed higher propensity for fraudulent patterns.
4. **country_United States**: Geolocation remains a significant risk factor.

### Business Recommendations
- **Dynamic Risk Rules**: Flag transactions where `time_since_signup` is < 30 minutes for manual review.
- **Velocity Thresholds**: Block users exceeding 3 transactions per hour.
- **Enhanced Verification**: Trigger MFA for transactions originating from high-risk countries or during night hours (midnight–5 AM).

---

## 🧪 Verification & Testing
- **Unit Tests**: 27 passing tests covering preprocessing, features, and model metrics.
- **Reproducibility**: All 4 notebooks have been executed with real data and contain embedded visualizations.
- **Docker-Ready**: Project structure follows modular package standards.

---

## 🚀 Getting Started
```bash
# 1. Install environment
python3 -m venv dibo_env
source dibo_env/bin/activate
pip install -r requirements.txt

# 2. Run FULL pipeline
python scripts/run_preprocessing.py
python scripts/run_training.py
python scripts/run_explainability.py

# 3. Run Tests
pytest tests/
```
