# 🛵 Food Delivery ETA Prediction: An Empirical Study on Model Complexity

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Latest-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-Latest-green)
![Status](https://img.shields.io/badge/Status-Completed-success)

## 📌 Project Overview

Predicting Estimated Time of Arrival (ETA) is a critical component of food delivery platforms. This project builds a complete end-to-end Machine Learning pipeline to predict delivery times based on external and internal factors (distance, weather, traffic, preparation time).

**🔍 The Core Academic Finding:** Beyond just building a predictive model, this project serves as an empirical demonstration of the **Occam's Razor principle** and the **No Free Lunch Theorem** in Machine Learning. Through rigorous hyperparameter tuning and model evaluation, we discovered that for this specific (highly linear) dataset, a simple **Linear Regression** model completely outperformed a highly complex, regularized **XGBoost** algorithm.

## 🚀 Key Features & Pipeline

- **Automated Execution:** A single orchestrator (`main.py`) triggers the entire pipeline.
- **Resource Optimization:** Implemented parameter caching (`configs/xgb_best_params.json`) to bypass redundant GridSearchCV operations.
- **Data Leakage Prevention:** Strict isolation of validation sets for XGBoost's Early Stopping mechanism.
- **Comprehensive Outputs:** Automatically generates model artifacts (`.pkl`), evaluation reports, and strategic EDA visualizations.

## 📁 Repository Structure

```text
.
├── configs/
│   └── xgb_best_params.json      # Cached optimal hyperparameters
├── data/
│   ├── X_train.csv, y_train.csv  # Training set
│   └── X_test.csv, y_test.csv    # Hold-out test set
├── src/
│   ├── preprocess.py             # Data cleaning, encoding, and scaling
│   ├── train_lr.py               # Linear Regression modeling
│   ├── train_xgb.py              # XGBoost modeling (with Early Stopping)
│   └── evaluate.py               # Metrics calculation & visualization
├── outputs/
│   ├── models/                   # Serialized production-ready models (.pkl)
│   ├── plots/                    # EDA and model comparison visualizations
│   ├── predictions/              # Predicted ETA values (.csv)
│   └── evaluation_report.txt     # Automated MAE/RMSE metrics report
├── scripts/
│   └── tune_hyperparameters.py   # Independent GridSearchCV research script
├── main.py                       # Master pipeline orchestrator
├── requirements.txt              # Environment dependencies
└── README.md
