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
│   └── xgb_best_params.json         # Model hyperparameters
├── data/
│   ├── .gitkeep                     # Keeps folder structure on Git
│   ├── data_raw.csv                 # Immutable raw data
│   ├── X_test.csv                   # Split features
│   ├── X_train.csv
│   ├── y_test.csv                   # Split targets
│   └── y_train.csv
├── outputs/
│   ├── models/
│   │   ├── .gitkeep
│   │   ├── linear_regression.pkl    # Trained model artifacts
│   │   └── xgboost_tuned.pkl
│   ├── plots/
│   │   ├── .gitkeep
│   │   ├── correlation_heatmap.png  # EDA & Evaluation charts
│   │   └── feature_importance.png
│   └── predictions/
│       ├── .gitkeep
│       ├── y_pred_linear.csv        # Model inference outputs
│       └── y_pred_xgb_tuned.csv
├── src/                             # Source code directory
│   ├── __init__.py                  # Makes src a Python package
│   ├── data_preprocessing.py        # Data cleaning & engineering
│   ├── eda.py                       # Exploratory Data Analysis
│   ├── evaluation.py                # Metrics & validation logic
│   ├── model_baseline.py            # Simple baseline model
│   ├── model_xgboost.py             # Main model architecture
│   └── tune_xgboost.py              # Hyperparameter tuning
├── .gitignore                       # Git ignore rules
├── environment.yml                  # Conda environment definition
├── main.py                          # Pipeline execution entry point
└── README.md                        # Project documentation
