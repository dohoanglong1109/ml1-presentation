import json
import joblib
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from pathlib import Path


def train_xgboost(
    data_dir: Path, config_dir: Path, pred_dir: Path, plot_dir: Path, model_dir: Path
):
    """Train XGBoost using saved best params, export results, plots, and model."""
    X_train = pd.read_csv(data_dir / "X_train.csv")
    X_test = pd.read_csv(data_dir / "X_test.csv")
    y_train = pd.read_csv(data_dir / "y_train.csv").values.ravel()

    pred_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    # 1. Default Model
    print(">> Training Default XGBoost...")
    xgb_default = xgb.XGBRegressor(objective="reg:squarederror", random_state=42)
    xgb_default.fit(X_train, y_train)
    pd.DataFrame(xgb_default.predict(X_test), columns=["prediction"]).to_csv(
        pred_dir / "y_pred_xgb_default.csv", index=False
    )

    # 2. Tuned Model
    config_path = config_dir / "xgb_best_params.json"
    best_params = {}
    if config_path.exists():
        with open(config_path, "r") as f:
            best_params = json.load(f)

    print(">> Training Tuned XGBoost...")
    xgb_tuned = xgb.XGBRegressor(
        objective="reg:squarederror", random_state=42, **best_params
    )
    xgb_tuned.fit(X_train, y_train)
    pd.DataFrame(xgb_tuned.predict(X_test), columns=["prediction"]).to_csv(
        pred_dir / "y_pred_xgb_tuned.csv", index=False
    )

    # Save the trained tuned model object (.pkl)
    joblib.dump(xgb_tuned, model_dir / "xgboost_tuned.pkl")
    print("   [OK] Tuned model saved to outputs/models/xgboost_tuned.pkl")

    # 3. Feature Importance Plot
    plt.figure(figsize=(10, 6))
    xgb.plot_importance(
        xgb_tuned, ax=plt.gca(), importance_type="weight", color="#CC79A7"
    )
    plt.title("XGBoost - Feature Importance Weights (Tuned)")
    plt.tight_layout()
    plt.savefig(plot_dir / "feature_importance.png", dpi=300)
    plt.close()
