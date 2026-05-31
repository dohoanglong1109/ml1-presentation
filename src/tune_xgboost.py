import json
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
from pathlib import Path


def run_hyperparameter_tuning(data_dir: Path, config_dir: Path):
    """Run RandomizedSearchCV to find best XGBoost params and save to JSON."""
    print(">> Starting XGBoost Hyperparameter Tuning (This may take a while)...")

    X_train = pd.read_csv(data_dir / "X_train.csv")
    y_train = pd.read_csv(data_dir / "y_train.csv").values.ravel()

    param_dist = {
        "max_depth": [3, 4, 5, 6, 8],
        "learning_rate": [0.01, 0.05, 0.1, 0.2, 0.3],
        "n_estimators": [50, 100, 150, 200],
        "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
        "reg_lambda": [1, 5, 10, 50, 100],
    }

    random_search = RandomizedSearchCV(
        estimator=xgb.XGBRegressor(objective="reg:squarederror", random_state=42),
        param_distributions=param_dist,
        n_iter=15,
        scoring="neg_mean_absolute_error",
        cv=5,
        n_jobs=-1,
        random_state=42,
    )

    random_search.fit(X_train, y_train)
    best_params = random_search.best_params_

    # Export params to JSON
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / "xgb_best_params.json"

    with open(config_path, "w") as f:
        json.dump(best_params, f, indent=4)

    print(f"   [OK] Tuning complete. Best parameters saved to {config_path}")
