import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression
from pathlib import Path


def train_baseline(data_dir: Path, pred_dir: Path, model_dir: Path):
    """Train Linear Regression baseline, export predictions and model."""
    print(">> Training Linear Regression baseline...")
    X_train = pd.read_csv(data_dir / "X_train.csv")
    y_train = pd.read_csv(data_dir / "y_train.csv").values.ravel()
    X_test = pd.read_csv(data_dir / "X_test.csv")

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Save predictions
    pred_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(y_pred, columns=["prediction"]).to_csv(
        pred_dir / "y_pred_linear.csv", index=False
    )

    # Save the trained model object (.pkl)
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_dir / "linear_regression.pkl")

    print("   [OK] Baseline model saved to outputs/models/linear_regression.pkl")
