import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from pathlib import Path


def evaluate_models(data_dir: Path, pred_dir: Path, plot_dir: Path):
    """Compute performance metrics (MAE, RMSE) and plot cross-model comparisons."""
    print(">> Executing performance evaluation...")
    y_true = pd.read_csv(data_dir / "y_test.csv").values.ravel()

    # Load all computed predictions
    y_linear = pd.read_csv(pred_dir / "y_pred_linear.csv", sep=";", decimal=".")[
        "prediction"
    ].values
    y_xgb_default = pd.read_csv(pred_dir / "y_pred_xgb_default.csv")[
        "prediction"
    ].values
    y_xgb_tuned = pd.read_csv(pred_dir / "y_pred_xgb_tuned.csv")["prediction"].values

    models = {
        "Linear Regression": y_linear,
        "XGBoost Default": y_xgb_default,
        "XGBoost Tuned": y_xgb_tuned,
    }

    print("\n" + "=" * 50)
    print(f"{'MODEL PERFORMANCE METRICS':^50}")
    print("=" * 50)
    for name, pred in models.items():
        mae = mean_absolute_error(y_true, pred)
        # Calculate RMSE by taking the square root of MSE
        rmse = np.sqrt(mean_squared_error(y_true, pred))

        print(f"{name:20} -> MAE: {mae:.4f} | RMSE: {rmse:.4f}")
    print("=" * 50 + "\n")

    # Generate unified model comparison plot
    plt.figure(figsize=(14, 6))
    x = range(len(y_true))

    # Plot Actual vs Predicted
    plt.plot(x, y_true, label="Actual", color="black", linewidth=1.5)
    plt.plot(x, y_linear, label="Linear Regression", color="#E69F00", linestyle="--")
    plt.plot(x, y_xgb_default, label="XGB Default", color="#CC79A7", linestyle=":")
    plt.plot(x, y_xgb_tuned, label="XGB Tuned", color="#56B4E9")

    plt.xlabel("Sample Index")
    plt.ylabel("Delivery Time (min)")
    plt.title("Model Prediction Performance Comparison")
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()

    plot_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_dir / "model_comparison.png", dpi=300)
    plt.close()
    print("   [OK] Final evaluation plot saved to outputs/plots/model_comparison.png")
