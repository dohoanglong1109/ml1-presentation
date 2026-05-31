import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from src.eda import run_eda
from src.data_preprocessing import preprocess_data
from src.model_baseline import train_baseline
from src.tune_xgboost import run_hyperparameter_tuning
from src.model_xgboost import train_xgboost
from src.evaluation import evaluate_models


def main():
    FORCE_RETUNE = False

    BASE_DIR = Path(__file__).parent
    RAW_DATA_PATH = BASE_DIR / "data" / "data_raw.csv"
    DATA_DIR = BASE_DIR / "data"
    CONFIG_DIR = BASE_DIR / "configs"
    OUTPUT_DIR = BASE_DIR / "outputs"

    PLOT_DIR = OUTPUT_DIR / "plots"
    PRED_DIR = OUTPUT_DIR / "predictions"
    MODEL_DIR = OUTPUT_DIR / "models"  # <-- THÊM MỚI TẠI ĐÂY

    print("=" * 60)
    print("      STARTING END-TO-END ETA PREDICTION PIPELINE")
    print("=" * 60)

    run_eda(RAW_DATA_PATH, PLOT_DIR)
    preprocess_data(RAW_DATA_PATH, DATA_DIR)

    # Truyền MODEL_DIR vào
    train_baseline(DATA_DIR, PRED_DIR, MODEL_DIR)

    if FORCE_RETUNE or not (CONFIG_DIR / "xgb_best_params.json").exists():
        run_hyperparameter_tuning(DATA_DIR, CONFIG_DIR)

    # Truyền MODEL_DIR vào
    train_xgboost(DATA_DIR, CONFIG_DIR, PRED_DIR, PLOT_DIR, MODEL_DIR)

    evaluate_models(DATA_DIR, PRED_DIR, PLOT_DIR)

    print("=" * 60)
    print("      PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    main()
