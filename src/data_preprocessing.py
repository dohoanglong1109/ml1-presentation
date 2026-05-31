import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pathlib import Path


def preprocess_data(raw_data_path: Path, data_dir: Path):
    """Clean data strictly avoiding Data Leakage, and scale features for Linear models."""
    print(">> Preprocessing data (Strict Pipeline)...")
    df = pd.read_csv(raw_data_path)

    # ---------------------------------------------------------
    # 1. DROP IDENTIFIERS (Fixing the Order_ID bug)
    # ---------------------------------------------------------
    if "Order_ID" in df.columns:
        df = df.drop(columns=["Order_ID"])
        print("   [INFO] Dropped 'Order_ID' to prevent overfitting.")

    target_col = (
        "Delivery_Time_min" if "Delivery_Time_min" in df.columns else df.columns[-1]
    )
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # ---------------------------------------------------------
    # 2. TRAIN-TEST SPLIT FIRST (Prevent Data Leakage)
    # ---------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Identify numerical and categorical columns
    num_cols = X_train.select_dtypes(include=[np.number]).columns
    cat_cols = X_train.select_dtypes(exclude=[np.number]).columns

    # ---------------------------------------------------------
    # 3. IMPUTATION (Learn strictly from Train, apply to Train & Test)
    # ---------------------------------------------------------
    train_medians = X_train[num_cols].median()
    train_modes = X_train[cat_cols].mode().iloc[0]

    X_train.loc[:, num_cols] = X_train[num_cols].fillna(train_medians)
    X_test.loc[:, num_cols] = X_test[num_cols].fillna(train_medians)

    X_train.loc[:, cat_cols] = X_train[cat_cols].fillna(train_modes)
    X_test.loc[:, cat_cols] = X_test[cat_cols].fillna(train_modes)

    # ---------------------------------------------------------
    # 4. ONE-HOT ENCODING (Align columns to handle unseen categories in Test)
    # ---------------------------------------------------------
    X_train = pd.get_dummies(X_train, columns=cat_cols, drop_first=True, dtype=int)
    X_test = pd.get_dummies(X_test, columns=cat_cols, drop_first=True, dtype=int)

    # Critical step: Ensure X_test has the exact same dummy columns as X_train
    X_train, X_test = X_train.align(X_test, join="left", axis=1, fill_value=0)

    # ---------------------------------------------------------
    # 5. FEATURE SCALING (Crucial for Linear Regression convergence)
    # ---------------------------------------------------------
    # Only scale original numerical columns, preserving One-Hot encoded 0/1 binary states
    scaler = StandardScaler()

    # Wrap scaled arrays in a pandas DataFrame with matching index and columns to prevent TypeError
    X_train[num_cols] = pd.DataFrame(
        scaler.fit_transform(X_train[num_cols]), columns=num_cols, index=X_train.index
    )

    X_test[num_cols] = pd.DataFrame(
        scaler.transform(X_test[num_cols]), columns=num_cols, index=X_test.index
    )

    # Ensure all column names are strings to avoid XGBoost feature naming errors
    X_train.columns = X_train.columns.astype(str)
    X_test.columns = X_test.columns.astype(str)
    # ---------------------------------------------------------
    # 6. EXPORT SPLITS
    # ---------------------------------------------------------
    data_dir.mkdir(parents=True, exist_ok=True)
    X_train.to_csv(data_dir / "X_train.csv", index=False)
    X_test.to_csv(data_dir / "X_test.csv", index=False)
    pd.DataFrame(y_train).to_csv(data_dir / "y_train.csv", index=False)
    pd.DataFrame(y_test).to_csv(data_dir / "y_test.csv", index=False)

    print(f"   [OK] Preprocessing done. Train: {X_train.shape} | Test: {X_test.shape}")
