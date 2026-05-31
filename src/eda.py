import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def run_eda(data_path: Path, output_dir: Path):
    """Load raw data, perform exploratory data analysis, and save combined plots."""
    print(">> Running Exploratory Data Analysis (EDA)...")
    df = pd.read_csv(data_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    target_col = (
        "Delivery_Time_min" if "Delivery_Time_min" in df.columns else df.columns[-1]
    )

    # 1. Target Variable Distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(df[target_col], kde=True, color="#56B4E9")
    plt.title("Target Variable Distribution (Delivery Time)")
    plt.xlabel("Minutes")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(output_dir / "target_distribution.png", dpi=300)
    plt.close()

    # 2. COMBINED Boxplots: Weather & Traffic vs Delivery Time
    categorical_features = ["Weather", "Traffic_Level"]
    valid_features = [f for f in categorical_features if f in df.columns]

    if len(valid_features) == 2:
        # Create a figure with 1 row and 2 columns
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        for ax, feature in zip(axes, valid_features):
            sns.boxplot(
                data=df,
                x=feature,
                y=target_col,
                hue=feature,
                palette="Set2",
                legend=False,
                ax=ax,
            )
            ax.set_title(f"Delivery Time by {feature}")
            ax.set_xlabel(feature)
            ax.set_ylabel("Delivery Time (min)")

        plt.tight_layout()
        plt.savefig(output_dir / "categorical_boxplots_combined.png", dpi=300)
        plt.close()
        print(
            "   [OK] Combined boxplot saved to outputs/plots/categorical_boxplots_combined.png"
        )

    # 3. Correlation Heatmap
    if "Order_ID" in df.columns:
        df_for_corr = df.drop(columns=["Order_ID"])
    else:
        df_for_corr = df.copy()

    num_cols = df_for_corr.select_dtypes(include=[np.number])
    if not num_cols.empty:
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            num_cols.corr(),
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            vmin=-1,
            vmax=1,
            square=True,
        )
        plt.title("Numerical Features Correlation Heatmap")
        plt.tight_layout()
        plt.savefig(output_dir / "correlation_heatmap.png", dpi=300)
        plt.close()
        print("   [OK] Correlation heatmap saved.")
