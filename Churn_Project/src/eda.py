import pandas as pd
import matplotlib.pyplot as plt
from config import FIGURES_DIR, TARGET_COLUMN


def save_missing_values_chart(df: pd.DataFrame) -> None:
    missing = df.isnull().mean().sort_values(ascending=False)
    missing = missing[missing > 0]

    if missing.empty:
        print("No missing values found.")
        return

    plt.figure(figsize=(12, 6))
    missing.plot(kind="bar")
    plt.title("Missing Value Ratio by Feature")
    plt.ylabel("Proportion Missing")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "missing_values.png")
    plt.close()


def save_target_chart(df: pd.DataFrame) -> None:
    if TARGET_COLUMN not in df.columns:
        return

    counts = df[TARGET_COLUMN].value_counts().sort_index()
    plt.figure(figsize=(6, 4))
    counts.plot(kind="bar")
    plt.title("Target Class Distribution")
    plt.xlabel("Churn")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "target_distribution.png")
    plt.close()


def detect_possible_leakage_columns(df: pd.DataFrame) -> list[str]:
    suspicious_keywords = [
        "churn", "cancel", "closed", "termination", "retention",
        "outcome", "future", "next_cycle", "post_event"
    ]
    flagged = []
    for col in df.columns:
        name = col.lower()
        if col != TARGET_COLUMN and any(word in name for word in suspicious_keywords):
            flagged.append(col)
    return flagged


def numeric_summary(df: pd.DataFrame) -> pd.DataFrame:
    return df.describe(include="number").T