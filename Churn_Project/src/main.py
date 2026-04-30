import joblib
import pandas as pd

from sklearn.model_selection import train_test_split

from config import TARGET_COLUMN, TEST_SIZE, RANDOM_STATE, TABLES_DIR, MODELS_DIR
from data_loader import load_data, basic_overview
from eda import save_missing_values_chart, save_target_chart, detect_possible_leakage_columns
from preprocess import split_features_target, build_preprocessor
from train import build_model_pipelines, tune_random_forest, tune_gradient_boosting
from evaluate import evaluate_model
from interpret import evaluate_by_segment


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the churn dataset before preprocessing and modelling.
    """

    # 1. Clean column names
    df.columns = df.columns.str.strip()

    # 2. Drop Customer ID because it should not be used for prediction
    if "Customer ID" in df.columns:
        df = df.drop(columns=["Customer ID"])

    # 3. Convert Yes/No columns into 1/0
    for col in df.columns:
        if df[col].dtype == "object":
            values = set(df[col].dropna().unique())
            if values.issubset({"Yes", "No"}):
                df[col] = df[col].map({"Yes": 1, "No": 0})

    # 4. Convert Churn if it is stored as Yes/No
    if TARGET_COLUMN in df.columns and df[TARGET_COLUMN].dtype == "object":
        df[TARGET_COLUMN] = df[TARGET_COLUMN].map({"Yes": 1, "No": 0})

    return df


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds extra useful features for the churn model.
    These help with the feature engineering part of the report.
    """

    if all(col in df.columns for col in ["App Logins", "Portal Logins", "Email Clicks"]):
        df["Digital_Engagement_Total"] = (
            df["App Logins"] + df["Portal Logins"] + df["Email Clicks"]
        )

    if all(col in df.columns for col in ["Calls Last Month", "Complaints Last Year"]):
        df["Service_Friction"] = (
            df["Calls Last Month"] + df["Complaints Last Year"]
        )

    if all(col in df.columns for col in ["Bill_Month_10", "Bill_Month_11", "Bill_Month_12"]):
        df["Avg_Bill_Last_3_Months"] = (
            df["Bill_Month_10"] + df["Bill_Month_11"] + df["Bill_Month_12"]
        ) / 3

    if all(col in df.columns for col in ["Electricity_Month_10", "Electricity_Month_11", "Electricity_Month_12"]):
        df["Avg_Electricity_Last_3_Months"] = (
            df["Electricity_Month_10"] + df["Electricity_Month_11"] + df["Electricity_Month_12"]
        ) / 3

    if all(col in df.columns for col in ["Gas_Month_10", "Gas_Month_11", "Gas_Month_12"]):
        df["Avg_Gas_Last_3_Months"] = (
            df["Gas_Month_10"] + df["Gas_Month_11"] + df["Gas_Month_12"]
        ) / 3

    return df


def main():
    # STEP 1: Load dataset
    df = load_data()

    # STEP 2: Clean dataset
    df = clean_dataset(df)

    # STEP 3: Add feature engineering
    df = add_engineered_features(df)

    # STEP 4: Basic overview and EDA
    basic_overview(df)

    save_missing_values_chart(df)
    save_target_chart(df)

    leakage_cols = detect_possible_leakage_columns(df)
    print("\n=== POSSIBLE LEAKAGE COLUMNS ===")
    print(leakage_cols)

    # STEP 5: Check target column exists
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found in dataset.")

    # STEP 6: Remove rows where target is missing
    df = df.dropna(subset=[TARGET_COLUMN])

    # STEP 7: Show class balance
    print("\n=== CHURN CLASS BALANCE ===")
    print(df[TARGET_COLUMN].value_counts())
    print(df[TARGET_COLUMN].value_counts(normalize=True))

    # STEP 8: Split into X and y
    X, y = split_features_target(df, TARGET_COLUMN)

    # STEP 9: Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    # STEP 10: Build preprocessing pipeline
    preprocessor, numeric_cols, categorical_cols = build_preprocessor(X_train)

    print("\n=== NUMERIC COLUMNS ===")
    print(numeric_cols)

    print("\n=== CATEGORICAL COLUMNS ===")
    print(categorical_cols)

    # STEP 11: Build model pipelines
    baseline, rf, gb = build_model_pipelines(preprocessor)

    # STEP 12: Train and evaluate baseline model
    baseline.fit(X_train, y_train)
    baseline_results = evaluate_model(
        baseline,
        X_test,
        y_test,
        "Logistic Regression Baseline"
    )

    # STEP 13: Train and tune Random Forest
    rf_search = tune_random_forest(rf, X_train, y_train)

    print("\n=== BEST RANDOM FOREST PARAMETERS ===")
    print(rf_search.best_params_)

    rf_results = evaluate_model(
        rf_search.best_estimator_,
        X_test,
        y_test,
        "Tuned Random Forest"
    )

    # STEP 14: Train and tune Gradient Boosting
    gb_search = tune_gradient_boosting(gb, X_train, y_train)

    print("\n=== BEST GRADIENT BOOSTING PARAMETERS ===")
    print(gb_search.best_params_)

    gb_results = evaluate_model(
        gb_search.best_estimator_,
        X_test,
        y_test,
        "Tuned Gradient Boosting"
    )

    # STEP 15: Save model comparison results
    results_df = pd.DataFrame([
        baseline_results,
        rf_results,
        gb_results
    ])

    results_df.to_csv(TABLES_DIR / "model_results.csv", index=False)

    print("\n=== MODEL COMPARISON RESULTS ===")
    print(results_df)

    # STEP 16: Choose best model based on F1 score
    best_model_name = results_df.sort_values("f1", ascending=False).iloc[0]["model"]

    if best_model_name == "Tuned Random Forest":
        best_model = rf_search.best_estimator_
    elif best_model_name == "Tuned Gradient Boosting":
        best_model = gb_search.best_estimator_
    else:
        best_model = baseline

    print(f"\nBest model selected: {best_model_name}")

    # STEP 17: Save best model
    joblib.dump(best_model, MODELS_DIR / "best_model.joblib")
    print("Best model saved to outputs/models/best_model.joblib")

    # STEP 18: Segment analysis for fairness/error analysis
    segment_columns = [
        "Region",
        "Gender",
        "Customer Type",
        "Tariff Type",
        "Tenure Type",
        "Meter Type"
    ]

    for segment in segment_columns:
        if segment in X_test.columns:
            segment_df = evaluate_by_segment(best_model, X_test, y_test, segment)
            segment_df.to_csv(
                TABLES_DIR / f"segment_{segment.replace(' ', '_').lower()}.csv",
                index=False
            )

    print("\nProject run complete.")


if __name__ == "__main__":
    main()