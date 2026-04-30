from data_loader import load_data, basic_overview
from eda import save_missing_values_chart, save_target_chart, detect_possible_leakage_columns
from sklearn.model_selection import train_test_split
from preprocess import build_preprocessor
from train import build_model_pipelines
from evaluate import evaluate_model

def clean_data(df):
    df.columns = df.columns.str.strip()

    if "Customer ID" in df.columns:
        df = df.drop(columns=["Customer ID"])

    if df["Churn"].dtype == "object":
        df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    return df

def add_freatures(df):
    if all(col in df.columns for col in ["App Logins", "Portal Logins", "Email Clicks"]):
        df["Digital_Engagement_Total"] = (
            df["App Logins"] + df["Portal Logins"] + df["Email Clicks"]
        )
    return df

def main():
    df = load_data()
    df = clean_data(df)
    df = add_freatures(df)

    X= df.drop(columns=["Churn"])
    y= df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    preprocesser, _, _ = build_preprocessor(X_train)
    baseline, rf, gb = build_model_pipelines(preprocesser)
    baseline.fit(X_train, y_train)
    evaluate_model(baseline, X_test, y_test, "Baseline")

if __name__ == "__main__":
    main()