from data_loader import load_data, basic_overview
from eda import save_missing_values_chart, save_target_chart, detect_possible_leakage_columns


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

    basic_overview(df)

    save_missing_values_chart(df)
    save_target_chart(df)

    leakage = detect_possible_leakage_columns(df)
    print("\n=== POSSIBLE LEAKAGE COLUMNS ===")
    print(leakage)


if __name__ == "__main__":
    main()