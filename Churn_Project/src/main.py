from data_loader import load_data, basic_overview
from eda import save_missing_values_chart, save_target_chart, detect_possible_leakage_columns


def main():
    df = load_data()
    basic_overview(df)

    save_missing_values_chart(df)
    save_target_chart(df)

    leakage = detect_possible_leakage_columns(df)
    print("\n=== POSSIBLE LEAKAGE COLUMNS ===")
    print(leakage)


if __name__ == "__main__":
    main()