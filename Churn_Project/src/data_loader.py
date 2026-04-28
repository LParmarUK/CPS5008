import pandas as pd
from config import DATA_FILE, TARGET_COLUMN


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_FILE)
    return df


def basic_overview(df: pd.DataFrame) -> None:
    print("\n=== SHAPE ===")
    print(df.shape)

    print("\n=== COLUMNS ===")
    print(df.columns.tolist())

    print("\n=== DTYPES ===")
    print(df.dtypes)

    print("\n=== MISSING VALUES ===")
    print(df.isnull().sum().sort_values(ascending=False))

    if TARGET_COLUMN in df.columns:
        print("\n=== TARGET DISTRIBUTION ===")
        print(df[TARGET_COLUMN].value_counts(dropna=False))
        print(df[TARGET_COLUMN].value_counts(normalize=True, dropna=False))