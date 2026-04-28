import pandas as pd
from pathlib import Path

DATA_PATH = Path("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")


def load_telco_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """
    Load the Telco Customer Churn dataset.
    """
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}")
    
    df = pd.read_csv(path)
    return df


def clean_telco_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean dataset before modeling
    """
    df = df.copy()

    # Customer ID is just an identifier, not useful for prediction
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    # TotalCharges sometimes comes as text because of blank values
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    
    # Fill missing TotalCharges with median
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

    # Convert Target column to 0/1
    df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})

    return df


if __name__ == "__main__":
    df = load_telco_data()
    df = clean_telco_data(df)

    print("Dataset loaded successfully")
    print(f"Shape: {df.shape}")
    print("\nColumns:")
    print(df.columns.tolist())
    print("\nMissing Values:")
    print(df.isna().sum())
    print("Target Distribution")
    print(df["Churn"].value_counts(normalize=True))
