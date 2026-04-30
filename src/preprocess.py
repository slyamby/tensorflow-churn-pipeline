import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


def split_features_target(df: pd.DataFrame):
    """
    split dataset into features (X) and target (Y)

    """
    X = df.drop(columns=["Churn"])
    y = df["Churn"]
    return X, y


def get_feature_types(X: pd.DataFrame):
    """
    Identify numerical and categorical columns
    """
    numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object","string"]).columns.tolist()

    return numerical_cols, categorical_cols


def build_preprocessing_pipeline(numerical_cols, categorical_cols):
    """
    Build preprocessing pipeline:
    - scale numerical features
    - one-hot encode categorical features

    """
    numerical_pipeline = Pipeline(steps=[
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline(steps=[
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_pipeline, numerical_cols),
            ("cat", categorical_pipeline, categorical_cols)
        ]
    )

    return preprocessor


def train_test_split_data(X,y):
    """
    Split into train and test sets
    """
    return train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )


if __name__ == "__main__":
    from src.load_data import load_telco_data, clean_telco_data

    df = load_telco_data()
    df = clean_telco_data(df)

    X, y = split_features_target(df)
    num_cols, cat_cols = get_feature_types(X)
    
    print("Numerical columns:", num_cols)
    print("\nCategorical columns:", cat_cols)

    X_train, X_test, y_train, y_test = train_test_split_data(X, y)

    print("\nTrain shape:", X_train.shape)
    print("Test shape:", X_test.shape)
