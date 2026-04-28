import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_classif

from load_data import load_telco_data, clean_telco_data
from preprocess import split_features_target, get_feature_types


def compute_mutual_information(X: pd.DataFrame, y: pd.Series):
    """
    Compute mutual information scores for all features.
    Converts all non-numeric columns into numeric codes.
    """

    X_encoded = X.copy()

    for col in X_encoded.columns:
        if not pd.api.types.is_numeric_dtype(X_encoded[col]):
            X_encoded[col] = pd.factorize(X_encoded[col])[0]

    mi_scores = mutual_info_classif(X_encoded, y, random_state=42)

    mi_series = pd.Series(mi_scores, index=X.columns)
    mi_series = mi_series.sort_values(ascending=False)

    return mi_series


if __name__ == "__main__":
    df = load_telco_data()
    df = clean_telco_data(df)

    X, y = split_features_target(df)

    mi_scores = compute_mutual_information(X, y)

    print("\nMutual Information Scores:")
    print(mi_scores)

    print("\nTop 10 Features:") 
    print(mi_scores.head(10))