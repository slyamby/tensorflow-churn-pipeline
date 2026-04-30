from __future__ import annotations

import joblib
import pandas as pd
import tensorflow as tf

MODEL_PATH = "models/telco_churn_tensorflow.keras"
PREPROCESSOR_PATH = "models/preprocessor.pkl"
TOP_FEATURES_PATH = "models/top_features.pkl"


def load_artifacts(
    model_path: str = MODEL_PATH,
    preprocessor_path: str = PREPROCESSOR_PATH,
    top_features_path: str = TOP_FEATURES_PATH,
):
    """Load the trained model and preprocessing artifacts."""
    model = tf.keras.models.load_model(model_path)
    preprocessor = joblib.load(preprocessor_path)
    top_features = joblib.load(top_features_path)
    return model, preprocessor, top_features


def prepare_input(input_df: pd.DataFrame, preprocessor, top_features: list[str]):
    """Select expected features and transform them for inference."""
    missing_features = [feature for feature in top_features if feature not in input_df.columns]
    if missing_features:
        missing = ", ".join(missing_features)
        raise ValueError(f"Missing required features: {missing}")

    transformed = preprocessor.transform(input_df[top_features])
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()
    return transformed


def predict_churn(input_df: pd.DataFrame, threshold: float = 0.5):
    """Run churn prediction for one or more input rows."""
    model, preprocessor, top_features = load_artifacts()
    transformed = prepare_input(input_df, preprocessor, top_features)
    probabilities = model.predict(transformed, verbose=0).ravel()
    predictions = (probabilities >= threshold).astype(int)

    results = input_df.copy()
    results["churn_probability"] = probabilities
    results["prediction"] = predictions
    return results
