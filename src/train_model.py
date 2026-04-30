import os
import joblib
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.utils.class_weight import compute_class_weight

from src.load_data import load_telco_data, clean_telco_data
from src.preprocess import (
    split_features_target,
    get_feature_types,
    build_preprocessing_pipeline,
    train_test_split_data
)
from src.feature_selection import compute_mutual_information


# ------------------------------
# 1. Configuration
# ------------------------------
TOP_N_FEATURES = 10
EPOCHS = 50
BATCH_SIZE = 32
MODEL_PATH = "models/telco_churn_tensorflow.keras"
PREPROCESSOR_PATH = "models/preprocessor.pkl"
TOP_FEATURES_PATH = "models/top_features.pkl"


# -------------------------------
# 2. Build Tensorflow Model
# -------------------------------
def build_tf_model(input_dim: int):
    """
    Build a simple neural network for binary classification.
    """

    model = models.Sequential([
        layers.Input(shape=(input_dim,)),

        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),

        layers.Dense(64, activation="relu"),
        layers.Dropout(0.3),

        layers.Dense(32, activation="relu"),
        layers.Dropout(0.2),

        layers.Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
    )

    return model


# ----------------------------
# 3. Main Training Workflow
# ----------------------------
def main():
    os.makedirs("models", exist_ok=True)

    # Load and clean data
    df = load_telco_data()
    df = clean_telco_data(df)

    # Split X and y
    X, y = split_features_target(df)

    # Compute mutual information scores
    mi_scores = compute_mutual_information(X, y)

    print("\nTop Mutual Information Features")
    print(mi_scores.head(TOP_N_FEATURES))

    # Select top N features
    top_features = mi_scores.head(TOP_N_FEATURES).index.tolist()

    joblib.dump(top_features, TOP_FEATURES_PATH)

    print("\nSelected Features")
    print(top_features)

    X_selected = X[top_features]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split_data(X_selected, y)

    # Identify feature types after selection
    numerical_cols, categorical_cols = get_feature_types(X_train)

    print("\nNumerical Columns")
    print(numerical_cols)

    print("\nCategorical Columns")
    print(categorical_cols)

    # Build preprocessing pipeline
    preprocessor = build_preprocessing_pipeline(
        numerical_cols=numerical_cols,
        categorical_cols=categorical_cols
    )

    # Fit preprocessing on training data only
    X_train_processed = preprocessor.fit_transform(X_train)

    # Transform test data using same fitted processor
    X_test_processed = preprocessor.transform(X_test)

    # Convert sparse matrix to dense array if needed
    if hasattr(X_train_processed, "toarray"):
        X_train_processed = X_train_processed.toarray()

    if hasattr(X_test_processed, "toarray"):
        X_test_processed = X_test_processed.toarray()

    print("\nProcessed Train Shape:", X_train_processed.shape)
    print("Processed Test Shape:", X_test_processed.shape)

    # Build Tensorflow model
    input_dim = X_train_processed.shape[1]
    model = build_tf_model(input_dim)

    model.summary()

    # Early stopping to reduce overfitting
    early_stopping = callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    )

    classes = np.unique(y_train)
    weights = compute_class_weight("balanced", classes=classes, y=y_train)
    class_weights = dict(zip(classes, weights))

    print("\nClass Weights:")
    print(class_weights)

    # True model
    history = model.fit(
        X_train_processed,
        y_train,
        validation_split=0.2,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stopping],
        class_weight=class_weights,
        verbose=1
    )

    joblib.dump(preprocessor, PREPROCESSOR_PATH)

    # Evaluate model
    test_loss, test_accuracy, test_auc = model.evaluate(X_test_processed, y_test, verbose=0)

    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test AUC: {test_auc:.4f}")

    # Predictions
    pred_probs = model.predict(X_test_processed).ravel()

    thresholds = [0.3, 0.35, 0.4, 0.45, 0.5]

    print("\nThreshold Comparison:")

    for threshold in thresholds:
        preds = (pred_probs >= threshold).astype(int)

        print(f"\nThreshold: {threshold}")
        print(classification_report(
            y_test,
            preds,
            target_names=["No Churn", "Churn"]
        ))

    print("\nClassification Report:")
    print(classification_report(y_test, preds, target_names=["No Churn", "Churn"]))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, preds))

    print("\nAccuracy Score:")
    print(accuracy_score(y_test, preds))

    # Save model
    model.save(MODEL_PATH)

    print(f"\nModel saved to: {MODEL_PATH}")

if __name__ == "__main__":
    main()
