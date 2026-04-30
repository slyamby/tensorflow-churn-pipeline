from __future__ import annotations

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score

from src.load_data import load_telco_data, clean_telco_data
from src.predict import load_artifacts, prepare_input
from src.preprocess import split_features_target, train_test_split_data


def evaluate_model(threshold: float = 0.5):
    """Evaluate the trained TensorFlow churn model on the held-out test set."""
    df = clean_telco_data(load_telco_data())
    X, y = split_features_target(df)

    model, preprocessor, top_features = load_artifacts()
    X_selected = X[top_features]
    _, X_test, _, y_test = train_test_split_data(X_selected, y)

    X_test_processed = prepare_input(X_test, preprocessor, top_features)
    pred_probs = model.predict(X_test_processed, verbose=0).ravel()
    preds = (pred_probs >= threshold).astype(int)

    metrics = {
        "accuracy": accuracy_score(y_test, preds),
        "auc": roc_auc_score(y_test, pred_probs),
        "confusion_matrix": confusion_matrix(y_test, preds),
        "classification_report": classification_report(
            y_test,
            preds,
            target_names=["No Churn", "Churn"],
        ),
    }
    return metrics


def main():
    metrics = evaluate_model()
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"AUC: {metrics['auc']:.4f}")
    print("\nClassification Report:")
    print(metrics["classification_report"])
    print("Confusion Matrix:")
    print(metrics["confusion_matrix"])


if __name__ == "__main__":
    main()
