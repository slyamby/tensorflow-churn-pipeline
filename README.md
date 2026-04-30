# Customer Churn Prediction Using TensorFlow

This project predicts telecom customer churn with a TensorFlow neural network and serves the model through a Streamlit app. It combines feature selection, structured preprocessing, threshold-based decisioning, and business-friendly output for retention use cases.

## Overview

The pipeline is designed to answer a practical question: which customers are most likely to leave, and how should the business respond?

Core components:

- Mutual information feature selection
- Scikit-learn preprocessing pipeline
- TensorFlow binary classification model
- Class weighting for imbalanced data
- Threshold tuning for churn recall
- Streamlit interface for interactive predictions

## Problem Statement

Customer churn directly affects revenue, customer lifetime value, and acquisition cost. The goal of this project is to identify at-risk customers early enough to support retention decisions.

The model is used to:

- Predict churn probability
- Assign a churn class based on a decision threshold
- Group customers into risk categories
- Suggest simple business actions based on risk

## Dataset

- Dataset: Telco Customer Churn Dataset
- Source: [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

The raw dataset is expected at:

```text
data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv
```

## Project Structure

```text
tensorflow-churn-pipeline/
├── app.py
├── README.md
├── requirements.txt
├── data/
│   └── raw/
├── models/
│   ├── telco_churn_tensorflow.keras
│   ├── preprocessor.pkl
│   └── top_features.pkl
└── src/
    ├── __init__.py
    ├── evaluate.py
    ├── feature_selection.py
    ├── load_data.py
    ├── predict.py
    ├── preprocess.py
    └── train_model.py
```

## Methodology

### 1. Data Preparation

The data preparation step:

- Drops `customerID`
- Converts `TotalCharges` to numeric
- Fills missing `TotalCharges` values with the median
- Maps `Churn` from `Yes`/`No` to `1`/`0`

### 2. Feature Selection

Mutual information is used to rank candidate features by predictive value. The current training flow keeps the top 10 features.

Top features used by the current model:

- `Contract`
- `tenure`
- `OnlineSecurity`
- `TechSupport`
- `OnlineBackup`
- `InternetService`
- `PaymentMethod`
- `DeviceProtection`
- `MonthlyCharges`
- `TotalCharges`

### 3. Preprocessing

The preprocessing pipeline is built with scikit-learn:

- Numerical columns are standardized with `StandardScaler`
- Categorical columns are encoded with `OneHotEncoder(handle_unknown="ignore")`

The fitted preprocessor is saved to `models/preprocessor.pkl`.

### 4. Model

The churn classifier is a feed-forward neural network:

```text
Input
→ Dense(128, relu)
→ Dropout(0.3)
→ Dense(64, relu)
→ Dropout(0.3)
→ Dense(32, relu)
→ Dropout(0.2)
→ Dense(1, sigmoid)
```

The model is trained with:

- Loss: `binary_crossentropy`
- Optimizer: `adam`
- Metrics: `accuracy`, `AUC`
- Early stopping on validation loss
- Class weights to reduce bias toward the majority class

### 5. Threshold Tuning

The training script compares multiple thresholds:

- `0.30`
- `0.35`
- `0.40`
- `0.45`
- `0.50`

This allows you to trade off false positives against churn recall depending on business goals.

## Current Results

From the current saved training flow:

- Accuracy: about `0.76`
- AUC: about `0.84`
- Churn recall at threshold `0.40`: about `0.86`

Representative confusion matrix at threshold `0.50`:

```text
[[800 235]
 [ 98 276]]
```

These numbers may change if you retrain the model.

## Business Interpretation

The current feature rankings suggest higher churn risk is associated with:

- Month-to-month contracts
- Lower tenure
- Lack of security or tech support services
- Certain payment method and internet service combinations

The Streamlit app translates predicted risk into practical actions such as retention outreach, targeted offers, and customer monitoring.

## Running the Project

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the model

Run from the project root:

```bash
python -m src.train_model
```

This saves:

- `models/telco_churn_tensorflow.keras`
- `models/preprocessor.pkl`
- `models/top_features.pkl`

### 3. Evaluate the saved model

```bash
python -m src.evaluate
```

### 4. Launch the Streamlit app

```bash
streamlit run app.py
```

## Future Improvements

- Add SHAP-based explainability
- Persist threshold selection as a model artifact
- Add tests for preprocessing and inference
- Improve calibration of predicted probabilities
- Package the app for cloud deployment
