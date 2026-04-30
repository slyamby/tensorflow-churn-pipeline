import streamlit as st
import pandas as pd
from src.predict import load_artifacts as load_prediction_artifacts
from src.predict import prepare_input

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="Customer Churn Prediction App",
    page_icon="📊",
    layout="wide"
)

# -----------------------------
# Load artifacts
# -----------------------------
@st.cache_resource
def load_artifacts():
    # CHANGE TRACKING: App cleanup. Reuse the shared inference loader from src.predict
    # so the Streamlit app and CLI inference path stay aligned.
    return load_prediction_artifacts()


model, preprocessor, top_features = load_artifacts()

# CHANGE TRACKING: App cleanup. Keep the feature importance values in one
# module-level constant to reduce inline duplication in the render path.
FEATURE_IMPORTANCE = {
    "Contract": 0.091574,
    "tenure": 0.076971,
    "OnlineSecurity": 0.060765,
    "TechSupport": 0.058751,
    "OnlineBackup": 0.055380,
    "InternetService": 0.054400,
    "PaymentMethod": 0.047580,
    "DeviceProtection": 0.046295,
    "MonthlyCharges": 0.045334,
    "TotalCharges": 0.043115,
}

# -----------------------------
# Header
# -----------------------------
st.title("📊 Customer Churn Prediction App")
st.markdown(
    """
    This app predicts whether a telecom customer is likely to churn using a
    TensorFlow neural network trained on selected high-value features.
    """
)

st.divider()

# -----------------------------
# Sidebar information
# -----------------------------
with st.sidebar:
    st.header("About this App")
    st.write(
        """
        This project uses:
        - Mutual Information feature selection
        - Scikit-learn preprocessing pipeline
        - TensorFlow neural network
        - Threshold tuning for churn detection
        """
    )

    st.subheader("Model Threshold")
    threshold = st.slider(
        "Prediction threshold",
        min_value=0.30,
        max_value=0.70,
        value=0.40,
        step=0.05
    )

    st.caption("Lower threshold catches more churners but may increase false alarms.")

# -----------------------------
# Input form
# -----------------------------
st.subheader("Customer Information")

with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        payment_method = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ]
        )
        monthly_charges = st.number_input("Monthly Charges", value=50.0, min_value=0.0)

    with col2:
        online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
        tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
        online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
        device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
        total_charges = st.number_input("Total Charges", value=500.0, min_value=0.0)

    submit_button = st.form_submit_button("Predict Churn")

def get_risk_category(probability):
    """
    Convert churn probability into a business-friendly risk category.
    """
    if probability < 0.30:
        return "Low Risk", "Customer is unlikely to churn."
    elif probability < 0.60:
        return "Medium Risk", "Customer shows some churn risk and should be monitored."
    else:
        return "High Risk", "Customer is at significant risk of churn."
    

def get_business_recommendations(risk_category):
    """
    Return business actions based on churn risk category.
    """
    if risk_category == "High Risk":
        return [
            "Assign customer to retention team immediately.",
            "Offer a personalized discount or loyalty incentive.",
            "Review service issues such as support, security, or billing complaints.",
            "Prioritize follow-up within 24–48 hours."
        ]

    elif risk_category == "Medium Risk":
        return [
            "Monitor customer activity closely.",
            "Send targeted engagement messages or product education.",
            "Offer optional service bundle recommendations.",
            "Schedule a satisfaction check-in."
        ]

    else:
        return [
            "Maintain regular communication.",
            "Encourage continued usage through loyalty messaging.",
            "Offer upsell or cross-sell only if customer engagement is strong.",
            "No urgent intervention required."
        ]


def build_input_frame():
    """Create a single-row input frame from the current form values."""
    return pd.DataFrame([{
        "Contract": contract,
        "tenure": tenure,
        "OnlineSecurity": online_security,
        "TechSupport": tech_support,
        "OnlineBackup": online_backup,
        "InternetService": internet_service,
        "PaymentMethod": payment_method,
        "DeviceProtection": device_protection,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
    }])


# -----------------------------
# Prediction logic
# -----------------------------
if submit_button:
    # CHANGE TRACKING: App cleanup. Centralize input shaping and preprocessing
    # through shared helpers instead of duplicating transform logic in the UI.
    input_data = build_input_frame()
    input_processed = prepare_input(input_data, preprocessor, top_features)

    prob = model.predict(input_processed, verbose=0)[0][0]
    prediction_label = "Churn" if prob >= threshold else "No Churn"
    
    risk_category, risk_message = get_risk_category(prob)
    recommendations = get_business_recommendations(risk_category)

    st.divider()
    st.subheader("Prediction Result")

    result_col1, result_col2, result_col3 = st.columns(3)

    with result_col1:
        st.metric("Churn Probability", f"{prob:.2%}")

    with result_col2:
        st.metric("Predicted Class", prediction_label)

    with result_col3:
        st.metric("Risk Category", risk_category)

    st.caption(f"Decision threshold used for classification: {threshold:.2f}")


    if risk_category == "High Risk":
        st.error(risk_message)
    elif risk_category == "Medium Risk":
        st.warning(risk_message)
    else:
        st.success(risk_message)
    
    st.subheader("Recommended Business Actions")

    for rec in recommendations:
        st.write(f"- {rec}")

    st.subheader("Top Churn Drivers Used by the Model")

    importance_df = pd.DataFrame({
        "Feature": FEATURE_IMPORTANCE.keys(),
        "Mutual Information Score": FEATURE_IMPORTANCE.values()
    })

    st.dataframe(importance_df, use_container_width=True)

    st.bar_chart(
    importance_df.set_index("Feature")
    )
