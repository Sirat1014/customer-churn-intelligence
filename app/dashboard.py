import streamlit as st
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split

# Page configuration
st.set_page_config(
    page_title="Customer Churn Intelligence",
    page_icon="📊",
    layout="wide"
)

# Load data
@st.cache_data
def load_data():
    return pd.read_csv("data/processed/cleaned_telco_churn.csv")

@st.cache_resource
def load_model():
    return joblib.load("models/logistic_regression_tuned.pkl")

df = load_data()
model = load_model()

# Recreate held-out test set
X = df.drop(columns=["customerID", "Churn"])
y = df["Churn"].map({"No": 0, "Yes": 1})

X_dev, X_test, y_dev, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Load test-set retention results
retention = pd.read_csv(
    "reports/customer_retention_priorities.csv"
)

# Dashboard heading
st.title("Customer Churn Intelligence")
st.caption("Predictive analytics and customer retention prioritisation")

st.divider()

# Executive overview
st.subheader("Executive Overview")

total_customers = len(retention)
high_risk = (retention["Risk_Category"] == "High").sum()
elevated_risk = (retention["Risk_Category"] == "Elevated").sum()
low_risk = (retention["Risk_Category"] == "Low").sum()

actual_churn = (
    retention["CustomerID"]
    .map(df.set_index("customerID")["Churn"])
    .eq("Yes")
    .sum()
)

col1, col2, col3, col4 = st.columns(4)

col1.metric("Customers Analysed", f"{total_customers:,}")
col2.metric("High-Risk Customers", f"{high_risk:,}")
col3.metric("Elevated-Risk Customers", f"{elevated_risk:,}")
col4.metric("Actual Churners", f"{actual_churn:,}")

st.divider()

# Risk distribution
st.subheader("Customer Risk Distribution")

risk_summary = pd.DataFrame({
    "Risk Category": ["High", "Elevated", "Low"],
    "Customers": [high_risk, elevated_risk, low_risk]
})

col1, col2 = st.columns(2)

with col1:
    st.bar_chart(
        risk_summary.set_index("Risk Category")
    )

with col2:
    st.dataframe(
        risk_summary,
        use_container_width=True,
        hide_index=True
    )

st.divider()

# Historical validation
st.subheader("Actual Churn Rate by Predicted Risk")

validation = pd.read_csv(
    "reports/retention_segment_validation.csv"
)

st.bar_chart(
    validation.set_index("Risk_Category")[
        "Actual_Churn_Rate_Percent"
    ]
)

st.caption(
    "Historical results from the held-out test set. "
    "These are not live customer predictions."
)


# Individual customer lookup
st.divider()
st.subheader("Individual Customer Risk Lookup")

customer_ids = retention["CustomerID"].tolist()

selected_customer = st.selectbox(
    "Search Customer ID",
    customer_ids
)

customer = retention[
    retention["CustomerID"] == selected_customer
].iloc[0]

col1, col2, col3 = st.columns(3)

col1.metric(
    "Churn Probability",
    f"{customer['Churn_Probability']:.1%}"
)

col2.metric(
    "Risk Category",
    customer["Risk_Category"]
)

col3.metric(
    "Predicted Churn",
    "Yes" if customer["Predicted_Churn"] == 1 else "No"
)

st.write("### Customer Details")

details = {
    "Tenure (months)": customer["tenure"],
    "Monthly Charges": customer["MonthlyCharges"],
    "Contract": customer["Contract"],
    "Internet Service": customer["InternetService"],
    "Payment Method": customer["PaymentMethod"],
}

st.json(details)

st.write("### Recommended Retention Action")
st.info(customer["Recommended_Action"])

st.caption(
    "Customer lookup uses the held-out historical test dataset, "
    "not live customer records."
)

# Model insights
st.divider()
st.subheader("Model Performance Insights")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Accuracy", "70%")
col2.metric("Churn Recall", "86.6%")
col3.metric("Churn Precision", "46.6%")
col4.metric("ROC-AUC", "0.8416")

st.caption(
    "Metrics are from the held-out test set using a 0.40 prediction threshold."
)

st.write("### Most Influential Churn Features")

importance = pd.read_csv(
    "reports/shap_feature_importance.csv"
).head(10)

importance = importance.sort_values(
    "Mean_Absolute_SHAP",
    ascending=True
)

st.bar_chart(
    importance.set_index("Feature")["Mean_Absolute_SHAP"]
)

st.caption(
    "SHAP importance represents average absolute feature influence "
    "on the model's output, not causal impact."
)

