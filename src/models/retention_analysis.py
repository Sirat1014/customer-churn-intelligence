import pandas as pd
import joblib

from sklearn.model_selection import train_test_split

# Load cleaned dataset
df = pd.read_csv("data/processed/cleaned_telco_churn.csv")

# Separate features and target
X = df.drop(columns=["customerID", "Churn"])
y = df["Churn"].map({"No": 0, "Yes": 1})

# Recreate original test split
X_dev, X_test, y_dev, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Load tuned model
model = joblib.load("models/logistic_regression_tuned.pkl")

# Predict churn probabilities
churn_probabilities = model.predict_proba(X_test)[:, 1]

# Create customer-level retention report
retention = X_test.copy()

retention.insert(
    0,
    "CustomerID",
    df.loc[X_test.index, "customerID"]
)

retention["Churn_Probability"] = churn_probabilities
retention["Predicted_Churn"] = (
    retention["Churn_Probability"] >= 0.40
).astype(int)

# Assign risk categories
retention["Risk_Category"] = pd.cut(
    retention["Churn_Probability"],
    bins=[-0.001, 0.40, 0.60, 1.001],
    labels=["Low", "Elevated", "High"],
    right=False
)

# Recommend retention actions
def recommend_action(row):

    if row["Risk_Category"] == "High":
        if row["Contract"] == "Month-to-month":
            return "Offer a suitable long-term contract incentive"

        if row["tenure"] <= 12:
            return "Provide onboarding and early customer support"

        return "Assign priority retention follow-up"

    if row["Risk_Category"] == "Elevated":
        if row["TechSupport"] == "No":
            return "Offer a technical support consultation"

        if row["OnlineSecurity"] == "No":
            return "Offer an online security service review"

        return "Send a personalised engagement offer"

    return "Continue regular customer engagement"

retention["Recommended_Action"] = retention.apply(
    recommend_action,
    axis=1
)

# Sort highest-risk customers first
retention = retention.sort_values(
    by="Churn_Probability",
    ascending=False
)

# Save customer-level report
retention.to_csv(
    "reports/customer_retention_priorities.csv",
    index=False
)

# Display risk distribution
print("\n--- CUSTOMER RETENTION PRIORITIES ---")

print(
    retention["Risk_Category"]
    .value_counts()
    .reindex(["High", "Elevated", "Low"], fill_value=0)
)

print("\n--- RECOMMENDED ACTIONS ---")
print(retention["Recommended_Action"].value_counts())

print("\n--- HIGHEST-RISK CUSTOMERS ---")
print(
    retention[
        ["CustomerID", "Churn_Probability",
         "Risk_Category", "Recommended_Action"]
    ].head(10).to_string(index=False)
)

print("\nRetention report saved successfully!")
