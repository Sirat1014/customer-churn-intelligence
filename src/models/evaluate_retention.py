import pandas as pd
import joblib

from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("data/processed/cleaned_telco_churn.csv")

X = df.drop(columns=["customerID", "Churn"])
y = df["Churn"].map({"No": 0, "Yes": 1})

# Recreate test split
X_dev, X_test, y_dev, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Load retention report
retention = pd.read_csv(
    "reports/customer_retention_priorities.csv"
)

# Match actual outcomes using customer IDs
actual_outcomes = df.set_index("customerID")["Churn"]

retention["Actual_Churn"] = (
    retention["CustomerID"].map(actual_outcomes)
)

# Calculate actual churn rate by risk category
evaluation = retention.groupby(
    "Risk_Category",
    observed=False
).agg(
    Total_Customers=("CustomerID", "count"),
    Actual_Churned=("Actual_Churn", lambda x: (x == "Yes").sum())
)

evaluation["Actual_Churn_Rate_Percent"] = (
    evaluation["Actual_Churned"]
    / evaluation["Total_Customers"]
) * 100

print("\n--- RETENTION SEGMENT VALIDATION ---")
print(evaluation.round(2))

evaluation.to_csv(
    "reports/retention_segment_validation.csv"
)

print("\nValidation report saved successfully!")
