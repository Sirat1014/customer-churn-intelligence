import pandas as pd

df = pd.read_csv("data/processed/cleaned_telco_churn.csv")

columns = [
    "InternetService",
    "PaymentMethod",
    "OnlineSecurity",
    "TechSupport",
    "SeniorCitizen",
    "Dependents"
]

for column in columns:
    print(f"\n--- CHURN BY {column} ---")

    result = df.groupby(column, dropna=False).agg(
        Total_Customers=("Churn", "count"),
        Churned_Customers=("Churn", lambda x: (x == "Yes").sum())
    )

    result["Churn_Rate_Percent"] = (
        result["Churned_Customers"]
        / result["Total_Customers"]
    ) * 100

    print(result.round(2))
