import pandas as pd

# Load cleaned dataset
df = pd.read_csv("data/processed/cleaned_telco_churn.csv")

# Churn rate by contract type
contract_analysis = df.groupby("Contract").agg(
    Total_Customers=("customerID", "count"),
    Churned_Customers=("Churn", lambda x: (x == "Yes").sum())
)

contract_analysis["Churn_Rate_Percent"] = (
    contract_analysis["Churned_Customers"]
    / contract_analysis["Total_Customers"]
) * 100

print("\n--- CHURN ANALYSIS BY CONTRACT ---")
print(contract_analysis.round(2))

# Average tenure by churn status
print("\n--- AVERAGE TENURE BY CHURN STATUS ---")
print(df.groupby("Churn")["tenure"].mean().round(2))

# Average monthly charges by churn status
print("\n--- MONTHLY CHARGES BY CHURN STATUS ---")
print(df.groupby("Churn")["MonthlyCharges"].mean().round(2))
