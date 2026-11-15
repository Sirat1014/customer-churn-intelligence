import pandas as pd

# Load dataset
df = pd.read_csv("data/raw/Telco-Customer-Churn.csv")

# Investigate blank TotalCharges
print("\n--- BLANK TOTAL CHARGES INVESTIGATION ---")

blank_mask = df["TotalCharges"].str.strip() == ""

blank_customers = df.loc[
    blank_mask,
    ["customerID", "tenure", "MonthlyCharges", "TotalCharges", "Churn"]
]

print(blank_customers)

print("\nNumber of blank records:", blank_mask.sum())

print("\nTenure distribution:")
print(df.loc[blank_mask, "tenure"].value_counts())
