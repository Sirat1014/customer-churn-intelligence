import pandas as pd
import matplotlib.pyplot as plt

# Load cleaned dataset
df = pd.read_csv("data/processed/cleaned_telco_churn.csv")

# Create reports directory if needed
import os
os.makedirs("reports", exist_ok=True)

# 1. Churn rate by contract
contract_churn = (
    df.groupby("Contract")["Churn"]
    .apply(lambda x: (x == "Yes").mean() * 100)
)

plt.figure(figsize=(9, 5))
contract_churn.plot(kind="bar")
plt.title("Customer Churn Rate by Contract Type")
plt.xlabel("Contract Type")
plt.ylabel("Churn Rate (%)")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("reports/churn_by_contract.png", dpi=300)
plt.close()

# 2. Tenure distribution by churn status
plt.figure(figsize=(9, 5))

for status in ["No", "Yes"]:
    df.loc[df["Churn"] == status, "tenure"].plot(
        kind="hist",
        bins=20,
        alpha=0.5,
        label=f"Churn: {status}"
    )

plt.title("Customer Tenure Distribution")
plt.xlabel("Tenure (Months)")
plt.ylabel("Number of Customers")
plt.legend()
plt.tight_layout()
plt.savefig("reports/tenure_distribution.png", dpi=300)
plt.close()

# 3. Monthly charges by churn status
plt.figure(figsize=(8, 5))

df.boxplot(column="MonthlyCharges", by="Churn")

plt.title("Monthly Charges by Churn Status")
plt.suptitle("")
plt.xlabel("Churn")
plt.ylabel("Monthly Charges")
plt.tight_layout()
plt.savefig("reports/monthly_charges_by_churn.png", dpi=300)
plt.close()

print("Three EDA visualizations created successfully!")
print("Check the reports/ folder.")
