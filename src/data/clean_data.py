import pandas as pd

# Load raw dataset
df = pd.read_csv("data/raw/Telco-Customer-Churn.csv")

# Convert blank TotalCharges values to missing values
df["TotalCharges"] = df["TotalCharges"].str.strip()

df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

# Handle blank TotalCharges for customers with zero tenure
df.loc[
    (df["tenure"] == 0) & (df["TotalCharges"].isna()),
    "TotalCharges"
] = 0.0

# Verify cleaning
print("\n--- CLEANING RESULTS ---")
print("Total rows:", len(df))
print("TotalCharges data type:", df["TotalCharges"].dtype)
print("Remaining missing values:", df["TotalCharges"].isna().sum())
print("Zero-tenure customers:", (df["tenure"] == 0).sum())

# Save cleaned dataset
df.to_csv("data/processed/cleaned_telco_churn.csv", index=False)

print("\nCleaned dataset saved successfully.")
