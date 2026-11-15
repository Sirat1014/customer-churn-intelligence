import pandas as pd

# Load cleaned dataset
df = pd.read_csv("data/processed/cleaned_telco_churn.csv")

print("\n--- CLEANED DATA VALIDATION ---")

# Dataset dimensions
print("\nDataset shape:", df.shape)

# Missing values
print("\nMissing values:")
print(df.isnull().sum().sum())

# Duplicate records
print("\nDuplicate rows:", df.duplicated().sum())

# Numeric summary
print("\nNumeric statistics:")
print(df[["tenure", "MonthlyCharges", "TotalCharges"]].describe())

# Target distribution
print("\nChurn distribution:")
print(df["Churn"].value_counts(normalize=True) * 100)

