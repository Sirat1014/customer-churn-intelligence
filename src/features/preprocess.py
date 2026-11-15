import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Load cleaned dataset
df = pd.read_csv("data/processed/cleaned_telco_churn.csv")

# Separate numerical and categorical columns
numerical_features = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "SeniorCitizen"
]

categorical_features = [
    column for column in df.columns
    if column not in numerical_features + ["customerID", "Churn"]
]

# Create preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ]
)

print("Numerical features:", numerical_features)
print("\nCategorical features:", categorical_features)
print("\nPreprocessing pipeline created successfully!")
