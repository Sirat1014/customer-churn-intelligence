import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score
)

df = pd.read_csv("data/processed/cleaned_telco_churn.csv")

X = df.drop(columns=["customerID", "Churn"])
y = df["Churn"].map({"No": 0, "Yes": 1})

# Recreate the original 80:20 split
X_dev, X_test, y_dev, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

numerical_features = [
    "tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"
]

categorical_features = [
    column for column in X.columns
    if column not in numerical_features
]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ]
)

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42
        ))
    ]
)

# Train on the full development set
model.fit(X_dev, y_dev)

# Predict probabilities on the untouched test set
probabilities = model.predict_proba(X_test)[:, 1]

# Apply selected threshold
threshold = 0.40
predictions = (probabilities >= threshold).astype(int)

print("\n--- FINAL TEST RESULTS: THRESHOLD 0.40 ---")
print(classification_report(y_test, predictions))

print("Confusion Matrix:")
print(confusion_matrix(y_test, predictions))

print("ROC-AUC:", round(roc_auc_score(y_test, probabilities), 4))

joblib.dump(model, "models/logistic_regression_tuned.pkl")

print("\nTuned model saved successfully!")
