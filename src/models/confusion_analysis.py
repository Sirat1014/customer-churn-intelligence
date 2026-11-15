import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import os

df = pd.read_csv("data/processed/cleaned_telco_churn.csv")

X = df.drop(columns=["customerID", "Churn"])
y = df["Churn"].map({"No": 0, "Yes": 1})

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

models = {
    "Logistic Regression": "models/logistic_regression.pkl",
    "Random Forest": "models/random_forest.pkl",
    "XGBoost": "models/xgboost.pkl"
}

os.makedirs("reports", exist_ok=True)

for name, path in models.items():
    model = joblib.load(path)

    predictions = model.predict(X_test)

    cm = confusion_matrix(y_test, predictions)

    print(f"\n--- {name} ---")
    print(cm)

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Stayed", "Churned"]
    )

    display.plot(cmap="Blues")
    plt.title(f"{name} - Confusion Matrix")
    plt.tight_layout()

    filename = name.lower().replace(" ", "_")
    plt.savefig(f"reports/{filename}_confusion_matrix.png", dpi=300)
    plt.close()

print("\nAll confusion matrices saved successfully!")
