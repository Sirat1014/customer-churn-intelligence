import pandas as pd

results = {
    "Model": [
        "Logistic Regression",
        "Random Forest",
        "XGBoost"
    ],
    "Accuracy": [0.74, 0.77, 0.81],
    "Churn_Precision": [0.50, 0.55, 0.67],
    "Churn_Recall": [0.78, 0.63, 0.52],
    "Churn_F1": [0.61, 0.59, 0.59],
    "ROC_AUC": [0.8416, 0.8208, 0.8405]
}

df = pd.DataFrame(results)

df.to_csv("reports/model_comparison.csv", index=False)

print("\n--- MODEL COMPARISON ---")
print(df.to_string(index=False))

print("\nComparison report saved successfully!")
