import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import os

from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("data/processed/cleaned_telco_churn.csv")

X = df.drop(columns=["customerID", "Churn"])
y = df["Churn"].map({"No": 0, "Yes": 1})

# Recreate the original test split
X_dev, X_test, y_dev, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Load tuned model
model = joblib.load("models/logistic_regression_tuned.pkl")

# Transform test data using the fitted preprocessor
preprocessor = model.named_steps["preprocessor"]
classifier = model.named_steps["classifier"]

X_test_transformed = preprocessor.transform(X_test)

# Get transformed feature names
feature_names = preprocessor.get_feature_names_out()

# SHAP explainer for Logistic Regression
explainer = shap.LinearExplainer(
    classifier,
    preprocessor.transform(X_dev)
)

shap_values = explainer(X_test_transformed)

# Create reports directory
os.makedirs("reports", exist_ok=True)

# Global feature importance
plt.figure()
shap.summary_plot(
    shap_values,
    X_test_transformed,
    feature_names=feature_names,
    show=False
)

plt.tight_layout()
plt.savefig("reports/shap_summary.png", dpi=300, bbox_inches="tight")
plt.close()

# Feature importance bar chart
plt.figure()
shap.summary_plot(
    shap_values,
    X_test_transformed,
    feature_names=feature_names,
    plot_type="bar",
    show=False
)

plt.tight_layout()
plt.savefig("reports/shap_feature_importance.png", dpi=300, bbox_inches="tight")
plt.close()

print("SHAP analysis completed successfully!")
print("Check the reports/ folder.")

importance = pd.DataFrame({
    "Feature": feature_names,
    "Mean_Absolute_SHAP": abs(shap_values.values).mean(axis=0)
})

importance = importance.sort_values(
    by="Mean_Absolute_SHAP",
    ascending=False
)

print("\n--- TOP 15 CHURN INFLUENCING FEATURES ---")
print(importance.head(15).to_string(index=False))

importance.to_csv("reports/shap_feature_importance.csv", index=False)


import numpy as np

# Convert transformed test data to a NumPy array
X_dense = (
    X_test_transformed.toarray()
    if hasattr(X_test_transformed, "toarray")
    else np.asarray(X_test_transformed)
)

shap_array = shap_values.values

print("\n--- DIRECTION OF SHAP INFLUENCE ---")

# Compare low and high values for numerical features
for feature in [
    "num__tenure",
    "num__MonthlyCharges",
    "num__TotalCharges"
]:
    index = list(feature_names).index(feature)
    values = X_dense[:, index]
    contributions = shap_array[:, index]

    low_limit = np.percentile(values, 25)
    high_limit = np.percentile(values, 75)

    low_mean = contributions[values <= low_limit].mean()
    high_mean = contributions[values >= high_limit].mean()

    print(f"\n{feature}")
    print("Low-value average SHAP:", round(low_mean, 4))
    print("High-value average SHAP:", round(high_mean, 4))

# Compare selected binary categories
for feature in [
    "cat__Contract_Month-to-month",
    "cat__Contract_Two year",
    "cat__InternetService_Fiber optic",
    "cat__PaymentMethod_Electronic check"
]:
    index = list(feature_names).index(feature)
    active = X_dense[:, index] == 1

    print(f"\n{feature}")
    print("Average SHAP when present:",
          round(shap_array[active, index].mean(), 4))
    print("Average SHAP when absent:",
          round(shap_array[~active, index].mean(), 4))

