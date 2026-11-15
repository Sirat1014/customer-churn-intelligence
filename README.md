# Customer Churn Prediction & Retention Intelligence

An end-to-end machine learning project that predicts customer churn risk, explains model predictions, and prioritises customers for retention outreach.

## Overview

This project uses the IBM Telco Customer Churn dataset to identify customers who may be at risk of leaving a subscription-based service.

It combines exploratory data analysis, machine learning, model explainability, customer risk segmentation, and an interactive Streamlit dashboard.

## Objectives

- Analyse historical customer churn patterns.
- Clean and prepare customer data.
- Train and compare classification models.
- Evaluate model performance.
- Tune the classification threshold.
- Explain predictions using SHAP.
- Segment customers by predicted churn risk.
- Generate illustrative retention recommendations.
- Build an interactive dashboard.

## Dataset

**IBM Telco Customer Churn Dataset**

- 7,043 customer records
- 21 columns
- Target variable: `Churn`

The dataset includes customer demographics, tenure, contract information, internet services, payment methods, and billing information.

Source: https://github.com/IBM/telco-customer-churn-on-icp4d

## Technologies

- Python 3.13
- Pandas and NumPy
- Scikit-learn
- XGBoost
- SHAP
- Matplotlib and Seaborn
- Streamlit
- Joblib

## Machine Learning Workflow

1. Data inspection and cleaning
2. Exploratory data analysis
3. Numerical feature scaling
4. Categorical feature encoding
5. Stratified train-test split
6. Logistic Regression, Random Forest, and XGBoost training
7. Model comparison
8. Classification threshold analysis
9. SHAP explainability
10. Customer risk segmentation
11. Streamlit dashboard

## Model Performance

Held-out test results for the tuned Logistic Regression model at a classification threshold of 0.40:

| Metric | Result |
|---|---:|
| Accuracy | 70.0% |
| Churn Precision | 46.6% |
| Churn Recall | 86.6% |
| Churn F1-score | 0.61 |
| ROC-AUC | 0.8416 |

The threshold was lowered to 0.40 to identify more potential churners. This increased recall but also increased false positives.

## Customer Risk Segmentation

Customers were grouped using their predicted churn probabilities:

| Risk Category | Probability |
|---|---|
| Low | Below 40% |
| Elevated | 40% to below 60% |
| High | 60% or above |

Observed results on the held-out test set:

| Risk Category | Customers | Actual Churn Rate |
|---|---:|---:|
| High | 489 | 53.99% |
| Elevated | 206 | 29.13% |
| Low | 714 | 7.00% |

These groups show different historical churn rates. Retention recommendations are illustrative and have not been experimentally validated.

## Model Explainability

SHAP was used to interpret Logistic Regression predictions.

Important features by mean absolute SHAP value included:

- Tenure
- Monthly Charges
- Total Charges
- Internet Service
- Contract Type

SHAP explains model behaviour; it does not establish causation.

## Project Structure

```text
customer-churn-intelligence/
├── app/
│   └── dashboard.py
├── data/
│   ├── raw/
│   └── processed/
├── models/
│   └── logistic_regression_tuned.pkl
├── reports/
├── src/
│   ├── data/
│   ├── features/
│   └── models/
├── .gitignore
├── README.md
└── requirements.txt


git clone git clone <https://github.com/Sirat1014/customer-churn-intelligence>
cd customer-churn-intelligence
## Setup and Installation

Clone the repository:

```bash
git clone https://github.com/Sirat1014/customer-churn-intelligence.git
cd customer-churn-intelligence
```

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the dashboard:

```bash
streamlit run app/dashboard.py
```

## Limitations

- Uses historical telecom data, not live customer records.
- Predictions are not guarantees of future behaviour.
- The selected threshold prioritises recall at the cost of more false positives.
- Retention recommendations are illustrative and require real-world testing.

## Future Improvements

- Improve probability calibration.
- Evaluate precision-recall performance.
- Add new-customer prediction functionality.
- Introduce cost-sensitive retention prioritisation.
- Deploy the dashboard.

## Author

Developed as a machine learning and data science portfolio project.
