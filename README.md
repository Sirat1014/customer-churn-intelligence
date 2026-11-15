# Customer Churn Prediction & Retention Intelligence

An end-to-end machine learning project that predicts customer churn risk, explains model predictions, and prioritises customers for retention outreach through an interactive Streamlit dashboard.

## Overview

This project uses the IBM Telco Customer Churn dataset to identify customers who may be at risk of leaving a subscription-based service.

It combines exploratory data analysis, machine learning, model evaluation, SHAP explainability, customer risk segmentation, and illustrative retention recommendations in an interactive dashboard.

## Objectives

- Analyse historical customer churn patterns.
- Clean and prepare customer data.
- Train and compare classification models.
- Evaluate model performance using multiple metrics.
- Tune the classification threshold to improve churn detection.
- Explain model predictions using SHAP.
- Segment customers by predicted churn risk.
- Generate illustrative retention recommendations.
- Build an interactive Streamlit dashboard.

## Dataset

**IBM Telco Customer Churn Dataset**

- 7,043 customer records
- 21 columns
- Target variable: `Churn`

The dataset contains customer demographics, tenure, contract information, internet services, payment methods, and billing information.

**Source:** [IBM Telco Customer Churn Dataset](https://github.com/IBM/telco-customer-churn-on-icp4d)

### Data Cleaning

- Removed unnecessary whitespace from column values.
- Converted `TotalCharges` into a numeric column.
- Handled 11 missing `TotalCharges` values by assigning 0.0 to customers with zero tenure.
- Validated the cleaned dataset for missing values and duplicate records.

The original raw dataset was preserved separately from the cleaned dataset.

## Technologies

- **Programming:** Python 3.13
- **Data Analysis:** Pandas, NumPy
- **Machine Learning:** Scikit-learn, XGBoost
- **Explainability:** SHAP
- **Visualisation:** Matplotlib, Seaborn
- **Dashboard:** Streamlit
- **Model Persistence:** Joblib

## Machine Learning Workflow

1. Data inspection and cleaning
2. Exploratory data analysis
3. Numerical feature scaling
4. Categorical feature encoding
5. Stratified train-test splitting
6. Training Logistic Regression, Random Forest, and XGBoost
7. Comparing model performance
8. Classification threshold analysis
9. SHAP-based model explainability
10. Customer churn risk segmentation
11. Retention recommendation generation
12. Interactive Streamlit dashboard development

## Model Performance

The tuned Logistic Regression model was evaluated on a held-out test set using a classification threshold of 0.40.

| Metric | Result |
|---|---:|
| Accuracy | 70.0% |
| Churn Precision | 46.6% |
| Churn Recall | 86.6% |
| Churn F1-score | 0.61 |
| ROC-AUC | 0.8416 |

### Why Logistic Regression?

Three classification algorithms were evaluated:

- Logistic Regression
- Random Forest
- XGBoost

The tuned Logistic Regression model was selected for the retention-focused analysis because of its strong churn recall and ROC-AUC.

The classification threshold was lowered to 0.40 to identify more potential churners. This reduced false negatives but increased false positives.

The model therefore prioritises identifying potential churners rather than maximising overall accuracy.

## Customer Risk Segmentation

Customers were grouped using their predicted churn probabilities.

| Risk Category | Probability |
|---|---|
| Low | Below 40% |
| Elevated | 40% to below 60% |
| High | 60% or above |

### Held-out Test Set Validation

| Risk Category | Customers | Actual Churn Rate |
|---|---:|---:|
| High | 489 | 53.99% |
| Elevated | 206 | 29.13% |
| Low | 714 | 7.00% |

The high-risk segment showed a substantially higher historical churn rate than the low-risk segment.

These results demonstrate how predicted probabilities can help prioritise retention outreach.

Retention recommendations are illustrative, rule-based suggestions and have not been experimentally validated.

## Model Explainability

SHAP (SHapley Additive exPlanations) was used to interpret Logistic Regression predictions.

Important features by mean absolute SHAP value included:

- Tenure
- Monthly Charges
- Total Charges
- Internet Service
- Contract Type

SHAP values help explain how features contribute to model predictions. They do not establish causal relationships.

## Interactive Dashboard

The Streamlit dashboard provides:

- Executive overview of customer risk.
- High, elevated, and low-risk customer distribution.
- Historical churn rates across predicted risk segments.
- Individual customer risk lookup.
- Predicted churn probability.
- Customer-specific retention recommendations.
- SHAP-based feature importance visualisation.

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
├── DATA_LICENSE
├── README.md
└── requirements.txt
```

## Setup and Installation

### 1. Clone the repository

```bash
git clone https://github.com/Sirat1014/customer-churn-intelligence.git
cd customer-churn-intelligence
```

### 2. Create a virtual environment

```bash
python3 -m venv .venv
```

### 3. Activate the virtual environment

macOS / Linux:

```bash
source .venv/bin/activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the dashboard

```bash
streamlit run app/dashboard.py
```

## Limitations

- Uses historical telecom customer data rather than live customer records.
- Predictions are estimates, not guarantees of future customer behaviour.
- The selected threshold prioritises recall at the cost of more false positives.
- Retention recommendations are illustrative and require real-world testing.
- Model performance may vary across different customer populations.

## Future Improvements

- Improve probability calibration.
- Evaluate precision-recall performance.
- Add new-customer prediction functionality.
- Introduce cost-sensitive retention prioritisation.
- Test retention strategies using real-world outcomes.
- Deploy the dashboard.

## Author

## Author

**Subhsirat Kaur Sehra**

BSc Computer Science Graduate | Aspiring Machine Learning Engineer

GitHub: [@Sirat1014](https://github.com/Sirat1014)
