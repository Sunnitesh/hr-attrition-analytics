# HR Attrition Analytics & Prediction

End-to-end analysis of employee attrition using the IBM HR Analytics dataset (1,470 employees), with a Power BI dashboard and a Streamlit app that predicts an employee's flight risk.

## What's in this repo

| File | Purpose |
|---|---|
| `HR.ipynb` | EDA, cleaning, model comparison, threshold tuning, model export |
| `app.py` | Streamlit app for attrition risk prediction (10-feature model) |
| `Attrition dashboard.pbix` | Power BI dashboard (needs Power BI Desktop) |
| `Attrition_overview.png`, `Prediction_UI_Overview.png` | Screenshots of the dashboard and the app |
| `hr_attrition_model.pkl`, `hr_attrition_scaler.pkl` | Full-feature Random Forest and its scaler |
| `hr_attrition_model_10.pkl`, `hr_attrition_scaler_10.pkl` | 10-feature Random Forest and scaler used by the app |
| `WA_Fn-UseC_-HR-Employee-Attrition.csv` / `.xlsx` | Raw dataset |
| `hr_attrition_clean.csv`, `hr_attrition_dashboard.csv` | Cleaned data for modelling and for the dashboard |

## Key findings

- About 16% of employees left (class imbalance).
- Employees working overtime left at 30.5% vs 10.4% for those who did not, roughly 3x.

## Models

Four models were compared: Logistic Regression, Logistic Regression (balanced), Decision Tree and Random Forest. Random Forest was chosen, and its decision threshold was lowered from 0.5 because missing a real leaver costs more than a false alarm.

| Model | Threshold | Recall (leavers) | Precision (leavers) |
|---|---|---|---|
| Random Forest, all features | 0.15 | 72% | 31% |
| Random Forest, 10 features (used in the app) | 0.30 | 70% | 31% |

The 10 features: Age, DistanceFromHome, EnvironmentSatisfaction, JobSatisfaction, MonthlyIncome, NumCompaniesWorked, TotalWorkingYears, WorkLifeBalance, YearsAtCompany, OverTime.

Note: thresholds were chosen on the test set (20% stratified split, `random_state=42`), so these numbers are slightly optimistic.

## Run the app

1. Install Python 3.12.
2. Install the libraries: `pip install -r requirements.txt`
3. From this folder run: `streamlit run app.py`

The app opens at `http://localhost:8501`. The risk cut-off can be changed from the sidebar (default 0.30).

## Notebook

Open `HR.ipynb` with Jupyter (`jupyter notebook`) from this folder, so the dataset path resolves.
