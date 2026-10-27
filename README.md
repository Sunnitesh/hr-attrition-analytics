# HR Attrition Analytics & Prediction

SQL- and machine-learning-based HR decision support system built on the IBM HR Analytics dataset (1,470 employees). Data is stored in a SQLite database, a leakage-free Random Forest estimates each employee's attrition risk, and a Streamlit app shows SQL-driven attrition analytics and a Risk Check tool.

## Pipeline (run in this order)

| Step | Script | What it does |
|---|---|---|
| 1 | `python setup_db.py` | Loads the CSV into SQLite (`hr.db`), removes 4 non-predictive columns with SQL, creates the `employees` and `predictions` tables |
| 2 | `python train_model.py` | Trains the 10-feature Random Forest without data leakage, saves the model, scaler and `model_info.json` |
| 3 | `streamlit run app.py` | Opens the app (Risk Check + SQL Overview) at http://localhost:8501 |

Install the libraries first: `pip install -r requirements.txt`

## SQL tables

- `employees`: historical IBM data, used for SQL `GROUP BY` queries (attrition rate by department, job role, overtime, job satisfaction and age band, plus the overall rate).
- `predictions`: every Risk Check made in the app is saved here (employee, inputs, risk %, label) and shown on the Overview tab.

## Leakage-free methodology

1. Train/test split (80/20, stratified, `random_state=42`) is done before any scaling.
2. The scaler is fit on training data only.
3. The decision threshold is chosen with 5-fold stratified cross-validation on the training set only (F2 score, which favours recall).
4. The model is evaluated once on the untouched test set. Exact numbers are printed by `train_model.py` and saved in `model_info.json`.

The model favours recall over precision on purpose: missing a real leaver costs more than reviewing a false alarm. A high-risk flag should start a supportive conversation, not an adverse action.

## Features used (10)

Age, DistanceFromHome, EnvironmentSatisfaction, JobSatisfaction, MonthlyIncome, NumCompaniesWorked, TotalWorkingYears, WorkLifeBalance, YearsAtCompany, OverTime.

## Other files

- `HR.ipynb`: earlier exploratory notebook (EDA and model comparison of Logistic Regression, Decision Tree and Random Forest).
- `Attrition dashboard.pbix`, `Attrition_overview.png`, `Prediction_UI_Overview.png`: Power BI dashboard and screenshots.
- `WA_Fn-UseC_-HR-Employee-Attrition.csv`: raw dataset.

## Limitations

- Fixed dataset of 1,470 records; the model does not retrain on newly saved predictions.
- The IBM dataset is synthetic, so results are for demonstration.
