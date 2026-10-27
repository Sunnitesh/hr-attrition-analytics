"""Stage 1 - Load the IBM HR dataset into SQLite and clean it using SQL.

Run once:  python setup_db.py
Creates hr.db with two tables:
  employees   - historical data (1,470 rows) used for SQL attrition analytics
  predictions - risk checks saved by the Streamlit app (starts empty)
"""
import sqlite3
import sys
from pathlib import Path

import pandas as pd

BASE = Path(__file__).parent
CSV = BASE / "WA_Fn-UseC_-HR-Employee-Attrition.csv"
DB = BASE / "hr.db"

# Columns that must exist in the CSV (REQ-4.1.4: halt if malformed)
REQUIRED = ["Age", "Attrition", "Department", "JobRole", "OverTime", "MonthlyIncome",
            "DistanceFromHome", "EnvironmentSatisfaction", "JobSatisfaction",
            "WorkLifeBalance", "NumCompaniesWorked", "TotalWorkingYears",
            "YearsAtCompany", "EmployeeCount", "StandardHours", "Over18", "EmployeeNumber"]
# Constant or identifier columns with no predictive value (REQ-4.1.2)
DROP_COLS = ["EmployeeCount", "StandardHours", "Over18", "EmployeeNumber"]


def main():
    df = pd.read_csv(CSV, encoding="utf-8-sig")
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        sys.exit(f"ERROR: CSV is missing expected columns: {missing}")

    con = sqlite3.connect(DB)
    cur = con.cursor()

    # Load raw CSV into the employees table (overwrite on every run)
    df.to_sql("employees", con, if_exists="replace", index=False)

    # Remove non-predictive columns using SQL statements
    for col in DROP_COLS:
        cur.execute(f'ALTER TABLE employees DROP COLUMN "{col}"')

    # Table that stores risk checks made from the app
    cur.execute("DROP TABLE IF EXISTS predictions")
    cur.execute("""
        CREATE TABLE predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            checked_at TEXT DEFAULT CURRENT_TIMESTAMP,
            employee_name TEXT,
            age INTEGER,
            distance_from_home INTEGER,
            environment_satisfaction INTEGER,
            job_satisfaction INTEGER,
            monthly_income INTEGER,
            num_companies_worked INTEGER,
            total_working_years INTEGER,
            work_life_balance INTEGER,
            years_at_company INTEGER,
            overtime TEXT,
            risk_probability REAL,
            threshold REAL,
            risk_label TEXT
        )
    """)
    con.commit()

    n = cur.execute("SELECT COUNT(*) FROM employees").fetchone()[0]
    cols = [r[1] for r in cur.execute("PRAGMA table_info(employees)")]
    print(f"Loaded {n} rows into 'employees' ({len(cols)} columns after SQL cleaning)")
    print("Created empty table 'predictions'")
    con.close()


if __name__ == "__main__":
    main()
