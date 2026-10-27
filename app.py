import json
import sqlite3
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

BASE = Path(__file__).parent
DB_PATH = BASE / "hr.db"

# 1. Page configuration
st.set_page_config(
    page_title="HR Attrition Predictor",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Custom UI styling
st.markdown("""
<style>
    [data-testid="stAppViewContainer"] {
        background: radial-gradient(circle at top left, #1e1b4b 0%, #0f172a 40%, #020617 100%) !important;
    }
    [data-testid="stHeader"] { background: transparent !important; }
    [data-testid="stSidebar"] {
        background: #090d16 !important;
        border-right: 1px solid rgba(139, 92, 246, 0.2) !important;
    }
    h1, h2, h3, h4, h5, h6, p, label, .stMarkdown { color: #f8fafc !important; }
    .hero-banner {
        background: linear-gradient(135deg, #6366f1 0%, #a855f7 50%, #ec4899 100%);
        padding: 2rem 1rem; border-radius: 18px;
        box-shadow: 0 10px 30px rgba(168, 85, 247, 0.35);
        margin-bottom: 2rem; text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.15);
    }
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: rgba(15, 23, 42, 0.65) !important;
        backdrop-filter: blur(12px) !important;
        -webkit-backdrop-filter: blur(12px) !important;
        border: 1px solid rgba(147, 197, 253, 0.15) !important;
        border-radius: 16px !important; padding: 1.2rem !important;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37) !important;
    }
    .stSlider > div [data-baseweb="slider"] { color: #a855f7 !important; }
    [data-testid="stMetricValue"] {
        color: #38bdf8 !important; font-size: 2.2rem !important; font-weight: 800 !important;
    }
    [data-testid="stMetricLabel"] > div { color: #c4b5fd !important; font-size: 1rem !important; }
    .stButton > button {
        background: linear-gradient(135deg, #ec4899 0%, #8b5cf6 100%) !important;
        color: #ffffff !important; border: none !important; border-radius: 12px !important;
        padding: 0.85rem 2rem !important; font-weight: 700 !important; font-size: 1.1rem !important;
        box-shadow: 0 4px 20px rgba(236, 72, 153, 0.45) !important;
    }
</style>
""", unsafe_allow_html=True)


# 3. Loaders -----------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load(BASE / "hr_attrition_model_10.pkl")
    scaler = joblib.load(BASE / "hr_attrition_scaler_10.pkl")
    return model, scaler


def load_model_info():
    f = BASE / "model_info.json"
    return json.loads(f.read_text()) if f.exists() else {}


def run_sql(query, params=()):
    """Run a SQL query on hr.db and return a DataFrame."""
    con = sqlite3.connect(DB_PATH)
    try:
        return pd.read_sql_query(query, con, params=params)
    finally:
        con.close()


def save_prediction(row):
    con = sqlite3.connect(DB_PATH)
    con.execute(
        """INSERT INTO predictions
           (employee_name, age, distance_from_home, environment_satisfaction, job_satisfaction,
            monthly_income, num_companies_worked, total_working_years, work_life_balance,
            years_at_company, overtime, risk_probability, threshold, risk_label)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""", row)
    con.commit()
    con.close()


try:
    model, scaler = load_artifacts()
except Exception as e:
    st.error(f"Error loading model files: {e}. Run train_model.py first.")
    st.stop()

if not DB_PATH.exists():
    st.error("Database hr.db not found. Run `python setup_db.py` first.")
    st.stop()

info = load_model_info()
default_threshold = float(info.get("threshold", 0.30))

# 4. Header
st.markdown("""
<div class="hero-banner">
    <h1 style="margin:0; font-size: 2.3rem; font-weight: 800; color: #ffffff !important;">🔮 HR Attrition Risk Predictor</h1>
    <p style="margin:0.4rem 0 0 0; font-size: 1.05rem; color: #e0e7ff !important;">SQL-backed Workforce Retention & Flight-Risk Analytics</p>
</div>
""", unsafe_allow_html=True)

# 5. Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    threshold = st.slider("Risk Cut-off Threshold", 0.05, 0.50,
                          min(max(default_threshold, 0.05), 0.50), 0.01)
    st.caption("Default threshold was chosen by 5-fold cross-validation on training data only. "
               "Lower cut-offs trigger retention warnings earlier.")
    st.markdown("---")
    st.markdown("### 📌 Model Profile")
    profile = ("**Algorithm:** Random Forest Classifier\n\n**Source:** IBM HR Analytics\n\n"
               "**Feature Depth:** 10 Core Drivers")
    if info:
        profile += (f"\n\n**Held-out test:** recall {info['recall']*100:.0f}%, "
                    f"precision {info['precision']*100:.0f}%, ROC-AUC {info['roc_auc']:.2f}")
    st.info(profile)

tab_risk, tab_overview = st.tabs(["🎯 Risk Check", "📊 Overview (SQL)"])

# ============================ TAB 1: RISK CHECK =============================
with tab_risk:
    col1, col2, col3 = st.columns(3)

    with col1:
        with st.container(border=True):
            st.markdown("<h4 style='color:#a78bfa !important;'>👤 Personal & Role</h4>", unsafe_allow_html=True)
            emp_name = st.text_input("Employee Name / ID (optional)", "")
            age = st.slider("Age", 18, 60, 30)
            monthly_income = st.number_input("Monthly Income ($)", 1000, 20000, 5000, step=250)
            overtime = st.selectbox("Overtime Status", ["No", "Yes"])

    with col2:
        with st.container(border=True):
            st.markdown("<h4 style='color:#38bdf8 !important;'>💼 Experience & Commute</h4>", unsafe_allow_html=True)
            years_at_company = st.slider("Years at Company", 0, 40, 5)
            total_working_years = st.slider("Total Working Experience", 0, 40, 10)
            num_companies_worked = st.slider("Past Companies Worked", 0, 10, 2)
            distance_from_home = st.slider("Commute Distance (km)", 1, 30, 5)

    with col3:
        with st.container(border=True):
            st.markdown("<h4 style='color:#f472b6 !important;'>📊 Satisfaction Metrics</h4>", unsafe_allow_html=True)
            job_satisfaction = st.select_slider(
                "Job Satisfaction", options=[1, 2, 3, 4], value=3,
                format_func=lambda x: {1: "🔴 Low", 2: "🟠 Moderate", 3: "🟡 High", 4: "🟢 Very High"}[x])
            work_life_balance = st.select_slider(
                "Work-Life Balance", options=[1, 2, 3, 4], value=3,
                format_func=lambda x: {1: "🔴 Bad", 2: "🟠 Good", 3: "🟡 Better", 4: "🟢 Best"}[x])
            environment_satisfaction = st.select_slider(
                "Work Environment", options=[1, 2, 3, 4], value=3,
                format_func=lambda x: {1: "🔴 Low", 2: "🟠 Moderate", 3: "🟡 High", 4: "🟢 Very High"}[x])

    st.markdown("<br>", unsafe_allow_html=True)
    _, center_col, _ = st.columns([1, 1.3, 1])
    with center_col:
        predict_clicked = st.button("🚀 Analyze Flight Risk", width="stretch")

    if predict_clicked:
        # Input validation: reject impossible combinations
        errors = []
        if years_at_company > total_working_years:
            errors.append("Years at Company cannot be more than Total Working Experience.")
        if total_working_years > age - 16:
            errors.append("Total Working Experience is not possible for this age.")
        if errors:
            for e in errors:
                st.warning(e)
        else:
            overtime_val = 1 if overtime == "Yes" else 0
            input_data = pd.DataFrame([[
                age, distance_from_home, environment_satisfaction, job_satisfaction,
                monthly_income, num_companies_worked, total_working_years,
                work_life_balance, years_at_company, overtime_val
            ]], columns=["Age", "DistanceFromHome", "EnvironmentSatisfaction", "JobSatisfaction",
                         "MonthlyIncome", "NumCompaniesWorked", "TotalWorkingYears",
                         "WorkLifeBalance", "YearsAtCompany", "OverTime"])
            probability = float(model.predict_proba(scaler.transform(input_data))[0][1])
            high = probability >= threshold
            label = "High" if high else "Low"

            # Save this check into the SQL predictions table
            save_prediction((emp_name.strip() or "Unnamed", age, distance_from_home,
                             environment_satisfaction, job_satisfaction, monthly_income,
                             num_companies_worked, total_working_years, work_life_balance,
                             years_at_company, overtime, probability, threshold, label))

            st.markdown("---")
            st.markdown("### 🎯 Assessment Results")
            with st.container(border=True):
                res_col1, res_col2 = st.columns([1, 2])
                with res_col1:
                    st.metric(label="Calculated Attrition Risk", value=f"{probability*100:.1f}%")
                    st.progress(min(max(probability, 0.0), 1.0))
                with res_col2:
                    if high:
                        st.error(f"🚨 **High Flight Risk Detected ({probability*100:.1f}%)**")
                        st.write("• Probability exceeds the current early-warning threshold.")
                        st.write("• **Recommended Action:** Schedule a supportive 1-on-1 retention discussion.")
                    else:
                        st.success(f"✅ **Low Flight Risk ({probability*100:.1f}%)**")
                        st.write("• Engagement and stability indicators look healthy.")
                        st.write("• **Recommended Action:** Maintain standard quarterly check-ins.")
                    st.caption("This is an estimated probability, not a certainty. A high-risk flag should "
                               "start a supportive conversation, not an adverse action.")
            st.success("Result saved to the SQL `predictions` table.")

# ============================ TAB 2: OVERVIEW (SQL) =========================
with tab_overview:
    st.markdown("### 📊 Workforce Attrition Overview")
    st.caption("Every number below is computed with SQL queries on hr.db (table: employees).")

    overall = run_sql("""SELECT COUNT(*) AS employees,
                                SUM(CASE WHEN Attrition='Yes' THEN 1 ELSE 0 END) AS left_count,
                                ROUND(100.0*SUM(CASE WHEN Attrition='Yes' THEN 1 ELSE 0 END)/COUNT(*),1) AS attrition_rate
                         FROM employees""").iloc[0]
    m1, m2, m3 = st.columns(3)
    m1.metric("Total Employees", int(overall["employees"]))
    m2.metric("Employees Who Left", int(overall["left_count"]))
    m3.metric("Overall Attrition Rate", f"{overall['attrition_rate']}%")

    def rate_query(group_expr, label):
        return run_sql(f"""SELECT {group_expr} AS {label},
                                  COUNT(*) AS employees,
                                  ROUND(100.0*SUM(CASE WHEN Attrition='Yes' THEN 1 ELSE 0 END)/COUNT(*),1) AS attrition_rate
                           FROM employees GROUP BY {label} ORDER BY attrition_rate DESC""")

    age_band = """CASE WHEN Age < 25 THEN '18-24' WHEN Age < 35 THEN '25-34'
                       WHEN Age < 45 THEN '35-44' WHEN Age < 55 THEN '45-54' ELSE '55+' END"""
    views = [
        ("By Department", rate_query("Department", "department"), "department"),
        ("By Job Role", rate_query("JobRole", "job_role"), "job_role"),
        ("By Overtime", rate_query("OverTime", "overtime"), "overtime"),
        ("By Job Satisfaction (1=Low, 4=Very High)", rate_query("JobSatisfaction", "job_satisfaction"), "job_satisfaction"),
        ("By Age Band", rate_query(age_band, "age_band"), "age_band"),
    ]
    for title, data, idx in views:
        with st.container(border=True):
            st.markdown(f"#### {title}")
            c1, c2 = st.columns([2, 1])
            c1.bar_chart(data.set_index(idx)["attrition_rate"], height=240)
            c2.dataframe(data, hide_index=True, width="stretch")

    st.markdown("### 🗂️ Saved Risk Checks (SQL `predictions` table)")
    summary = run_sql("""SELECT COUNT(*) AS total_checked,
                                COALESCE(SUM(CASE WHEN risk_label='High' THEN 1 ELSE 0 END),0) AS high_risk
                         FROM predictions""").iloc[0]
    p1, p2 = st.columns(2)
    p1.metric("Employees Checked", int(summary["total_checked"]))
    p2.metric("Flagged High Risk", int(summary["high_risk"]))
    history = run_sql("""SELECT checked_at, employee_name, age, overtime, monthly_income,
                                ROUND(risk_probability*100,1) AS risk_percent, risk_label
                         FROM predictions ORDER BY id DESC LIMIT 20""")
    if history.empty:
        st.info("No risk checks saved yet. Use the Risk Check tab to add one.")
    else:
        st.dataframe(history, hide_index=True, width="stretch")
