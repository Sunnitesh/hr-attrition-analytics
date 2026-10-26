
import streamlit as st
import joblib
import numpy as np

# 1. Page configuration
st.set_page_config(
    page_title="HR Attrition Predictor",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Rich Animated Background & Custom UI Styling
st.markdown("""
<style>
    /* Main Background: Multi-stop rich gradient */
    [data-testid="stAppViewContainer"] {
        background: radial-gradient(circle at top left, #1e1b4b 0%, #0f172a 40%, #020617 100%) !important;
    }

    /* Top Navigation Header */
    [data-testid="stHeader"] {
        background: transparent !important;
    }

    /* Sidebar Background */
    [data-testid="stSidebar"] {
        background: #090d16 !important;
        border-right: 1px solid rgba(139, 92, 246, 0.2) !important;
    }

    /* Base Typography Colors */
    h1, h2, h3, h4, h5, h6, p, label, .stMarkdown {
        color: #f8fafc !important;
    }

    /* Hero Banner */
    .hero-banner {
        background: linear-gradient(135deg, #6366f1 0%, #a855f7 50%, #ec4899 100%);
        padding: 2rem 1rem;
        border-radius: 18px;
        box-shadow: 0 10px 30px rgba(168, 85, 247, 0.35);
        margin-bottom: 2rem;
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.15);
    }

    /* Modern Glassmorphic Input Cards */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: rgba(15, 23, 42, 0.65) !important;
        backdrop-filter: blur(12px) !important;
        -webkit-backdrop-filter: blur(12px) !important;
        border: 1px solid rgba(147, 197, 253, 0.15) !important;
        border-radius: 16px !important;
        padding: 1.2rem !important;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37) !important;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }

    div[data-testid="stVerticalBlockBorderWrapper"]:hover {
        border-color: rgba(168, 85, 247, 0.45) !important;
        transform: translateY(-2px);
    }

    /* Interactive Inputs & Sliders styling */
    .stSlider > div [data-baseweb="slider"] {
        color: #a855f7 !important;
    }
    
    /* Result Box Metric */
    [data-testid="stMetricValue"] {
        color: #38bdf8 !important;
        font-size: 2.2rem !important;
        font-weight: 800 !important;
    }
    [data-testid="stMetricLabel"] > div {
        color: #c4b5fd !important;
        font-size: 1rem !important;
    }

    /* Action Trigger Button */
    .stButton > button {
        background: linear-gradient(135deg, #ec4899 0%, #8b5cf6 100%) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.85rem 2rem !important;
        font-weight: 700 !important;
        font-size: 1.1rem !important;
        letter-spacing: 0.5px !important;
        box-shadow: 0 4px 20px rgba(236, 72, 153, 0.45) !important;
        transition: all 0.3s ease !important;
    }
    .stButton > button:hover {
        transform: scale(1.02) !important;
        box-shadow: 0 6px 25px rgba(139, 92, 246, 0.6) !important;
    }
</style>
""", unsafe_allow_html=True)

# 3. Model & Scaler Loader
@st.cache_resource
def load_artifacts():
    from pathlib import Path
    base = Path(__file__).parent
    model = joblib.load(base / "hr_attrition_model_10.pkl")
    scaler = joblib.load(base / "hr_attrition_scaler_10.pkl")
    return model, scaler

try:
    model, scaler = load_artifacts()
except Exception as e:
    st.error(f"Error loading model files: {e}")
    st.stop()

# 4. Header Banner
st.markdown("""
<div class="hero-banner">
    <h1 style="margin:0; font-size: 2.3rem; font-weight: 800; color: #ffffff !important;">🔮 HR Attrition Risk Predictor</h1>
    <p style="margin:0.4rem 0 0 0; font-size: 1.05rem; color: #e0e7ff !important; opacity: 0.95;">AI-Powered Workforce Retention & Flight-Risk Analytics</p>
</div>
""", unsafe_allow_html=True)

# 5. Sidebar Controls
with st.sidebar:
    st.header("⚙️ Settings")
    threshold = st.slider("Risk Cut-off Threshold", 0.05, 0.50, 0.30, 0.01)
    st.caption("Lower cut-offs trigger proactive retention warnings earlier.")
    
    st.markdown("---")
    st.markdown("### 📌 Model Profile")
    st.info("**Algorithm:** Random Forest Classifier\n\n**Source:** IBM HR Analytics\n\n**Feature Depth:** 10 Core Drivers")

# 6. Glassmorphic Input Columns
col1, col2, col3 = st.columns(3)

with col1:
    with st.container(border=True):
        st.markdown("<h4 style='color:#a78bfa !important; margin-bottom:1rem;'>👤 Personal & Role</h4>", unsafe_allow_html=True)
        age = st.slider("Age", 18, 60, 30)
        monthly_income = st.number_input("Monthly Income ($)", 1000, 20000, 5000, step=250)
        overtime = st.selectbox("Overtime Status", ["No", "Yes"])

with col2:
    with st.container(border=True):
        st.markdown("<h4 style='color:#38bdf8 !important; margin-bottom:1rem;'>💼 Experience & Commute</h4>", unsafe_allow_html=True)
        years_at_company = st.slider("Years at Company", 0, 40, 5)
        total_working_years = st.slider("Total Working Experience", 0, 40, 10)
        num_companies_worked = st.slider("Past Companies Worked", 0, 10, 2)
        distance_from_home = st.slider("Commute Distance (km)", 1, 30, 5)

with col3:
    with st.container(border=True):
        st.markdown("<h4 style='color:#f472b6 !important; margin-bottom:1rem;'>📊 Satisfaction Metrics</h4>", unsafe_allow_html=True)
        job_satisfaction = st.select_slider(
            "Job Satisfaction", 
            options=[1, 2, 3, 4], 
            value=3,
            format_func=lambda x: {1: "🔴 Low", 2: "🟠 Moderate", 3: "🟡 High", 4: "🟢 Very High"}[x]
        )
        work_life_balance = st.select_slider(
            "Work-Life Balance", 
            options=[1, 2, 3, 4], 
            value=3,
            format_func=lambda x: {1: "🔴 Poor", 2: "🟠 Fair", 3: "🟡 Good", 4: "🟢 Best"}[x]
        )
        environment_satisfaction = st.select_slider(
            "Work Environment", 
            options=[1, 2, 3, 4], 
            value=3,
            format_func=lambda x: {1: "🔴 Low", 2: "🟠 Moderate", 3: "🟡 High", 4: "🟢 Very High"}[x]
        )

st.markdown("<br>", unsafe_allow_html=True)

# 7. Action Button
_, center_col, _ = st.columns([1, 1.3, 1])
with center_col:
    predict_clicked = st.button("🚀 Analyze Flight Risk", use_container_width=True)

# 8. Output Display
if predict_clicked:
    overtime_val = 1 if overtime == "Yes" else 0

    input_data = np.array([[
        age, distance_from_home, environment_satisfaction,
        job_satisfaction, monthly_income, num_companies_worked,
        total_working_years, work_life_balance,
        years_at_company, overtime_val
    ]])

    input_scaled = scaler.transform(input_data)
    probability = float(model.predict_proba(input_scaled)[0][1])

    st.markdown("---")
    st.markdown("### 🎯 Assessment Results")

    with st.container(border=True):
        res_col1, res_col2 = st.columns([1, 2])

        with res_col1:
            st.metric(label="Calculated Attrition Risk", value=f"{probability*100:.1f}%")
            st.progress(min(max(probability, 0.0), 1.0))

        with res_col2:
            if probability >= threshold:
                st.error(f"🚨 **High Flight Risk Detected ({probability*100:.1f}%)**")
                st.write("• Probability exceeds the current early-warning threshold.")
                st.write("• **Recommended Action:** Schedule a 1-on-1 retention discussion regarding workload and compensation.")
            else:
                st.success(f"✅ **Low Flight Risk ({probability*100:.1f}%)**")
                st.write("• Employee engagement and stability indicators look healthy.")
                st.write("• **Recommended Action:** Maintain standard quarterly development check-ins.")