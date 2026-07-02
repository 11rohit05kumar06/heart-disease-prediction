import os
import streamlit as st
import pandas as pd
import joblib

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
    /* Main title banner */
    .title-banner {
        background: linear-gradient(90deg, #ff4b4b 0%, #b91d73 100%);
        padding: 25px 30px;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 25px;
    }
    .title-banner h1 {
        color: white;
        font-size: 2.2rem;
        margin: 0;
        white-space: nowrap;
    }
    .title-banner p {
        color: #ffe0e0;
        margin: 8px 0 0 0;
        font-size: 1rem;
    }

    /* Section headers */
    .section-header {
        background: rgba(255, 75, 75, 0.1);
        border-left: 4px solid #ff4b4b;
        padding: 8px 15px;
        border-radius: 5px;
        margin: 15px 0 10px 0;
        font-weight: 600;
        font-size: 1.05rem;
    }

    /* Predict button */
    div.stButton > button {
        background: linear-gradient(90deg, #ff4b4b 0%, #b91d73 100%);
        color: white;
        font-size: 1.2rem;
        font-weight: bold;
        padding: 12px 0;
        width: 100%;
        border: none;
        border-radius: 10px;
        transition: 0.3s;
    }
    div.stButton > button:hover {
        transform: scale(1.02);
        box-shadow: 0 5px 20px rgba(255, 75, 75, 0.4);
        color: white;
    }

    /* Result cards */
    .result-danger {
        background: rgba(255, 75, 75, 0.15);
        border: 2px solid #ff4b4b;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        font-size: 1.3rem;
        font-weight: bold;
        color: #ff4b4b;
    }
    .result-safe {
        background: rgba(33, 195, 84, 0.15);
        border: 2px solid #21c354;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        font-size: 1.3rem;
        font-weight: bold;
        color: #21c354;
    }
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "heart_disease_model.pkl")
model = joblib.load(MODEL_PATH)

# ---------------- HEADER ----------------
st.markdown("""
<div class="title-banner">
    <h1>❤️ Heart Disease Prediction System</h1>
    <p>AI-powered risk assessment based on clinical health indicators</p>
</div>
""", unsafe_allow_html=True)

# ---------------- INPUT FORM ----------------
col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown('<div class="section-header">🧍 Personal Information</div>', unsafe_allow_html=True)
    age = st.slider("Age", 20, 100, 50)
    sex = st.selectbox("Sex", [0, 1], format_func=lambda x: ["Female", "Male"][x])

    st.markdown('<div class="section-header">🩺 Vitals & Blood Work</div>', unsafe_allow_html=True)
    trestbps = st.slider("Resting Blood Pressure (mm Hg)", 80, 200, 120)
    chol = st.slider("Cholesterol (mg/dl)", 100, 600, 200)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1], format_func=lambda x: ["No", "Yes"][x])
    thalach = st.slider("Maximum Heart Rate Achieved", 60, 220, 150)

with col2:
    st.markdown('<div class="section-header">💢 Chest Pain & ECG</div>', unsafe_allow_html=True)
    cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3],
        format_func=lambda x: ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"][x])
    restecg = st.selectbox("Resting ECG Results", [0, 1, 2],
        format_func=lambda x: ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"][x])

    st.markdown('<div class="section-header">🏃 Exercise & Heart Tests</div>', unsafe_allow_html=True)
    exang = st.selectbox("Exercise Induced Angina", [0, 1], format_func=lambda x: ["No", "Yes"][x])
    oldpeak = st.slider("ST Depression (Oldpeak)", 0.0, 10.0, 1.0, 0.1)
    slope = st.selectbox("Slope of Peak Exercise ST", [0, 1, 2],
        format_func=lambda x: ["Upsloping", "Flat", "Downsloping"][x])
    ca = st.selectbox("Number of Major Vessels (0–4)", [0, 1, 2, 3, 4])
    thal = st.selectbox("Thalassemia", [0, 1, 2, 3],
        format_func=lambda x: ["Unknown", "Normal", "Fixed Defect", "Reversible Defect"][x])

# ---------------- CREATE DATAFRAME ----------------
input_data = pd.DataFrame([[
    age, sex, cp, trestbps, chol, fbs, restecg,
    thalach, exang, oldpeak, slope, ca, thal
]], columns=[
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
    'restecg', 'thalach', 'exang', 'oldpeak',
    'slope', 'ca', 'thal'
])

# ---------------- PREDICTION ----------------
st.write("")
if st.button("🔍  Predict Heart Disease"):
    prediction = model.predict(input_data)

    # Try to get probability (works for RF/XGBoost/LogReg)
    try:
        proba = model.predict_proba(input_data)[0][1] * 100
    except Exception:
        proba = None

    st.write("")
    res_col1, res_col2 = st.columns([2, 1])

    with res_col1:
        if prediction[0] == 1:
            st.markdown('<div class="result-danger">⚠️ Heart Disease Detected<br><small>Please consult a cardiologist</small></div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="result-safe">✅ No Heart Disease Detected<br><small>Keep maintaining a healthy lifestyle</small></div>', unsafe_allow_html=True)

    with res_col2:
        if proba is not None:
            st.metric("Risk Probability", f"{proba:.1f}%")
            st.progress(int(proba))

    with st.expander("📋 View Entered Patient Data"):
        st.dataframe(input_data, use_container_width=True)

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:gray; font-size:0.85rem;'>"
    "⚕️ Disclaimer: This tool is for educational purposes only and is not a substitute for professional medical advice."
    "</p>", unsafe_allow_html=True
)
