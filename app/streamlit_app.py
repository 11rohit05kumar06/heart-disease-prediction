import os
import streamlit as st
import pandas as pd
import joblib

# --- 1. PAGE CONFIG ---
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- 2. CUSTOM CSS: purple/pink 3D-glow dashboard theme ---
st.markdown("""
<style>
    /* App background - deep purple/indigo glow gradient */
    .stApp {
        background:
            radial-gradient(circle at 15% 10%, rgba(236,72,153,0.18) 0%, transparent 45%),
            radial-gradient(circle at 85% 0%, rgba(139,92,246,0.20) 0%, transparent 50%),
            radial-gradient(circle at 50% 100%, rgba(59,130,246,0.12) 0%, transparent 55%),
            linear-gradient(160deg, #170a2e 0%, #0d0620 55%, #0a0518 100%);
        color: #f1f5f9;
    }

    h1, h2, h3, h4 { color: #ffffff !important; font-family: 'Segoe UI', sans-serif; }

    /* Gradient hero title */
    .hero-title {
        font-size: 3rem;
        font-weight: 800;
        line-height: 1.1;
        background: linear-gradient(90deg, #f472b6, #c084fc 60%, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 0;
    }

    /* Glowing heart icon w/ orbit rings */
    .heart-wrap {
        position: relative;
        width: 100%;
        height: 170px;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    .orbit {
        position: absolute;
        border: 1.5px solid rgba(236,72,153,0.35);
        border-radius: 50%;
    }
    .orbit-1 { width: 210px; height: 100px; animation: spin 14s linear infinite; }
    .orbit-2 { width: 150px; height: 190px; animation: spin 20s linear infinite reverse; }
    @keyframes spin { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }
    .heart-emoji {
        font-size: 5.2rem;
        filter: drop-shadow(0 0 25px rgba(236,72,153,0.65)) drop-shadow(0 0 45px rgba(139,92,246,0.4));
        z-index: 2;
        animation: pulse 2.4s ease-in-out infinite;
    }
    @keyframes pulse { 0%,100% { transform: scale(1); } 50% { transform: scale(1.08); } }

    /* Card styling */
    .card {
        background: linear-gradient(145deg, rgba(30,20,55,0.85), rgba(20,12,40,0.85));
        border: 1px solid rgba(168,85,247,0.25);
        border-radius: 16px;
        padding: 20px 20px 6px 20px;
        margin-bottom: 20px;
        box-shadow: 0 8px 24px rgba(0,0,0,0.45), inset 0 1px 0 rgba(255,255,255,0.03);
    }
    .card h3 { font-size: 1.05rem !important; margin-bottom: 8px; }

    label { color: #c4b5fd !important; font-size: 0.85rem; }

    .stSelectbox > div > div, .stNumberInput input {
        background-color: rgba(15,10,30,0.7) !important;
        color: white !important;
        border: 1px solid rgba(168,85,247,0.3) !important;
        border-radius: 8px !important;
    }

    .stSlider [data-baseweb="slider"] div div { background: linear-gradient(90deg,#f472b6,#818cf8); }

    /* Predict button */
    .stButton > button {
        background: linear-gradient(90deg, #ec4899, #a855f7);
        color: white;
        font-size: 1.1rem;
        font-weight: 700;
        border-radius: 30px;
        padding: 12px 40px;
        border: none;
        box-shadow: 0 0 20px rgba(236,72,153,0.55);
        width: 100%;
        transition: all 0.2s ease-in-out;
    }
    .stButton > button:hover {
        box-shadow: 0 0 32px rgba(236,72,153,0.85);
        transform: translateY(-2px);
    }

    /* Result boxes */
    .result-box-danger {
        background: rgba(220, 38, 38, 0.15);
        border: 2px solid #ef4444;
        border-radius: 14px;
        padding: 22px;
        text-align: center;
        box-shadow: 0 0 25px rgba(239,68,68,0.25);
    }
    .result-box-safe {
        background: rgba(34, 197, 94, 0.15);
        border: 2px solid #22c55e;
        border-radius: 14px;
        padding: 22px;
        text-align: center;
        box-shadow: 0 0 25px rgba(34,197,94,0.25);
    }

    /* Circular gauge */
    .gauge-wrap { display:flex; flex-direction:column; align-items:center; justify-content:center; }
    .gauge-label { color:#c4b5fd; font-size:0.95rem; margin-bottom:6px; }
</style>
""", unsafe_allow_html=True)

# --- 3. LOAD MODEL ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "heart_disease_model.pkl")

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}. Please check file path.")
    st.stop()

# --- 4. HERO HEADER ---
col_title, col_img = st.columns([2, 1])
with col_title:
    st.markdown(
        "<div class='hero-title'>❤️ Heart Disease<br>Prediction System</div>",
        unsafe_allow_html=True
    )
    st.markdown(
        "<p style='color:#a78bfa; margin-top:10px;'>AI-powered risk assessment from clinical parameters</p>",
        unsafe_allow_html=True
    )
with col_img:
    st.markdown("""
        <div class="heart-wrap">
            <div class="orbit orbit-1"></div>
            <div class="orbit orbit-2"></div>
            <div class="heart-emoji">🫀</div>
        </div>
    """, unsafe_allow_html=True)

st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

# --- 5. INPUTS (4 CARDS) ---
c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown("<div class='card'><h3>👤 Personal Information</h3>", unsafe_allow_html=True)
    age = st.slider("Age", 20, 100, 42)
    sex = st.selectbox("Sex", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
    st.markdown("</div>", unsafe_allow_html=True)

with c2:
    st.markdown("<div class='card'><h3>📉 Chest Pain & ECG</h3>", unsafe_allow_html=True)
    cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3],
                      format_func=lambda x: ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"][x])
    restecg = st.selectbox("Resting ECG Results", [0, 1, 2],
                           format_func=lambda x: ["Normal", "ST-T wave abnormality", "Left ventricular hypertrophy"][x])
    st.markdown("</div>", unsafe_allow_html=True)

with c3:
    st.markdown("<div class='card'><h3>🩸 Vitals & Blood Work</h3>", unsafe_allow_html=True)
    trestbps = st.slider("Resting Blood Pressure (mm Hg)", 80, 200, 120)
    chol = st.slider("Cholesterol (mg/dl)", 100, 600, 200)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    thalach = st.slider("Maximum Heart Rate Achieved", 60, 220, 150)
    st.markdown("</div>", unsafe_allow_html=True)

with c4:
    st.markdown("<div class='card'><h3>🏃 Exercise & Heart Tests</h3>", unsafe_allow_html=True)
    exang = st.selectbox("Exercise Induced Angina", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    oldpeak = st.number_input("ST Depression (Oldpeak)", 0.0, 10.0, 1.0, 0.1)
    slope = st.selectbox("Slope of Peak Exercise ST", [0, 1, 2],
                         format_func=lambda x: ["Upsloping", "Flat", "Downsloping"][x])
    ca = st.selectbox("Number of Major Vessels (0-4)", [0, 1, 2, 3, 4])
    thal = st.selectbox("Thalassemia", [0, 1, 2, 3],
                        format_func=lambda x: ["Unknown", "Normal", "Fixed Defect", "Reversible Defect"][x])
    st.markdown("</div>", unsafe_allow_html=True)

# --- 6. PREDICTION BUTTON ---
st.markdown("<br>", unsafe_allow_html=True)
b1, b2, b3 = st.columns([1, 2, 1])
with b2:
    predict_clicked = st.button("Predict Heart Disease")

# --- 7. RESULTS ---
if predict_clicked:
    input_data = pd.DataFrame([[
        age, sex, cp, trestbps, chol, fbs, restecg,
        thalach, exang, oldpeak, slope, ca, thal
    ]], columns=[
        'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
        'restecg', 'thalach', 'exang', 'oldpeak',
        'slope', 'ca', 'thal'
    ])

    prediction = model.predict(input_data)[0]

    risk_prob = 0.0
    if hasattr(model, "predict_proba"):
        risk_prob = model.predict_proba(input_data)[0][1] * 100

    st.markdown("<br>", unsafe_allow_html=True)

    res_col1, res_col2 = st.columns([2, 1])

    with res_col1:
        if prediction == 1:
            st.markdown("""
            <div class="result-box-danger">
                <h2 style="color: #ef4444; margin:0;">⚠️ Heart Disease Detected</h2>
                <p style="color: #fca5a5; margin-top:10px;">Please consult a cardiologist immediately.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="result-box-safe">
                <h2 style="color: #22c55e; margin:0;">✅ No Heart Disease Detected</h2>
                <p style="color: #86efac; margin-top:10px;">Patient appears healthy based on current metrics.</p>
            </div>
            """, unsafe_allow_html=True)

    with res_col2:
        gauge_color = "#ef4444" if risk_prob >= 50 else "#22d3ee"
        st.markdown(f"""
        <div class="gauge-wrap">
            <div class="gauge-label">Risk Probability</div>
            <div style="position:relative;width:150px;height:150px;">
                <div style="width:150px;height:150px;border-radius:50%;
                            background:conic-gradient({gauge_color} {risk_prob}%, rgba(255,255,255,0.08) {risk_prob}% 100%);
                            display:flex;align-items:center;justify-content:center;
                            box-shadow:0 0 25px {gauge_color}55;">
                    <div style="width:112px;height:112px;border-radius:50%;
                                background:#150f2e;display:flex;align-items:center;justify-content:center;">
                        <span style="font-size:1.6rem;font-weight:800;color:white;">{risk_prob:.1f}%</span>
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Data Table
    with st.expander("📝 View Entered Patient Data"):
        st.dataframe(input_data)
