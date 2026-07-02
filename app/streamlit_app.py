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

# --- 2. CUSTOM CSS: Matching the 3D-glow dark dashboard theme ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@600;700;800&display=swap');

    /* App background - Deep space blue matching the image */
    .stApp {
        background-color: #030614;
        background-image: 
            radial-gradient(circle at 80% 20%, rgba(255, 60, 100, 0.15) 0%, transparent 40%),
            radial-gradient(circle at 20% 80%, rgba(50, 100, 250, 0.1) 0%, transparent 40%);
        color: #f1f5f9;
    }

    h1, h2, h3, h4 { color: #ffffff !important; font-family: 'Montserrat', sans-serif; }

    /* Single-line Gradient hero title */
    .hero-title {
        font-family: 'Montserrat', sans-serif;
        font-size: 2.8rem;
        font-weight: 800;
        white-space: nowrap; /* Forces single line */
        background: linear-gradient(90deg, #ff4b6b, #ffffff 40%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 0;
        padding-top: 20px;
    }

    /* Glowing heart icon wrap */
    .heart-wrap {
        position: relative;
        width: 100%;
        height: 150px;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    .orbit {
        position: absolute;
        border: 1.5px solid rgba(255, 75, 107, 0.35);
        border-radius: 50%;
    }
    .orbit-1 { width: 180px; height: 90px; animation: spin 12s linear infinite; }
    .orbit-2 { width: 130px; height: 160px; animation: spin 18s linear infinite reverse; }
    @keyframes spin { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }
    .heart-emoji {
        font-size: 5rem;
        filter: drop-shadow(0 0 25px rgba(255, 75, 107, 0.65)) drop-shadow(0 0 45px rgba(59, 130, 246, 0.4));
        z-index: 2;
        animation: pulse 2s ease-in-out infinite;
    }
    @keyframes pulse { 0%,100% { transform: scale(1); } 50% { transform: scale(1.05); } }

    /* Glassmorphism Cards */
    .card {
        background: rgba(18, 24, 45, 0.6);
        border: 1px solid rgba(80, 140, 255, 0.25);
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 20px;
        backdrop-filter: blur(12px);
        box-shadow: 0 8px 24px rgba(0,0,0,0.5);
    }
    .card h3 { 
        font-size: 1.1rem !important; 
        margin-bottom: 12px;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    label { color: #cbd5e1 !important; font-size: 0.85rem; }

    /* Dropdowns styling - matching the dark/blue rim style */
    .stSelectbox > div > div, .stNumberInput input {
        background-color: rgba(10, 15, 30, 0.8) !important;
        color: white !important;
        border: 1px solid rgba(80, 140, 255, 0.4) !important;
        border-radius: 6px !important;
    }
    /* Hide clear 'x' icon on selectboxes if it appears */
    span[data-baseweb="icon"] { display: none !important; }

    /* Fix Slider visibility (Red glowing line + white text) */
    .stSlider [data-baseweb="slider"] div[data-testid="stThumbValue"],
    .stSlider [data-baseweb="slider"] div[data-testid="stTickBarMin"],
    .stSlider [data-baseweb="slider"] div[data-testid="stTickBarMax"] {
        color: #ffffff !important;
        font-weight: 700;
        font-size: 0.9rem;
    }
    .stSlider [data-baseweb="slider"] div[data-testid="stSliderTrack"] > div { 
        background: linear-gradient(90deg, #ff4b6b, #ff7e5f) !important; 
        box-shadow: 0 0 10px rgba(255, 75, 107, 0.8);
    }

    /* Predict button */
    .stButton > button {
        background: linear-gradient(90deg, #f43f5e, #fb923c);
        color: white;
        font-size: 1.2rem;
        font-weight: 700;
        border-radius: 12px;
        padding: 12px 0;
        border: none;
        box-shadow: 0 4px 20px rgba(244, 63, 94, 0.5);
        width: 100%;
        transition: all 0.2s ease-in-out;
    }
    .stButton > button:hover {
        box-shadow: 0 6px 25px rgba(244, 63, 94, 0.8);
        transform: translateY(-2px);
    }

    /* Result boxes */
    .result-box-danger {
        background: rgba(30, 10, 15, 0.6);
        border: 2px solid #ef4444;
        border-radius: 12px;
        padding: 24px;
        text-align: center;
        box-shadow: 0 0 25px rgba(239,68,68,0.3);
        backdrop-filter: blur(10px);
    }
    .result-box-safe {
        background: rgba(10, 30, 15, 0.6);
        border: 2px solid #22c55e;
        border-radius: 12px;
        padding: 24px;
        text-align: center;
        box-shadow: 0 0 25px rgba(34,197,94,0.3);
        backdrop-filter: blur(10px);
    }

    /* Circular gauge */
    .gauge-wrap { display:flex; flex-direction:column; align-items:center; justify-content:center; height: 100%; }
    .gauge-label { color:#cbd5e1; font-size:1rem; margin-bottom:10px; font-weight:600;}
    
    /* Spacer to push table down */
    .table-spacer { margin-top: 60px; }
</style>
""", unsafe_allow_html=True)

# --- 3. LOAD MODEL ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "heart_disease_model.pkl")

@st.cache_resource
def load_model():
    # If testing without a model file, comment the joblib line and return a dummy model
    return joblib.load(MODEL_PATH) 

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}. Please check file path.")
    st.stop()

# --- 4. HERO HEADER ---
# Adjusted column ratio to give the single-line title plenty of room
col_title, col_img = st.columns([3.5, 1])
with col_title:
    st.markdown(
        "<div class='hero-title'>❤️ Heart Disease Prediction System</div>",
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

st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

# --- 5. INPUTS (4 CARDS) ---
c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown("<div class='card'><h3>🧍 Personal Information</h3>", unsafe_allow_html=True)
    age = st.slider("Age", 20, 120, 42)
    sex = st.selectbox("Sex", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
    st.markdown("</div>", unsafe_allow_html=True)

with c2:
    st.markdown("<div class='card'><h3>🩺 Chest Pain & ECG</h3>", unsafe_allow_html=True)
    cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3],
                      format_func=lambda x: ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"][x])
    restecg = st.selectbox("Resting ECG Results", [0, 1, 2],
                           format_func=lambda x: ["Normal", "ST-T wave abnormality", "Left ventricular hypertrophy"][x])
    st.markdown("</div>", unsafe_allow_html=True)

with c3:
    st.markdown("<div class='card'><h3>🩸 Vitals & Blood Work</h3>", unsafe_allow_html=True)
    trestbps = st.slider("Resting Blood Pressure (mm Hg)", 80, 200, 120)
    chol = st.slider("Cholesterol (mg/dl)", 100, 600, 200)
    fbs = st.slider("Fasting Blood Sugar > 120 mg/dl", 0, 300, 150) # Updated to slider to match image visual
    thalach = st.slider("Maximum Heart Rate Achieved", 60, 220, 150)
    st.markdown("</div>", unsafe_allow_html=True)

with c4:
    st.markdown("<div class='card'><h3>🏃 Exercise & Heart Tests</h3>", unsafe_allow_html=True)
    exang = st.selectbox("Exercise Induced Angina", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    oldpeak = st.selectbox("ST Depression (Oldpeak)", [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0], index=1)
    slope = st.selectbox("Slope of Peak Exercise ST", [0, 1, 2],
                         format_func=lambda x: ["Upsloping", "Flat", "Downsloping"][x])
    ca = st.selectbox("Number of Major Vessels (0-4)", [0, 1, 2, 3, 4])
    thal = st.selectbox("Thalassemia", [0, 1, 2, 3],
                        format_func=lambda x: ["Unknown", "Normal", "Fixed Defect", "Reversible Defect"][x])
    st.markdown("</div>", unsafe_allow_html=True)

# --- 6. PREDICTION BUTTON ---
st.markdown("<br>", unsafe_allow_html=True)
b1, b2, b3 = st.columns([1, 1.5, 1])
with b2:
    predict_clicked = st.button("Predict Heart Disease")

# --- 7. RESULTS ---
if predict_clicked:
    # Prepare Fbs category based on slider value
    fbs_category = 1 if fbs > 120 else 0

    input_data = pd.DataFrame([[
        age, sex, cp, trestbps, chol, fbs_category, restecg,
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

    res_col1, res_col2 = st.columns([2.5, 1])

    with res_col1:
        if prediction == 1:
            st.markdown("""
            <div class="result-box-danger">
                <h2 style="color: #ef4444; margin:0; font-size: 1.8rem;">⚠️ Heart Disease Detected</h2>
                <p style="color: #fca5a5; margin-top:10px; font-size: 1.1rem;">Please consult a cardiologist</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="result-box-safe">
                <h2 style="color: #22c55e; margin:0; font-size: 1.8rem;">✅ No Heart Disease Detected</h2>
                <p style="color: #86efac; margin-top:10px; font-size: 1.1rem;">Patient appears healthy based on current metrics.</p>
            </div>
            """, unsafe_allow_html=True)

    with res_col2:
        # Match gauge aesthetic from image (blue and red ring)
        gauge_primary = "#ef4444" 
        gauge_secondary = "#3b82f6" 
        
        st.markdown(f"""
        <div class="gauge-wrap">
            <div class="gauge-label">Risk Probability</div>
            <div style="position:relative;width:140px;height:140px;">
                <div style="width:140px;height:140px;border-radius:50%;
                            background:conic-gradient({gauge_primary} {risk_prob}%, {gauge_secondary} {risk_prob}% 100%);
                            display:flex;align-items:center;justify-content:center;
                            box-shadow:0 0 20px rgba(0,0,0,0.5);">
                    <div style="width:108px;height:108px;border-radius:50%;
                                background:#080b1a;display:flex;align-items:center;justify-content:center;">
                        <span style="font-size:1.8rem;font-weight:800;color:white;">{risk_prob:.1f}%</span>
                    </div>
                </div>
            </div>
            <div style="width: 80%; height: 6px; background: linear-gradient(90deg, #ef4444 {risk_prob}%, #3b82f6 {risk_prob}%); margin-top: 15px; border-radius: 3px;"></div>
        </div>
        """, unsafe_allow_html=True)

    # Adding a clear CSS separation to prevent UI overlap
    st.markdown("<div class='table-spacer'></div>", unsafe_allow_html=True)
    
    # Data Table
    with st.expander("﹀ View Entered Patient Data", expanded=True):
        st.dataframe(input_data, use_container_width=True)
