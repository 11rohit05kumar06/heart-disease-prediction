import os
import streamlit as st
import pandas as pd
import joblib
import base64

# --- 1. PAGE CONFIG & CUSTOM CSS ---
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="collapsed" # Hide sidebar for cleaner look
)

# Inject Custom CSS for the Dark Dashboard Look
st.markdown("""
<style>
    /* Main Background */
    .stApp {
        background-color: #0f172a; /* Deep Blue/Slate */
        color: #f8fafc;
    }
    
    /* Headers */
    h1, h2, h3, h4 {
        color: #ffffff !important;
        font-family: 'Segoe UI', sans-serif;
    }
    
    /* Card Styling (Simulated with containers) */
    .card {
        background-color: #1e293b;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.5);
    }
    
    /* Input Labels */
    label {
        color: #94a3b8 !important;
        font-size: 0.9rem;
    }
    
    /* Selectbox and Slider styling overrides */
    .stSelectbox > div > div {
        background-color: #0f172a;
        color: white;
        border: 1px solid #334155;
    }
    
    /* The Big Predict Button */
    .stButton > button {
        background: linear-gradient(90deg, #ff4b4b, #ff9090);
        color: white;
        font-size: 1.2rem;
        font-weight: bold;
        border-radius: 30px;
        padding: 10px 40px;
        border: none;
        box-shadow: 0 0 15px rgba(255, 75, 75, 0.5);
        width: 100%;
    }
    .stButton > button:hover {
        box-shadow: 0 0 25px rgba(255, 75, 75, 0.8);
    }

    /* Result Box Styling */
    .result-box-danger {
        background-color: rgba(220, 38, 38, 0.2);
        border: 2px solid #ef4444;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
    }
    .result-box-safe {
        background-color: rgba(34, 197, 94, 0.2);
        border: 2px solid #22c55e;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# --- 2. LOAD MODEL ---
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

# --- 3. HEADER ---
col_title, col_img = st.columns([2, 1])
with col_title:
    st.markdown("<h1 style='font-size: 3rem; margin-bottom: 0;'>❤️ Heart Disease<br>Prediction System</h1>", unsafe_allow_html=True)
with col_img:
    # Using a placeholder heart image URL or emoji. 
    # You can replace this URL with your own hosted image if you want the exact 3D heart.
    st.markdown("<div style='text-align: right; font-size: 5rem;'>🫀</div>", unsafe_allow_html=True)

st.markdown("---")

# --- 4. INPUTS (4 COLUMNS) ---
# We use st.container to create the "Card" look
c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("👤 Personal Information")
    age = st.slider("Age", 20, 100, 42)
    sex = st.selectbox("Sex", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
    st.markdown("</div>", unsafe_allow_html=True)

with c2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("📉 Chest Pain & ECG")
    cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3], 
                      format_func=lambda x: ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"][x])
    restecg = st.selectbox("Resting ECG Results", [0, 1, 2], 
                           format_func=lambda x: ["Normal", "ST-T wave abnormality", "Left ventricular hypertrophy"][x])
    st.markdown("</div>", unsafe_allow_html=True)

with c3:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("🩸 Vitals & Blood Work")
    trestbps = st.slider("Resting Blood Pressure (mm Hg)", 80, 200, 120)
    chol = st.slider("Cholesterol (mg/dl)", 100, 600, 200)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    thalach = st.slider("Maximum Heart Rate Achieved", 60, 220, 150)
    st.markdown("</div>", unsafe_allow_html=True)

with c4:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("🏃 Exercise & Heart Tests")
    exang = st.selectbox("Exercise Induced Angina", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    oldpeak = st.number_input("ST Depression (Oldpeak)", 0.0, 10.0, 1.0, 0.1)
    slope = st.selectbox("Slope of Peak Exercise ST", [0, 1, 2], 
                         format_func=lambda x: ["Upsloping", "Flat", "Downsloping"][x])
    ca = st.selectbox("Number of Major Vessels (0-4)", [0, 1, 2, 3, 4])
    thal = st.selectbox("Thalassemia", [0, 1, 2, 3], 
                        format_func=lambda x: ["Unknown", "Normal", "Fixed Defect", "Reversible Defect"][x])
    st.markdown("</div>", unsafe_allow_html=True)

# --- 5. PREDICTION BUTTON ---
st.markdown("<br>", unsafe_allow_html=True)
b1, b2, b3 = st.columns([1, 2, 1])
with b2:
    predict_clicked = st.button("Predict Heart Disease")

# --- 6. RESULTS ---
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
    
    # Try to get probability if model supports it (RandomForest/XGBoost usually do)
    risk_prob = 0.0
    if hasattr(model, "predict_proba"):
        risk_prob = model.predict_proba(input_data)[0][1] * 100
    
    st.markdown("---")
    
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
        st.markdown("<div style='text-align: center; margin-top: 20px;'>", unsafe_allow_html=True)
        st.metric("Risk Probability", f"{risk_prob:.1f}%")
        st.progress(risk_prob / 100)
        st.markdown("</div>", unsafe_allow_html=True)

    # Data Table
    with st.expander("📝 View Entered Patient Data"):
        st.dataframe(input_data)
