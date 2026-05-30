import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load(r'../models/heart_disease_model.pkl')

# App title
st.title("❤️ Heart Disease Prediction System")

st.write("Enter patient details below:")

# Input fields
age = st.slider("Age", 20, 100, 50)

sex = st.selectbox("Sex", {
    0: "Female",
    1: "Male"
}.keys(), format_func=lambda x: {
    0: "Female",
    1: "Male"
}[x])

cp = st.selectbox("Chest Pain Type (cp)", {
    0: "Typical Angina",
    1: "Atypical Angina",
    2: "Non-anginal Pain",
    3: "Asymptomatic"
}.keys(), format_func=lambda x: {
    0: "Typical Angina",
    1: "Atypical Angina",
    2: "Non-anginal Pain",
    3: "Asymptomatic"
}[x])

trestbps = st.slider("Resting Blood Pressure (trestbps)", 80, 200, 120)
chol = st.slider("Cholesterol (chol)", 100, 600, 200)

fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (fbs)", {
    0: "No",
    1: "Yes"
}.keys(), format_func=lambda x: {
    0: "No",
    1: "Yes"
}[x])

restecg = st.selectbox("Resting ECG Results (restecg)", {
    0: "Normal",
    1: "ST-T wave abnormality",
    2: "Left ventricular hypertrophy"
}.keys(), format_func=lambda x: {
    0: "Normal",
    1: "ST-T wave abnormality",
    2: "Left ventricular hypertrophy"
}[x])

thalach = st.slider("Maximum Heart Rate Achieved (thalach)", 60, 220, 150)

exang = st.selectbox("Exercise Induced Angina (exang)", {
    0: "No",
    1: "Yes"
}.keys(), format_func=lambda x: {
    0: "No",
    1: "Yes"
}[x])

oldpeak = st.slider("Oldpeak", 0.0, 10.0, 1.0, 0.1)

slope = st.selectbox("Slope", {
    0: "Upsloping",
    1: "Flat",
    2: "Downsloping"
}.keys(), format_func=lambda x: {
    0: "Upsloping",
    1: "Flat",
    2: "Downsloping"
}[x])

ca = st.selectbox("Number of Major Vessels (ca)", [0, 1, 2, 3, 4])

thal = st.selectbox("Thal", {
    0: "Unknown",
    1: "Normal",
    2: "Fixed Defect",
    3: "Reversible Defect"
}.keys(), format_func=lambda x: {
    0: "Unknown",
    1: "Normal",
    2: "Fixed Defect",
    3: "Reversible Defect"
}[x])

# Create dataframe
input_data = pd.DataFrame([[
    age, sex, cp, trestbps, chol, fbs, restecg,
    thalach, exang, oldpeak, slope, ca, thal
]], columns=[
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
    'restecg', 'thalach', 'exang', 'oldpeak',
    'slope', 'ca', 'thal'
])

# Prediction button
if st.button("Predict Heart Disease"):
    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.error("⚠️ Prediction: Heart Disease Detected")
    else:
        st.success("✅ Prediction: No Heart Disease Detected")

    st.write("Input Data:")
    st.dataframe(input_data)