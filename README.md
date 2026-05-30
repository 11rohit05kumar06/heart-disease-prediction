❤️ **Heart Disease Prediction System**
A complete Machine Learning project that predicts whether a patient is likely to have heart disease based on medical attributes. This project covers the full ML workflow including data preprocessing, exploratory data analysis (EDA), feature engineering, model building, model evaluation, model saving, and deployment using Streamlit.

📌 **Project Description**
Heart disease is one of the leading causes of death worldwide. Early prediction can help doctors and patients take preventive action at the right time.

The objective of this project is to build a machine learning classification model that predicts the presence of heart disease using patient health data such as age, blood pressure, cholesterol, chest pain type, and other medical indicators.

This project was developed as part of a Machine Learning Internship Assessment and demonstrates an end-to-end ML pipeline from raw dataset to deployment-ready prediction application.

🎯 **Problem Statement**
The goal of this project is to predict whether a patient has heart disease or not based on clinical and health-related features.
Domain: Healthcare
Project Type: Classification
Target Variable: target
Expected Outcome: Predict whether heart disease is present (1) or absent (0)

🗂️ **Dataset Information**
Dataset Name: Heart Disease Dataset
Dataset File: heart.csv
Number of Columns: 14
Target Column: target
Features Used
age - Age of the patient
sex - Gender of the patient (0 = Female, 1 = Male)
cp - Chest pain type
trestbps - Resting blood pressure
chol - Serum cholesterol level
fbs - Fasting blood sugar > 120 mg/dl
restecg - Resting electrocardiographic results
thalach - Maximum heart rate achieved
exang - Exercise induced angina
oldpeak - ST depression induced by exercise
slope - Slope of peak exercise ST segment
ca - Number of major vessels colored by fluoroscopy
thal - Thalassemia type
target - Output (0 = No Heart Disease, 1 = Heart Disease)

⚙️ **Technologies Used**
Python
Pandas
NumPy
Matplotlib
Seaborn
Scikit-learn
XGBoost
Joblib
Streamlit
Git
GitHub

🔄 **Project Workflow**
This project follows the complete machine learning lifecycle:
Problem Statement Definition
Data Collection
Exploratory Data Analysis (EDA)
Data Preprocessing
Train-Test Split
Model Building
Accuracy Optimization
Model Evaluation
Model Saving
Prediction Application using Streamlit

📊 **Exploratory Data Analysis (EDA)**
The following EDA steps were performed:
Shape of dataset
Column and data type analysis
Statistical summary
Missing value analysis
Duplicate record analysis
Count plots
Histograms
Box plots
Correlation heatmap
Confusion matrix visualization
Model accuracy comparison chart
Key Insights
The dataset contains important medical features useful for heart disease prediction.
Correlation analysis helped identify relationships between input features and the target variable.
Box plots and distribution plots helped in understanding data spread and possible outliers.

🧹 **Data Preprocessing**
The following preprocessing steps were applied:
Checked for missing values
Checked duplicate rows
Cleaned the data
Selected input and target columns
Applied feature scaling for Logistic Regression
Performed train-test split using 80:20 ratio
Prepared data for multiple classification models

✨ **Feature Engineering**
The project includes the following feature preparation steps:
Feature selection based on relevant medical attributes
Scaling of features for Logistic Regression
Direct handling of numeric features
Use of structured input data for Streamlit prediction interface

🤖 **Models Used**
The following machine learning models were trained and evaluated:
Logistic Regression
Random Forest Classifier
XGBoost Classifier

📈 **Model Evaluation**
The models were compared using the following classification metrics:
Accuracy
Precision
Recall
F1-score
Confusion Matrix
**Model Accuracy**
Logistic Regression Accuracy: 80%
Random Forest Accuracy: 99%
XGBoost Accuracy: 99%
Final Selected Model
Random Forest Classifier
or replace with your actual final selected model
The final model was selected based on better performance in accuracy and classification results.

💾 **Model Saving**
The final trained model was saved using Joblib.
Saved Model File:
models/heart_disease_model.pkl

🖥️ **Streamlit Prediction Application**
A prediction interface was created using Streamlit.
Application Features
User-friendly interface
Input fields for all required medical features
Prediction button
Prediction result display
Real-time heart disease prediction
Prediction Output
Heart Disease Detected
No Heart Disease Detected

📁 **Project Structure**
Bash
heart-disease-prediction/
│
├── app/
│   └── streamlit_app.py
│
├── data/
│   └── heart.csv
│
├── models/
│   └── heart_disease_model.pkl
│
├── notebook/
│   └── heart_disease_analysis.ipynb
│
├── screenshots/
│   ├── home_page.png
│   ├── prediction_result.png
│   ├── confusion_matrix.png
│   └── accuracy_comparison.png
│
├── requirements.txt
├── README.md
└── .gitignore

🚀 **Installation Steps**
**Clone the repository:**
Bash
git clone https://github.com/11rohit05kumar06/heart-disease-prediction.git
**Move into the project folder:**
Bash
cd heart-disease-prediction
**Install all required dependencies:**
Bash
pip install -r requirements.txt
**Run the Streamlit app:**
Bash
streamlit run app/streamlit_app.py

▶️ **Usage Instructions**
Open the Streamlit app in your browser
Enter patient details in the given input fields
Click on Predict Heart Disease
View the prediction result

🧪 **Sample Test Inputs**
**Sample Patient 1 - Likely Heart Disease**
Age: 58
Sex: Female
Chest Pain Type: Typical Angina
Resting Blood Pressure: 100
Cholesterol: 248
Fasting Blood Sugar: No
Resting ECG: Normal
Maximum Heart Rate Achieved: 122
Exercise Induced Angina: No
Oldpeak: 1.0
Slope: Flat
Number of Major Vessels: 0
Thal: Fixed Defect
**Sample Patient 2 - Likely No Heart Disease**
Age: 52
Sex: Male
Chest Pain Type: Typical Angina
Resting Blood Pressure: 125
Cholesterol: 212
Fasting Blood Sugar: No
Resting ECG: ST-T wave abnormality
Maximum Heart Rate Achieved: 168
Exercise Induced Angina: No
Oldpeak: 1.0
Slope: Downsloping
Number of Major Vessels: 2
Thal: Reversible Defect

⚠️ **Challenges Faced**
Understanding medical feature meanings
Selecting the best classification model
Improving model performance
Organizing files properly for GitHub
Building a working prediction app using Streamlit

✅ **Conclusion**
This project successfully demonstrates a complete end-to-end machine learning workflow for predicting heart disease. Multiple models were trained and compared, and the best-performing model was selected and deployed using Streamlit for real-time prediction.
