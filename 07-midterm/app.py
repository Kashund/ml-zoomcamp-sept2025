import streamlit as st
import requests

st.title("30-Day Readmission Risk Predictor")

st.sidebar.header("Patient Features")

age = st.sidebar.number_input("Age", 0, 120, 72)
sex = st.sidebar.selectbox("Sex", ["Male", "Female"])
bmi = st.sidebar.number_input("BMI", 10.0, 60.0, 29.2)
systolic_bp = st.sidebar.number_input("Systolic BP", 80, 220, 140)
diastolic_bp = st.sidebar.number_input("Diastolic BP", 40, 130, 85)
glucose = st.sidebar.number_input("Glucose", 50, 400, 155)
cholesterol = st.sidebar.number_input("Cholesterol", 100, 400, 210)
creatinine = st.sidebar.number_input("Creatinine", 0.1, 10.0, 1.1)
diabetes = st.sidebar.selectbox("Diabetes", [0, 1])
hypertension = st.sidebar.selectbox("Hypertension", [0, 1])
diagnosis = st.sidebar.selectbox("Diagnosis", ["Normal", "Pneumonia", "Heart Failure", "Sepsis"])

if st.button("Predict readmission risk"):
    payload = {
        "age": age,
        "sex": sex,
        "bmi": bmi,
        "systolic_bp": systolic_bp,
        "diastolic_bp": diastolic_bp,
        "glucose": glucose,
        "cholesterol": cholesterol,
        "creatinine": creatinine,
        "diabetes": diabetes,
        "hypertension": hypertension,
        "diagnosis": diagnosis,
    }

    res = requests.post("http://localhost:8000/predict", json=payload)
    if res.status_code == 200:
        data = res.json()
        st.write(f"**Readmission probability:** {data['readmit_proba']:.3f}")
        st.write(f"**Predicted class:** {data['readmit_pred']}")
    else:
        st.error(f"Error from API: {res.status_code}")