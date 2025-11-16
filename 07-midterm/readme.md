# Readmission Risk Prediction – ML Zoomcamp Midterm Project
Author: Kashun Davis 
Course: ML Zoomcamp 2025  
Model: Class-Weighted Logistic Regression  
Deployment: FastAPI + Docker + Streamlit

---

## 🩺 Project Overview

This project predicts whether a patient will experience a **30-day hospital readmission** based on demographics, vitals, lab results, and diagnoses.  
The dataset is **synthetic clinical tabular data** that resembles EHR patient records while being safe to share.

This repository follows the complete ML Zoomcamp end‑to‑end ML workflow:
- EDA and leakage‑safe preprocessing  
- Baseline → tuned → class‑weighted models  
- Threshold tuning for imbalanced data  
- Feature importance + fairness slices  
- Deployment with FastAPI  
- Containerization with Docker  
- Optional Streamlit web app UI  

---

## 📂 Project Structure
```
readmission_midterm_project/
│
├── data/
│   └── synthetic_clinical_dataset.csv
│
├── artifacts/
│   ├── dv.joblib
│   └── model.joblib
|   └── metrics.json
│
├── notebooks/
│   └── readmission_midterm_notebook.ipynb
    └── readmission_midterm_notebook.md
│
├── train.py
├── predict.py
├── serve.py
├── app.py
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## 🧠 Dataset Location  
Place dataset here:  
```
data/synthetic_clinical_dataset.csv
```

---

## 🛠 Create Virtual Environment & Install Dependencies
```bash
python -m venv venv
source venv/bin/activate     # Windows: venv\Scriptsctivate
pip install --upgrade pip
pip install -r requirements.txt
```

---

# 🏋️‍♂️ 1. Train the Model (Generates Artifacts)
```bash
python train.py   --input_csv data/synthetic_clinical_dataset.csv   --out_dir artifacts
```

This creates:
```
artifacts/dv.joblib
artifacts/model.joblib
artifacts/metrics.json
```

---

# 📦 2. Batch Predictions (Predict on Full CSV)

```bash
python predict.py   --dv_path artifacts/dv.joblib   --model_path artifacts/model.joblib   --input_csv data/synthetic_clinical_dataset.csv   --out_csv predictions.csv
```

Output includes:
- `readmit_proba`
- `readmit_pred`  

---

# 🚀 3. Run the FastAPI Model Server (Web API)

```bash
uvicorn serve:app --reload --port 8000
```

### Swagger UI (interactive):
📌 **http://localhost:8000/docs**

### Example JSON body:
```json
{
  "age": 72,
  "sex": "Female",
  "bmi": 29.2,
  "systolic_bp": 140,
  "diastolic_bp": 85,
  "glucose": 155,
  "cholesterol": 210,
  "creatinine": 1.1,
  "diabetes": 1,
  "hypertension": 1,
  "diagnosis": "Heart Failure"
}
```

### curl request:
```bash
curl -X POST "http://localhost:8000/predict"   -H "Content-Type: application/json"   -d '{"age":72,"sex":"Female","bmi":29.2,"systolic_bp":140,"diastolic_bp":85,"glucose":155,"cholesterol":210,"creatinine":1.1,"diabetes":1,"hypertension":1,"diagnosis":"Heart Failure"}'
```

---

# 🐳 4. Deploy With Docker

Build the container:
```bash
docker build -t readmission-api .
```

Run it:
```bash
docker run -p 8000:8000 readmission-api
```

API available again at:
📌 http://localhost:8000/docs

---

# 💻 5. Optional End‑User Web App (Streamlit UI)

Run:
```bash
streamlit run app.py
```

Streamlit provides:
- Sidebars for entering patient information  
- A “Predict” button  
- Risk probability + class output from FastAPI  

---

# 🧪 6. End‑to‑End Testing Checklist

| Step | Expected Result |
|------|-----------------|
| Training | Creates artifacts/ directory |
| API launch | Uvicorn running at http://localhost:8000 |
| Swagger | Shows `/predict` |
| Docker run | API works identically on port 8000 |
| Streamlit UI | Can call API and display prediction |

---

# ⚠️ Limitations
- Dataset is synthetic  
- Model not clinically validated  
- Probability calibration not applied  
- Performance depends on imbalance handling  
- Requires fairness considerations if used in production  

---

# 🎉 Final Note
A full **end‑to‑end ML system**:
- Notebook → Model → API → Docker → Web UI 