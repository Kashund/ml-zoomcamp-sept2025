from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Readmission Risk API")

# Resolve artifacts directory relative to this file
BASE_DIR = Path(__file__).parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"

dv_path = ARTIFACTS_DIR / "dv.joblib"
model_path = ARTIFACTS_DIR / "model.joblib"

dv = joblib.load(dv_path)
model = joblib.load(model_path)


class Patient(BaseModel):
    age: float
    sex: str
    bmi: float
    systolic_bp: float
    diastolic_bp: float
    glucose: float
    cholesterol: float
    creatinine: float
    diabetes: int  # 0/1 from client, will be cast to str before transform
    hypertension: int  # 0/1
    diagnosis: str


@app.post("/predict")
def predict(p: Patient):
    data = p.dict()

    # Convert types to match training pipeline
    data["diabetes"] = str(data["diabetes"])
    data["hypertension"] = str(data["hypertension"])
    data["sex"] = str(data["sex"])
    data["diagnosis"] = str(data["diagnosis"])

    row_df = pd.DataFrame([data])

    Xd = dv.transform(row_df.to_dict(orient="records"))
    proba = model.predict_proba(Xd)[0, 1]

    # NOTE: Using 0.5 threshold here.
    # You could replace with your tuned threshold from the notebook if desired.
    pred = int(proba >= 0.5)

    return {
        "readmit_proba": float(proba),
        "readmit_pred": pred,
    }