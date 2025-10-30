import os, pickle
from typing import Dict, Any
from fastapi import FastAPI

app = FastAPI(title="lead-conversion-predictor")

# Load the saved pipeline (v1 for local FastAPI)
MODEL_PATH = os.getenv("MODEL_PATH", "pipeline_v1.bin")
with open(MODEL_PATH, "rb") as f_in:
    pipeline = pickle.load(f_in)

def predict_single(x: Dict[str, Any]) -> float:
    return float(pipeline.predict_proba([x])[0, 1])

@app.get("/ping")
def ping():
    return {"status": "ok"}

@app.post("/predict")
def predict(payload: Dict[str, Any]):
    p = predict_single(payload)
    return {"probability": p, "convert": p >= 0.5}