import pickle, json

with open("pipeline_v1.bin", "rb") as f_in:
    pipeline = pickle.load(f_in)

record = {
    "lead_source": "paid_ads",
    "number_of_courses_viewed": 2,
    "annual_income": 79276.0
}

proba = float(pipeline.predict_proba([record])[0, 1])
print(json.dumps({"probability": proba}, indent=2))