import argparse
from pathlib import Path

import joblib
import pandas as pd


def main(args):
    dv = joblib.load(args.dv_path)
    model = joblib.load(args.model_path)

    df = pd.read_csv(args.input_csv)
    df.columns = [c.strip().lower() for c in df.columns]

    feat_cols = [
        "age",
        "sex",
        "bmi",
        "systolic_bp",
        "diastolic_bp",
        "glucose",
        "cholesterol",
        "creatinine",
        "diabetes",
        "hypertension",
        "diagnosis",
    ]

    X = df[feat_cols].copy()

    # Ensure categorical columns are strings as in training
    for c in ["sex", "diabetes", "hypertension", "diagnosis"]:
        X[c] = X[c].astype(str)

    # Transform with DictVectorizer
    Xd = dv.transform(X.to_dict(orient="records"))

    proba = model.predict_proba(Xd)[:, 1]
    pred = (proba >= 0.5).astype(int)

    out = df.copy()
    out["readmit_proba"] = proba
    out["readmit_pred"] = pred

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    print(f"Wrote predictions to: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dv_path", required=True, type=str, help="Path to dv.joblib")
    parser.add_argument("--model_path", required=True, type=str, help="Path to model.joblib")
    parser.add_argument("--input_csv", required=True, type=str, help="CSV file with patient rows")
    parser.add_argument("--out_csv", default="predictions.csv", type=str, help="Output CSV for predictions")
    args = parser.parse_args()
    main(args)