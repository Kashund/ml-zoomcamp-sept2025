import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
import joblib
import warnings

warnings.filterwarnings("ignore")

RANDOM_STATE = 42


def load_and_prepare(input_csv: str):
    """
    Load the synthetic clinical dataset and apply the same preprocessing
    as in the midterm notebook.
    """
    df = pd.read_csv(input_csv)
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
    target_col = "readmission_30d"

    # Simple numeric imputation (median), as in the notebook
    num_cols = ["age", "bmi", "systolic_bp", "diastolic_bp", "glucose", "cholesterol", "creatinine"]
    for c in num_cols:
        if c in df.columns:
            df[c] = df[c].fillna(df[c].median())

    # Categorical to string
    cat_cols = ["sex", "diabetes", "hypertension", "diagnosis"]
    for c in cat_cols:
        if c in df.columns:
            df[c] = df[c].astype(str)

    X = df[feat_cols].copy()
    y = df[target_col].astype(int)

    return df, X, y, feat_cols, target_col


def main(args):
    df, X, y, feat_cols, target_col = load_and_prepare(args.input_csv)

    print("Class balance (counts):")
    print(y.value_counts())
    print("\nClass balance (proportions):")
    print(y.value_counts(normalize=True))

    # Train-test split with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    # DictVectorizer
    dv = DictVectorizer(sparse=True)
    X_train_d = dv.fit_transform(X_train.to_dict(orient="records"))
    X_test_d = dv.transform(X_test.to_dict(orient="records"))

    # Final model: class-weighted Logistic Regression (winner from notebook)
    model = LogisticRegression(
        max_iter=1000,
        solver="liblinear",
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )

    model.fit(X_train_d, y_train)

    # Evaluation on test set
    y_prob = model.predict_proba(X_test_d)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    roc = roc_auc_score(y_test, y_prob)
    f1 = f1_score(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print("\n=== Test metrics (class-weighted Logistic Regression) ===")
    print(f"ROC AUC : {roc:.3f}")
    print(f"F1      : {f1:.3f}")
    print(f"Accuracy: {acc:.3f}")
    print("Confusion matrix:")
    print(cm)
    print("\nClassification report:")
    print(classification_report(y_test, y_pred))

    # Save artifacts
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dv_path = out_dir / "dv.joblib"
    model_path = out_dir / "model.joblib"
    joblib.dump(dv, dv_path)
    joblib.dump(model, model_path)

    print(f"\nSaved DictVectorizer to {dv_path}")
    print(f"Saved model to {model_path}")

    meta = {
        "model_type": "logreg_balanced",
        "roc_auc": roc,
        "f1": f1,
        "accuracy": acc,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "feature_count": int(len(dv.get_feature_names_out())),
        "features": feat_cols,
        "target": target_col,
    }

    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(json.dumps(meta, indent=2))
    print(f"\nSaved metrics to {metrics_path}")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", required=True, type=str, help="Path to synthetic_clinical_dataset.csv")
    parser.add_argument("--out_dir", default="artifacts", type=str, help="Directory to save dv/model/metrics")
    args = parser.parse_args()
    main(args)