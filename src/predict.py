"""
Phase 5 — Inference pipeline.

Loads saved artifacts and predicts mortality for a single patient.
Only transform() is ever called here — never fit().
"""
import sys
import json
import pickle
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from src.config import ARTIFACTS_DIR, MODEL_PATH, PREPROCESSOR_PATH, THRESHOLD_PATH


def load_artifacts():
    """Load preprocessor, model, and threshold once at startup."""
    with open(PREPROCESSOR_PATH, "rb") as f:
        preprocessor = pickle.load(f)

    model = XGBClassifier()
    model.load_model(str(MODEL_PATH))

    with open(THRESHOLD_PATH) as f:
        thresholds = json.load(f)
    threshold = thresholds["youden_threshold"]

    return preprocessor, model, threshold


# Module-level cache — loaded once, reused on every call
_preprocessor, _model, _threshold = None, None, None


def _get_artifacts():
    global _preprocessor, _model, _threshold
    if _model is None:
        _preprocessor, _model, _threshold = load_artifacts()
    return _preprocessor, _model, _threshold


def predict(patient_dict: dict) -> dict:
    """
    Predict ICU mortality for one patient.

    Parameters
    ----------
    patient_dict : raw feature values matching the training data columns
                   (before any preprocessing — same fields as mimic_train_X)

    Returns
    -------
    dict with:
        mortality_probability  float 0-1
        binary_prediction      0 or 1
        risk_level             "Low" / "Medium" / "High"
        threshold_used         float
    """
    preprocessor, model, threshold = _get_artifacts()

    X = pd.DataFrame([patient_dict])

    # Apply saved preprocessing (transform only — medians and encoder
    # were fitted on training data and are baked into the pipeline)
    X_transformed = preprocessor.transform(X)

    prob = float(model.predict_proba(X_transformed)[0, 1])
    prediction = int(prob >= threshold)

    if prob >= 0.6:
        risk = "High"
    elif prob >= 0.3:
        risk = "Medium"
    else:
        risk = "Low"

    return {
        "mortality_probability": round(prob, 4),
        "binary_prediction":     prediction,
        "risk_level":            risk,
        "threshold_used":        threshold,
    }


# ── Quick smoke test ───────────────────────────────────────────────────────
if __name__ == "__main__":
    import pandas as pd
    from src.config import TRAIN_X_PATH, DIAGNOSES_PATH
    from src.icd9_features import process_icd9_data

    print("Loading a real patient from training data for smoke test...")
    train_x   = pd.read_csv(TRAIN_X_PATH, index_col=0)
    diagnoses = pd.read_csv(DIAGNOSES_PATH)

    if "ICD9_diagnosis" in train_x.columns:
        train_x = train_x.drop(columns=["ICD9_diagnosis"])

    train_x = process_icd9_data(diagnoses, train_x)

    # Take first patient, drop icustay_id (not a model feature)
    sample = train_x.iloc[0].drop("icustay_id").to_dict()

    print(f"  icustay_id : {train_x.iloc[0]['icustay_id']}")
    result = predict(sample)

    print("\nPrediction result:")
    for k, v in result.items():
        print(f"  {k:<25} {v}")
