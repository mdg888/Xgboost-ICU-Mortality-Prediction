"""
Phase 4 — Model training pipeline.

Replicates R workflow:
  - 5-fold stratified CV  (set.seed(1110))
  - XGBoost with same hyperparameters
  - Threshold optimisation via Youden's J
  - Final model trained on full training set
  - Artifacts saved to artifacts/
"""
import sys
import json
import pickle
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

from src.config import (
    TRAIN_X_PATH, TRAIN_Y_PATH, DIAGNOSES_PATH,
    ARTIFACTS_DIR, MODEL_PATH, PREPROCESSOR_PATH, THRESHOLD_PATH,
    XGB_PARAMS, RANDOM_SEED, CV_FOLDS,
)
from src.icd9_features import process_icd9_data
from src.preprocessing import build_pipeline
from src.evaluate import compute_metrics, optimize_threshold


def load_and_enrich() -> tuple[pd.DataFrame, pd.Series]:
    print("Loading data...")
    train_x   = pd.read_csv(TRAIN_X_PATH, index_col=0)
    train_y   = pd.read_csv(TRAIN_Y_PATH, index_col=0)
    diagnoses = pd.read_csv(DIAGNOSES_PATH)

    train_y = train_y.rename(columns={"HOSPITAL_EXPIRE_FLAG": "Death"})

    # Remove raw ICD9 text column (matches R)
    if "ICD9_diagnosis" in train_x.columns:
        train_x = train_x.drop(columns=["ICD9_diagnosis"])

    print("Processing ICD-9 features...")
    train_x = process_icd9_data(diagnoses, train_x)

    # Merge target
    train = train_x.merge(train_y[["icustay_id", "Death"]], on="icustay_id", how="left")

    y = train["Death"].astype(int)
    X = train.drop(columns=["Death", "icustay_id"])

    print(f"  Dataset: {X.shape[0]} rows, {X.shape[1]} raw features")
    print(f"  Class balance — 0: {(y==0).sum()}  1: {(y==1).sum()}")
    return X, y


def run_cv(X: pd.DataFrame, y: pd.Series) -> tuple:
    """5-fold stratified CV — returns OOF probabilities and per-fold metrics."""
    neg = (y == 0).sum()
    pos = (y == 1).sum()
    scale_pos_weight = neg / pos

    params = {**XGB_PARAMS, "scale_pos_weight": scale_pos_weight}
    # early_stopping_rounds requires an eval set during fit; use a 10% val split
    early_stop_rounds = params.pop("early_stopping_rounds")

    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)

    oof_probs  = np.zeros(len(y))
    fold_metrics = []

    print(f"\nRunning {CV_FOLDS}-fold stratified CV (seed={RANDOM_SEED})...")
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # Fit preprocessing pipeline on fold train data only (no leakage)
        pipe = build_pipeline()
        X_tr_t  = pipe.fit_transform(X_tr)
        X_val_t = pipe.transform(X_val)

        model = XGBClassifier(**params, early_stopping_rounds=early_stop_rounds)
        model.fit(
            X_tr_t, y_tr,
            eval_set=[(X_val_t, y_val)],
            verbose=False,
        )

        probs = model.predict_proba(X_val_t)[:, 1]
        oof_probs[val_idx] = probs

        m = compute_metrics(y_val.values, probs, threshold=0.5)
        fold_metrics.append(m)
        print(f"  Fold {fold}: AUC={m['roc_auc']:.4f}  "
              f"Sens={m['sensitivity']:.3f}  Spec={m['specificity']:.3f}  "
              f"F1={m['f1']:.3f}")

    return oof_probs, fold_metrics


def summarise_cv(fold_metrics: list):
    print("\n-- CV Summary (mean ± std) --")
    keys = list(fold_metrics[0].keys())
    for k in keys:
        vals = [m[k] for m in fold_metrics]
        print(f"  {k:<15} {np.mean(vals):.4f} ± {np.std(vals):.4f}")


def train_final(X: pd.DataFrame, y: pd.Series):
    """Train final model on full dataset, save artifacts."""
    neg = (y == 0).sum()
    pos = (y == 1).sum()
    scale_pos_weight = neg / pos

    params = {**XGB_PARAMS, "scale_pos_weight": scale_pos_weight}
    # No early stopping for final model (no held-out eval set)
    params.pop("early_stopping_rounds")

    print("\nFitting final preprocessing pipeline on full training data...")
    pipe = build_pipeline()
    X_t = pipe.fit_transform(X)

    print("Training final XGBoost model...")
    model = XGBClassifier(**params)
    model.fit(X_t, y, verbose=False)

    ARTIFACTS_DIR.mkdir(exist_ok=True)

    with open(PREPROCESSOR_PATH, "wb") as f:
        pickle.dump(pipe, f)
    model.save_model(str(MODEL_PATH))
    print(f"  Saved preprocessor -> {PREPROCESSOR_PATH}")
    print(f"  Saved model        -> {MODEL_PATH}")

    return pipe, model


def main():
    X, y = load_and_enrich()

    # --- Cross-validation ---
    oof_probs, fold_metrics = run_cv(X, y)
    summarise_cv(fold_metrics)

    # --- Threshold optimisation on OOF predictions ---
    print("\n-- Threshold Optimisation (OOF predictions) --")
    opt = optimize_threshold(y.values, oof_probs)

    for name in ("youden", "f1", "balanced"):
        r = opt[name]
        print(f"  {name:<10}  threshold={r['threshold']:.2f}  "
              f"sens={r['sensitivity']:.3f}  spec={r['specificity']:.3f}  "
              f"youden={r['youden']:.3f}")

    optimal_threshold = opt["youden"]["threshold"]
    print(f"\n  -> Using Youden threshold: {optimal_threshold}")

    # OOF metrics at optimal threshold
    oof_metrics = compute_metrics(y.values, oof_probs, threshold=optimal_threshold)
    print("\n-- OOF Metrics at Youden Threshold --")
    for k, v in oof_metrics.items():
        print(f"  {k:<15} {v:.4f}")

    # Save threshold
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    with open(THRESHOLD_PATH, "w") as f:
        json.dump({
            "youden_threshold":   opt["youden"]["threshold"],
            "f1_threshold":       opt["f1"]["threshold"],
            "balanced_threshold": opt["balanced"]["threshold"],
        }, f, indent=2)
    print(f"\n  Saved thresholds   -> {THRESHOLD_PATH}")

    # --- Final model ---
    pipe, model = train_final(X, y)

    print("\nPhase 4 complete. Artifacts saved to artifacts/")


if __name__ == "__main__":
    main()
