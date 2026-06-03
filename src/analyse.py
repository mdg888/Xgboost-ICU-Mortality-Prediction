"""
Phase 1 analysis — loads, processes, and reports on the MIMIC data
exactly as the R project does, producing console output for verification.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np

from src.config import (
    TRAIN_X_PATH, TRAIN_Y_PATH, TEST_X_PATH, TEST_Y_PATH,
    DIAGNOSES_PATH, VITAL_SIGNS, CATEGORICAL_COLS,
)
from src.icd9_features import process_icd9_data
from src.preprocessing import build_pipeline


def load_raw_data():
    print("=" * 60)
    print("1. LOADING RAW DATA")
    print("=" * 60)

    train_x = pd.read_csv(TRAIN_X_PATH, index_col=0)
    train_y = pd.read_csv(TRAIN_Y_PATH, index_col=0)
    test_x  = pd.read_csv(TEST_X_PATH,  index_col=0)
    test_y  = pd.read_csv(TEST_Y_PATH)
    diagnoses = pd.read_csv(DIAGNOSES_PATH)

    # Rename target column to match R convention
    train_y = train_y.rename(columns={"HOSPITAL_EXPIRE_FLAG": "Death"})
    test_y  = test_y.rename(columns={"HOSPITAL_EXPIRE_FLAG": "Death", "ID": "icustay_id"})

    print(f"  train_X shape : {train_x.shape}")
    print(f"  train_y shape : {train_y.shape}")
    print(f"  test_X  shape : {test_x.shape}")
    print(f"  diagnoses     : {diagnoses.shape}")
    print(f"  train_X cols  : {list(train_x.columns)}\n")

    return train_x, train_y, test_x, test_y, diagnoses


def analyse_raw(train_x: pd.DataFrame, train_y: pd.DataFrame):
    print("=" * 60)
    print("2. RAW DATA OVERVIEW")
    print("=" * 60)

    print("\n  -- Missing values (train_X, top 15) --")
    missing = train_x.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False).head(15)
    for col, cnt in missing.items():
        pct = cnt / len(train_x) * 100
        print(f"    {col:<35} {cnt:>5}  ({pct:.1f}%)")

    print("\n  -- Target distribution --")
    vc = train_y["Death"].value_counts().sort_index()
    total = len(train_y)
    for val, cnt in vc.items():
        print(f"    Death={val}: {cnt:>5}  ({cnt/total*100:.1f}%)")
    neg = vc.get(0, 0)
    pos = vc.get(1, 0)
    print(f"  scale_pos_weight (neg/pos): {neg/pos:.4f}")

    print("\n  -- Categorical column cardinality --")
    for col in CATEGORICAL_COLS:
        if col in train_x.columns:
            n = train_x[col].nunique()
            top3 = train_x[col].value_counts().head(3).to_dict()
            print(f"    {col:<25} {n} levels  top3={top3}")

    print("\n  -- Vital signs range (train_X) --")
    for vital in VITAL_SIGNS:
        cols = [f"{vital}_Min", f"{vital}_Max", f"{vital}_Mean"]
        existing = [c for c in cols if c in train_x.columns]
        if existing:
            row = train_x[existing].describe().loc[["min", "max", "mean"]].round(2)
            print(f"    {vital}")
            print(row.to_string(header=True) + "\n")


def analyse_icd9(diagnoses: pd.DataFrame, train_x: pd.DataFrame):
    print("=" * 60)
    print("3. ICD-9 FEATURE ENGINEERING")
    print("=" * 60)

    print(f"  Diagnoses rows : {len(diagnoses):,}")
    print(f"  Unique patients: {diagnoses['SUBJECT_ID'].nunique():,}")
    print(f"  Unique admissions: {diagnoses['HADM_ID'].nunique():,}")

    # Remove ICD9_diagnosis column (matches R)
    if "ICD9_diagnosis" in train_x.columns:
        train_x = train_x.drop(columns=["ICD9_diagnosis"])

    enriched = process_icd9_data(diagnoses, train_x)
    icd9_new_cols = [c for c in enriched.columns if c not in train_x.columns]
    print(f"\n  New ICD-9 feature columns ({len(icd9_new_cols)}): {icd9_new_cols}")

    # How many rows matched
    matched = enriched["total_diagnoses"].notna().sum()
    print(f"  Rows matched to diagnoses: {matched:,} / {len(enriched):,} "
          f"({matched/len(enriched)*100:.1f}%)")

    # Category prevalence
    has_cols = [c for c in enriched.columns if c.startswith("has_")]
    print("\n  ICD-9 category prevalence (% of ICU stays):")
    prev = enriched[has_cols].mean().sort_values(ascending=False)
    for col, val in prev.items():
        print(f"    {col:<35} {val*100:.1f}%")

    return enriched


def analyse_preprocessing(train_x_enriched: pd.DataFrame, train_y: pd.DataFrame):
    print("\n" + "=" * 60)
    print("4. PREPROCESSING PIPELINE")
    print("=" * 60)

    # Merge target
    train = train_x_enriched.merge(train_y, on="icustay_id", how="left")
    y = train["Death"]
    X = train.drop(columns=["Death"])

    pipeline = build_pipeline()

    # Fit on training data only
    # The pipeline expects icustay_id to pass through — keep it, then drop after
    X_ids = X["icustay_id"].copy()
    X_model = X.drop(columns=["icustay_id"])

    X_transformed = pipeline.fit_transform(X_model)

    # Get feature names
    ct = pipeline.named_steps["encode"]
    feature_names = list(ct.get_feature_names_out())

    print(f"\n  Input features  : {X_model.shape[1]}")
    print(f"  Output features : {len(feature_names)}")
    print(f"  Training rows   : {X_transformed.shape[0]}")

    # Check for any remaining NaNs
    nan_count = np.isnan(X_transformed).sum()
    print(f"  NaN values after pipeline: {nan_count}")

    # Show a few feature names
    print(f"\n  First 10 feature names : {feature_names[:10]}")
    print(f"  Last  10 feature names : {feature_names[-10:]}")

    # Age stats
    # Age is in the passthrough columns; find it
    if "Age" in feature_names:
        age_idx = feature_names.index("Age")
        age_vals = X_transformed[:, age_idx]
        print(f"\n  Age stats — min:{age_vals.min():.0f}  max:{age_vals.max():.0f}  "
              f"mean:{age_vals.mean():.1f}  median:{np.median(age_vals):.1f}")

    return pipeline, X_transformed, feature_names, y


def main():
    train_x, train_y, test_x, test_y, diagnoses = load_raw_data()
    analyse_raw(train_x, train_y)

    train_x_enriched = analyse_icd9(diagnoses, train_x.copy())
    pipeline, X_arr, feature_names, y = analyse_preprocessing(train_x_enriched, train_y)

    print("\n" + "=" * 60)
    print("5. SUMMARY")
    print("=" * 60)
    print(f"  Final feature matrix : {X_arr.shape}")
    print(f"  Target classes       : {sorted(y.unique())}")
    pos = (y == 1).sum()
    neg = (y == 0).sum()
    print(f"  Class balance        : {neg} negative / {pos} positive  "
          f"(scale_pos_weight = {neg/pos:.3f})")
    print("\nPhase 1 analysis complete. Ready for Phase 4 model training.\n")


if __name__ == "__main__":
    main()
