"""
ICD-9 feature engineering — exact port of R process_icd9_data().
"""
import pandas as pd
from src.config import ICD9_RANGES, ALL_ICD9_CATEGORIES


def _classify_icd9(code) -> str:
    if not isinstance(code, str) or code == "":
        return "other"
    if code.startswith("E"):
        return "external_causes"
    if code.startswith("V"):
        return "supplementary"
    try:
        num = int(str(code)[:3])
    except (ValueError, TypeError):
        return "other"
    for lo, hi, cat in ICD9_RANGES:
        if lo <= num <= hi:
            return cat
    return "other"


def process_icd9_data(diagnoses_df: pd.DataFrame, main_df: pd.DataFrame) -> pd.DataFrame:
    """
    Join ICD-9 derived features onto main_df.

    diagnoses_df columns: SUBJECT_ID, HADM_ID, SEQ_NUM, ICD9_CODE
    main_df must have: subject_id, hadm_id
    """
    # Patient-level hospital stay count (mirrors R's hospital_stays)
    hospital_stays = (
        diagnoses_df.groupby("SUBJECT_ID")["HADM_ID"]
        .nunique()
        .reset_index(name="total_hospital_stays")
    )

    diag = diagnoses_df.copy()
    diag["icd9_category"] = diag["ICD9_CODE"].astype(str).apply(_classify_icd9)
    diag["high_priority"] = (diag["SEQ_NUM"] <= 3).astype(int)

    # Base aggregation per patient-admission
    agg = (
        diag.groupby(["SUBJECT_ID", "HADM_ID"])
        .agg(
            total_diagnoses=("ICD9_CODE", "count"),
            high_priority_diagnoses=("high_priority", "sum"),
        )
        .reset_index()
    )

    # Binary presence flag for each ICD-9 category
    cat_flags = (
        diag.groupby(["SUBJECT_ID", "HADM_ID"])["icd9_category"]
        .apply(lambda s: s.value_counts())
        .unstack(fill_value=0)
        .reset_index()
    )
    # Ensure all expected category columns exist (fill 0 for absent ones)
    for cat in ALL_ICD9_CATEGORIES:
        col = cat
        if col not in cat_flags.columns:
            cat_flags[col] = 0
        else:
            cat_flags[col] = (cat_flags[col] > 0).astype(int)

    cat_flags = cat_flags[["SUBJECT_ID", "HADM_ID"] + ALL_ICD9_CATEGORIES]
    # Rename to has_<category>
    cat_flags = cat_flags.rename(
        columns={c: f"has_{c}" for c in ALL_ICD9_CATEGORIES}
    )

    agg = agg.merge(cat_flags, on=["SUBJECT_ID", "HADM_ID"], how="left")
    agg = agg.merge(hospital_stays, on="SUBJECT_ID", how="left")

    # Join onto main data
    result = main_df.merge(
        agg,
        left_on=["subject_id", "hadm_id"],
        right_on=["SUBJECT_ID", "HADM_ID"],
        how="left",
    )
    result.drop(columns=["SUBJECT_ID", "HADM_ID"], errors="ignore", inplace=True)

    # Fill NAs in ICD-9 columns with 0 (patients with no diagnosis records)
    icd9_cols = [c for c in agg.columns if c not in ("SUBJECT_ID", "HADM_ID")]
    result[icd9_cols] = result[icd9_cols].fillna(0)

    return result
