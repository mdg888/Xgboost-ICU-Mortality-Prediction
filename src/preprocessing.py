"""
sklearn-compatible transformers — exact port of R preprocess_mimic_data().
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from src.config import VITAL_SIGNS, CATEGORICAL_COLS, DROP_COLS


class DropColumnsTransformer(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.drop(columns=[c for c in DROP_COLS if c in X.columns])


class AgeCalculator(TransformerMixin, BaseEstimator):
    """year(ADMITTIME) - year(DOB) → Age; drop date columns."""

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        if "ADMITTIME" in X.columns and "DOB" in X.columns:
            admit_year = pd.to_datetime(X["ADMITTIME"], errors="coerce").dt.year
            dob_year   = pd.to_datetime(X["DOB"], errors="coerce").dt.year
            X["Age"]   = admit_year - dob_year
            # MIMIC privacy shift: ages > 89 are set to ~300; map to 91
            X["Age"] = X["Age"].clip(lower=0).where(X["Age"] <= 200, 91)
        return X.drop(columns=["ADMITTIME", "DOB"], errors="ignore")


class GenderBinariser(TransformerMixin, BaseEstimator):
    """M → 1, anything else → 0."""

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        if "GENDER" in X.columns:
            X["GENDER"] = (X["GENDER"] == "M").astype(int)
        return X


class VitalVariabilityTransformer(TransformerMixin, BaseEstimator):
    """Replace _Min/_Max pairs with _Variability = Max - Min (NA → 0). Keep _Mean."""

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for vital in VITAL_SIGNS:
            min_col = f"{vital}_Min"
            max_col = f"{vital}_Max"
            var_col = f"{vital}_Variability"
            if min_col in X.columns and max_col in X.columns:
                X[var_col] = (X[max_col] - X[min_col]).fillna(0)
                X.drop(columns=[min_col, max_col], inplace=True)
        return X


class MedianImputer(TransformerMixin, BaseEstimator):
    """Fit medians on train; apply to train and test. Stored in self.medians_."""

    def fit(self, X: pd.DataFrame, y=None):
        num_cols = X.select_dtypes(include="number").columns
        self.medians_ = X[num_cols].median()
        self.feature_names_in_ = list(X.columns)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col, med in self.medians_.items():
            if col in X.columns:
                X[col] = X[col].fillna(med)
        return X


def build_pipeline() -> Pipeline:
    """
    Assemble the full preprocessing pipeline.
    OneHotEncoder is fitted inside ColumnTransformer so unseen levels at
    inference are handled with handle_unknown='ignore' (zeros — matches R).
    """
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    col_transformer = ColumnTransformer(
        transformers=[
            ("ohe", ohe, CATEGORICAL_COLS),
        ],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )

    pipe = Pipeline(steps=[
        ("drop",        DropColumnsTransformer()),
        ("age",         AgeCalculator()),
        ("gender",      GenderBinariser()),
        ("variability", VitalVariabilityTransformer()),
        ("impute",      MedianImputer()),
        ("encode",      col_transformer),
    ])
    return pipe


def get_feature_names(pipeline: Pipeline, X_sample: pd.DataFrame) -> list[str]:
    """Return feature names after the full pipeline (for inspection)."""
    # Run all steps up to but not including the ColumnTransformer
    X_partial = X_sample.copy()
    for name, step in pipeline.steps[:-1]:
        X_partial = step.transform(X_partial)
    ct = pipeline.named_steps["encode"]
    return list(ct.get_feature_names_out())
