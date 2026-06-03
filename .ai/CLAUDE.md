# XGBoost ICU Mortality Prediction — Project Intelligence

> Source R file: https://github.com/mdg888/Xgboost-ICU-Mortality-Prediction/blob/main/Boosted%20Trees.R
> Migration target: Python + Gradio + Hugging Face Spaces

---

## Phase 1 — R Project Analysis

### 1. High-Level Overview

Binary classification task: predict in-hospital patient mortality (`HOSPITAL_EXPIRE_FLAG`) from ICU stay data derived from the **MIMIC-III** clinical database (Kaggle competition format). The model is a gradient-boosted tree (XGBoost) trained through the `tidymodels` framework in R, with threshold optimisation via Youden's J statistic, and cross-validated on 5 stratified folds.

---

### 2. Input Dataset Structure

| File | Role | Key columns |
|------|------|-------------|
| `mimic_train_X.csv` | Training features | `icustay_id`, `subject_id`, `hadm_id`, `ADMITTIME`, `DOB`, `GENDER`, `RELIGION`, `ADMISSION_TYPE`, `ETHNICITY`, `INSURANCE`, `FIRST_CAREUNIT`, `MARITAL_STATUS`, `DIAGNOSIS`, vital sign min/mean/max columns |
| `mimic_train_y.csv` | Training labels | `icustay_id`, `HOSPITAL_EXPIRE_FLAG` (0/1) |
| `mimic_test_X.csv` | Test features | Same structure as train X |
| `mimic_kaggle_death_sample_submission.csv` | Test label template | `ID` (→ `icustay_id`), `HOSPITAL_EXPIRE_FLAG` |
| `MIMIC_diagnoses.csv` | Supplementary ICD-9 diagnoses | `SUBJECT_ID`, `HADM_ID`, `ICD9_CODE`, `SEQ_NUM` |

**Vital sign columns** (each with `_Min`, `_Mean`, `_Max` suffix):
`HeartRate`, `SysBP`, `DiasBP`, `MeanBP`, `RespRate`, `TempC`, `SpO2`, `Glucose`

---

### 3. Every Preprocessing Step

#### 3a. ICD-9 Column Removal from Main Data
```r
if("ICD9_diagnosis" %in% names(training_data_mimic_x)) {
  training_data_mimic_x <- training_data_mimic_x %>% select(-ICD9_diagnosis)
}
```
**Why:** The raw X files contain a text `ICD9_diagnosis` column that would cause factor-level errors. Structured diagnosis features are re-created from `MIMIC_diagnoses.csv` instead.

#### 3b. Target Rename
```r
training_data_mimic_y <- training_data_mimic_y %>%
  rename(Death = HOSPITAL_EXPIRE_FLAG) %>%
  select(-any_of("...1"))
```
**Why:** Standardises column name; drops unnamed index column that `read_csv` may introduce.

#### 3c. Column Removal Inside `preprocess_mimic_data`
```r
remove_vec <- c("...1", "subject_id", "hadm_id", "DIAGNOSIS", "Diff", "ICD9_diagnosis")
df_processed <- data %>% select(-any_of(remove_vec))
```
**Why:** `subject_id` / `hadm_id` are identifiers only; `DIAGNOSIS` is free text; `Diff` is undefined; `ICD9_diagnosis` already removed. `icustay_id` is **kept** here as the join key.

#### 3d. Age Derivation from Dates
```r
df_processed <- df_processed %>%
  mutate(
    Addtime = ifelse(!is.na(ADMITTIME), year(as.Date(ADMITTIME)), NA),
    dob     = ifelse(!is.na(DOB),       year(as.Date(DOB)),       NA),
    Age     = Addtime - dob
  ) %>%
  select(-any_of(c("ADMITTIME", "DOB", "Addtime", "dob")))
```
**Why:** MIMIC stores admission time and date-of-birth as datetime strings. Year-level age is sufficient and avoids HIPAA-shifted date arithmetic.

#### 3e. Gender Binarisation
```r
df_processed <- df_processed %>%
  mutate(GENDER = ifelse(GENDER == "M", 1, 0))
```
**Why:** XGBoost requires numeric input; `M → 1`, everything else → `0`.

#### 3f. Vital Sign Variability Features
```r
for(vital in vital_signs) {
  df_processed[[var_col]] <- df_processed[[max_col]] - df_processed[[min_col]]
  df_processed[[var_col]][is.na(df_processed[[var_col]])] <- 0
  df_processed <- df_processed %>% select(-all_of(c(min_col, max_col)))
}
```
`vital_signs = c("HeartRate","SysBP","DiasBP","MeanBP","RespRate","TempC","SpO2","Glucose")`

**Why:** Replaces the Min/Max pair with a single variability (range) feature. Missing variability is imputed as 0 (assumes stable readings). Only the `_Mean` column for each vital sign is retained alongside the new `_Variability` column.

#### 3g. Categorical Dummy Encoding (one-hot)
```r
categorical_cols <- c("RELIGION","ADMISSION_TYPE","ETHNICITY",
                       "INSURANCE","FIRST_CAREUNIT","MARITAL_STATUS")
df_processed <- dummy_cols(df_processed,
                            select_columns = col,
                            remove_selected_columns = TRUE)
```
**Why:** `fastDummies::dummy_cols` creates binary columns for every level; original string columns are dropped.

#### 3h. Missing Value Imputation (median)
```r
train_data_with_target <- train_data_with_target %>%
  mutate(across(where(is.numeric), ~ifelse(is.na(.), median(., na.rm = TRUE), .)))
```
**Why:** Median imputation is robust to outliers in clinical data. Applied **after** the train/test split (on each set independently), which is correct — no leakage.

#### 3i. Target as Factor
```r
train_data_with_target <- train_data_with_target %>%
  mutate(Death = factor(Death))
```
**Why:** `tidymodels` classification mode requires a factor outcome.

---

### 4. Every Feature Engineering Step

#### 4a. ICD-9 Category Features (`process_icd9_data`)

Steps:
1. Count distinct hospital admissions per patient (`total_hospital_stays`).
2. Extract first 3 characters of `ICD9_CODE` → `icd9_first3`.
3. Map to 19 clinical categories using ICD-9 numeric ranges and E/V prefix rules → `icd9_category`.
4. Flag `high_priority = 1` if `SEQ_NUM <= 3` (primary/secondary diagnosis).
5. Aggregate per `(SUBJECT_ID, HADM_ID)`:
   - `total_diagnoses` — total rows
   - `high_priority_diagnoses` — sum of priority flags
   - 19 binary `has_<category>` columns
6. Left-join back to main data on `(subject_id, hadm_id)`.
7. Fill NAs in ICD-9 columns with 0 (patients with no diagnosis records).

**ICD-9 category mapping:**

| Range | Category |
|-------|----------|
| 1–139 | infectious |
| 140–239 | neoplasms |
| 240–279 | endocrine |
| 280–289 | blood |
| 290–319 | mental |
| 320–389 | nervous |
| 390–459 | circulatory |
| 460–519 | respiratory |
| 520–579 | digestive |
| 580–629 | genitourinary |
| 630–679 | pregnancy |
| 680–709 | skin |
| 710–739 | musculoskeletal |
| 740–759 | congenital |
| 760–779 | perinatal |
| 780–799 | symptoms |
| 800–999 | injury |
| E… | external_causes |
| V… | supplementary |

#### 4b. Vital Sign Variability
(See §3f above)  
New columns: `HeartRate_Variability`, `SysBP_Variability`, `DiasBP_Variability`, `MeanBP_Variability`, `RespRate_Variability`, `TempC_Variability`, `SpO2_Variability`, `Glucose_Variability`

#### 4c. Derived Age Feature
(See §3d above)
New column: `Age = year(ADMITTIME) - year(DOB)`

---

### 5. Missing Value Handling

| Location | Method | Code reference |
|----------|--------|----------------|
| ICD-9 join NAs | Fill with 0 | `main_data_with_icd9[icd9_cols] <- lapply(..., function(x) ifelse(is.na(x), 0, x))` |
| Vital variability NAs | Fill with 0 | `df_processed[[var_col]][is.na(...)] <- 0` |
| All remaining numeric NAs | Median imputation | `mutate(across(where(is.numeric), ~ifelse(is.na(.), median(., na.rm=TRUE), .)))` |

**Important:** Median imputation is computed separately on train and test sets. This is correct because test medians are computed from test data only (no leakage from train). In production/inference, train medians must be stored and applied.

---

### 6. Data Cleaning Logic

- Drop unnamed index column (`...1`) from CSVs.
- Drop identifier columns that must not be model inputs: `subject_id`, `hadm_id`.
- Drop free-text columns: `DIAGNOSIS`, `ICD9_diagnosis`.
- Drop `Diff` (undefined column, likely an artifact).
- `ICD9_diagnosis` is stripped before ICD-9 feature engineering begins to avoid factor-level conflicts.

---

### 7. Train/Test Split Methodology

The dataset is a **pre-split Kaggle competition set**. There is no explicit `initial_split` call. The researcher uses:
- `mimic_train_X/Y` → training
- `mimic_test_X/Y` → holdout evaluation / submission

Cross-validation is performed only on training data:
```r
set.seed(1110)
cv_folds <- vfold_cv(train_data_with_target, v = 5, strata = Death)
```
- 5-fold, stratified on `Death` to preserve class balance across folds.

---

### 8. Model Training Methodology

Framework: `tidymodels` workflow wrapping `xgboost`.

```r
bt_spec_balanced <- boost_tree(
  trees        = 1500,
  min_n        = 15,
  tree_depth   = 8,
  learn_rate   = 0.02,
  loss_reduction = 0.05,
  sample_size  = 0.8
) %>%
  set_mode("classification") %>%
  set_engine("xgboost",
             objective    = "binary:logistic",
             eval_metric  = "auc",
             early_stopping = 100)
```

Recipe: `Death ~ .` with `icustay_id` set to `"id"` role (excluded from model matrix but kept in data frame).

Training:
1. `fit_resamples` with 5-fold CV to obtain unbiased performance estimates.
2. `fit(data = train_data_with_target)` — final model trained on **all** training data.
3. Predictions on test set using `predict(..., type = "prob")`.

---

### 9. Hyperparameters Used

| Hyperparameter | Value | tidymodels arg | XGBoost arg |
|----------------|-------|---------------|-------------|
| Number of trees | 1500 | `trees` | `nrounds` |
| Min node size | 15 | `min_n` | `min_child_weight` |
| Tree depth | 8 | `tree_depth` | `max_depth` |
| Learning rate | 0.02 | `learn_rate` | `eta` |
| Min loss reduction | 0.05 | `loss_reduction` | `gamma` |
| Row subsample | 0.8 | `sample_size` | `subsample` |
| Objective | — | engine arg | `binary:logistic` |
| Eval metric | — | engine arg | `auc` |
| Early stopping rounds | 100 | engine arg | `early_stopping_rounds` |

Note: `scale_pos_weight` is **calculated** (`neg/pos`) but **not passed** to the engine in the code. It is a computed variable only.

---

### 10. Evaluation Metrics Used

CV metrics (via `yardstick::metric_set`):
- `accuracy`
- `roc_auc`
- `sensitivity` (recall for positive class)
- `specificity`
- `f_meas` (F1 score)
- `precision`
- `recall`
- `mcc` (Matthews Correlation Coefficient)
- `bal_accuracy` (balanced accuracy)
- `kap` (Cohen's kappa)

Threshold optimisation metrics (computed over thresholds 0.10 → 0.90, step 0.05):
- Youden's J = sensitivity + specificity − 1  ← **used for final threshold**
- F1 score
- Balanced accuracy

Final submission: raw `death_probability` (`.pred_1`) at the Youden-optimal threshold.

---

### 11. Assumptions Made by the Original Author

1. **Year-level age is sufficient** — only `year()` of ADMITTIME and DOB are used; day-level precision is ignored.
2. **Missing variability = stable readings** — variability NAs are set to 0, implying no variation.
3. **ICD-9 join is by patient + admission** — assumes `subject_id`/`hadm_id` in main data matches `SUBJECT_ID`/`HADM_ID` in diagnoses exactly.
4. **Test set medians are valid for test imputation** — author imputes test NAs with test-set medians (not train medians). This is a mild assumption; in production, train medians should be used.
5. **`scale_pos_weight` is not applied** — it is computed but never passed to the XGBoost engine, so class imbalance is not explicitly handled by weighting.
6. **All categorical levels present in train are present in test** — `fastDummies` does not align columns across train/test; a column mismatch would silently occur if levels differ.
7. **ICD-9 codes are stable** — the same patient-admission appears in `MIMIC_diagnoses.csv`; patients with no record get all-zero diagnosis features.
8. **Gender is binary** — `M → 1`, all other values → `0`.

---

## Phase 2 — Python Architecture

### Folder Structure

```
xgboost-icu/
├── .ai/
│   └── CLAUDE.md                  # This file
├── data/
│   └── .gitkeep                   # Raw CSV files go here (not committed)
├── artifacts/
│   ├── model.json                 # Saved XGBoost model
│   ├── preprocessor.pkl           # Saved sklearn pipeline
│   └── train_medians.pkl          # Median values from training set
├── src/
│   ├── __init__.py
│   ├── config.py                  # Constants, column lists, hyperparameters
│   ├── icd9_features.py           # ICD-9 feature engineering
│   ├── preprocessing.py           # sklearn-compatible transformers
│   ├── pipeline.py                # Full sklearn Pipeline assembly
│   ├── train.py                   # Training script (CLI entry point)
│   ├── evaluate.py                # Metrics, threshold optimisation
│   └── predict.py                 # Inference pipeline
├── app.py                         # Gradio UI (Hugging Face Spaces entry point)
├── requirements.txt
└── README.md
```

### File Responsibilities

| File | Responsibility |
|------|---------------|
| `config.py` | All magic numbers and column lists (vital signs, categorical cols, hyperparameters, ICD-9 ranges) |
| `icd9_features.py` | `process_icd9_data(diagnoses_df, main_df)` — exact port of R `process_icd9_data` |
| `preprocessing.py` | Custom sklearn `TransformerMixin` classes: `AgeCalculator`, `GenderBinariser`, `VitalVariabilityTransformer`, `MedianImputer` (fit on train, apply to test) |
| `pipeline.py` | Assembles `sklearn.pipeline.Pipeline` from all transformers + `DictVectorizer` / `OneHotEncoder` for categoricals |
| `train.py` | Loads CSVs → runs ICD-9 features → fits pipeline → trains XGBoost → saves artifacts |
| `evaluate.py` | CV with `StratifiedKFold`, metric collection, threshold optimisation (Youden/F1/balanced) |
| `predict.py` | `predict(patient_dict)` → loads artifacts → applies pipeline → returns probability + class |
| `app.py` | Gradio `Blocks` UI — inputs, validation, prediction, display |

### Data Flow

```
Raw CSVs
   │
   ├─ ICD-9 join (icd9_features.py)
   │
   ▼
Merged DataFrame
   │
   ├─ sklearn Pipeline (preprocessing.py / pipeline.py)
   │    ├─ Drop ID/text columns
   │    ├─ Age derivation
   │    ├─ Gender binarisation
   │    ├─ Vital variability (max - min, NA→0)
   │    ├─ Median imputation (fit on train only)
   │    └─ One-hot encoding (fit on train only)
   │
   ▼
Feature Matrix (numpy array / DataFrame)
   │
   ├─ Training path → XGBClassifier.fit()  → artifacts/model.json
   └─ Inference path → XGBClassifier.predict_proba()  → probability
```

### Training Workflow

1. `python src/train.py --train-x data/mimic_train_X.csv --train-y data/mimic_train_y.csv --diagnoses data/MIMIC_diagnoses.csv`
2. Load and ICD-9-enrich training data.
3. Fit pipeline (imputer + encoder) on training data.
4. `StratifiedKFold(n_splits=5)` CV with metric collection.
5. Threshold optimisation on OOF predictions.
6. Final `XGBClassifier.fit()` on full training set.
7. Save `model.json`, `preprocessor.pkl`, `train_medians.pkl`.

### Prediction Workflow

1. Load `preprocessor.pkl` and `model.json`.
2. Accept raw patient dict (same fields as train X).
3. Apply preprocessor (transform only, no fit).
4. `model.predict_proba(X)[:, 1]` → mortality probability.
5. Apply Youden threshold → binary prediction.
6. Return `{"probability": float, "prediction": int, "risk_level": str}`.

### Deployment Workflow

1. Push repo to Hugging Face Spaces (Gradio SDK).
2. `requirements.txt` installs dependencies.
3. `app.py` is the entry point — loads artifacts from `artifacts/`.
4. No training at runtime; artifacts committed to repo or loaded from HF Hub.

---

## Phase 3 — Feature Engineering Migration (Python)

### 3.1 ICD-9 Feature Engineering

**R original:**
```r
process_icd9_data <- function(extra_data, main_data) {
  hospital_stays <- extra_data %>%
    group_by(SUBJECT_ID) %>%
    summarise(total_hospital_stays = n_distinct(HADM_ID))
  ...
}
```

**Python equivalent (`src/icd9_features.py`):**
```python
import pandas as pd
import numpy as np

ICD9_RANGES = [
    (1,   139, "infectious"),
    (140, 239, "neoplasms"),
    (240, 279, "endocrine"),
    (280, 289, "blood"),
    (290, 319, "mental"),
    (320, 389, "nervous"),
    (390, 459, "circulatory"),
    (460, 519, "respiratory"),
    (520, 579, "digestive"),
    (580, 629, "genitourinary"),
    (630, 679, "pregnancy"),
    (680, 709, "skin"),
    (710, 739, "musculoskeletal"),
    (740, 759, "congenital"),
    (760, 779, "perinatal"),
    (780, 799, "symptoms"),
    (800, 999, "injury"),
]

ALL_CATEGORIES = [r[2] for r in ICD9_RANGES] + ["external_causes", "supplementary", "other"]

def _classify_icd9(code: str) -> str:
    if not isinstance(code, str) or code == "":
        return "other"
    if code.startswith("E"):
        return "external_causes"
    if code.startswith("V"):
        return "supplementary"
    try:
        num = int(code[:3])
    except ValueError:
        return "other"
    for lo, hi, cat in ICD9_RANGES:
        if lo <= num <= hi:
            return cat
    return "other"

def process_icd9_data(diagnoses_df: pd.DataFrame, main_df: pd.DataFrame) -> pd.DataFrame:
    # Hospital stay counts per patient
    hospital_stays = (
        diagnoses_df.groupby("SUBJECT_ID")["HADM_ID"]
        .nunique()
        .reset_index(name="total_hospital_stays")
    )

    diag = diagnoses_df.copy()
    diag["icd9_category"] = diag["ICD9_CODE"].apply(_classify_icd9)
    diag["high_priority"] = (diag["SEQ_NUM"] <= 3).astype(int)

    # Aggregate per patient-admission
    agg = diag.groupby(["SUBJECT_ID", "HADM_ID"]).agg(
        total_diagnoses=("ICD9_CODE", "count"),
        high_priority_diagnoses=("high_priority", "sum"),
    )
    # Binary category presence
    for cat in ALL_CATEGORIES:
        agg[f"has_{cat}"] = diag[diag["icd9_category"] == cat].groupby(
            ["SUBJECT_ID", "HADM_ID"]
        ).size().gt(0).astype(int)
    agg = agg.reset_index()

    # Add hospital stay counts
    agg = agg.merge(hospital_stays, on="SUBJECT_ID", how="left")

    # Join to main data
    result = main_df.merge(
        agg,
        left_on=["subject_id", "hadm_id"],
        right_on=["SUBJECT_ID", "HADM_ID"],
        how="left",
    )
    result.drop(columns=["SUBJECT_ID", "HADM_ID"], errors="ignore", inplace=True)

    # Fill missing ICD-9 cols with 0
    icd9_cols = [c for c in agg.columns if c not in ("SUBJECT_ID", "HADM_ID")]
    result[icd9_cols] = result[icd9_cols].fillna(0)

    return result
```

**Differences from R:**
- `has_<category>` groupby approach is equivalent but expressed as a loop over categories.
- `n_distinct(HADM_ID)` inside `group_by(SUBJECT_ID, HADM_ID)` always returns 1; the R code accidentally computes 1 per row before the left-join with `hospital_stays` fixes it. Python version correctly uses the patient-level `hospital_stays` from the start.

---

### 3.2 Preprocessing Transformers

**R `preprocess_mimic_data` → Python `src/preprocessing.py`:**

```python
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

VITAL_SIGNS = ["HeartRate","SysBP","DiasBP","MeanBP","RespRate","TempC","SpO2","Glucose"]
CATEGORICAL_COLS = ["RELIGION","ADMISSION_TYPE","ETHNICITY","INSURANCE","FIRST_CAREUNIT","MARITAL_STATUS"]
DROP_COLS = ["subject_id","hadm_id","DIAGNOSIS","Diff","ICD9_diagnosis"]

class DropColumnsTransformer(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None): return self
    def transform(self, X):
        return X.drop(columns=[c for c in DROP_COLS if c in X.columns])

class AgeCalculator(TransformerMixin, BaseEstimator):
    """year(ADMITTIME) - year(DOB) → Age; drops date columns."""
    def fit(self, X, y=None): return self
    def transform(self, X):
        X = X.copy()
        if "ADMITTIME" in X.columns and "DOB" in X.columns:
            X["Age"] = (
                pd.to_datetime(X["ADMITTIME"], errors="coerce").dt.year
                - pd.to_datetime(X["DOB"], errors="coerce").dt.year
            )
        return X.drop(columns=["ADMITTIME","DOB"], errors="ignore")

class GenderBinariser(TransformerMixin, BaseEstimator):
    """M → 1, else 0."""
    def fit(self, X, y=None): return self
    def transform(self, X):
        X = X.copy()
        if "GENDER" in X.columns:
            X["GENDER"] = (X["GENDER"] == "M").astype(int)
        return X

class VitalVariabilityTransformer(TransformerMixin, BaseEstimator):
    """Replace _Min/_Max pairs with _Variability = Max - Min (NA → 0)."""
    def fit(self, X, y=None): return self
    def transform(self, X):
        X = X.copy()
        for vital in VITAL_SIGNS:
            min_col, max_col, var_col = f"{vital}_Min", f"{vital}_Max", f"{vital}_Variability"
            if min_col in X.columns and max_col in X.columns:
                X[var_col] = (X[max_col] - X[min_col]).fillna(0)
                X.drop(columns=[min_col, max_col], inplace=True)
        return X

class MedianImputer(TransformerMixin, BaseEstimator):
    """Fit medians on train; apply to train and test."""
    def fit(self, X, y=None):
        self.medians_ = X.select_dtypes(include="number").median()
        return self
    def transform(self, X):
        X = X.copy()
        for col, med in self.medians_.items():
            if col in X.columns:
                X[col] = X[col].fillna(med)
        return X
```

---

### 3.3 One-Hot Encoding

**R:** `fastDummies::dummy_cols` — creates all dummies, no drop-first.  
**Python:** `sklearn.preprocessing.OneHotEncoder(handle_unknown="ignore", sparse_output=False)` fitted inside a `ColumnTransformer`.

**Difference:** sklearn's `handle_unknown="ignore"` fills unseen categories with zeros at inference (equivalent to R's behaviour since R would produce NaN columns for unseen levels).

---

## Phase 4 — Model Training Pipeline

### `src/train.py` (outline)

```python
# Key parameters matching R exactly:
XGB_PARAMS = {
    "n_estimators":     1500,
    "min_child_weight": 15,
    "max_depth":        8,
    "learning_rate":    0.02,
    "gamma":            0.05,
    "subsample":        0.8,
    "objective":        "binary:logistic",
    "eval_metric":      "auc",
    "early_stopping_rounds": 100,
    "random_state":     1110,   # matches set.seed(1110)
}
```

Training steps:
1. Load CSVs.
2. `process_icd9_data(diagnoses, train_x)`.
3. Merge with `train_y` on `icustay_id`.
4. Fit preprocessing pipeline on train X.
5. `StratifiedKFold(n_splits=5, shuffle=True, random_state=1110)`.
6. For each fold: transform, fit XGB, collect OOF probabilities.
7. Compute all metrics (accuracy, ROC-AUC, sensitivity, specificity, F1, MCC, balanced accuracy, kappa).
8. Threshold optimisation (Youden's J) on OOF predictions.
9. Final fit on all training data.
10. Save `model.json`, `preprocessor.pkl`, `optimal_threshold.json`.

### Metric parity with R `yardstick`

| R metric | Python equivalent |
|----------|-------------------|
| `roc_auc` | `sklearn.metrics.roc_auc_score` |
| `accuracy` | `sklearn.metrics.accuracy_score` |
| `sensitivity` | `recall_score(pos_label=1)` |
| `specificity` | `recall_score(pos_label=0)` |
| `f_meas` | `f1_score` |
| `precision` | `precision_score` |
| `mcc` | `matthews_corrcoef` |
| `bal_accuracy` | `balanced_accuracy_score` |
| `kap` | `cohen_kappa_score` |

### Threshold optimisation

```python
def find_youden_threshold(y_true, y_prob, thresholds=None):
    if thresholds is None:
        thresholds = np.arange(0.10, 0.91, 0.05)
    best_j, best_t = -1, 0.5
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        sens = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
        spec = recall_score(y_true, y_pred, pos_label=0, zero_division=0)
        j = sens + spec - 1
        if j > best_j:
            best_j, best_t = j, t
    return best_t, best_j
```

---

## Phase 5 — Inference Pipeline

### `src/predict.py`

```python
import json, pickle
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

def load_artifacts(artifacts_dir="artifacts"):
    with open(f"{artifacts_dir}/preprocessor.pkl","rb") as f:
        preprocessor = pickle.load(f)
    model = XGBClassifier()
    model.load_model(f"{artifacts_dir}/model.json")
    with open(f"{artifacts_dir}/optimal_threshold.json") as f:
        threshold = json.load(f)["youden_threshold"]
    return preprocessor, model, threshold

def predict(patient_dict: dict, artifacts_dir="artifacts") -> dict:
    preprocessor, model, threshold = load_artifacts(artifacts_dir)
    X = pd.DataFrame([patient_dict])
    X_transformed = preprocessor.transform(X)
    prob = model.predict_proba(X_transformed)[0, 1]
    prediction = int(prob >= threshold)
    risk = "High" if prob >= 0.7 else "Medium" if prob >= 0.4 else "Low"
    return {
        "mortality_probability": round(float(prob), 4),
        "binary_prediction": prediction,
        "risk_level": risk,
        "threshold_used": threshold,
    }
```

**Critical requirement:** The preprocessor must be fitted on training data and saved as `preprocessor.pkl`. At inference, only `transform()` is called, never `fit()`. Train medians are baked into `MedianImputer.medians_` inside the saved pipeline.

---

## Phase 6 — Gradio UI Design

### Target Users
- Healthcare professionals reviewing patient risk
- Recruiters / evaluators assessing ML portfolio

### UI Wireframe

```
┌─────────────────────────────────────────────────────┐
│  ICU Mortality Prediction  (XGBoost · MIMIC-III)    │
├───────────────────────┬─────────────────────────────┤
│  PATIENT DEMOGRAPHICS │  VITAL SIGNS                │
│  Age          [  ]    │  Heart Rate Mean    [  ]    │
│  Gender       [M/F]   │  SysBP Mean         [  ]    │
│  Admission    [  ]    │  DiasBP Mean        [  ]    │
│  Insurance    [  ]    │  MeanBP Mean        [  ]    │
│  Ethnicity    [  ]    │  RespRate Mean      [  ]    │
│  Religion     [  ]    │  TempC Mean         [  ]    │
│  Marital      [  ]    │  SpO2 Mean          [  ]    │
│  Care Unit    [  ]    │  Glucose Mean       [  ]    │
│                       │                             │
│  LAB VALUES           │  VITAL VARIABILITY          │
│  (lab columns…)       │  (max-min per vital)        │
├───────────────────────┴─────────────────────────────┤
│              [ Predict Mortality ]                  │
├─────────────────────────────────────────────────────┤
│  RESULT                                             │
│  Mortality Probability: ██████░░░░  62%             │
│  Risk Level: ● HIGH                                 │
│  Binary Prediction: Predicted Death                 │
├─────────────────────────────────────────────────────┤
│  SUBMITTED FEATURES (collapsible table)             │
└─────────────────────────────────────────────────────┘
```

### Component List

| Component | Type | Purpose |
|-----------|------|---------|
| Age | `gr.Number` | Integer 0–120 |
| Gender | `gr.Radio(["M","F"])` | Binarised to 1/0 |
| ADMISSION_TYPE | `gr.Dropdown` | One-hot encoded |
| INSURANCE | `gr.Dropdown` | One-hot encoded |
| ETHNICITY | `gr.Dropdown` | One-hot encoded |
| RELIGION | `gr.Dropdown` | One-hot encoded |
| MARITAL_STATUS | `gr.Dropdown` | One-hot encoded |
| FIRST_CAREUNIT | `gr.Dropdown` | One-hot encoded |
| Vital sign means | `gr.Number` × 8 | Continuous |
| Vital sign min/max | `gr.Number` × 16 | For variability calc |
| Predict button | `gr.Button` | Triggers inference |
| Probability bar | `gr.Label` / `gr.HTML` | Visual probability |
| Risk level | `gr.Textbox` | High/Medium/Low |
| Feature table | `gr.DataFrame` | Input summary |

### User Workflow
1. Fill in patient demographics and vital signs.
2. Click **Predict Mortality**.
3. View mortality probability (0–100%), risk level, and binary outcome.
4. Expand feature table to verify submitted values.

---

## Phase 7 — Gradio Implementation Notes

Key implementation requirements:
- Model loading happens **once** at app startup (not per request) using `@functools.lru_cache` or module-level init.
- Input validation: numeric fields clipped to physiological ranges; required fields flagged before prediction.
- Vital sign min/max pairs accepted as inputs; variability computed inside prediction function (matching pipeline).
- Separate functions: `load_model()`, `validate_inputs()`, `build_patient_dict()`, `run_prediction()`.
- Error handling: return user-friendly message on pipeline failure.

---

## Phase 8 — Hugging Face Deployment

### `requirements.txt`
```
pandas>=2.0
numpy>=1.24
scikit-learn>=1.3
xgboost>=2.0
gradio>=4.0
```

### Spaces configuration (`README.md` frontmatter)
```yaml
---
title: ICU Mortality Prediction
emoji: 🏥
colorFrom: blue
colorTo: red
sdk: gradio
sdk_version: "4.x"
app_file: app.py
pinned: false
---
```

### Artifacts strategy
- `artifacts/model.json`, `artifacts/preprocessor.pkl`, `artifacts/optimal_threshold.json` committed to repo.
- Dataset CSVs **not** committed (too large / sensitive).

---

## Phase 9 — Code Review Checklist

### 1. Feature Engineering Parity with R

| Check | Status | Notes |
|-------|--------|-------|
| ICD-9 categories match | ✓ | All 19 ranges replicated |
| `high_priority` (SEQ_NUM ≤ 3) | ✓ | |
| `total_hospital_stays` per patient | ✓ | R bug noted and corrected |
| Vital variability = max − min | ✓ | NA → 0 |
| Age = year(admit) − year(DOB) | ✓ | |
| Gender M → 1 | ✓ | |
| Categorical one-hot | ✓ | `handle_unknown="ignore"` |
| Median imputation train-only | ✓ | `MedianImputer.fit()` on train only |
| ICD-9 NAs → 0 | ✓ | |

### 2. Data Leakage Risks

| Risk | Mitigation |
|------|-----------|
| Median imputation leakage | `MedianImputer` fitted on train only; saved and applied at test time |
| One-hot encoder leakage | `ColumnTransformer` fitted on train only |
| ICD-9 data leakage | ICD-9 features joined before split — this is the same as R; acceptable because the split is pre-defined by Kaggle |
| Test set median imputation (R bug) | **Fixed in Python**: use train medians for test imputation |

### 3. Reproducibility

- `random_state=1110` matches R `set.seed(1110)`.
- `XGBClassifier` seeded.
- `StratifiedKFold(shuffle=True, random_state=1110)`.
- Model saved as `model.json` (deterministic).

### 4. Known Issues to Fix Before Production

1. **R's `scale_pos_weight` is never applied** — add `scale_pos_weight = neg/pos` to `XGB_PARAMS` to properly handle class imbalance.
2. **Column alignment after one-hot** — after fitting on train, use `preprocessor.get_feature_names_out()` to ensure test matrix has identical columns.
3. **`total_hospital_stays` bug in R** — fixed in Python (per-patient count, not per-admission count of 1).
4. **Age can be negative** — MIMIC shifts dates for privacy; add a clip `Age = max(0, Age)` and consider mapping ages > 89 to a fixed value (MIMIC convention: 300 → 91.4).
5. **`early_stopping_rounds` requires eval set** — in sklearn API, pass `eval_set=[(X_val, y_val)]` to `fit()` or remove early stopping for final training run.

---

## Key Decisions Log

| Decision | Rationale |
|----------|-----------|
| Separate `icd9_features.py` from preprocessing | ICD-9 join requires two DataFrames; sklearn transformers expect single X |
| `MedianImputer` fitted only on train | Prevent leakage; fix R test-set median bug |
| `handle_unknown="ignore"` in OHE | Matches R behaviour for unseen categorical levels at inference |
| Artifacts committed to repo | Enables zero-training Hugging Face deployment |
| Youden threshold stored in JSON | Allows threshold tuning without retraining |
