---
title: ICU Mortality Prediction
emoji: 🏥
colorFrom: blue
colorTo: red
sdk: gradio
sdk_version: "5.29.0"
app_file: app.py
pinned: false
---

# ICU Mortality Prediction

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Hugging%20Face-yellow)](https://huggingface.co/spaces/mdg8888/icu-mortality-prediction)

**[Try the live demo](https://huggingface.co/spaces/mdg8888/icu-mortality-prediction)**

## What is this?

This project predicts the probability of in-hospital death for ICU patients using a gradient-boosted tree model (XGBoost). It was originally built in R as part of a university data science project using the MIMIC-III clinical database, and has been fully migrated to Python and deployed as an interactive web app on Hugging Face Spaces.

### Background

ICU (Intensive Care Unit) mortality prediction is a well-studied problem in clinical machine learning. Early identification of high-risk patients allows clinical teams to allocate resources more effectively and intervene sooner. The MIMIC-III database (Medical Information Mart for Intensive Care) is one of the most widely used publicly available clinical datasets, containing records from over 40,000 ICU admissions at Beth Israel Deaconess Medical Center.

### What the original R project did

The original project was built in R using the `tidymodels` framework. It:
- Loaded raw MIMIC-III ICU data from CSV files
- Engineered features from ICD-9 diagnosis codes, vital signs, and patient demographics
- Trained an XGBoost classifier with 5-fold cross-validation
- Optimised the classification threshold using Youden's J statistic
- Produced a submission CSV for the Kaggle competition format

### What this Python project adds

This migration reproduces the entire R pipeline in Python with several improvements:
- **Proper train-only fitting** — the preprocessing pipeline (median imputation, one-hot encoding) is fitted on training data only and saved as an artifact, preventing data leakage when applied to new patients
- **`scale_pos_weight` applied** — the R project calculated but never used the class weight; this version applies it correctly to handle the 88/12 class imbalance
- **Age edge case handling** — MIMIC-III shifts ages above 89 to ~300 for privacy; this is correctly mapped back to 91
- **Deployed as a live web app** — anyone can enter patient data and get a prediction instantly

The app takes a patient's vital signs and demographics as input and returns:
- A **mortality probability** (0–100%)
- A **risk level** (Low / Medium / High)
- A **binary prediction** (Survival / Death) based on an optimised threshold

## Model

- **Algorithm:** XGBoost (`n_estimators=1500`, `max_depth=8`, `learning_rate=0.02`)
- **Training data:** MIMIC-III via Kaggle competition format (20,885 ICU stays)
- **Validation:** 5-fold stratified cross-validation (seed=1110)
- **Class imbalance:** handled via `scale_pos_weight` (7.9 — 88.8% survive, 11.2% die)
- **Threshold:** optimised via Youden's J statistic (threshold = 0.30)

### Cross-Validation Results

| Metric | Mean | Std |
|--------|------|-----|
| ROC-AUC | 0.887 | ±0.005 |
| Sensitivity | 0.640 | ±0.039 |
| Specificity | 0.902 | ±0.012 |
| Balanced Accuracy | 0.771 | ±0.014 |
| F1 Score | 0.529 | ±0.008 |
| MCC | 0.468 | ±0.010 |

### At Youden Threshold (0.30)

| Metric | Value |
|--------|-------|
| Sensitivity | 0.810 |
| Specificity | 0.790 |
| Balanced Accuracy | 0.800 |

## Features (74 total)

| Group | Features |
|-------|----------|
| Vital signs | Heart Rate, Systolic BP, Diastolic BP, Mean BP, Respiratory Rate, Temperature, SpO2, Glucose — each with Mean and Variability (Max − Min) |
| Demographics | Age, Gender |
| Categorical | Admission Type, Insurance, Religion, Marital Status, Ethnicity, First Care Unit (one-hot encoded) |
| ICD-9 diagnoses | 19 binary disease category flags, total diagnoses count, high-priority diagnosis count, total hospital stays |

## Project Structure

```
icu-mortality-prediction/
│
├── app.py                        # Gradio web app — entry point for Hugging Face Spaces
├── requirements.txt              # Python package dependencies
├── README.md                     # This file
│
├── src/                          # Core Python package
│   ├── __init__.py
│   ├── config.py                 # Central config — all file paths, column names,
│   │                             # ICD-9 ranges, and XGBoost hyperparameters
│   │
│   ├── icd9_features.py          # ICD-9 feature engineering
│   │                             # Joins MIMIC_diagnoses.csv onto main data,
│   │                             # maps codes to 19 clinical categories,
│   │                             # creates binary presence flags per admission
│   │
│   ├── preprocessing.py          # sklearn-compatible transformers:
│   │                             #   AgeCalculator       — year(admit) - year(DOB)
│   │                             #   GenderBinariser     — M→1, F→0
│   │                             #   VitalVariability    — Max - Min per vital sign
│   │                             #   MedianImputer       — fit on train, apply to test
│   │                             #   OneHotEncoder       — 6 categorical columns
│   │                             # Assembled into a single sklearn Pipeline
│   │
│   ├── train.py                  # Full training pipeline:
│   │                             #   1. Load + ICD-9 enrich data
│   │                             #   2. 5-fold stratified CV with metrics
│   │                             #   3. Threshold optimisation (Youden/F1/balanced)
│   │                             #   4. Final model fit on all training data
│   │                             #   5. Save artifacts to artifacts/
│   │
│   ├── evaluate.py               # Metric functions mirroring R's yardstick:
│   │                             # accuracy, ROC-AUC, sensitivity, specificity,
│   │                             # F1, precision, MCC, balanced accuracy, kappa
│   │                             # + Youden threshold optimisation
│   │
│   ├── predict.py                # Inference pipeline:
│   │                             #   Loads artifacts once at startup (cached)
│   │                             #   Applies preprocessor.transform() only
│   │                             #   Returns probability, binary prediction, risk level
│   │
│   └── analyse.py                # Exploratory analysis script — prints data stats,
│                                 # missing values, class balance, ICD-9 prevalence.
│                                 # Not used in production, run locally for inspection.
│
├── artifacts/                    # Saved model artifacts (committed to repo)
│   ├── model.json                # XGBoost model weights
│   ├── preprocessor.pkl          # Fitted sklearn pipeline (medians + encoder baked in)
│   └── optimal_threshold.json   # Youden / F1 / balanced thresholds
│
└── data/                         # Raw MIMIC-III CSV files (NOT committed — too large)
    ├── mimic_train_X.csv
    ├── mimic_train_y.csv
    ├── mimic_test_X.csv
    ├── mimic_kaggle_death_sample_submission.csv
    └── extra_data/
        └── MIMIC_diagnoses.csv
```

### Data Flow

```
Raw CSVs (data/)
      │
      ▼
ICD-9 join (icd9_features.py)
      │  Maps diagnosis codes → 19 category flags per patient-admission
      ▼
Preprocessing Pipeline (preprocessing.py)
      │  Age derivation → Gender binarisation → Vital variability
      │  → Median imputation (train medians) → One-hot encoding
      ▼
Feature Matrix (20,885 rows × 74 features)
      │
      ├──► Training (train.py) ──► artifacts/ (model + preprocessor + threshold)
      │
      └──► Inference (predict.py) ──► mortality probability + risk level
```

## Results

<!-- Add screenshots here once captured -->
*Example predictions available in the live demo — click "High-Risk Patient" or "Low-Risk Patient" to see the model in action.*

## How to Run Locally

```bash
git clone https://huggingface.co/spaces/mdg8888/icu-mortality-prediction
cd icu-mortality-prediction
pip install -r requirements.txt
python app.py
```

Then open `http://127.0.0.1:7860` in your browser.

To retrain the model from scratch (requires MIMIC-III data files in `data/`):
```bash
python src/train.py
```

To run the data analysis:
```bash
python src/analyse.py
```

## Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.11 | Core language |
| XGBoost | Gradient boosted tree model |
| scikit-learn | Preprocessing pipeline |
| pandas / numpy | Data manipulation |
| Gradio | Interactive web UI |
| Hugging Face Spaces | Cloud deployment |

## Data Source

[MIMIC-III Clinical Database](https://mimic.mit.edu/) — a freely accessible critical care database developed by the MIT Lab for Computational Physiology, containing de-identified health data from over 40,000 ICU admissions. Data accessed via Kaggle competition format.

## Disclaimer

This is a portfolio project and is **not intended for clinical use**.
