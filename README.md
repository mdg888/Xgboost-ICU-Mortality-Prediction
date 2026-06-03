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

XGBoost model trained on MIMIC-III clinical ICU data to predict in-hospital mortality.

## Model

- **Algorithm:** XGBoost (`n_estimators=1500`, `max_depth=8`, `learning_rate=0.02`)
- **Training data:** MIMIC-III via Kaggle competition format (20,885 ICU stays)
- **Validation:** 5-fold stratified cross-validation

| Metric | Value |
|--------|-------|
| ROC-AUC | 0.887 |
| Sensitivity | 0.640 |
| Specificity | 0.902 |
| Balanced Accuracy | 0.771 |

Threshold optimised via Youden's J statistic (threshold = 0.30).

## Features

- 8 vital signs (Heart Rate, BP, Respiratory Rate, Temperature, SpO2, Glucose)
- Patient demographics (age, gender, admission type, insurance, ethnicity, care unit)
- ICD-9 diagnosis category features derived from MIMIC-III diagnoses

## Disclaimer

This is a portfolio project and is **not intended for clinical use**.
