"""
ICU Mortality Prediction — Gradio UI
Phase 7: Portfolio showcase app powered by XGBoost trained on MIMIC-III data.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import gradio as gr
import pandas as pd

from src.predict import predict

# ── Categorical options (exact levels from training data) ──────────────────
ADMISSION_TYPES  = ["EMERGENCY", "ELECTIVE", "URGENT"]
INSURANCE_TYPES  = ["Medicare", "Medicaid", "Private", "Government", "Self Pay"]
RELIGIONS        = ["CATHOLIC", "PROTESTANT QUAKER", "JEWISH", "NOT SPECIFIED",
                    "UNOBTAINABLE", "OTHER"]
MARITAL_STATUSES = ["MARRIED", "SINGLE", "WIDOWED", "DIVORCED",
                    "SEPARATED", "LIFE PARTNER", "UNKNOWN (DEFAULT)"]
ETHNICITIES      = ["WHITE", "BLACK", "HISPANIC OR LATINO", "ASIAN",
                    "MULTI RACE ETHNICITY", "OTHER", "UNKNOWN"]
CARE_UNITS       = ["MICU", "SICU", "CSRU", "CCU", "TSICU"]

VITAL_LABELS = [
    ("HeartRate",  "Heart Rate",        "bpm",   0,   300),
    ("SysBP",      "Systolic BP",       "mmHg",  0,   350),
    ("DiasBP",     "Diastolic BP",      "mmHg",  0,   200),
    ("MeanBP",     "Mean BP",           "mmHg",  0,   300),
    ("RespRate",   "Respiratory Rate",  "/min",  0,   80),
    ("TempC",      "Temperature",       "C",     25,  45),
    ("SpO2",       "SpO2",              "%",     0,   100),
    ("Glucose",    "Glucose",           "mg/dL", 0,   2500),
]


def build_patient_dict(
    age, gender,
    admission_type, insurance, religion, marital_status, ethnicity, first_careunit,
    hr_min, hr_max,
    sbp_min, sbp_max,
    dbp_min, dbp_max,
    mbp_min, mbp_max,
    rr_min, rr_max,
    temp_min, temp_max,
    spo2_min, spo2_max,
    gluc_min, gluc_max,
):
    hr_mean   = (hr_min   + hr_max)   / 2
    sbp_mean  = (sbp_min  + sbp_max)  / 2
    dbp_mean  = (dbp_min  + dbp_max)  / 2
    mbp_mean  = (mbp_min  + mbp_max)  / 2
    rr_mean   = (rr_min   + rr_max)   / 2
    temp_mean = (temp_min + temp_max) / 2
    spo2_mean = (spo2_min + spo2_max) / 2
    gluc_mean = (gluc_min + gluc_max) / 2
    return {
        "Age":              age,
        "GENDER":           gender,
        "ADMISSION_TYPE":   admission_type,
        "INSURANCE":        insurance,
        "RELIGION":         religion,
        "MARITAL_STATUS":   marital_status,
        "ETHNICITY":        ethnicity,
        "FIRST_CAREUNIT":   first_careunit,
        "HeartRate_Min":    hr_min,   "HeartRate_Max":   hr_max,   "HeartRate_Mean":   hr_mean,
        "SysBP_Min":        sbp_min,  "SysBP_Max":       sbp_max,  "SysBP_Mean":       sbp_mean,
        "DiasBP_Min":       dbp_min,  "DiasBP_Max":       dbp_max,  "DiasBP_Mean":      dbp_mean,
        "MeanBP_Min":       mbp_min,  "MeanBP_Max":       mbp_max,  "MeanBP_Mean":      mbp_mean,
        "RespRate_Min":     rr_min,   "RespRate_Max":     rr_max,   "RespRate_Mean":    rr_mean,
        "TempC_Min":        temp_min, "TempC_Max":        temp_max, "TempC_Mean":       temp_mean,
        "SpO2_Min":         spo2_min, "SpO2_Max":         spo2_max, "SpO2_Mean":        spo2_mean,
        "Glucose_Min":      gluc_min, "Glucose_Max":      gluc_max, "Glucose_Mean":     gluc_mean,
        # ICD-9 fields default to 0 for manual entry (no diagnosis lookup)
        "total_diagnoses":          0,
        "high_priority_diagnoses":  0,
        "total_hospital_stays":     1,
        "has_infectious": 0, "has_neoplasms": 0, "has_endocrine": 0,
        "has_blood": 0, "has_mental": 0, "has_nervous": 0,
        "has_circulatory": 0, "has_respiratory": 0, "has_digestive": 0,
        "has_genitourinary": 0, "has_pregnancy": 0, "has_skin": 0,
        "has_musculoskeletal": 0, "has_congenital": 0, "has_perinatal": 0,
        "has_symptoms": 0, "has_injury": 0, "has_external_causes": 0,
        "has_supplementary": 0, "has_other": 0,
    }


def validate(patient_dict: dict) -> str | None:
    vitals = ["HeartRate", "SysBP", "DiasBP", "MeanBP", "RespRate", "TempC", "SpO2", "Glucose"]
    for v in vitals:
        mn = patient_dict[f"{v}_Min"]
        mx = patient_dict[f"{v}_Max"]
        if mn is None or mx is None:
            return f"{v}: Min and Max are required."
        if mn > mx:
            return f"{v}: Min ({mn}) cannot exceed Max ({mx})."
    if patient_dict["Age"] is None or not (0 <= patient_dict["Age"] <= 120):
        return "Age must be between 0 and 120."
    return None


def run_prediction(*args):
    patient = build_patient_dict(*args)

    error = validate(patient)
    if error:
        return (
            "⚠️ Input error",
            error,
            "",
            pd.DataFrame(),
        )

    try:
        result = predict(patient)
    except Exception as e:
        return ("Error", str(e), "", pd.DataFrame())

    prob       = result["mortality_probability"]
    prediction = result["binary_prediction"]
    risk       = result["risk_level"]

    # Probability display
    pct = prob * 100
    bar_filled = int(pct / 5)
    bar = "█" * bar_filled + "░" * (20 - bar_filled)
    prob_display = f"{bar}  {pct:.1f}%"

    # Risk badge
    risk_colors = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}
    risk_display = f"{risk_colors[risk]} {risk} Risk"

    # Outcome
    outcome = "Predicted: Death" if prediction == 1 else "Predicted: Survival"

    # Feature summary table
    vitals_summary = []
    for key, label, unit, _, _ in VITAL_LABELS:
        vitals_summary.append({
            "Feature": label,
            "Min": patient[f"{key}_Min"],
            "Max": patient[f"{key}_Max"],
            "Mean": patient[f"{key}_Mean"],
            "Unit": unit,
        })
    summary_df = pd.DataFrame(vitals_summary)

    return prob_display, risk_display, outcome, summary_df


# ── UI Layout ──────────────────────────────────────────────────────────────
with gr.Blocks(title="ICU Mortality Prediction") as demo:

    gr.Markdown("""
# ICU Mortality Prediction
**XGBoost model trained on MIMIC-III clinical data**

Predicts in-hospital mortality risk for ICU patients.
Enter patient vitals and demographics below, then click **Predict**.

> *Portfolio project — not for clinical use.*
""")

    with gr.Row():
        # ── Left column: Demographics ──────────────────────────────────────
        with gr.Column(scale=1):
            gr.Markdown("### Patient Demographics")

            age    = gr.Number(label="Age (years)", value=65, minimum=0, maximum=120, precision=0)
            gender = gr.Radio(["M", "F"], label="Gender", value="M")

            admission_type  = gr.Dropdown(ADMISSION_TYPES,  label="Admission Type",  value="EMERGENCY")
            insurance       = gr.Dropdown(INSURANCE_TYPES,  label="Insurance",       value="Medicare")
            religion        = gr.Dropdown(RELIGIONS,        label="Religion",        value="NOT SPECIFIED")
            marital_status  = gr.Dropdown(MARITAL_STATUSES, label="Marital Status",  value="MARRIED")
            ethnicity       = gr.Dropdown(ETHNICITIES,      label="Ethnicity",       value="WHITE")
            first_careunit  = gr.Dropdown(CARE_UNITS,       label="First Care Unit", value="MICU")

        # ── Right column: Vital signs ──────────────────────────────────────
        with gr.Column(scale=2):
            gr.Markdown("### Vital Signs — enter Min and Max for each")

            vital_inputs = []
            for key, label, unit, vmin, vmax in VITAL_LABELS:
                gr.Markdown(f"**{label}** ({unit})")
                with gr.Row():
                    mn = gr.Number(label="Min", minimum=vmin, maximum=vmax)
                    mx = gr.Number(label="Max", minimum=vmin, maximum=vmax)
                if key == "TempC":
                    gr.Markdown(f"*Please enter values between {vmin}°C and {vmax}°C.*")
                vital_inputs += [mn, mx]

    predict_btn = gr.Button("Predict Mortality", variant="primary", size="lg")

    gr.Markdown("---")
    gr.Markdown("### Result")

    with gr.Row():
        prob_out    = gr.Textbox(label="Mortality Probability", interactive=False)
        risk_out    = gr.Textbox(label="Risk Level",            interactive=False)
        outcome_out = gr.Textbox(label="Prediction",            interactive=False)

    gr.Markdown("### Submitted Vital Signs")
    table_out = gr.DataFrame(label="", interactive=False)

    # Wire up
    all_inputs = [
        age, gender,
        admission_type, insurance, religion, marital_status, ethnicity, first_careunit,
        *vital_inputs,
    ]

    predict_btn.click(
        fn=run_prediction,
        inputs=all_inputs,
        outputs=[prob_out, risk_out, outcome_out, table_out],
    )

    gr.Markdown("### Example Patients — click to load")
    gr.Examples(
        label="",
        examples=[
            # High-risk: 91yo widowed female, emergency CCU admission
            [91, "F", "EMERGENCY", "Medicare", "NOT SPECIFIED", "WIDOWED", "WHITE", "CCU",
             56, 77, 83, 148, 46, 91, 59, 98,
             16, 28, 35.7, 37.8, 94, 100, 137, 164],
            # Low-risk: 70yo single female, emergency MICU admission
            [70, "F", "EMERGENCY", "Medicare", "PROTESTANT QUAKER", "SINGLE", "WHITE", "MICU",
             89, 145, 74, 127, 42, 90, 59, 94,
             15, 30, 35.1, 36.9, 90, 99, 111, 230],
        ],
        inputs=all_inputs,
        outputs=[prob_out, risk_out, outcome_out, table_out],
        fn=run_prediction,
        cache_examples=False,
        example_labels=["High-Risk Patient (91F, CCU)", "Low-Risk Patient (70F, MICU)"],
    )

    gr.Markdown("""
---
**Model details:** XGBoost (`n_estimators=1500`, `max_depth=8`, `learning_rate=0.02`)
| CV ROC-AUC | Sensitivity | Specificity | Balanced Accuracy |
|---|---|---|---|
| 0.887 | 0.640 | 0.902 | 0.771 |

Threshold optimised via Youden's J statistic (threshold = 0.30).
Data source: [MIMIC-III](https://mimic.mit.edu/) via Kaggle competition format.
""")


if __name__ == "__main__":
    demo.launch()
