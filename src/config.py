from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
ARTIFACTS_DIR = ROOT / "artifacts"

TRAIN_X_PATH      = DATA_DIR / "mimic_train_X.csv"
TRAIN_Y_PATH      = DATA_DIR / "mimic_train_y.csv"
TEST_X_PATH       = DATA_DIR / "mimic_test_X.csv"
TEST_Y_PATH       = DATA_DIR / "mimic_kaggle_death_sample_submission.csv"
DIAGNOSES_PATH    = DATA_DIR / "extra_data" / "MIMIC_diagnoses.csv"

MODEL_PATH        = ARTIFACTS_DIR / "model.json"
PREPROCESSOR_PATH = ARTIFACTS_DIR / "preprocessor.pkl"
THRESHOLD_PATH    = ARTIFACTS_DIR / "optimal_threshold.json"

# ── Columns ────────────────────────────────────────────────────────────────
VITAL_SIGNS = [
    "HeartRate", "SysBP", "DiasBP", "MeanBP",
    "RespRate", "TempC", "SpO2", "Glucose",
]

CATEGORICAL_COLS = [
    "RELIGION", "ADMISSION_TYPE", "ETHNICITY",
    "INSURANCE", "FIRST_CAREUNIT", "MARITAL_STATUS",
]

DROP_COLS = [
    "subject_id", "hadm_id", "DIAGNOSIS", "Diff", "ICD9_diagnosis",
]

# ── ICD-9 category ranges ──────────────────────────────────────────────────
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

ALL_ICD9_CATEGORIES = (
    [r[2] for r in ICD9_RANGES] + ["external_causes", "supplementary", "other"]
)

# ── Model hyperparameters (matching R exactly) ─────────────────────────────
XGB_PARAMS = {
    "n_estimators":          1500,
    "min_child_weight":      15,       # R: min_n = 15
    "max_depth":             8,        # R: tree_depth = 8
    "learning_rate":         0.02,     # R: learn_rate = 0.02
    "gamma":                 0.05,     # R: loss_reduction = 0.05
    "subsample":             0.8,      # R: sample_size = 0.8
    "objective":             "binary:logistic",
    "eval_metric":           "auc",
    "early_stopping_rounds": 100,
    "random_state":          1110,     # R: set.seed(1110)
}

RANDOM_SEED = 1110
CV_FOLDS    = 5

# Threshold grid matches R: seq(0.1, 0.9, by = 0.05)
THRESHOLD_GRID = [round(t, 2) for t in [x / 100 for x in range(10, 91, 5)]]
