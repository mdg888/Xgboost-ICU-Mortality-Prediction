"""
Metrics and threshold optimisation — mirrors R yardstick metric_set and
optimize_threshold() with Youden/F1/balanced criteria.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, roc_auc_score, recall_score, f1_score,
    precision_score, matthews_corrcoef, balanced_accuracy_score,
    cohen_kappa_score,
)

from src.config import THRESHOLD_GRID


def compute_metrics(y_true, y_prob, threshold: float) -> dict:
    """Compute all metrics matching R's custom_metrics metric_set."""
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "accuracy":     accuracy_score(y_true, y_pred),
        "roc_auc":      roc_auc_score(y_true, y_prob),
        "sensitivity":  recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "specificity":  recall_score(y_true, y_pred, pos_label=0, zero_division=0),
        "f1":           f1_score(y_true, y_pred, zero_division=0),
        "precision":    precision_score(y_true, y_pred, zero_division=0),
        "recall":       recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "mcc":          matthews_corrcoef(y_true, y_pred),
        "bal_accuracy": balanced_accuracy_score(y_true, y_pred),
        "kappa":        cohen_kappa_score(y_true, y_pred),
    }


def optimize_threshold(y_true, y_prob) -> dict:
    """
    Find optimal thresholds via Youden's J, F1, and balanced accuracy.
    Mirrors R's optimize_threshold() over seq(0.1, 0.9, by=0.05).
    """
    rows = []
    for t in THRESHOLD_GRID:
        y_pred = (y_prob >= t).astype(int)
        sens = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
        spec = recall_score(y_true, y_pred, pos_label=0, zero_division=0)
        acc  = accuracy_score(y_true, y_pred)
        f1   = f1_score(y_true, y_pred, zero_division=0)
        rows.append({
            "threshold":    t,
            "sensitivity":  sens,
            "specificity":  spec,
            "accuracy":     acc,
            "f1_score":     f1,
            "youden":       sens + spec - 1,
            "balanced_acc": (sens + spec) / 2,
        })

    df = pd.DataFrame(rows)
    return {
        "youden":   df.loc[df["youden"].idxmax()].to_dict(),
        "f1":       df.loc[df["f1_score"].idxmax()].to_dict(),
        "balanced": df.loc[df["balanced_acc"].idxmax()].to_dict(),
        "all":      df,
    }
