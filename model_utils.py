"""Shared schema and evaluation helpers for the classification assignment."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


PROJECT_DIR = Path(__file__).resolve().parent
TARGET_COLUMN = "diagnosis"
POSITIVE_LABEL = "M"
CLASS_LABELS = ["B", "M"]

BASE_FEATURES = [
    "radius",
    "texture",
    "perimeter",
    "area",
    "smoothness",
    "compactness",
    "concavity",
    "concave points",
    "symmetry",
    "fractal dimension",
]
FEATURE_COLUMNS = [
    f"{statistic} {feature}"
    for statistic in ("mean", "se", "worst")
    for feature in BASE_FEATURES
]

MODEL_SPECS = {
    "Logistic Regression": "logistic_regression",
    "Decision Tree": "decision_tree",
    "K-Nearest Neighbors": "knn",
    "Gaussian Naive Bayes": "gaussian_nb",
    "Support Vector Machine": "support_vector_machine",
}


def validate_test_data(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a clean copy after checking the expected CSV schema."""
    missing = [column for column in FEATURE_COLUMNS + [TARGET_COLUMN] if column not in frame.columns]
    if missing:
        raise ValueError("Missing required column(s): " + ", ".join(missing))

    clean = frame.loc[:, FEATURE_COLUMNS + [TARGET_COLUMN]].copy()
    for column in FEATURE_COLUMNS:
        clean[column] = pd.to_numeric(clean[column], errors="coerce")
    if clean[FEATURE_COLUMNS].isna().any().any():
        raise ValueError("Feature columns must contain numeric values and no blank cells.")
    if clean[TARGET_COLUMN].isna().any():
        raise ValueError("The diagnosis target column contains blank values.")

    clean[TARGET_COLUMN] = clean[TARGET_COLUMN].astype(str).str.strip().str.upper()
    unexpected = sorted(set(clean[TARGET_COLUMN]) - set(CLASS_LABELS))
    if unexpected:
        raise ValueError("Diagnosis values must be B or M; found: " + ", ".join(unexpected))
    if len(clean) < 2:
        raise ValueError("Upload at least two labeled rows to calculate evaluation metrics.")
    if clean[TARGET_COLUMN].nunique() < 2:
        raise ValueError("The uploaded file must include both diagnosis classes (B and M) to calculate AUC.")
    return clean


def positive_class_scores(model: Any, features: pd.DataFrame) -> np.ndarray:
    """Get scores for the malignant class, regardless of estimator class order."""
    classes = list(model.classes_)
    positive_index = classes.index(POSITIVE_LABEL)
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(features))[:, positive_index]
    decision = np.asarray(model.decision_function(features))
    return decision if positive_index == 1 else -decision


def evaluate_predictions(
    truth: pd.Series,
    predictions: np.ndarray,
    positive_scores: np.ndarray,
) -> dict[str, float]:
    """Calculate the six assignment metrics with malignant as the positive class."""
    y_binary = (truth == POSITIVE_LABEL).astype(int)
    return {
        "Accuracy": float(accuracy_score(truth, predictions)),
        "AUC": float(roc_auc_score(y_binary, positive_scores)),
        "Precision": float(precision_score(truth, predictions, pos_label=POSITIVE_LABEL, zero_division=0)),
        "Recall": float(recall_score(truth, predictions, pos_label=POSITIVE_LABEL, zero_division=0)),
        "F1": float(f1_score(truth, predictions, pos_label=POSITIVE_LABEL, zero_division=0)),
        "MCC": float(matthews_corrcoef(truth, predictions)),
    }
