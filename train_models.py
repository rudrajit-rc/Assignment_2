"""Fit and save the five required classifiers, then report fixed-test metrics."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

from model_utils import (
    FEATURE_COLUMNS,
    MODEL_SPECS,
    PROJECT_DIR,
    TARGET_COLUMN,
    evaluate_predictions,
    positive_class_scores,
    validate_test_data,
)


def build_models() -> dict[str, object]:
    """Create comparable, reproducible model pipelines."""
    return {
        "Logistic Regression": Pipeline(
            [
                ("scale", StandardScaler()),
                ("classifier", LogisticRegression(C=1.0, max_iter=3000, random_state=42)),
            ]
        ),
        "Decision Tree": DecisionTreeClassifier(
            criterion="gini", max_depth=5, min_samples_leaf=2, random_state=42
        ),
        "K-Nearest Neighbors": Pipeline(
            [
                ("scale", StandardScaler()),
                ("classifier", KNeighborsClassifier(n_neighbors=7, weights="uniform")),
            ]
        ),
        "Gaussian Naive Bayes": GaussianNB(),
        "Support Vector Machine": Pipeline(
            [
                ("scale", StandardScaler()),
                ("classifier", SVC(C=1.0, kernel="rbf", probability=True, random_state=42)),
            ]
        ),
    }


def main() -> None:
    train_path = PROJECT_DIR / "train_data.csv"
    test_path = PROJECT_DIR / "test_data.csv"
    model_dir = PROJECT_DIR / "model"
    model_dir.mkdir(exist_ok=True)

    train = pd.read_csv(train_path)
    test = validate_test_data(pd.read_csv(test_path))
    missing_train = [name for name in FEATURE_COLUMNS + [TARGET_COLUMN] if name not in train.columns]
    if missing_train:
        raise ValueError("Training CSV is missing required column(s): " + ", ".join(missing_train))
    train = train.loc[:, FEATURE_COLUMNS + [TARGET_COLUMN]].copy()
    for column in FEATURE_COLUMNS:
        train[column] = pd.to_numeric(train[column], errors="raise")
    train[TARGET_COLUMN] = train[TARGET_COLUMN].astype(str).str.strip().str.upper()
    if set(train[TARGET_COLUMN]) != {"B", "M"}:
        raise ValueError("Training data must include B and M diagnosis values.")

    X_train, y_train = train[FEATURE_COLUMNS], train[TARGET_COLUMN]
    X_test, y_test = test[FEATURE_COLUMNS], test[TARGET_COLUMN]
    scores = []

    for display_name, model in build_models().items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        probabilities = positive_class_scores(model, X_test)
        metrics = evaluate_predictions(y_test, predictions, probabilities)
        scores.append({"Model": display_name, **metrics})

        artifact = {
            "model": model,
            "feature_columns": FEATURE_COLUMNS,
            "target_column": TARGET_COLUMN,
            "positive_label": "M",
            "display_name": display_name,
        }
        joblib.dump(artifact, model_dir / f"{MODEL_SPECS[display_name]}.joblib")

    results = pd.DataFrame(scores).set_index("Model")
    results.to_csv(PROJECT_DIR / "results.csv", float_format="%.6f")
    with (PROJECT_DIR / "model" / "model_config.json").open("w", encoding="utf-8") as stream:
        json.dump(
            {
                "positive_class": "M (malignant)",
                "negative_class": "B (benign)",
                "test_rows": len(test),
                "training_rows": len(train),
                "features": FEATURE_COLUMNS,
            },
            stream,
            indent=2,
        )

    print("Fixed holdout evaluation (malignant is the positive class)")
    print(f"Train rows: {len(train)} | Test rows: {len(test)} | Features: {len(FEATURE_COLUMNS)}")
    print(results.to_string(float_format=lambda value: f"{value:.4f}"))
    print(f"\nSaved five fitted models in: {model_dir}")


if __name__ == "__main__":
    main()
