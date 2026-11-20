"""Prove the self-written NB matches Weka's NaiveBayesMultinomial on identical Weka vectors."""

import csv
from pathlib import Path

import numpy as np

from .data import write_json
from .models.naive_bayes import MultinomialNaiveBayes
from .tree_workflow import classification_metrics
from .weka_io import CLASSES, read_vector_arff


def _label(cell: str) -> int:
    # Weka writes nominal predictions as "<index>:<value>", e.g. "2:spam".
    return CLASSES[cell.split(":", 1)[-1].strip()]


def read_weka_predictions(csv_path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with open(csv_path, encoding="utf-8", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row.get("inst#", "").strip()]
    if not rows:
        raise ValueError("No predictions found in Weka CSV")
    rows.sort(key=lambda row: int(row["inst#"]))
    actual = np.array([_label(r["actual"]) for r in rows], dtype=np.int64)
    predicted = np.array([_label(r["predicted"]) for r in rows], dtype=np.int64)
    confidence = np.array([float(r["prediction"]) for r in rows], dtype=np.float64)
    return actual, predicted, confidence


def compare_with_weka(train_vec: str | Path, test_vec: str | Path, weka_predictions_csv: str | Path,
                      output_dir: str | Path) -> dict:
    X_train, y_train, train_names = read_vector_arff(train_vec)
    X_test, y_test, test_names = read_vector_arff(test_vec)
    if train_names != test_names:
        raise ValueError("Train and test vector attributes differ")
    model = MultinomialNaiveBayes().fit(X_train, y_train)
    proba = model.predict_proba(X_test)
    predicted = model.predict(X_test)
    actual, weka_predicted, weka_confidence = read_weka_predictions(weka_predictions_csv)
    if len(actual) != len(y_test) or not np.array_equal(actual, y_test):
        raise ValueError("Weka predictions do not align with the test vectors (count or labels differ)")
    python_confidence = proba[np.arange(len(predicted)), predicted]
    disagreements = (np.flatnonzero(predicted != weka_predicted) + 1).tolist()
    result = {
        "n_test": int(len(y_test)), "n_features": len(train_names),
        "agreement": int(np.sum(predicted == weka_predicted)), "disagreements": disagreements,
        # Weka prints the predicted-class probability rounded to 3 decimals.
        "max_abs_proba_diff": float(np.max(np.abs(python_confidence - weka_confidence))),
        "python_metrics": classification_metrics(y_test, predicted),
        "weka_metrics": classification_metrics(actual, weka_predicted),
    }
    write_json(Path(output_dir) / "nb_python_vs_weka.json", result)
    return result
