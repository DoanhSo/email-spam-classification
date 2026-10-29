"""Decision Tree handoff workflow; consumes precomputed, train-fitted features.

This does not replace the teammates' text preprocessing, feature selection,
shared evaluation or ensemble modules.
"""

import csv
import hashlib
import json
import platform
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from .artifacts import load_tree_bundle, save_tree_bundle
from .config import ProjectConfig
from .models.decision_tree import DecisionTreeClassifier, binary_labels, binary_matrix


@dataclass(frozen=True)
class MatrixSplit:
    X: np.ndarray
    y: np.ndarray
    sample_ids: tuple[str, ...]
    group_ids: tuple[str, ...]


def _strings(values: Any, size: int, name: str, *, unique: bool) -> tuple[str, ...]:
    array = np.asarray(values)
    if array.ndim != 1 or len(array) != size or array.dtype.kind not in "US":
        raise ValueError(f"{name} must contain {size} strings")
    result = tuple(str(value) for value in array)
    if any(not s.strip() for s in result) or (unique and len(set(result)) != size):
        raise ValueError(f"{name} must be nonempty{' and unique' if unique else ''}")
    return result


def load_matrices(path: str | Path, splits: tuple[str, ...]) -> tuple[tuple[str, ...], dict[str, MatrixSplit]]:
    """NPZ contract includes IDs/groups; never reads an unrequested test split."""
    result = {}
    archive = np.load(path, allow_pickle=False)
    if not isinstance(archive, np.lib.npyio.NpzFile):
        raise ValueError("Expected an NPZ archive, not an individual NPY array")
    with archive:
        try:
            raw_names = archive["feature_names"]
            if raw_names.ndim != 1:
                raise ValueError("feature_names must be a 1-D string array")
            names = _strings(raw_names, len(raw_names), "feature_names", unique=True)
            if not names:
                raise ValueError("At least one feature is required")
            for split in splits:
                X = binary_matrix(archive[f"X_{split}"], len(names))
                y = binary_labels(archive[f"y_{split}"], len(X), require_both=split == "train")
                ids = _strings(archive[f"ids_{split}"], len(X), "sample IDs", unique=True)
                groups = _strings(archive[f"groups_{split}"], len(X), "group IDs", unique=False)
                result[split] = MatrixSplit(X, y, ids, groups)
        except KeyError as exc:
            raise ValueError(f"Missing NPZ field: {exc}") from exc
    for position, first in enumerate(splits):
        for second in splits[position + 1:]:
            if set(result[first].sample_ids) & set(result[second].sample_ids):
                raise ValueError(f"Overlapping sample IDs: {first}/{second}")
            if set(result[first].group_ids) & set(result[second].group_ids):
                raise ValueError(f"Overlapping duplicate groups: {first}/{second}")
    return names, result


def classification_metrics(y: Any, prediction: Any) -> dict[str, int | float]:
    """Minimal tree-specific metrics, to be integrated with Dương's evaluator."""
    truth = np.asarray(y)
    binary_labels(truth, len(truth))
    pred = binary_labels(prediction, len(truth))
    tn = int(np.sum((truth == 0) & (pred == 0)))
    fp = int(np.sum((truth == 0) & (pred == 1)))
    fn = int(np.sum((truth == 1) & (pred == 0)))
    tp = int(np.sum((truth == 1) & (pred == 1)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "n_samples": len(truth), "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "accuracy": (tn + tp) / len(truth), "precision_spam": precision,
        "recall_spam": recall,
        "f1_spam": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
    }


def _json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def _csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def train_tree(
    data_path: str | Path, config: ProjectConfig, output: str | Path, *,
    benchmark: bool = False, depths: list[int] | None = None,
    leaf_sizes: list[int] | None = None,
) -> dict[str, Any]:
    """Select on validation only; output directories must not already exist."""
    names, splits = load_matrices(data_path, ("train", "validation"))
    train, validation = splits["train"], splits["validation"]
    if len(names) > config.select_k:
        raise ValueError("More feature columns than configured select_k; check the handoff config")
    candidates = [
        {**config.tree, "max_depth": depth, "min_samples_leaf": leaf}
        for depth in (depths if depths is not None else [config.tree["max_depth"]])
        for leaf in (leaf_sizes if leaf_sizes is not None else [config.tree["min_samples_leaf"]])
    ]
    if not candidates:
        raise ValueError("At least one candidate is required")
    for params in candidates:
        DecisionTreeClassifier(**params)
    reference_class = None
    if benchmark:
        # The sole production-source import of a library tree is this opt-in benchmark.
        from sklearn.tree import DecisionTreeClassifier as ReferenceTree

        reference_class = ReferenceTree
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=False)
    scores = []
    best = None
    for params in candidates:
        started = perf_counter()
        model = DecisionTreeClassifier(**params).fit(train.X, train.y)
        fit_seconds = perf_counter() - started
        started = perf_counter()
        prediction = model.predict(validation.X)
        predict_seconds = perf_counter() - started
        metrics = classification_metrics(validation.y, prediction)
        scores.append({**params, **metrics, "fit_seconds": fit_seconds, "predict_seconds": predict_seconds})
        rank = (metrics["f1_spam"], metrics["precision_spam"], -model.depth_, -len(model.nodes_))
        if best is None or rank > best[0]:
            best = (rank, model, prediction, metrics, fit_seconds, predict_seconds)
    _, model, prediction, metrics, fit_seconds, predict_seconds = best
    fingerprint = hashlib.sha256(Path(data_path).read_bytes()).hexdigest()
    summary = {
        "model": "self_written_binary_tree", "split": "validation",
        "data_sha256": fingerprint, "parameters": model.get_params(), "metrics": metrics,
        "depth": model.depth_, "nodes": len(model.nodes_), "leaves": model.n_leaves_,
        "fit_seconds": fit_seconds, "predict_seconds": predict_seconds,
        "note": "Input features must already be fit on train only. No text transformer is included.",
    }
    save_tree_bundle(destination / "tree.json", model, names, {
        "data_sha256": fingerprint, "seed": config.seed,
        "train_sample_ids": train.sample_ids, "train_group_ids": train.group_ids,
        "validation_sample_ids": validation.sample_ids, "validation_group_ids": validation.group_ids,
        "feature_origin": "external_train_fitted_matrix",
    })
    if reference_class is not None:
        params = model.get_params()
        # Our depth=0 is a stump; sklearn requires depth>=1. Disable splitting instead.
        if params["max_depth"] == 0:
            params = {**params, "max_depth": 1, "min_samples_split": len(train.y) + 1}
        reference = reference_class(criterion="gini", random_state=config.seed, **params)
        started = perf_counter()
        reference.fit(train.X, train.y)
        reference_fit = perf_counter() - started
        ref_pred = reference.predict(validation.X)
        _json(destination / "benchmark.json", {
            "split": "validation", "sklearn_version": version("scikit-learn"),
            "self_written": metrics, "sklearn": classification_metrics(validation.y, ref_pred),
            "prediction_agreement": float(np.mean(prediction == ref_pred)),
            "sklearn_fit_seconds": reference_fit,
            "note": "Feature tie-breaking and zero-gain split policy may differ.",
        })
    probabilities = model.predict_proba(validation.X)[:, 1]
    _csv(destination / "predictions.validation.csv", [
        {"sample_id": sid, "y_true": int(y), "y_pred": int(p), "spam_fraction": float(prob)}
        for sid, y, p, prob in zip(validation.sample_ids, validation.y, prediction, probabilities)
    ])
    _csv(destination / "candidates.validation.csv", scores)
    _json(destination / "summary.json", summary)
    _json(destination / "rules.json", model.export_rules(names))
    _json(destination / "config.resolved.json", asdict(config))
    _json(destination / "environment.json", {
        "python": platform.python_version(), "numpy": version("numpy"), "PyYAML": version("PyYAML"),
    })
    return summary


def evaluate_tree(bundle_path: str | Path, data_path: str | Path, split: str) -> dict[str, Any]:
    bundle = load_tree_bundle(bundle_path)
    names, splits = load_matrices(data_path, (split,))
    if names != bundle.feature_names:
        raise ValueError("Feature names/order differ from the training bundle")
    batch = splits[split]
    if split in ("test", "external_test"):
        for origin in ("train", "validation"):
            if set(batch.sample_ids) & set(bundle.metadata.get(f"{origin}_sample_ids", [])):
                raise ValueError(f"Evaluation sample IDs overlap {origin}")
            if set(batch.group_ids) & set(bundle.metadata.get(f"{origin}_group_ids", [])):
                raise ValueError(f"Evaluation groups overlap {origin}")
    return {
        "split": split, "metrics": classification_metrics(batch.y, bundle.model.predict(batch.X)),
        "data_sha256": hashlib.sha256(Path(data_path).read_bytes()).hexdigest(),
    }


def create_demo_archive(path: str | Path) -> None:
    """Synthetic binary features, not Enron or measured spam-filter quality."""
    rng = np.random.default_rng(42)
    data: dict[str, Any] = {"feature_names": np.array(["free", "url", "meeting", "money"])}
    for split, size in (("train", 120), ("validation", 40), ("test", 40)):
        X = rng.integers(0, 2, size=(size, 4), dtype=np.uint8)
        y = ((X[:, 0] == 1) & ((X[:, 1] == 1) | (X[:, 3] == 1))).astype(np.int64)
        data.update({
            f"X_{split}": X, f"y_{split}": y,
            f"ids_{split}": np.array([f"synthetic-{split}-{i}" for i in range(size)]),
            # These are independent synthetic numeric records, not email text.
            # Equal binary vectors do not imply duplicate emails.
            f"groups_{split}": np.array([f"synthetic-record-{split}-{i}" for i in range(size)]),
        })
    with Path(path).open("xb") as stream:
        np.savez_compressed(stream, **data)
