"""Multinomial Naive Bayes on word counts, NumPy only.

Smoothing matches Weka's NaiveBayesMultinomial so results can be compared exactly:
P(c) = (N_c + 1) / (N + 2) and P(w|c) = (count_wc + alpha) / (sum_w count_wc + alpha * V).
Scores are computed in log space (no multinomial coefficient) and normalised with log-sum-exp.
Class ties prefer ham (0).
"""

from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray

from .base import NotFittedError
from .decision_tree import binary_labels


def count_matrix(X: Any, n_features: int | None = None) -> NDArray[np.float64]:
    array = np.asarray(X)
    if array.dtype.kind not in "biuf" or array.ndim != 2 or array.shape[1] == 0:
        raise ValueError("X must be a dense 2-D numeric matrix with at least one feature")
    array = array.astype(np.float64, copy=False)
    if not np.isfinite(array).all() or (array < 0).any():
        raise ValueError("X must contain finite, non-negative counts")
    if n_features is not None and array.shape[1] != n_features:
        raise ValueError(f"Expected {n_features} features, got {array.shape[1]}")
    return array


class MultinomialNaiveBayes:
    def __init__(self, alpha: float = 1.0):
        if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not alpha > 0:
            raise ValueError("alpha must be a positive number")
        self.alpha = float(alpha)

    def fit(self, X: Any, y: Any) -> "MultinomialNaiveBayes":
        counts = count_matrix(X)
        labels = binary_labels(y, counts.shape[0], require_both=True)
        class_counts = np.bincount(labels, minlength=2).astype(np.float64)
        self.class_log_prior_ = np.log((class_counts + 1) / (class_counts.sum() + 2))
        word_counts = np.vstack([counts[labels == c].sum(axis=0) for c in (0, 1)])
        denominators = word_counts.sum(axis=1, keepdims=True) + self.alpha * counts.shape[1]
        self.feature_log_prob_ = np.log((word_counts + self.alpha) / denominators)
        return self

    def _check_fitted(self) -> None:
        if not hasattr(self, "feature_log_prob_"):
            raise NotFittedError("Fit the model before prediction")

    def predict_proba(self, X: Any) -> NDArray[np.float64]:
        self._check_fitted()
        counts = count_matrix(X, self.feature_log_prob_.shape[1])
        scores = counts @ self.feature_log_prob_.T + self.class_log_prior_
        scores -= scores.max(axis=1, keepdims=True)
        probabilities = np.exp(scores)
        return probabilities / probabilities.sum(axis=1, keepdims=True)

    def predict(self, X: Any) -> NDArray[np.int64]:
        return (self.predict_proba(X)[:, 1] > 0.5).astype(np.int64)

    def explain(self, x_row: Any, feature_names: Sequence[str], n: int = 5) -> list[tuple[str, float]]:
        self._check_fitted()
        row = count_matrix([x_row], self.feature_log_prob_.shape[1])[0]
        if len(feature_names) != row.size:
            raise ValueError("feature_names must match the number of features")
        contribution = row * (self.feature_log_prob_[1] - self.feature_log_prob_[0])
        present = np.flatnonzero(row > 0)
        order = present[np.argsort(-contribution[present], kind="stable")]
        return [(str(feature_names[i]), float(contribution[i])) for i in order[:n]]

    def to_dict(self) -> dict[str, Any]:
        self._check_fitted()
        return {
            "alpha": self.alpha,
            "class_log_prior": self.class_log_prior_.tolist(),
            "feature_log_prob": self.feature_log_prob_.tolist(),
        }

    @classmethod
    def from_dict(cls, state: dict[str, Any]) -> "MultinomialNaiveBayes":
        try:
            model = cls(state["alpha"])
            prior = np.asarray(state["class_log_prior"], dtype=np.float64)
            table = np.asarray(state["feature_log_prob"], dtype=np.float64)
        except (KeyError, TypeError) as exc:
            raise ValueError("Malformed Naive Bayes state") from exc
        if prior.shape != (2,) or table.ndim != 2 or table.shape[0] != 2 or not np.isfinite(table).all():
            raise ValueError("Malformed Naive Bayes state")
        model.class_log_prior_, model.feature_log_prob_ = prior, table
        return model
