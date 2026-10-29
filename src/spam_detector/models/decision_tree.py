"""CART-style greedy Gini splits for dense binary word-presence features.

All model learning and prediction are implemented here using NumPy only.
Feature ties prefer the lowest column index; class ties prefer ham (0).
Zero-gain splits are not accepted, so a pure XOR cannot be learned at its root.
"""

from dataclasses import asdict, dataclass
from numbers import Integral
from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray

from .base import NotFittedError


def validate_parameters(
    max_depth: int | None, min_samples_split: int, min_samples_leaf: int
) -> None:
    for name, value, lower in (
        ("max_depth", max_depth, 0),
        ("min_samples_split", min_samples_split, 2),
        ("min_samples_leaf", min_samples_leaf, 1),
    ):
        if name == "max_depth" and value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, Integral) or value < lower:
            raise ValueError(f"{name} must be an integer >= {lower}")


def binary_matrix(X: Any, n_features: int | None = None) -> NDArray[np.uint8]:
    """Reject continuous/sparse input rather than silently changing semantics."""
    array = np.asarray(X)
    if array.ndim != 2 or array.shape[1] == 0:
        raise ValueError("X must be a dense 2-D matrix with at least one feature")
    if array.dtype.kind not in "biuf" or not np.isfinite(array).all():
        raise ValueError("X must contain finite numeric binary values")
    if not np.all((array == 0) | (array == 1)):
        raise ValueError("X must contain only 0/1; binarize selected features upstream")
    if n_features is not None and array.shape[1] != n_features:
        raise ValueError(f"Expected {n_features} features, got {array.shape[1]}")
    return array.astype(np.uint8, copy=False)


def binary_labels(y: Any, n_samples: int, *, require_both: bool = False) -> NDArray[np.int64]:
    array = np.asarray(y)
    if array.ndim != 1 or len(array) != n_samples or n_samples == 0:
        raise ValueError("y must be a nonempty 1-D array matching X rows")
    if array.dtype.kind not in "biuf" or not np.isfinite(array).all():
        raise ValueError("y must contain finite numeric 0/1 labels")
    if not np.all((array == 0) | (array == 1)):
        raise ValueError("y must contain only 0 (ham) and 1 (spam)")
    if require_both and np.unique(array).size != 2:
        raise ValueError("Training requires both classes 0 and 1")
    return array.astype(np.int64, copy=False)


def gini(spam_count: int, n_samples: int) -> float:
    """Binary impurity; callers pass a nonempty population."""
    p = spam_count / n_samples
    return 2.0 * p * (1.0 - p)


@dataclass(frozen=True)
class Node:
    n_samples: int
    spam_count: int
    depth: int
    feature: int | None = None
    left: int | None = None
    right: int | None = None

    @property
    def prediction(self) -> int:
        return int(2 * self.spam_count > self.n_samples)


class DecisionTreeClassifier:
    """Binary classifier; max_depth=0 intentionally creates a root-only tree."""

    def __init__(
        self, *, max_depth: int | None = 10,
        min_samples_split: int = 10, min_samples_leaf: int = 5,
    ) -> None:
        validate_parameters(max_depth, min_samples_split, min_samples_leaf)
        self.max_depth = None if max_depth is None else int(max_depth)
        self.min_samples_split = int(min_samples_split)
        self.min_samples_leaf = int(min_samples_leaf)

    def get_params(self) -> dict[str, int | None]:
        return {
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
            "min_samples_leaf": self.min_samples_leaf,
        }

    def fit(self, X: Any, y: Any) -> "DecisionTreeClassifier":
        validate_parameters(**self.get_params())
        matrix = binary_matrix(X)
        labels = binary_labels(y, len(matrix), require_both=True)
        nodes = [Node(len(labels), int(labels.sum()), 0)]
        pending = [(0, np.arange(len(labels)))]
        # Iterative construction avoids Python recursion limits for deep trees.
        while pending:
            node_id, indices = pending.pop()
            node = nodes[node_id]
            if (node.spam_count in (0, node.n_samples)
                or node.n_samples < self.min_samples_split
                or (self.max_depth is not None and node.depth >= self.max_depth)):
                continue
            local_X, local_y = matrix[indices], labels[indices]
            right_n = local_X.sum(axis=0, dtype=np.int64)
            right_spam = local_X[local_y == 1].sum(axis=0, dtype=np.int64)
            left_n = node.n_samples - right_n
            left_spam = node.spam_count - right_spam
            valid = (left_n >= self.min_samples_leaf) & (right_n >= self.min_samples_leaf)
            columns = np.flatnonzero(valid)
            if not len(columns):
                continue
            p_left = left_spam[columns] / left_n[columns]
            p_right = right_spam[columns] / right_n[columns]
            impurity = (
                left_n[columns] * 2 * p_left * (1 - p_left)
                + right_n[columns] * 2 * p_right * (1 - p_right)
            ) / node.n_samples
            best = int(np.argmin(impurity))
            if gini(node.spam_count, node.n_samples) - impurity[best] <= 1e-12:
                continue
            feature = int(columns[best])
            right_mask = local_X[:, feature] == 1
            left_id, right_id = len(nodes), len(nodes) + 1
            nodes.extend([
                Node(int(left_n[feature]), int(left_spam[feature]), node.depth + 1),
                Node(int(right_n[feature]), int(right_spam[feature]), node.depth + 1),
            ])
            nodes[node_id] = Node(
                node.n_samples, node.spam_count, node.depth, feature, left_id, right_id
            )
            pending.extend([(right_id, indices[right_mask]), (left_id, indices[~right_mask])])
        # Publish a fitted state only after successful training.
        self.nodes_ = tuple(nodes)
        self.n_features_in_ = matrix.shape[1]
        self.classes_ = np.array([0, 1], dtype=np.int64)
        self.depth_ = max(node.depth for node in nodes)
        self.n_leaves_ = sum(node.feature is None for node in nodes)
        return self

    def _check_fitted(self) -> None:
        if not hasattr(self, "nodes_"):
            raise NotFittedError("Call fit before prediction or export")

    def _leaf_ids(self, X: Any) -> NDArray[np.int64]:
        self._check_fitted()
        matrix = binary_matrix(X, self.n_features_in_)
        result = np.empty(len(matrix), dtype=np.int64)
        pending = [(0, np.arange(len(matrix)))]
        while pending:
            node_id, rows = pending.pop()
            if not len(rows):
                continue
            node = self.nodes_[node_id]
            if node.feature is None:
                result[rows] = node_id
            else:
                mask = matrix[rows, node.feature] == 1
                pending.extend([(node.left, rows[~mask]), (node.right, rows[mask])])
        return result

    def predict(self, X: Any) -> NDArray[np.int64]:
        leaves = self._leaf_ids(X)
        return np.array([self.nodes_[i].prediction for i in leaves], dtype=np.int64)

    def predict_proba(self, X: Any) -> NDArray[np.float64]:
        leaves = self._leaf_ids(X)
        p = np.array([self.nodes_[i].spam_count / self.nodes_[i].n_samples for i in leaves])
        return np.column_stack((1 - p, p))

    def export_rules(self, feature_names: Sequence[str]) -> list[dict[str, Any]]:
        self._check_fitted()
        names = list(feature_names)
        if len(names) != self.n_features_in_ or any(not isinstance(s, str) for s in names):
            raise ValueError("feature_names must contain one string per feature")
        rules = []
        pending: list[tuple[int, list[dict[str, Any]]]] = [(0, [])]
        while pending:
            node_id, conditions = pending.pop()
            node = self.nodes_[node_id]
            if node.feature is None:
                rules.append({
                    "conditions": conditions, "prediction": node.prediction,
                    "n_samples": node.n_samples, "spam_count": node.spam_count,
                    "spam_fraction": node.spam_count / node.n_samples,
                })
            else:
                for child, present in ((node.right, True), (node.left, False)):
                    condition = {"feature_index": node.feature, "word": names[node.feature], "present": present}
                    pending.append((child, conditions + [condition]))
        return rules

    def to_dict(self) -> dict[str, Any]:
        self._check_fitted()
        return {
            "format_version": 1, "model_type": "binary_decision_tree",
            "parameters": self.get_params(), "n_features": self.n_features_in_,
            "nodes": [asdict(node) for node in self.nodes_],
        }

    @classmethod
    def from_dict(cls, state: dict[str, Any]) -> "DecisionTreeClassifier":
        """Validate the entire graph before exposing a loaded model."""
        try:
            if state["format_version"] != 1 or state["model_type"] != "binary_decision_tree":
                raise ValueError("Unsupported model format")
            model = cls(**state["parameters"])
            width = state["n_features"]
            if type(width) is not int or width < 1:
                raise ValueError("Invalid feature count")
            nodes = tuple(Node(**item) for item in state["nodes"])
            if not nodes or nodes[0].depth != 0:
                raise ValueError("Invalid root")
            seen: set[int] = set()
            pending = [0]
            while pending:
                index = pending.pop()
                if type(index) is not int or not 0 <= index < len(nodes) or index in seen:
                    raise ValueError("Invalid or repeated child node")
                seen.add(index)
                node = nodes[index]
                if (any(type(x) is not int for x in (node.n_samples, node.spam_count, node.depth))
                    or node.n_samples < 1 or not 0 <= node.spam_count <= node.n_samples
                    or node.depth < 0
                    or (model.max_depth is not None and node.depth > model.max_depth)):
                    raise ValueError("Invalid node population or depth")
                if node.feature is None:
                    if node.left is not None or node.right is not None:
                        raise ValueError("Leaf has children")
                    continue
                if type(node.feature) is not int or not 0 <= node.feature < width:
                    raise ValueError("Invalid split feature")
                children = (node.left, node.right)
                if any(type(i) is not int or not 0 <= i < len(nodes) for i in children):
                    raise ValueError("Invalid child index")
                left, right = (nodes[i] for i in children)
                if (left.n_samples + right.n_samples != node.n_samples
                    or left.spam_count + right.spam_count != node.spam_count
                    or any(child.depth != node.depth + 1 for child in (left, right))
                    or min(left.n_samples, right.n_samples) < model.min_samples_leaf
                    or node.n_samples < model.min_samples_split):
                    raise ValueError("Inconsistent children")
                pending.extend(children)
            if len(seen) != len(nodes) or not 0 < nodes[0].spam_count < nodes[0].n_samples:
                raise ValueError("Unreachable nodes or missing training class")
            model.nodes_, model.n_features_in_ = nodes, width
            model.classes_ = np.array([0, 1], dtype=np.int64)
            model.depth_ = max(node.depth for node in nodes)
            model.n_leaves_ = sum(node.feature is None for node in nodes)
            return model
        except (KeyError, TypeError, AttributeError) as exc:
            raise ValueError("Malformed decision tree state") from exc
