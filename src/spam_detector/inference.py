"""Text prediction using an immutable, checksummed Decision Tree release."""

import json
from pathlib import Path
from typing import Sequence

import numpy as np

from .artifacts import load_tree_bundle
from .data import sha256
from .features import BinaryTextFeatures
from .preprocessing import PREPROCESSING_VERSION, preprocess


class TextPredictor:
    def __init__(self, release_dir: str | Path):
        root = Path(release_dir)
        manifest = json.loads((root / "release.json").read_text(encoding="utf-8"))
        if manifest.get("format_version") != 1 or manifest.get("preprocessing_version") != PREPROCESSING_VERSION:
            raise ValueError("Unsupported release/preprocessing version")
        for name in ("tree.json", "features.json"):
            if sha256(root / name) != manifest["checksums"][name]:
                raise ValueError(f"Release checksum mismatch: {name}")
        self.bundle = load_tree_bundle(root / "tree.json")
        self.features = BinaryTextFeatures.load(root / "features.json")
        if self.bundle.feature_names != tuple(self.features.get_feature_names()):
            raise ValueError("Release model and feature order do not match")
        self.manifest = manifest

    def predict_texts(self, texts: Sequence[str]) -> list[dict]:
        if isinstance(texts, str) or any(not isinstance(t, str) or not t.strip() for t in texts):
            raise ValueError("Supply a sequence of nonempty email strings")
        X = self.features.transform(texts)
        labels, probabilities = self.bundle.model.predict(X), self.bundle.model.predict_proba(X)
        vocabulary = set(self.features.get_feature_names())
        results = []
        for text, row, label, prob in zip(texts, X, labels, probabilities):
            tokens = set(preprocess(text).split())
            matched = sorted(tokens & vocabulary)
            # Follow exactly the fitted tree for a compact explanation.
            index, conditions = 0, []
            while self.bundle.model.nodes_[index].feature is not None:
                node = self.bundle.model.nodes_[index]
                present = bool(row[node.feature])
                conditions.append({"word": self.bundle.feature_names[node.feature], "present": present})
                index = node.right if present else node.left
            leaf = self.bundle.model.nodes_[index]
            results.append({
                "label": int(label), "class_name": "spam" if label else "ham",
                "spam_fraction": float(prob[1]), "score_kind": "uncalibrated_leaf_fraction",
                "matched_features": len(matched), "unique_tokens": len(tokens),
                "selected_feature_coverage": len(matched) / len(tokens) if tokens else 0.0,
                "warning": "No selected vocabulary matches; prediction uses learned absence branches" if not matched else None,
                "path": conditions, "leaf_training_samples": leaf.n_samples,
            })
        return results


def load_predictor(release_dir: str | Path) -> TextPredictor:
    return TextPredictor(release_dir)
