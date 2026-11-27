"""Checksummed Naive Bayes release (model + train-fitted features) and text prediction."""

import json
from pathlib import Path
from typing import Any, Sequence

from .data import sha256, write_json
from .features import BinaryTextFeatures
from .models.naive_bayes import MultinomialNaiveBayes
from .preprocessing import PREPROCESSING_VERSION, preprocess

FILES = ("nb.json", "features.json")


def save_nb_release(release_dir: str | Path, model: MultinomialNaiveBayes, features: BinaryTextFeatures,
                    metadata: dict[str, Any] | None = None) -> None:
    root = Path(release_dir)
    names = features.get_feature_names()
    state = model.to_dict()
    if len(state["feature_log_prob"][0]) != len(names):
        raise ValueError("Model and feature vocabulary sizes differ")
    write_json(root / "nb.json", {"bundle_version": 1, "feature_names": names, "model": state})
    features.save(root / "features.json")
    write_json(root / "release.json", {
        "format_version": 1, "preprocessing_version": PREPROCESSING_VERSION,
        "model": "self_written_multinomial_nb", "metadata": metadata or {},
        "checksums": {name: sha256(root / name) for name in FILES},
    })


class NBTextPredictor:
    def __init__(self, release_dir: str | Path):
        root = Path(release_dir)
        manifest = json.loads((root / "release.json").read_text(encoding="utf-8"))
        if manifest.get("format_version") != 1 or manifest.get("preprocessing_version") != PREPROCESSING_VERSION:
            raise ValueError("Unsupported release/preprocessing version")
        for name in FILES:
            if sha256(root / name) != manifest["checksums"][name]:
                raise ValueError(f"Release checksum mismatch: {name}")
        bundle = json.loads((root / "nb.json").read_text(encoding="utf-8"))
        self.model = MultinomialNaiveBayes.from_dict(bundle["model"])
        self.features = BinaryTextFeatures.load(root / "features.json")
        self.feature_names = self.features.get_feature_names()
        if bundle["feature_names"] != self.feature_names:
            raise ValueError("Release model and feature order do not match")
        self.manifest = manifest

    def predict_texts(self, texts: Sequence[str]) -> list[dict]:
        if isinstance(texts, str) or any(not isinstance(t, str) or not t.strip() for t in texts):
            raise ValueError("Supply a sequence of nonempty email strings")
        X = self.features.transform(texts)
        probabilities = self.model.predict_proba(X)[:, 1]
        vocabulary = set(self.feature_names)
        results = []
        for text, row, spam in zip(texts, X, probabilities):
            tokens = set(preprocess(text).split())
            matched = len(tokens & vocabulary)
            label = int(spam > 0.5)
            results.append({
                "label": label, "class_name": "spam" if label else "ham",
                "score_kind": "probability_uncalibrated", "score_value": float(spam),
                "matched_features": matched, "unique_tokens": len(tokens),
                "top_words": [{"word": w, "contribution": c}
                              for w, c in self.model.explain(row, self.feature_names, label=label)],
                "warning": "No vocabulary word matches; prediction falls back to class priors" if not matched else None,
            })
        return results
