"""Train-only binary vocabulary and deterministic Chi-square word selection."""

import json
from pathlib import Path
from typing import Sequence

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import chi2

from .data import write_json
from .preprocessing import PREPROCESSING_VERSION, preprocess


class BinaryTextFeatures:
    def __init__(self, k: int = 500, min_df: int = 2):
        if type(k) is not int or k < 1 or type(min_df) is not int or min_df < 1:
            raise ValueError("k and min_df must be positive integers")
        self.k, self.min_df = k, min_df

    @staticmethod
    def _vectorizer(**kwargs):
        return CountVectorizer(binary=True, lowercase=False, token_pattern=r"(?u)\b[a-z]{2,}\b", **kwargs)

    def fit(self, texts: Sequence[str], y) -> "BinaryTextFeatures":
        labels = np.asarray(y)
        if labels.shape != (len(texts),) or set(labels.tolist()) != {0, 1}:
            raise ValueError("Feature fitting requires matching labels with both classes")
        vectorizer = self._vectorizer(min_df=self.min_df)
        matrix = vectorizer.fit_transform([preprocess(text) for text in texts])
        names = vectorizer.get_feature_names_out()
        scores, _ = chi2(matrix, labels)
        scores = np.nan_to_num(scores, nan=0.0, posinf=0.0)
        selected = np.argsort(-scores, kind="stable")[:min(self.k, len(names))]
        # Stable lexical column order, while membership is determined by Chi-square.
        selected = np.sort(selected)
        self.feature_names_ = tuple(str(word) for word in names[selected])
        self.chi2_scores_ = tuple(float(score) for score in scores[selected])
        self.vocabulary_size_ = len(names)
        self._transformer = self._vectorizer(vocabulary={word: i for i, word in enumerate(self.feature_names_)})
        return self

    def transform(self, texts: Sequence[str]) -> np.ndarray:
        if not hasattr(self, "_transformer"):
            raise ValueError("Fit the feature pipeline before transform")
        if not len(texts):
            return np.empty((0, len(self.feature_names_)), dtype=np.uint8)
        matrix = self._transformer.transform([preprocess(text) for text in texts])
        return (matrix > 0).astype(np.uint8).toarray()

    def get_feature_names(self) -> list[str]:
        if not hasattr(self, "feature_names_"):
            raise ValueError("Features not fitted")
        return list(self.feature_names_)

    def save(self, path: str | Path) -> None:
        write_json(Path(path), {
            "format_version": 1, "preprocessing_version": PREPROCESSING_VERSION,
            "k": self.k, "min_df": self.min_df,
            "feature_names": self.get_feature_names(), "chi2_scores": self.chi2_scores_,
            "vocabulary_size": self.vocabulary_size_,
            "preprocessing": {"stemming": False, "stopword_removal": False, "ngram": "unigram"},
        })

    @classmethod
    def load(cls, path: str | Path) -> "BinaryTextFeatures":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        if raw.get("format_version") != 1 or raw.get("preprocessing_version") != PREPROCESSING_VERSION:
            raise ValueError("Unsupported feature/preprocessing format")
        model = cls(raw["k"], raw["min_df"])
        names = raw["feature_names"]
        if (not isinstance(names, list) or not names or len(names) > model.k
            or any(not isinstance(word, str) or not word for word in names)
            or len(set(names)) != len(names)):
            raise ValueError("Invalid saved feature vocabulary")
        model.feature_names_ = tuple(names)
        model.chi2_scores_ = tuple(raw["chi2_scores"])
        model.vocabulary_size_ = raw["vocabulary_size"]
        model._transformer = model._vectorizer(vocabulary={word: i for i, word in enumerate(names)})
        return model


def build_features(records: list[dict], output: str | Path, k: int) -> BinaryTextFeatures:
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=False)
    train = [r for r in records if r["split"] == "train"]
    features = BinaryTextFeatures(k).fit([r["text"] for r in train], [r["label"] for r in train])
    features.save(destination / "features.json")
    development = {}
    for split in ("train", "validation", "test", "external_test"):
        rows = [r for r in records if r["split"] == split]
        if not rows:
            raise ValueError(f"No records in {split}")
        archive = {
            "feature_names": np.array(features.get_feature_names()),
            f"X_{split}": features.transform([r["text"] for r in rows]),
            f"y_{split}": np.array([r["label"] for r in rows]),
            f"ids_{split}": np.array([r["sample_id"] for r in rows]),
            f"groups_{split}": np.array([r["group_id"] for r in rows]),
        }
        if split in ("train", "validation"):
            development.update(archive)
        else:
            np.savez_compressed(destination / f"{split}.npz", **archive)
    np.savez_compressed(destination / "development.npz", **development)
    return features
