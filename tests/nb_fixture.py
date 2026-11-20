"""Tiny NB release + web config for tests that must not depend on Weka-produced artifacts."""

from pathlib import Path

import yaml

from spam_detector.features import BinaryTextFeatures
from spam_detector.models.naive_bayes import MultinomialNaiveBayes
from spam_detector.nb_release import save_nb_release

ROOT = Path(__file__).resolve().parents[1]
TEXTS = ["win free money now", "free prize claim money", "cheap pills free offer",
         "meeting schedule for monday", "please review the gas contract", "monday meeting notes attached"]
LABELS = [1, 1, 1, 0, 0, 0]


def make_nb_config(tmp_path: Path) -> Path:
    features = BinaryTextFeatures(k=20, min_df=1).fit(TEXTS, LABELS)
    model = MultinomialNaiveBayes().fit(features.transform(TEXTS), LABELS)
    release = tmp_path / "nb-release"
    save_nb_release(release, model, features, {"note": "test"})
    raw = yaml.safe_load((ROOT / "configs/web.yaml").read_text(encoding="utf-8"))
    raw["default_model"] = "nb"
    raw["models"]["nb"]["release"] = str(release)
    raw["models"]["tree"]["release"] = str(ROOT / "artifacts/tree-enron-v1")
    path = tmp_path / "web.yaml"
    path.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")
    return path
