import csv
import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("bs4")
pytest.importorskip("sklearn")

from spam_detector.artifacts import save_tree_bundle
from spam_detector.data import assign_splits, clean_csv, read_records, sha256, write_csv, write_json
from spam_detector.enron_training import cross_validate
from spam_detector.features import BinaryTextFeatures
from spam_detector.inference import TextPredictor
from spam_detector.models.decision_tree import DecisionTreeClassifier
from spam_detector.preprocessing import PREPROCESSING_VERSION, preprocess


def test_preprocessing_keeps_markers_and_removes_scripts():
    cleaned = preprocess('<b>FREE</b> https://example.com me@host.com $100 42 <script>hidden</script>')
    assert all(token in cleaned.split() for token in ("free", "specialurl", "specialemail", "specialmoney", "specialnumber"))
    assert "hidden" not in cleaned


def test_dedup_conflicts_and_data_accounting(tmp_path):
    raw = tmp_path / "emails.csv"
    with raw.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["text", "label_num", "label"])
        writer.writeheader()
        for text, label in (("Hello world", 0), (" HELLO   world ", 0), ("bad offer", 0),
                            ("bad offer", 1), ("", 1), ("FREE prize", 1)):
            writer.writerow({"text": text, "label_num": label, "label": "spam" if label else "ham"})
    rows, report = clean_csv(raw, "enron")
    assert len(rows) == 2
    assert report["duplicate_rows_removed"] == 1
    assert report["conflicting_rows_removed"] == 2
    assert report["empty_or_no_tokens_removed"] == 1
    assert report["accounting_valid"]


def test_splits_are_stable_under_row_shuffle_and_disjoint():
    records = [{"sample_id": f"id-{i:03}", "group_id": f"group-{i}", "label": i % 2} for i in range(100)]
    first = assign_splits(records)
    second = assign_splits(records[::-1])
    assert first == second
    assert {s: sum(r["split"] == s for r in first) for s in ("train", "validation", "test")} == {"train": 60, "validation": 20, "test": 20}


def test_train_only_features_and_json_roundtrip(tmp_path):
    texts = ["meeting report", "meeting budget", "free prize", "free offer"]
    features = BinaryTextFeatures(k=100, min_df=1).fit(texts, [0, 0, 1, 1])
    names = features.get_feature_names()
    before = features.transform(texts)
    assert "secretvalidationtoken" not in names
    assert not features.transform(["secretvalidationtoken"]).any()
    path = tmp_path / "features.json"
    features.save(path)
    loaded = BinaryTextFeatures.load(path)
    np.testing.assert_array_equal(before, loaded.transform(texts))
    assert loaded.get_feature_names() == names


def test_text_release_checksums_model_and_raw_text_roundtrip(tmp_path):
    texts = ["meeting report", "meeting budget", "free prize", "free offer"]
    features = BinaryTextFeatures(10, min_df=1).fit(texts, [0, 0, 1, 1])
    model = DecisionTreeClassifier(min_samples_split=2, min_samples_leaf=1).fit(features.transform(texts), [0, 0, 1, 1])
    features.save(tmp_path / "features.json")
    save_tree_bundle(tmp_path / "tree.json", model, features.get_feature_names())
    write_json(tmp_path / "release.json", {
        "format_version": 1, "preprocessing_version": PREPROCESSING_VERSION,
        "checksums": {name: sha256(tmp_path / name) for name in ("tree.json", "features.json")},
    })
    predictor = TextPredictor(tmp_path)
    np.testing.assert_array_equal([r["label"] for r in predictor.predict_texts(texts)], model.predict(features.transform(texts)))
    assert predictor.predict_texts(["unknownword"])[0]["warning"]
    with pytest.raises(ValueError):
        predictor.predict_texts([" "])
    (tmp_path / "features.json").write_text("{}")
    with pytest.raises(ValueError, match="checksum"):
        TextPredictor(tmp_path)


def test_cv_refits_fresh_features_on_train_only(monkeypatch):
    calls = []
    original = BinaryTextFeatures.fit
    def recording_fit(self, texts, y):
        calls.append(list(texts))
        return original(self, texts, y)
    monkeypatch.setattr(BinaryTextFeatures, "fit", recording_fit)
    records = [{"sample_id": f"s{i}", "group_id": f"g{i}", "label": i % 2,
                "split": "train", "text": ("meeting report " if i % 2 == 0 else "free offer ") + str(i)} for i in range(30)]
    records.append({"sample_id": "heldout", "group_id": "heldout", "label": 1,
                    "split": "test", "text": "secretvalidationtoken"})
    result = cross_validate(records, [{"name": "candidate", "k": 5,
                                       "parameters": {"max_depth": 3, "min_samples_split": 2, "min_samples_leaf": 1}}])
    assert len(calls) == 5 and all(len(batch) == 24 for batch in calls)
    assert not any("secretvalidationtoken" in text for batch in calls for text in batch)
    assert len(result["fold_assignments"]) == 30
    assert result["summary"]["candidate"]["f1_spam"]["mean"] == 1


def test_prepared_csv_supports_long_email_bodies_in_a_fresh_process(tmp_path):
    csv.field_size_limit(131072)
    body = "long body " * 20000
    write_csv(tmp_path / "records.csv", [{"text": body, "label": 0}])
    write_json(tmp_path / "data_quality.json", {"dataset_fingerprint": sha256(tmp_path / "records.csv")})
    assert read_records(tmp_path)[0]["text"] == body


def test_complete_real_data_workflow_without_network(tmp_path, monkeypatch):
    from spam_detector.enron_training import run_enron_training
    raw = tmp_path / "raw"
    paths = {}
    for source, count in (("enron", 80), ("spamassassin", 20)):
        folder = raw / source
        folder.mkdir(parents=True)
        path = folder / "source.csv"
        rows = []
        for i in range(count):
            label = i % 2
            body = ("free prize offer " if label else "project meeting report ") + source + f" record {i}"
            rows.append({"text": body, "label_num": label, "label": "spam" if label else "ham"}
                        if source == "enron" else {"Body": body, "Label": label})
        write_csv(path, rows)
        write_json(folder / "manifest.json", {"dataset": source, "file_sha256": sha256(path)})
        paths[source] = path
    monkeypatch.setattr("spam_detector.enron_training.download_dataset", lambda source, raw_dir: paths[source])
    output = tmp_path / "real-run"
    result = run_enron_training(output, raw, progress=lambda message: None)
    assert result["round_trip_verified"]
    assert result["evaluation"]["test"]["metrics"]["f1_spam"] == 1
    assert (output / "REPORT.md").exists()
    assert (output / "handoff" / "tree.validation.csv").exists()
    assert (output / "figures" / "confusion-matrices.png").exists()
    assert len((output / "candidates.validation.csv").read_text().splitlines()) == 19


def test_published_model_bundle_is_complete_and_predicts_text():
    root = Path(__file__).resolve().parents[1] / "artifacts" / "tree-enron-v1"
    predictor = TextPredictor(root)
    outputs = predictor.predict_texts(["Subject: meeting agenda and report", "Claim your free prize today"])
    assert len(outputs) == 2
    assert all(row["label"] in (0, 1) and 0 <= row["spam_fraction"] <= 1 for row in outputs)
    assert predictor.bundle.model.n_features_in_ == 1000
