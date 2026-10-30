import copy
import json
from pathlib import Path

import pytest

pytest.importorskip("bs4")
from spam_detector.inference import TextPredictor
from spam_detector.web_service import PredictionService, load_web_config, registry_fingerprint

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def config():
    return load_web_config(ROOT / "configs/web.yaml", ROOT)


def test_service_matches_existing_text_predictor(config):
    service = PredictionService(config)
    subject, body = "Budget", "Please review the attached Enron report."
    expected = TextPredictor(ROOT / "artifacts/tree-enron-v1").predict_texts([f"Subject: {subject}\n{body}"])[0]
    result = service.predict(subject, body)
    assert result["results"][0]["label"] == expected["label"]
    assert result["results"][0]["score_value"] == expected["spam_fraction"]
    assert result["results"][0]["path"] == expected["path"]
    assert "body" not in result and "subject" not in result
    assert service.ready_ids == ["tree"]
    assert not service.ensemble_available()


@pytest.mark.parametrize("body", ["", " \n\t", None, 1])
def test_empty_or_invalid_body(config, body):
    with pytest.raises(ValueError):
        PredictionService(config).predict("", body)


def test_combined_length_and_subject_rules(config):
    config["max_characters"] = 20
    service = PredictionService(config)
    assert service.prepare_email("", "Subject: a\nbody") == "Subject: a\nbody"
    assert service.prepare_email(" a ", "body") == "Subject: a\nbody"
    with pytest.raises(ValueError, match="giới hạn"):
        service.predict("long subject", "01234567890123456789")


def test_model_missing_load_error_and_ensemble_gate(config):
    config["models"]["tree"]["release"] = str(ROOT / "missing-release")
    service = PredictionService(config)
    assert service.statuses["tree"]["status"] == "load_error"
    assert service.statuses["nb"]["status"] == "not_configured"
    for mode in ("single", "compare", "ensemble", "invalid"):
        with pytest.raises(ValueError):
            service.predict("", "hello email", mode)


def test_compare_only_returns_real_models_and_oov_warning(config):
    result = PredictionService(config).predict("", "zzzzzzqqqqqq", mode="compare")
    assert len(result["results"]) == 1
    assert result["results"][0]["warning"]
    assert result["results"][0]["matched_features"] == 0


def test_runtime_error_and_malformed_prediction_are_contained(config):
    class Broken:
        def predict_texts(self, texts):
            raise RuntimeError("private email content")
    result = PredictionService(config, loader=lambda spec: Broken()).predict("", "hello")
    assert not result["results"] and len(result["errors"]) == 1
    assert "private" not in json.dumps(result)
    class BadScore:
        def predict_texts(self, texts):
            return [{"label": 1, "spam_fraction": float("nan")}]
    assert PredictionService(config, loader=lambda spec: BadScore()).predict("", "hello")["errors"]


def test_ensemble_tie_missing_release_and_partial_failure(config):
    for spec in config["models"].values():
        spec["release"] = config["models"]["tree"]["release"]
    class Fake:
        def __init__(self, adapter): self.adapter = adapter
        def predict_texts(self, texts):
            return [{"label": int(self.adapter in ("tree", "nb")), "spam_fraction": 0.75,
                     "score_kind": "margin" if self.adapter == "svm" else "probability_uncalibrated",
                     "score_value": -1.0 if self.adapter == "svm" else 0.75}]
    service = PredictionService(config, loader=lambda spec: Fake(spec["adapter"]))
    config["ensemble"] = {"weight_source": "validation_f1",
                          "weights": {key: 0.25 for key in service.ready_ids},
                          "release_fingerprints": {key: status["fingerprint"] for key, status in service.statuses.items()}}
    assert service.ensemble_available()
    result = service.predict("", "example", "ensemble")
    assert result["ensemble"]["score_value"] == 0.5
    assert result["ensemble"]["label"] == 1
    original = service.predictors["nb"]
    service.predictors["nb"] = None
    broken = service.predict("", "example", "ensemble")
    assert broken["ensemble"] is None and len(broken["errors"]) == 1
    service.predictors["nb"] = original
    config["ensemble"]["release_fingerprints"]["nb"] = "wrong"
    assert not service.ensemble_available()
    with pytest.raises(ValueError): service.predict("", "example", "ensemble")


def test_cache_fingerprint_changes_for_changed_artifact_bytes(config, tmp_path):
    root = tmp_path / "release"
    root.mkdir()
    config["models"]["tree"]["release"] = str(root)
    before = registry_fingerprint(config)
    (root / "tree.json").write_text("changed")
    assert registry_fingerprint(config) != before


def test_invalid_config_and_session_isolation(config, tmp_path):
    path = tmp_path / "web.yaml"
    path.write_text("max_characters: true")
    with pytest.raises(ValueError): load_web_config(path, ROOT)
    first, second = PredictionService(config), PredictionService(copy.deepcopy(config))
    output = first.predict("", "Enron report")
    assert not hasattr(first, "last_input") and not hasattr(second, "last_input")
    assert second.predict("", "Enron report")["results"][0]["label"] == output["results"][0]["label"]
