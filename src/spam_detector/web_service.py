"""UI-independent prediction registry; no training, shared email cache or I/O logs."""

import hashlib
import json
import math
from pathlib import Path
from time import perf_counter
from typing import Any, Callable
from uuid import uuid4

import yaml

from .data import sha256
from .inference import TextPredictor

MODEL_IDS = ("tree", "nb", "lr", "svm")
SCORE_KINDS = {"leaf_fraction", "probability_uncalibrated", "margin", "vote_score"}


def load_web_config(path: str | Path, project_root: str | Path) -> dict:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or set(raw) - {"max_characters", "default_model", "models", "ensemble"}:
        raise ValueError("Invalid web configuration")
    if type(raw.get("max_characters")) is not int or raw["max_characters"] < 1:
        raise ValueError("max_characters must be a positive integer")
    models = raw.get("models")
    if not isinstance(models, dict) or set(models) != set(MODEL_IDS) or raw.get("default_model") not in models:
        raise ValueError("Configure the four model IDs and a valid default")
    resolved = {}
    for model_id, spec in models.items():
        if (not isinstance(spec, dict) or set(spec) != {"name", "adapter", "release"}
            or not isinstance(spec["name"], str) or not spec["name"]
            or not isinstance(spec["adapter"], str)
            or (spec["release"] is not None and not isinstance(spec["release"], str))):
            raise ValueError("Invalid model configuration")
        resolved[model_id] = {**spec, "release": str((Path(project_root) / spec["release"]).resolve()) if spec["release"] else None}
    return {**raw, "models": resolved}


def registry_fingerprint(config: dict) -> str:
    """Any changed model bytes invalidate the session's loaded resources/results."""
    digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode())
    for spec in config["models"].values():
        if spec["release"]:
            root = Path(spec["release"])
            for filename in ("release.json", "tree.json", "features.json"):
                path = root / filename
                digest.update((sha256(path) if path.is_file() else "missing").encode())
    return digest.hexdigest()


def default_loader(spec: dict):
    if spec["adapter"] != "tree":
        raise ValueError("Adapter not implemented")
    return TextPredictor(spec["release"])


class PredictionService:
    def __init__(self, config: dict, loader: Callable = default_loader):
        self.config = config
        self.statuses: dict[str, dict] = {}
        self.predictors = {}
        for model_id, spec in config["models"].items():
            status = {"name": spec["name"], "status": "not_configured", "release_id": None, "fingerprint": None}
            if spec["release"]:
                try:
                    predictor = loader(spec)
                    fingerprint = sha256(Path(spec["release"]) / "release.json")
                    self.predictors[model_id] = predictor
                    status.update(status="ready", release_id=Path(spec["release"]).name, fingerprint=fingerprint)
                except Exception:
                    # Do not surface local paths, email text or internal exception details.
                    status["status"] = "load_error"
            self.statuses[model_id] = status

    @property
    def ready_ids(self) -> list[str]:
        return [model_id for model_id in MODEL_IDS if model_id in self.predictors]

    def ensemble_available(self) -> bool:
        spec = self.config.get("ensemble")
        if not isinstance(spec, dict) or set(self.ready_ids) != set(MODEL_IDS):
            return False
        try:
            if (spec["weight_source"] != "validation_f1" or set(spec["weights"]) != set(MODEL_IDS)
                or set(spec["release_fingerprints"]) != set(MODEL_IDS)):
                return False
            weights = list(spec["weights"].values())
            if any(type(w) not in (int, float) or not math.isfinite(w) or w < 0 for w in weights):
                return False
            return math.isclose(sum(weights), 1, abs_tol=1e-9, rel_tol=0) and all(
                spec["release_fingerprints"][key] == self.statuses[key]["fingerprint"] for key in MODEL_IDS)
        except (KeyError, TypeError, AttributeError):
            return False

    def prepare_email(self, subject: str, body: str) -> str:
        if not isinstance(subject, str) or not isinstance(body, str) or not body.strip():
            raise ValueError("Vui lòng nhập nội dung email trước khi kiểm tra.")
        combined = f"Subject: {subject.strip()}\n{body}" if subject.strip() else body
        if len(combined) > self.config["max_characters"]:
            raise ValueError(f"Email vượt giới hạn {self.config['max_characters']:,} ký tự. Vui lòng rút gọn nội dung.")
        return combined

    def _predict_one(self, model_id: str, text: str) -> dict:
        started = perf_counter()
        outputs = self.predictors[model_id].predict_texts([text])
        if not isinstance(outputs, list) or len(outputs) != 1 or not isinstance(outputs[0], dict):
            raise ValueError("Invalid prediction output")
        raw = outputs[0]
        label = raw.get("label")
        if type(label) is not int or label not in (0, 1):
            raise ValueError("Invalid label")
        kind = "leaf_fraction" if model_id == "tree" else raw.get("score_kind")
        score = raw.get("spam_fraction") if model_id == "tree" else raw.get("score_value")
        if (kind not in SCORE_KINDS or type(score) not in (float, int) or not math.isfinite(score)
            or (kind != "margin" and not 0 <= score <= 1)):
            raise ValueError("Invalid score")
        return {
            "model_id": model_id, "model_name": self.statuses[model_id]["name"],
            "release_id": self.statuses[model_id]["release_id"], "label": label,
            "score_kind": kind, "score_value": float(score),
            "elapsed_ms": (perf_counter() - started) * 1000,
            "path": raw.get("path", []), "leaf_training_samples": raw.get("leaf_training_samples"),
            "matched_features": raw.get("matched_features"), "unique_tokens": raw.get("unique_tokens"),
            "warning": "Không có từ nào khớp bộ đặc trưng. Dự đoán dựa trên các nhánh vắng từ đã học." if raw.get("warning") else None,
        }

    def predict(self, subject: str, body: str, mode: str = "single", model_id: str = "tree") -> dict:
        text = self.prepare_email(subject, body)
        if mode not in ("single", "compare", "ensemble"):
            raise ValueError("Chế độ kiểm tra không hợp lệ.")
        if mode == "single" and model_id not in self.ready_ids:
            raise ValueError("Mô hình đã chọn chưa sẵn sàng.")
        if mode == "ensemble" and not self.ensemble_available():
            raise ValueError("Ensemble cần đủ 4 mô hình và trọng số validation khớp release.")
        ids = [model_id] if mode == "single" else self.ready_ids
        if not ids:
            raise ValueError("Chưa có mô hình sẵn sàng. Vui lòng kiểm tra cấu hình release.")
        results, errors = [], []
        for key in ids:
            try:
                results.append(self._predict_one(key, text))
            except Exception:
                errors.append({"model_id": key, "message": f"{self.statuses[key]['name']} chưa xử lý được lần kiểm tra này. Vui lòng thử lại."})
        ensemble = None
        if mode == "ensemble" and not errors and len(results) == 4:
            weights = self.config["ensemble"]["weights"]
            score = sum(weights[row["model_id"]] * row["label"] for row in results)
            ensemble = {"label": int(score >= 0.5), "score_value": score, "score_kind": "vote_score"}
        return {
            "request_id": uuid4().hex[:12], "mode": mode, "characters": len(text),
            "input_fingerprint": hashlib.sha256(text.encode()).hexdigest(),
            "results": results, "errors": errors, "ensemble": ensemble,
        }
