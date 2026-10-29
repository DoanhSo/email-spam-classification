"""Portable JSON tree bundle, with the exact feature ordering attached."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from .models.decision_tree import DecisionTreeClassifier


@dataclass(frozen=True)
class TreeBundle:
    model: DecisionTreeClassifier
    feature_names: tuple[str, ...]
    metadata: dict[str, Any]


def save_tree_bundle(
    path: str | Path, model: DecisionTreeClassifier, feature_names: Sequence[str],
    metadata: dict[str, Any] | None = None,
) -> None:
    names = tuple(feature_names)
    model.export_rules(names)  # validates fitted state and name count
    if len(set(names)) != len(names) or any(not word for word in names):
        raise ValueError("Feature names must be nonempty and unique")
    payload = {
        "bundle_version": 1, "feature_names": names,
        "model": model.to_dict(), "metadata": metadata or {},
    }
    encoded = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(encoded, encoding="utf-8")


def load_tree_bundle(path: str | Path) -> TreeBundle:
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        if raw["bundle_version"] != 1 or not isinstance(raw["metadata"], dict):
            raise ValueError("Unsupported or malformed bundle")
        model = DecisionTreeClassifier.from_dict(raw["model"])
        if not isinstance(raw["feature_names"], list):
            raise ValueError("Invalid feature names")
        names = tuple(raw["feature_names"])
        model.export_rules(names)
        if len(set(names)) != len(names) or any(not word for word in names):
            raise ValueError("Invalid feature names")
        return TreeBundle(model, names, raw["metadata"])
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError("Malformed tree bundle") from exc
