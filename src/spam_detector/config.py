"""Validated configuration for the implemented Decision Tree workstream."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .models.decision_tree import validate_parameters


@dataclass(frozen=True)
class ProjectConfig:
    seed: int = 42
    split_ratios: tuple[float, ...] = (0.6, 0.2, 0.2)
    select_k: int = 500
    tree: dict[str, Any] = field(default_factory=lambda: {
        "max_depth": 10, "min_samples_split": 10, "min_samples_leaf": 5,
    })

    def __post_init__(self) -> None:
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError("seed must be an integer in [0, 2**32)")
        if (len(self.split_ratios) != 3
            or any(type(x) not in (int, float) for x in self.split_ratios)
            or not all(np.isfinite(x) and 0 < x < 1 for x in self.split_ratios)
            or not np.isclose(sum(self.split_ratios), 1, rtol=0, atol=1e-9)):
            raise ValueError("split_ratios must be three positive fractions summing to 1")
        if type(self.select_k) is not int or self.select_k < 1:
            raise ValueError("select_k must be a positive integer")
        try:
            validate_parameters(**self.tree)
        except TypeError as exc:
            raise ValueError("Invalid tree parameters") from exc


def load_config(path: str | Path) -> ProjectConfig:
    """Read only implemented settings; reserved models cannot be run yet."""
    try:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("Configuration must be a YAML mapping")
        allowed = {"seed", "data", "features", "models"}
        if raw.keys() - allowed:
            raise ValueError(f"Unknown configuration sections: {raw.keys() - allowed}")
        data = raw.get("data", {})
        features = raw.get("features", {}).get("tree", {})
        models = raw.get("models", {})
        if (set(data) - {"split_ratios"} or set(features) - {"representation", "select_k"}
            or set(raw.get("features", {})) - {"tree"} or set(models) - {"tree"}):
            raise ValueError("Unknown or not-yet-supported configuration keys")
        if features.get("representation", "binary") != "binary":
            raise ValueError("Decision Tree currently requires binary features")
        defaults = ProjectConfig()
        return ProjectConfig(
            seed=raw.get("seed", defaults.seed),
            split_ratios=tuple(data.get("split_ratios", defaults.split_ratios)),
            select_k=features.get("select_k", defaults.select_k),
            tree={**defaults.tree, **models.get("tree", {})},
        )
    except (AttributeError, TypeError, yaml.YAMLError) as exc:
        raise ValueError("Invalid YAML configuration structure") from exc
