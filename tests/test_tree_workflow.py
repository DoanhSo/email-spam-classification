import json

import numpy as np
import pytest

from spam_detector.cli import main
from spam_detector.config import ProjectConfig, load_config
from spam_detector.tree_workflow import (
    classification_metrics, create_demo_archive, evaluate_tree, load_matrices, train_tree,
)


def archive_with(path, change):
    create_demo_archive(path)
    with np.load(path, allow_pickle=False) as raw:
        values = {key: raw[key] for key in raw.files}
    change(values)
    np.savez(path, **values)


def test_config_defaults_and_yaml_validation(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("models:\n  tree:\n    max_depth: 3\n", encoding="utf-8")
    assert load_config(path).tree["max_depth"] == 3
    for content in ("[]", "features: {tree: {representation: tfidf}}", "seed: true", "models: {nb: {}}"):
        path.write_text(content, encoding="utf-8")
        with pytest.raises(ValueError):
            load_config(path)


@pytest.mark.parametrize("params", [
    {"split_ratios": (0.6, 0.3, 0.2)}, {"split_ratios": (0.8, 0.2, 0)},
    {"select_k": 0}, {"select_k": True}, {"seed": -1},
])
def test_invalid_common_configuration(params):
    with pytest.raises(ValueError):
        ProjectConfig(**params)


def test_metrics_match_hand_calculation():
    scores = classification_metrics([0, 0, 0, 1, 1], [0, 0, 1, 0, 1])
    assert (scores["tn"], scores["fp"], scores["fn"], scores["tp"]) == (2, 1, 1, 1)
    assert scores["precision_spam"] == scores["recall_spam"] == scores["f1_spam"] == 0.5


def test_full_training_loading_and_heldout_evaluation(tmp_path):
    data = tmp_path / "demo.npz"
    create_demo_archive(data)
    output = tmp_path / "run"
    summary = train_tree(data, ProjectConfig(), output)
    assert summary["split"] == "validation"
    assert summary["metrics"]["f1_spam"] == 1.0
    evaluation = evaluate_tree(output / "tree.json", data, "test")
    assert evaluation["metrics"]["f1_spam"] == 1.0
    assert len(json.loads((output / "rules.json").read_text())) > 1
    with pytest.raises(FileExistsError):
        train_tree(data, ProjectConfig(), output)


def test_training_does_not_read_test_arrays(tmp_path):
    data = tmp_path / "demo.npz"
    archive_with(data, lambda raw: raw.update(X_test=np.array(["not-a-matrix"])))
    train_tree(data, ProjectConfig(), tmp_path / "run")


@pytest.mark.parametrize("key", ["ids_validation", "groups_validation"])
def test_train_validation_leakage_is_rejected(tmp_path, key):
    data = tmp_path / "demo.npz"
    other = key.replace("validation", "train")
    archive_with(data, lambda raw: raw[key].__setitem__(0, raw[other][0]))
    with pytest.raises(ValueError, match="Overlapping"):
        load_matrices(data, ("train", "validation"))


def test_test_overlap_and_feature_order_are_rejected(tmp_path):
    data = tmp_path / "demo.npz"
    create_demo_archive(data)
    output = tmp_path / "run"
    train_tree(data, ProjectConfig(), output)
    with np.load(data) as raw:
        values = {key: raw[key] for key in raw.files}
    values["groups_test"][0] = values["groups_train"][0]
    broken = tmp_path / "broken.npz"
    np.savez(broken, **values)
    with pytest.raises(ValueError, match="overlap train"):
        evaluate_tree(output / "tree.json", broken, "test")
    values["feature_names"] = values["feature_names"][::-1]
    np.savez(broken, **values)
    with pytest.raises(ValueError, match="Feature names/order"):
        evaluate_tree(output / "tree.json", broken, "test")


def test_tuning_selects_only_on_validation(tmp_path):
    data = tmp_path / "demo.npz"
    create_demo_archive(data)
    output = tmp_path / "run"
    summary = train_tree(data, ProjectConfig(), output, depths=[0, 3], leaf_sizes=[1, 5])
    assert summary["parameters"]["max_depth"] == 3
    assert len((output / "candidates.validation.csv").read_text().splitlines()) == 5


def test_cli_config_demo_and_actionable_failures(tmp_path, capsys):
    path = tmp_path / "config.yaml"
    path.write_text("{}", encoding="utf-8")
    assert main(["check-config", "--config", str(path)]) == 0
    assert main(["demo-tree", "--config", str(path), "--output", str(tmp_path / "demo")]) == 0
    assert "synthetic_only" in capsys.readouterr().out
    assert main(["train-tree", "--config", str(path), "--data", str(tmp_path / "missing"),
                 "--output", str(tmp_path / "run")]) == 2
    assert "Error:" in capsys.readouterr().err


def test_opt_in_sklearn_benchmark(tmp_path):
    pytest.importorskip("sklearn.tree")
    data = tmp_path / "demo.npz"
    create_demo_archive(data)
    output = tmp_path / "run"
    train_tree(data, ProjectConfig(), output, benchmark=True)
    result = json.loads((output / "benchmark.json").read_text())
    assert result["prediction_agreement"] == 1.0


def test_individual_npy_array_and_scalar_feature_names_are_rejected(tmp_path):
    data = tmp_path / "matrix.npy"
    np.save(data, np.zeros((2, 1)))
    with pytest.raises(ValueError, match="NPZ"):
        load_matrices(data, ("train",))
    data = tmp_path / "bad_names.npz"
    np.savez(data, feature_names=np.array("scalar"))
    with pytest.raises(ValueError, match="1-D"):
        load_matrices(data, ("train",))
