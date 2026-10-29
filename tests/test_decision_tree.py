import copy

import numpy as np
import pytest

from spam_detector.artifacts import load_tree_bundle, save_tree_bundle
from spam_detector.models.base import NotFittedError
from spam_detector.models.decision_tree import DecisionTreeClassifier, gini


def tree(**params):
    return DecisionTreeClassifier(min_samples_split=2, min_samples_leaf=1, **params)


def test_hand_calculated_best_split_and_rules():
    # Parent Gini=0.5. Column 0 creates pure leaves; column 1 has gain 0.
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.uint8)
    y = np.array([0, 0, 1, 1])
    model = tree().fit(X, y)
    assert gini(2, 4) == 0.5
    assert model.nodes_[0].feature == 0
    assert model.depth_ == 1 and model.n_leaves_ == 2
    np.testing.assert_array_equal(model.predict(X), y)
    np.testing.assert_array_equal(model.predict_proba(X), [[1, 0], [1, 0], [0, 1], [0, 1]])
    rules = model.export_rules(["free", "url"])
    assert [r["prediction"] for r in rules] == [0, 1]
    assert rules[0]["conditions"] == [{"feature_index": 0, "word": "free", "present": False}]
    assert sum(r["n_samples"] for r in rules) == len(y)


def test_weighted_impurity_does_not_overvalue_a_tiny_pure_child():
    # Column 0: pure one-row branch + mixed seven-row branch, gain ~0.0714.
    # Column 1: balanced 3:1 and 1:3 children, gain 0.125 -> must win.
    X = np.array([[1, 1], [0, 1], [0, 1], [0, 1], [0, 0], [0, 0], [0, 0], [0, 0]])
    y = [1, 1, 1, 0, 1, 0, 0, 0]
    assert tree(max_depth=1).fit(X, y).nodes_[0].feature == 1


def test_feature_tie_prefers_first_column():
    model = tree().fit([[0, 0], [0, 0], [1, 1], [1, 1]], [0, 0, 1, 1])
    assert model.nodes_[0].feature == 0


@pytest.mark.parametrize("X,y", [
    ([[0], [0], [0], [0]], [0, 1, 0, 1]),
    ([[0, 0], [0, 1], [1, 0], [1, 1]], [0, 1, 1, 0]),
])
def test_constant_or_zero_gain_features_stop_at_tied_ham_leaf(X, y):
    model = tree().fit(X, y)
    assert len(model.nodes_) == 1
    np.testing.assert_array_equal(model.predict(X), np.zeros(len(y)))
    np.testing.assert_array_equal(model.predict_proba(X), np.full((len(y), 2), 0.5))


def test_depth_zero_is_stump_and_leaves_preserve_empirical_proportions():
    model = tree(max_depth=0).fit([[0], [0], [1], [1], [1]], [0, 0, 1, 1, 1])
    assert model.depth_ == 0
    np.testing.assert_allclose(model.predict_proba([[0], [1]]), [[0.4, 0.6]] * 2)


def test_minimum_leaf_blocks_invalid_split():
    model = DecisionTreeClassifier(min_samples_split=2, min_samples_leaf=2).fit(
        [[1], [0], [0], [0]], [1, 0, 0, 0]
    )
    assert model.n_leaves_ == 1


def test_depth_and_leaf_limits_on_nontrivial_tree():
    rng = np.random.default_rng(7)
    X = rng.integers(0, 2, size=(250, 12))
    y = rng.integers(0, 2, size=250)
    model = DecisionTreeClassifier(max_depth=3, min_samples_split=8, min_samples_leaf=4).fit(X, y)
    assert model.depth_ <= 3
    assert all(n.n_samples >= 4 for n in model.nodes_ if n.feature is None)
    np.testing.assert_allclose(model.predict_proba(X).sum(axis=1), 1)
    np.testing.assert_array_equal(model.predict(X), model.predict(X))


@pytest.mark.parametrize("params", [
    {"max_depth": -1}, {"max_depth": True}, {"max_depth": 1.5},
    {"min_samples_split": 1}, {"min_samples_split": False}, {"min_samples_leaf": 0},
])
def test_invalid_parameters(params):
    with pytest.raises(ValueError):
        DecisionTreeClassifier(**params)


@pytest.mark.parametrize("X,y", [
    ([], []), ([[0], [1]], [1]), ([[0], [1]], [0, 2]),
    ([[0], [1]], [0, float("nan")]), ([[0.1], [1]], [0, 1]),
    ([[float("inf")], [1]], [0, 1]), ([[0], [1]], [[0], [1]]),
    ([[0], [1]], [1, 1]), ([["0"], ["1"]], [0, 1]),
])
def test_bad_training_inputs_are_rejected(X, y):
    with pytest.raises(ValueError):
        tree().fit(X, y)


def test_prediction_contract_and_empty_batch():
    model = tree()
    with pytest.raises(NotFittedError):
        model.predict([[0]])
    model.fit([[0], [1]], [0, 1])
    with pytest.raises(ValueError, match="Expected 1 features"):
        model.predict([[0, 1]])
    assert model.predict(np.empty((0, 1))).shape == (0,)
    assert model.predict_proba(np.empty((0, 1))).shape == (0, 2)
    with pytest.raises(ValueError):
        model.export_rules([])


def test_rejected_refit_preserves_previous_valid_model():
    model = tree().fit([[0], [1]], [0, 1])
    with pytest.raises(ValueError):
        model.fit([[0.5], [1]], [0, 1])
    np.testing.assert_array_equal(model.predict([[0], [1]]), [0, 1])


def test_bundle_round_trip_includes_order_names_and_rules(tmp_path):
    X, y = [[0, 1], [1, 0], [1, 1]], [0, 1, 1]
    model = tree().fit(X, y)
    path = tmp_path / "tree.json"
    save_tree_bundle(path, model, ["free", "url"], {"origin": "unit_test"})
    loaded = load_tree_bundle(path)
    assert loaded.feature_names == ("free", "url")
    assert loaded.metadata["origin"] == "unit_test"
    np.testing.assert_array_equal(model.predict(X), loaded.model.predict(X))
    np.testing.assert_allclose(model.predict_proba(X), loaded.model.predict_proba(X))
    assert model.export_rules(loaded.feature_names) == loaded.model.export_rules(loaded.feature_names)


@pytest.mark.parametrize("corruption", ["cycle", "population", "feature", "orphan", "version"])
def test_corrupt_state_is_rejected(corruption):
    state = copy.deepcopy(tree().fit([[0], [1]], [0, 1]).to_dict())
    if corruption == "cycle":
        state["nodes"][0]["left"] = 0
    elif corruption == "population":
        state["nodes"][1]["n_samples"] = 99
    elif corruption == "feature":
        state["nodes"][0]["feature"] = 99
    elif corruption == "orphan":
        state["nodes"].append(state["nodes"][1])
    else:
        state["format_version"] = 99
    with pytest.raises(ValueError):
        DecisionTreeClassifier.from_dict(state)


def test_sklearn_agrees_when_best_split_is_unique():
    reference_module = pytest.importorskip("sklearn.tree")
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = [0, 0, 1, 1]
    own = tree(max_depth=2).fit(X, y)
    reference = reference_module.DecisionTreeClassifier(max_depth=2, random_state=42).fit(X, y)
    np.testing.assert_array_equal(own.predict(X), reference.predict(X))
    np.testing.assert_allclose(own.predict_proba(X), reference.predict_proba(X))
