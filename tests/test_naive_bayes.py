import numpy as np
import pytest

from spam_detector.models.base import NotFittedError
from spam_detector.models.naive_bayes import MultinomialNaiveBayes

X = np.array([[2, 1, 0], [0, 1, 3], [1, 0, 0]])
Y = np.array([0, 1, 0])


def fitted():
    return MultinomialNaiveBayes().fit(X, Y)


def test_hand_computed_parameters():
    model = fitted()
    assert np.allclose(np.exp(model.class_log_prior_), [3 / 5, 2 / 5])
    assert np.allclose(np.exp(model.feature_log_prob_[0]), [4 / 7, 2 / 7, 1 / 7])
    assert np.allclose(np.exp(model.feature_log_prob_[1]), [1 / 7, 2 / 7, 4 / 7])


def test_hand_computed_proba():
    model = fitted()
    assert np.allclose(model.predict_proba([[1, 0, 1]]), [[0.6, 0.4]])
    assert np.isclose(model.predict_proba([[0, 0, 2]])[0, 1], 6.4 / 7.0)
    assert model.predict([[1, 0, 1], [0, 0, 2]]).tolist() == [0, 1]


def test_long_document_is_finite():
    proba = fitted().predict_proba([[100000, 0, 100000]])
    assert np.all(np.isfinite(proba)) and np.isclose(proba.sum(), 1.0)


@pytest.mark.parametrize("bad_X, bad_y", [([[1, -1, 0], [0, 1, 0]], [0, 1]), ([[1, 0, 0], [0, 1, 0]], [1, 1])])
def test_rejects_negative_and_single_class(bad_X, bad_y):
    with pytest.raises(ValueError):
        MultinomialNaiveBayes().fit(bad_X, bad_y)


def test_not_fitted():
    with pytest.raises(NotFittedError):
        MultinomialNaiveBayes().predict([[1, 0, 0]])


def test_dict_roundtrip():
    model = fitted()
    clone = MultinomialNaiveBayes.from_dict(model.to_dict())
    sample = np.array([[1, 0, 1], [0, 2, 5], [3, 3, 3]])
    assert np.array_equal(clone.predict_proba(sample), model.predict_proba(sample))


def test_explain_orders_by_spam_contribution():
    result = fitted().explain([0, 1, 3], ["a", "b", "c"])
    assert result[0][0] == "c"
    assert "a" not in [word for word, _ in result]
