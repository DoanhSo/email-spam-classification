"""Small common interface, with optional probability/score capabilities."""

from typing import Any, Protocol, Self

import numpy as np
from numpy.typing import NDArray


class NotFittedError(ValueError):
    """Prediction was requested before a model was fitted."""


class Classifier(Protocol):
    def fit(self, X: Any, y: Any) -> Self: ...

    def predict(self, X: Any) -> NDArray[np.int64]: ...


class ProbabilityClassifier(Classifier, Protocol):
    def predict_proba(self, X: Any) -> NDArray[np.float64]: ...


class ScoredClassifier(Classifier, Protocol):
    def decision_function(self, X: Any) -> NDArray[np.float64]: ...
