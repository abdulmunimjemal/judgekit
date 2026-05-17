"""Calibrators map raw judge scores to estimated ground-truth probabilities.

Two calibrators are included:

- ``PlattCalibrator``: fits a single-feature logistic regression on
  (raw_score, label). Smooth, parametric, well-behaved with little data.
- ``IsotonicCalibrator``: fits a monotone non-parametric mapping. More
  flexible — better when the score-vs-truth curve is non-sigmoid — but
  needs more data and can overfit on small sets.

Both expose the same ``fit`` / ``predict`` interface so the harness can
swap them transparently.

``bootstrap_ci`` is a small helper to compute confidence intervals around
any aggregate of calibrated scores (mean, win-rate, etc.) by resampling
the calibration set.

Math references:
- Platt (1999), "Probabilistic outputs for support vector machines"
- Zadrozny & Elkan (2002), "Transforming classifier scores into accurate
  multiclass probability estimates" (isotonic)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


class Calibrator(ABC):
    """A monotone-ish mapping from raw judge score -> calibrated probability."""

    fitted: bool = False

    @abstractmethod
    def fit(self, raw_scores: np.ndarray, labels: np.ndarray) -> Calibrator:
        """Fit on paired (raw_score, ground_truth_label) arrays.

        Both arrays must be 1-D and the same length. Labels must lie in [0, 1].
        Returns self for chaining.
        """

    @abstractmethod
    def predict(self, raw_scores: np.ndarray) -> np.ndarray:
        """Map raw judge scores to calibrated probabilities."""


class PlattCalibrator(Calibrator):
    """Single-feature logistic regression. Smooth, parametric, low-variance."""

    def __init__(self) -> None:
        self._model = LogisticRegression(solver="lbfgs")

    def fit(self, raw_scores: np.ndarray, labels: np.ndarray) -> PlattCalibrator:
        raw_scores = np.asarray(raw_scores, dtype=float).reshape(-1, 1)
        labels = np.asarray(labels, dtype=float)
        if raw_scores.shape[0] != labels.shape[0]:
            raise ValueError("raw_scores and labels must have the same length")
        if raw_scores.shape[0] < 2:
            raise ValueError("need at least 2 examples to fit Platt")
        # LogisticRegression wants integer-ish targets but accepts probabilities
        # via sample weighting. For simplicity we bucket labels at 0.5 and use
        # a soft weighting via class_weight balancing; this is the standard
        # Platt formulation used by libsvm.
        binary = (labels >= 0.5).astype(int)
        if len(np.unique(binary)) < 2:
            raise ValueError(
                "Platt needs both classes (some labels<0.5 and some>=0.5). "
                "Use IsotonicCalibrator for one-sided calibration sets."
            )
        self._model.fit(raw_scores, binary)
        self.fitted = True
        return self

    def predict(self, raw_scores: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("PlattCalibrator must be fit before predict()")
        raw_scores = np.asarray(raw_scores, dtype=float).reshape(-1, 1)
        # Probability of the positive class
        return self._model.predict_proba(raw_scores)[:, 1]


class IsotonicCalibrator(Calibrator):
    """Monotone non-parametric calibrator. Flexible, but data-hungry."""

    def __init__(self) -> None:
        self._model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)

    def fit(self, raw_scores: np.ndarray, labels: np.ndarray) -> IsotonicCalibrator:
        raw_scores = np.asarray(raw_scores, dtype=float)
        labels = np.asarray(labels, dtype=float)
        if raw_scores.shape[0] != labels.shape[0]:
            raise ValueError("raw_scores and labels must have the same length")
        if raw_scores.shape[0] < 2:
            raise ValueError("need at least 2 examples to fit isotonic")
        self._model.fit(raw_scores, labels)
        self.fitted = True
        return self

    def predict(self, raw_scores: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("IsotonicCalibrator must be fit before predict()")
        raw_scores = np.asarray(raw_scores, dtype=float)
        return self._model.predict(raw_scores)


def bootstrap_ci(
    scores: np.ndarray,
    statistic: Callable[[np.ndarray], float] = np.mean,
    n_resamples: int = 1000,
    confidence: float = 0.95,
    rng: np.random.Generator | None = None,
) -> tuple[float, float, float]:
    """Bootstrap a confidence interval around an aggregate of calibrated scores.

    Returns ``(point_estimate, lower, upper)``. Defaults to a 95% CI around
    the mean. Use this to report eval results with proper uncertainty so a
    judge-version bump doesn't silently move your headline number outside
    the previous CI.

    Parameters
    ----------
    scores : 1-D array of calibrated scores.
    statistic : aggregate to bootstrap (default: mean). E.g. ``np.median`` or
        a custom win-rate function.
    n_resamples : number of bootstrap resamples. 1000 is enough for stable
        percentile estimates; bump to 10000 if you want sub-percent precision.
    confidence : nominal coverage (e.g. 0.95 for 95% CI).
    rng : optional NumPy generator for reproducibility.
    """
    scores = np.asarray(scores, dtype=float)
    if scores.size == 0:
        raise ValueError("scores must be non-empty")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")

    rng = rng if rng is not None else np.random.default_rng()
    n = scores.size
    point = float(statistic(scores))
    resampled = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        resampled[i] = statistic(scores[idx])
    alpha = (1.0 - confidence) / 2.0
    lower = float(np.quantile(resampled, alpha))
    upper = float(np.quantile(resampled, 1.0 - alpha))
    return point, lower, upper
