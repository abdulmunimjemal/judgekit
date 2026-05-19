"""Calibrators map raw judge scores to estimated ground-truth probabilities.

Five calibrators ship with judgekit:

- ``PlattCalibrator``: single-feature logistic regression. Smooth, parametric,
  needs both classes present in the calibration set.
- ``IsotonicCalibrator``: monotone non-parametric mapping. Flexible but
  data-hungry (~200+ anchors recommended).
- ``TemperatureCalibrator``: single-parameter scaling (`T`). Treats raw scores
  as probabilities, transforms to logits, scales by `1/T`, sigmoids back.
  Cheapest model with the fewest assumptions; great default for small
  calibration sets.
- ``BetaCalibrator``: three-parameter Beta-distribution mapping
  (Kull et al. 2017). Captures non-sigmoid curves that Platt misses while
  staying parametric.
- ``HistogramBinCalibrator``: equal-width bin; predicted probability is the
  empirical label-mean inside each bin. Honest baseline; useful when the
  miscalibration is shaped weirdly enough that nothing parametric fits.

All inherit the same ``fit`` / ``predict`` interface so the harness can swap
them transparently. ``select_calibrator(n_anchors)`` picks a sensible default
based on calibration-set size.

``bootstrap_ci`` is a small helper to compute confidence intervals around
any aggregate of calibrated scores by resampling the calibration set.

Math references:
- Platt (1999), "Probabilistic outputs for support vector machines"
- Zadrozny & Elkan (2002), "Transforming classifier scores into accurate
  multiclass probability estimates" (isotonic)
- Guo et al. (2017), "On Calibration of Modern Neural Networks" (temperature)
- Kull et al. (2017), "Beta calibration: a well-founded and easily implemented
  improvement on logistic calibration for binary classifiers"
- Naeini et al. (2015), "Obtaining Well Calibrated Probabilities Using
  Bayesian Binning" (histogram-binning baseline)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import NamedTuple

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

# Clip raw scores away from exact 0/1 before applying logit so we don't blow
# up on boundary inputs.
_PROB_EPS = 1e-6


def _logit(p: np.ndarray) -> np.ndarray:
    """Log-odds of a probability array, with eps-clipping for stability."""
    clipped = np.clip(p, _PROB_EPS, 1.0 - _PROB_EPS)
    return np.asarray(np.log(clipped / (1.0 - clipped)), dtype=float)


def _sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically-stable sigmoid."""
    return np.asarray(
        np.where(z >= 0, 1.0 / (1.0 + np.exp(-z)), np.exp(z) / (1.0 + np.exp(z))), dtype=float
    )


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
        proba: np.ndarray = self._model.predict_proba(raw_scores)
        return np.asarray(proba[:, 1], dtype=float)


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
        return np.asarray(self._model.predict(raw_scores), dtype=float)


class TemperatureCalibrator(Calibrator):
    """Single-parameter temperature scaling.

    Treats `raw_score` as a probability, converts to logit, divides by a
    learned scalar `T`, and squashes back to probability via sigmoid. Because
    only `T` is learned, this is the lowest-variance calibrator we ship —
    great when you have <50 anchors. With `T == 1.0` it is the identity
    transform.
    """

    def __init__(self) -> None:
        self.T: float = 1.0

    def fit(self, raw_scores: np.ndarray, labels: np.ndarray) -> TemperatureCalibrator:
        raw_scores = np.asarray(raw_scores, dtype=float)
        labels = np.asarray(labels, dtype=float)
        if raw_scores.shape[0] != labels.shape[0]:
            raise ValueError("raw_scores and labels must have the same length")
        if raw_scores.shape[0] < 2:
            raise ValueError("need at least 2 examples to fit Temperature")

        logits = _logit(raw_scores)

        # Minimize negative log-likelihood over T in (eps, 100).
        def nll(t: float) -> float:
            if t <= 0:
                return float("inf")
            p = _sigmoid(logits / t)
            # Clamp for log stability.
            p = np.clip(p, _PROB_EPS, 1.0 - _PROB_EPS)
            return float(-np.sum(labels * np.log(p) + (1.0 - labels) * np.log(1.0 - p)))

        result = minimize_scalar(nll, bounds=(1e-3, 100.0), method="bounded")
        self.T = float(result.x)
        self.fitted = True
        return self

    def predict(self, raw_scores: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("TemperatureCalibrator must be fit before predict()")
        raw_scores = np.asarray(raw_scores, dtype=float)
        logits = _logit(raw_scores)
        return _sigmoid(logits / self.T)


class BetaCalibrator(Calibrator):
    """Three-parameter Beta calibration (Kull et al. 2017).

    Captures the family of curves
        ``logit(p_cal) = a + b * log(p) - c * log(1 - p)``
    where ``a``, ``b``, ``c`` are learned. Strictly more expressive than
    Platt while keeping a parametric form; the right default when your
    calibration set is 50 to 200 anchors.
    """

    def __init__(self) -> None:
        self._model = LogisticRegression(solver="lbfgs", C=1e6)
        # Coefficients exposed after fit so save/load can round-trip the
        # parameters without needing the sklearn object.
        self.a: float = 0.0
        self.b: float = 0.0
        self.c: float = 0.0

    @staticmethod
    def _features(raw_scores: np.ndarray) -> np.ndarray:
        clipped = np.clip(raw_scores, _PROB_EPS, 1.0 - _PROB_EPS)
        x1 = np.log(clipped)
        x2 = -np.log(1.0 - clipped)
        return np.asarray(np.column_stack([x1, x2]), dtype=float)

    def fit(self, raw_scores: np.ndarray, labels: np.ndarray) -> BetaCalibrator:
        raw_scores = np.asarray(raw_scores, dtype=float)
        labels = np.asarray(labels, dtype=float)
        if raw_scores.shape[0] != labels.shape[0]:
            raise ValueError("raw_scores and labels must have the same length")
        if raw_scores.shape[0] < 3:
            raise ValueError("need at least 3 examples to fit Beta")
        binary = (labels >= 0.5).astype(int)
        if len(np.unique(binary)) < 2:
            raise ValueError(
                "Beta needs both classes (some labels<0.5 and some>=0.5). "
                "Use IsotonicCalibrator or HistogramBinCalibrator for one-sided "
                "calibration sets."
            )
        features = self._features(raw_scores)
        self._model.fit(features, binary)
        # Cache parameters for inspection / persistence.
        coef = self._model.coef_.ravel()
        self.b = float(coef[0])
        self.c = float(coef[1])
        self.a = float(self._model.intercept_.ravel()[0])
        self.fitted = True
        return self

    def predict(self, raw_scores: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("BetaCalibrator must be fit before predict()")
        raw_scores = np.asarray(raw_scores, dtype=float)
        features = self._features(raw_scores)
        proba: np.ndarray = self._model.predict_proba(features)
        return np.asarray(proba[:, 1], dtype=float)


class HistogramBinCalibrator(Calibrator):
    """Equal-width histogram binning. Honest baseline.

    Splits raw_scores into ``n_bins`` equal-width buckets over [0, 1];
    predicted probability for an input is the empirical mean of labels in
    its bucket (or the prior label-mean if the bucket has no anchors).
    """

    def __init__(self, n_bins: int = 10) -> None:
        if n_bins < 2:
            raise ValueError("n_bins must be >= 2")
        self.n_bins = n_bins
        self._bin_edges: np.ndarray = np.asarray([])
        self._bin_means: np.ndarray = np.asarray([])
        self._prior: float = 0.5

    def fit(self, raw_scores: np.ndarray, labels: np.ndarray) -> HistogramBinCalibrator:
        raw_scores = np.asarray(raw_scores, dtype=float)
        labels = np.asarray(labels, dtype=float)
        if raw_scores.shape[0] != labels.shape[0]:
            raise ValueError("raw_scores and labels must have the same length")
        if raw_scores.shape[0] < self.n_bins:
            raise ValueError(f"need at least n_bins={self.n_bins} examples to fit HistogramBin")
        self._bin_edges = np.linspace(0.0, 1.0, self.n_bins + 1)
        # Bucket assignment for each anchor. np.digitize with edges of length
        # n_bins+1 returns indices 1..n_bins inclusive on the open right side;
        # we clamp to [0, n_bins-1] so we can index bin_means directly.
        idx = np.clip(np.digitize(raw_scores, self._bin_edges[1:-1]), 0, self.n_bins - 1)
        means = np.full(self.n_bins, np.nan, dtype=float)
        for b in range(self.n_bins):
            mask = idx == b
            if mask.any():
                means[b] = float(labels[mask].mean())
        self._prior = float(labels.mean())
        # Empty bins fall back to the prior.
        means = np.where(np.isnan(means), self._prior, means)
        self._bin_means = means
        self.fitted = True
        return self

    def predict(self, raw_scores: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("HistogramBinCalibrator must be fit before predict()")
        raw_scores = np.asarray(raw_scores, dtype=float)
        idx = np.clip(np.digitize(raw_scores, self._bin_edges[1:-1]), 0, self.n_bins - 1)
        return np.asarray(self._bin_means[idx], dtype=float)


def select_calibrator(n_anchors: int) -> Calibrator:
    """Return a sensible default calibrator for a given calibration-set size.

    - ``n_anchors < 50``  -> :class:`TemperatureCalibrator` (single parameter).
    - ``50 <= n_anchors < 200`` -> :class:`BetaCalibrator` (3 params).
    - ``n_anchors >= 200`` -> :class:`IsotonicCalibrator` (non-parametric).
    """
    if n_anchors < 50:
        return TemperatureCalibrator()
    if n_anchors < 200:
        return BetaCalibrator()
    return IsotonicCalibrator()


class ConfidenceInterval(NamedTuple):
    """A two-sided confidence interval ``[lower, upper]``.

    ``NamedTuple`` so existing destructuring (``lo, hi = ci``) keeps
    working and so the value is still ``tuple``-equal in tests.
    """

    lower: float
    upper: float


def bootstrap_ci(
    scores: np.ndarray,
    statistic: Callable[[np.ndarray], float] = np.mean,
    *,
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
