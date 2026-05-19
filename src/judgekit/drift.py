"""Distribution-drift monitors for judge score streams.

When the judge model is updated upstream (Anthropic / OpenAI roll a new
checkpoint), the score *distribution* shifts even if the prompts and
calibration set haven't changed. judgekit treats this as a calibration
invalidation signal: re-calibrate, or refuse to ship.

Four divergence / distance measures are exposed:

- ``kl_divergence``: classical Kullback-Leibler ``KL(P || Q)``. Sensitive to
  changes in the tail; asymmetric.
- ``psi``: Population Stability Index, common in credit-risk monitoring.
  Symmetric variant of KL, with widely-cited thresholds (<0.10 stable,
  0.10-0.25 watch, >0.25 drifted).
- ``ks_test``: two-sample Kolmogorov-Smirnov test. Non-parametric; returns
  a (statistic, p-value) pair. Wrapped from ``scipy.stats.ks_2samp``.
- ``wasserstein``: 1-Wasserstein (earth-mover) distance. Wrapped from
  ``scipy.stats.wasserstein_distance``.

The ``DriftMonitor`` class is the orchestrator. By default it computes PSI
and KL and verdicts on PSI thresholds. Pass ``method="all"`` to also compute
KS and Wasserstein; pass ``method="ks"`` or ``method="wasserstein"`` to use
those alone as the verdict.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.stats import ks_2samp, wasserstein_distance

# A small epsilon so empty bins don't blow up log(0).
_EPS = 1e-12

DriftMethod = Literal["psi", "ks", "wasserstein", "all"]


def _histogram(scores: np.ndarray, bins: int = 10) -> np.ndarray:
    """Probability mass per bin over [0, 1]. Returns shape (bins,)."""
    scores = np.asarray(scores, dtype=float)
    counts, _ = np.histogram(scores, bins=bins, range=(0.0, 1.0))
    total = counts.sum()
    if total == 0:
        return np.full(bins, 1.0 / bins)
    return np.asarray(counts / total, dtype=float)


def kl_divergence(p_scores: np.ndarray, q_scores: np.ndarray, bins: int = 10) -> float:
    """KL(P || Q) over score histograms.

    Both inputs are 1-D arrays of scores in [0, 1]. Returns a non-negative
    float; 0 means identical distributions.
    """
    p = _histogram(p_scores, bins=bins) + _EPS
    q = _histogram(q_scores, bins=bins) + _EPS
    return float(np.sum(p * np.log(p / q)))


def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index.

    PSI(E, A) = sum_i (A_i - E_i) * log(A_i / E_i) where A_i, E_i are bin
    probabilities. Symmetric in E and A. ``expected`` is the reference
    distribution (typically captured at calibration time), ``actual`` is
    the new sample being judged.
    """
    e = _histogram(expected, bins=bins) + _EPS
    a = _histogram(actual, bins=bins) + _EPS
    return float(np.sum((a - e) * np.log(a / e)))


def ks_test(reference: np.ndarray, sample: np.ndarray) -> tuple[float, float]:
    """Two-sample Kolmogorov-Smirnov test.

    Returns ``(statistic, p_value)``. The statistic is the supremum of the
    absolute differences between the two empirical CDFs; the p-value is
    the probability of seeing a statistic at least that extreme under the
    null hypothesis that both samples come from the same distribution.

    Rule-of-thumb threshold: ``p_value < 0.01`` -> drifted.
    """
    reference = np.asarray(reference, dtype=float)
    sample = np.asarray(sample, dtype=float)
    if reference.size == 0 or sample.size == 0:
        raise ValueError("reference and sample must both be non-empty")
    result = ks_2samp(reference, sample)
    return float(result.statistic), float(result.pvalue)


def wasserstein(reference: np.ndarray, sample: np.ndarray) -> float:
    """1-Wasserstein (earth-mover) distance.

    Wraps ``scipy.stats.wasserstein_distance``. Interpretable as the
    "area between the two CDFs" — i.e. the minimum amount of probability
    mass you'd need to move to turn one distribution into the other.
    """
    reference = np.asarray(reference, dtype=float)
    sample = np.asarray(sample, dtype=float)
    if reference.size == 0 or sample.size == 0:
        raise ValueError("reference and sample must both be non-empty")
    return float(wasserstein_distance(reference, sample))


@dataclass
class DriftStatus:
    """Result of a drift check.

    ``psi`` and ``kl`` are always populated. ``ks_statistic``, ``ks_pvalue``,
    and ``wasserstein_distance`` are populated when the monitor's ``method``
    is ``"ks"``, ``"wasserstein"``, or ``"all"``.
    """

    psi: float
    kl: float
    verdict: str  # "stable" | "watch" | "drifted"
    psi_threshold_warn: float
    psi_threshold_fail: float
    ks_statistic: float | None = None
    ks_pvalue: float | None = None
    wasserstein_distance: float | None = None

    @property
    def is_drifted(self) -> bool:
        return self.verdict == "drifted"


class DriftMonitor:
    """Snapshots a reference score distribution and compares new batches to it.

    Usage::

        monitor = DriftMonitor(reference_scores=calibration_raw_scores)
        status = monitor.check(new_judge_scores)
        if status.is_drifted:
            raise CalibrationStaleError(...)

    ``method`` controls how the verdict is produced:

    - ``"psi"`` (default): verdict on PSI thresholds. KL is also reported.
    - ``"ks"``: verdict on KS p-value (drifted if p < ks_p_threshold).
    - ``"wasserstein"``: verdict on Wasserstein distance threshold.
    - ``"all"``: compute every measure; verdict combines PSI and KS
      (drifted if either says drifted; PSI's `watch` tier wins ties).
    """

    def __init__(
        self,
        reference_scores: np.ndarray,
        *,
        bins: int = 10,
        psi_warn: float = 0.10,
        psi_fail: float = 0.25,
        method: DriftMethod = "psi",
        ks_p_threshold: float = 0.01,
        wasserstein_threshold: float = 0.10,
    ) -> None:
        ref = np.asarray(reference_scores, dtype=float)
        if ref.size < 10:
            raise ValueError(
                f"reference needs >=10 samples for a meaningful histogram, got {ref.size}"
            )
        self.reference = ref
        self.bins = bins
        self.psi_warn = psi_warn
        self.psi_fail = psi_fail
        self.method: DriftMethod = method
        self.ks_p_threshold = ks_p_threshold
        self.wasserstein_threshold = wasserstein_threshold

    def check(self, new_scores: np.ndarray) -> DriftStatus:
        new = np.asarray(new_scores, dtype=float)
        if new.size == 0:
            raise ValueError("new_scores must be non-empty")

        psi_v = psi(self.reference, new, bins=self.bins)
        kl_v = kl_divergence(self.reference, new, bins=self.bins)

        ks_stat: float | None = None
        ks_p: float | None = None
        wasser: float | None = None
        if self.method in ("ks", "all"):
            ks_stat, ks_p = ks_test(self.reference, new)
        if self.method in ("wasserstein", "all"):
            wasser = wasserstein(self.reference, new)

        verdict = self._verdict(psi_v, ks_p, wasser)

        return DriftStatus(
            psi=psi_v,
            kl=kl_v,
            verdict=verdict,
            psi_threshold_warn=self.psi_warn,
            psi_threshold_fail=self.psi_fail,
            ks_statistic=ks_stat,
            ks_pvalue=ks_p,
            wasserstein_distance=wasser,
        )

    def _verdict(self, psi_v: float, ks_p: float | None, wasser: float | None) -> str:
        if self.method == "psi":
            return self._psi_verdict(psi_v)
        if self.method == "ks":
            assert ks_p is not None  # set by check()
            return "drifted" if ks_p < self.ks_p_threshold else "stable"
        if self.method == "wasserstein":
            assert wasser is not None
            return "drifted" if wasser >= self.wasserstein_threshold else "stable"
        # "all": combine PSI + KS. Drifted if either; otherwise PSI's tier.
        psi_tier = self._psi_verdict(psi_v)
        ks_tier = "stable"
        if ks_p is not None and ks_p < self.ks_p_threshold:
            ks_tier = "drifted"
        if psi_tier == "drifted" or ks_tier == "drifted":
            return "drifted"
        if psi_tier == "watch":
            return "watch"
        return "stable"

    def _psi_verdict(self, psi_v: float) -> str:
        if psi_v >= self.psi_fail:
            return "drifted"
        if psi_v >= self.psi_warn:
            return "watch"
        return "stable"
