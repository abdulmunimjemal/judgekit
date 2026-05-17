"""Distribution-drift monitors for judge score streams.

When the judge model is updated upstream (Anthropic / OpenAI roll a new
checkpoint), the score *distribution* shifts even if the prompts and
calibration set haven't changed. judgekit treats this as a calibration
invalidation signal: re-calibrate, or refuse to ship.

Two divergence measures are exposed:

- ``kl_divergence``: classical Kullback-Leibler ``KL(P || Q)``. Sensitive
  to changes in the tail; asymmetric.
- ``psi``: Population Stability Index, common in credit-risk monitoring.
  Symmetric variant of KL, with widely-cited thresholds:
    * < 0.10 — no significant shift
    * 0.10 to 0.25 — moderate shift, worth investigating
    * > 0.25 — material shift, do not trust calibration

The ``DriftMonitor`` class is a thin orchestrator around these so the
harness can call ``monitor.check(new_scores)`` and get back a status.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# A small epsilon so empty bins don't blow up log(0).
_EPS = 1e-12


def _histogram(scores: np.ndarray, bins: int = 10) -> np.ndarray:
    """Probability mass per bin over [0, 1]. Returns shape (bins,)."""
    scores = np.asarray(scores, dtype=float)
    counts, _ = np.histogram(scores, bins=bins, range=(0.0, 1.0))
    total = counts.sum()
    if total == 0:
        return np.full(bins, 1.0 / bins)
    return counts / total


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


@dataclass
class DriftStatus:
    psi: float
    kl: float
    verdict: str  # "stable" | "watch" | "drifted"
    psi_threshold_warn: float
    psi_threshold_fail: float

    @property
    def is_drifted(self) -> bool:
        return self.verdict == "drifted"


class DriftMonitor:
    """Snapshots a reference score distribution and compares new batches to it.

    Usage:
        monitor = DriftMonitor(reference_scores=calibration_raw_scores)
        status = monitor.check(new_judge_scores)
        if status.is_drifted:
            raise CalibrationStaleError(...)

    Thresholds default to the PSI rules-of-thumb from the credit-risk
    literature; override per-domain when you have your own ground truth.
    """

    def __init__(
        self,
        reference_scores: np.ndarray,
        bins: int = 10,
        psi_warn: float = 0.10,
        psi_fail: float = 0.25,
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

    def check(self, new_scores: np.ndarray) -> DriftStatus:
        new = np.asarray(new_scores, dtype=float)
        if new.size == 0:
            raise ValueError("new_scores must be non-empty")
        psi_v = psi(self.reference, new, bins=self.bins)
        kl_v = kl_divergence(self.reference, new, bins=self.bins)
        if psi_v >= self.psi_fail:
            verdict = "drifted"
        elif psi_v >= self.psi_warn:
            verdict = "watch"
        else:
            verdict = "stable"
        return DriftStatus(
            psi=psi_v,
            kl=kl_v,
            verdict=verdict,
            psi_threshold_warn=self.psi_warn,
            psi_threshold_fail=self.psi_fail,
        )
