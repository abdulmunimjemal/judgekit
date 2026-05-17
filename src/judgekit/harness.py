"""JudgeHarness — the only object most users need to touch.

Ties a Judge, a CalibrationSet, a Calibrator, and a DriftMonitor together
into a single ``.evaluate(items)`` call that returns calibrated scores
with confidence intervals — or raises ``CalibrationStaleError`` when the
score distribution has drifted past the configured threshold.

The "refuse to ship" stance is deliberate. If you want a warning-only mode,
pass ``strict=False`` and inspect the ``EvalResult.drift`` field yourself.

Typical lifecycle:

    judge = MyLLMJudge(...)
    calset = CalibrationSet(examples=[...])
    harness = JudgeHarness(judge, calset).fit()

    # In CI, before publishing eval results:
    result = harness.evaluate(new_eval_items)
    print(result.point_estimate, result.confidence_interval)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from judgekit.calibration import Calibrator, IsotonicCalibrator, bootstrap_ci
from judgekit.drift import DriftMonitor, DriftStatus
from judgekit.judge import CalibrationSet, Judge


class CalibrationStaleError(RuntimeError):
    """Raised when score-distribution drift exceeds the harness threshold."""

    def __init__(self, status: DriftStatus) -> None:
        self.status = status
        super().__init__(
            f"Judge calibration is stale: PSI={status.psi:.4f} "
            f"(fail threshold={status.psi_threshold_fail:.2f}). "
            "Re-fit on a fresh calibration set before publishing eval results."
        )


@dataclass
class EvalResult:
    """One harness.evaluate() output."""

    calibrated_scores: np.ndarray
    raw_scores: np.ndarray
    point_estimate: float
    confidence_interval: tuple[float, float]
    confidence: float
    drift: DriftStatus
    n: int = field(init=False)

    def __post_init__(self) -> None:
        self.n = int(self.calibrated_scores.size)

    def __repr__(self) -> str:  # pragma: no cover — repr only
        lo, hi = self.confidence_interval
        return (
            f"EvalResult(n={self.n}, mean={self.point_estimate:.4f}, "
            f"CI{int(self.confidence * 100)}=[{lo:.4f}, {hi:.4f}], "
            f"drift={self.drift.verdict}, psi={self.drift.psi:.4f})"
        )


class JudgeHarness:
    """Orchestrates judge → calibrator → drift monitor."""

    def __init__(
        self,
        judge: Judge,
        calibration_set: CalibrationSet,
        calibrator: Calibrator | None = None,
        confidence: float = 0.95,
        psi_warn: float = 0.10,
        psi_fail: float = 0.25,
        strict: bool = True,
    ) -> None:
        if not 0.0 < confidence < 1.0:
            raise ValueError("confidence must be in (0, 1)")
        if len(calibration_set) < 10:
            raise ValueError(
                f"calibration_set needs >=10 examples, got {len(calibration_set)}"
            )
        self.judge = judge
        self.calibration_set = calibration_set
        self.calibrator: Calibrator = calibrator if calibrator is not None else IsotonicCalibrator()
        self.confidence = confidence
        self.psi_warn = psi_warn
        self.psi_fail = psi_fail
        self.strict = strict
        self._drift_monitor: DriftMonitor | None = None
        self._calibration_raw: np.ndarray | None = None
        self._fitted: bool = False

    @property
    def fitted(self) -> bool:
        return self._fitted

    def fit(self) -> JudgeHarness:
        """Run the judge on the calibration set and fit calibrator + drift baseline."""
        labels = np.asarray(self.calibration_set.labels(), dtype=float)
        raw = np.asarray(
            [self.judge(item).score for item in self.calibration_set.items()],
            dtype=float,
        )
        self.calibrator.fit(raw, labels)
        self._drift_monitor = DriftMonitor(
            reference_scores=raw,
            psi_warn=self.psi_warn,
            psi_fail=self.psi_fail,
        )
        self._calibration_raw = raw
        self._fitted = True
        return self

    def evaluate(
        self,
        items: list[str],
        *,
        rng: np.random.Generator | None = None,
        n_resamples: int = 1000,
    ) -> EvalResult:
        """Score new items through the judge → calibrator pipeline.

        Raises ``CalibrationStaleError`` when strict=True and PSI exceeds fail
        threshold.
        """
        if not self._fitted:
            raise RuntimeError("JudgeHarness.fit() must be called before evaluate()")
        assert self._drift_monitor is not None  # for type checkers
        if not items:
            raise ValueError("items must be non-empty")

        raw = np.asarray([self.judge(item).score for item in items], dtype=float)
        drift = self._drift_monitor.check(raw)
        if self.strict and drift.is_drifted:
            raise CalibrationStaleError(drift)

        calibrated = self.calibrator.predict(raw)
        point, lo, hi = bootstrap_ci(
            calibrated,
            n_resamples=n_resamples,
            confidence=self.confidence,
            rng=rng,
        )
        return EvalResult(
            calibrated_scores=calibrated,
            raw_scores=raw,
            point_estimate=point,
            confidence_interval=(lo, hi),
            confidence=self.confidence,
            drift=drift,
        )
