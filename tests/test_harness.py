"""End-to-end tests for the JudgeHarness orchestrator."""

from __future__ import annotations

import numpy as np
import pytest

from judgekit import (
    CalibrationSet,
    CalibrationStaleError,
    IsotonicCalibrator,
    JudgeHarness,
    JudgeOutput,
    LabeledExample,
    PlattCalibrator,
)


class _LinearJudge:
    """Mock judge: returns raw=label + Gaussian noise, clamped to [0,1]."""

    def __init__(self, noise: float = 0.05, seed: int = 0) -> None:
        self._rng = np.random.default_rng(seed)
        self._noise = noise
        self._mapping: dict[str, float] = {}

    def register(self, item: str, true: float) -> None:
        self._mapping[item] = true

    def __call__(self, item: str) -> JudgeOutput:
        true = self._mapping[item]
        noisy = float(np.clip(true + self._rng.normal(0, self._noise), 0.0, 1.0))
        return JudgeOutput(score=noisy)


class _ShiftedJudge(_LinearJudge):
    """Like _LinearJudge but biased upward to simulate a model-update shift."""

    def __init__(self, bias: float = 0.3, seed: int = 0) -> None:
        super().__init__(seed=seed)
        self._bias = bias

    def __call__(self, item: str) -> JudgeOutput:
        out = super().__call__(item)
        return JudgeOutput(score=float(np.clip(out.score + self._bias, 0.0, 1.0)))


def _build_calibration_set(judge: _LinearJudge, n: int = 60, seed: int = 0) -> CalibrationSet:
    rng = np.random.default_rng(seed)
    examples = []
    for i in range(n):
        label = float(rng.uniform(0, 1))
        item = f"cal-{i}"
        judge.register(item, label)
        examples.append(LabeledExample(item=item, label=label))
    return CalibrationSet(examples=examples)


def test_harness_happy_path_with_isotonic() -> None:
    judge = _LinearJudge(noise=0.05, seed=1)
    calset = _build_calibration_set(judge, n=400)
    harness = JudgeHarness(judge=judge, calibration_set=calset).fit()
    assert harness.fitted is True

    # Eval items drawn from the SAME distribution as the calibration set so
    # the drift monitor stays stable (this is what production teams want:
    # eval-time inputs broadly resemble the calibration anchors).
    eval_rng = np.random.default_rng(2)
    items = []
    for i in range(200):
        label = float(eval_rng.uniform(0, 1))
        item = f"eval-{i}"
        judge.register(item, label)
        items.append(item)

    result = harness.evaluate(items, rng=eval_rng)
    assert result.n == len(items)
    assert result.calibrated_scores.shape == (len(items),)
    assert 0.0 <= result.point_estimate <= 1.0
    lo, hi = result.confidence_interval
    assert lo <= result.point_estimate <= hi
    assert result.drift.verdict in ("stable", "watch")


def test_harness_raises_on_drift_in_strict_mode() -> None:
    judge = _LinearJudge(noise=0.05, seed=2)
    calset = _build_calibration_set(judge, n=80)
    harness = JudgeHarness(judge=judge, calibration_set=calset).fit()

    # Swap in a shifted judge to simulate an upstream model rev.
    shifted = _ShiftedJudge(bias=0.4, seed=3)
    items = []
    for i, label in enumerate([0.1, 0.2, 0.3, 0.4, 0.5] * 20):
        item = f"shifted-{i}"
        shifted.register(item, label)
        items.append(item)
    harness.judge = shifted  # type: ignore[assignment]

    with pytest.raises(CalibrationStaleError):
        harness.evaluate(items)


def test_harness_non_strict_returns_drift_status() -> None:
    judge = _LinearJudge(noise=0.05, seed=4)
    calset = _build_calibration_set(judge, n=80)
    harness = JudgeHarness(judge=judge, calibration_set=calset, strict=False).fit()

    shifted = _ShiftedJudge(bias=0.4, seed=5)
    items = []
    for i, label in enumerate([0.1, 0.4, 0.7] * 30):
        item = f"shifted-{i}"
        shifted.register(item, label)
        items.append(item)
    harness.judge = shifted  # type: ignore[assignment]

    result = harness.evaluate(items)
    assert result.drift.verdict in ("watch", "drifted")
    # CI is still produced — strict=False does not raise.
    assert result.calibrated_scores.shape == (len(items),)


def test_harness_rejects_tiny_calibration_set() -> None:
    judge = _LinearJudge(seed=6)
    examples = [LabeledExample(item=f"x{i}", label=0.5) for i in range(5)]
    with pytest.raises(ValueError, match=">=10"):
        JudgeHarness(judge=judge, calibration_set=CalibrationSet(examples=examples))


def test_harness_evaluate_before_fit_raises() -> None:
    judge = _LinearJudge(seed=7)
    calset = _build_calibration_set(judge, n=20)
    harness = JudgeHarness(judge=judge, calibration_set=calset)
    with pytest.raises(RuntimeError, match="fit"):
        harness.evaluate(["whatever"])


def test_harness_with_platt_calibrator() -> None:
    judge = _LinearJudge(noise=0.05, seed=8)
    calset = _build_calibration_set(judge, n=160)
    harness = JudgeHarness(
        judge=judge,
        calibration_set=calset,
        calibrator=PlattCalibrator(),
    ).fit()
    # Eval items from the same distribution to keep drift stable.
    eval_rng = np.random.default_rng(9)
    items = []
    for i in range(60):
        label = float(eval_rng.uniform(0, 1))
        item = f"p-{i}"
        judge.register(item, label)
        items.append(item)
    result = harness.evaluate(items, rng=eval_rng)
    assert isinstance(harness.calibrator, PlattCalibrator)
    assert result.point_estimate > 0  # sanity


def test_harness_default_calibrator_is_isotonic() -> None:
    judge = _LinearJudge(seed=10)
    calset = _build_calibration_set(judge, n=20)
    harness = JudgeHarness(judge=judge, calibration_set=calset)
    assert isinstance(harness.calibrator, IsotonicCalibrator)
