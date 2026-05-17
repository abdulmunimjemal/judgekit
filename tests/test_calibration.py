"""Tests for the calibration module."""

from __future__ import annotations

import numpy as np
import pytest

from judgekit.calibration import (
    IsotonicCalibrator,
    PlattCalibrator,
    bootstrap_ci,
)


def test_platt_fits_and_predicts_a_clean_separation() -> None:
    # Clear monotone relationship: raw=0 -> label=0, raw=1 -> label=1.
    rng = np.random.default_rng(0)
    raw = rng.uniform(0, 1, size=200)
    labels = (raw + rng.normal(0, 0.05, size=200) > 0.5).astype(float)

    cal = PlattCalibrator().fit(raw, labels)
    preds = cal.predict(np.array([0.05, 0.5, 0.95]))

    assert cal.fitted is True
    assert preds.shape == (3,)
    # Monotone in raw score.
    assert preds[0] < preds[1] < preds[2]
    # Endpoints close to truth.
    assert preds[0] < 0.2
    assert preds[2] > 0.8


def test_platt_rejects_one_class_calibration_sets() -> None:
    raw = np.array([0.1, 0.2, 0.3, 0.4])
    labels = np.array([0.0, 0.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="both classes"):
        PlattCalibrator().fit(raw, labels)


def test_platt_predict_before_fit_raises() -> None:
    with pytest.raises(RuntimeError, match="must be fit"):
        PlattCalibrator().predict(np.array([0.5]))


def test_platt_length_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="same length"):
        PlattCalibrator().fit(np.array([0.1, 0.9]), np.array([0.0]))


def test_isotonic_is_monotone() -> None:
    raw = np.linspace(0, 1, 100)
    labels = raw  # perfect linear
    cal = IsotonicCalibrator().fit(raw, labels)
    preds = cal.predict(np.array([0.0, 0.25, 0.5, 0.75, 1.0]))
    # Monotone non-decreasing.
    assert np.all(np.diff(preds) >= 0)
    # Output clipped to [0, 1].
    assert preds.min() >= 0.0
    assert preds.max() <= 1.0


def test_isotonic_handles_noisy_data() -> None:
    rng = np.random.default_rng(42)
    raw = rng.uniform(0, 1, size=200)
    labels = np.clip(raw + rng.normal(0, 0.1, size=200), 0, 1)
    cal = IsotonicCalibrator().fit(raw, labels)
    # Approx identity recovery.
    preds = cal.predict(np.array([0.25, 0.5, 0.75]))
    assert abs(preds[0] - 0.25) < 0.15
    assert abs(preds[1] - 0.5) < 0.15
    assert abs(preds[2] - 0.75) < 0.15


def test_bootstrap_ci_covers_truth_for_simple_mean() -> None:
    rng = np.random.default_rng(7)
    scores = rng.normal(0.6, 0.1, size=200)
    point, lo, hi = bootstrap_ci(scores, n_resamples=500, rng=rng)
    assert lo <= point <= hi
    # Point estimate is close to the sample mean (which is close to 0.6 for n=200).
    assert abs(point - float(scores.mean())) < 1e-9
    # The 95% CI on a large-ish sample should be tight and near the truth.
    assert abs(point - 0.6) < 0.05


def test_bootstrap_ci_with_custom_statistic() -> None:
    scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    point, lo, hi = bootstrap_ci(scores, statistic=np.median, n_resamples=300, rng=np.random.default_rng(1))
    assert lo <= point <= hi
    assert abs(point - 0.5) < 0.2


def test_bootstrap_rejects_empty() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        bootstrap_ci(np.array([]))


def test_bootstrap_rejects_bad_confidence() -> None:
    with pytest.raises(ValueError, match="confidence"):
        bootstrap_ci(np.array([0.1, 0.2]), confidence=1.5)
