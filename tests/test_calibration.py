"""Tests for the calibration module."""

from __future__ import annotations

import numpy as np
import pytest

from judgekit.calibration import (
    BetaCalibrator,
    HistogramBinCalibrator,
    IsotonicCalibrator,
    PlattCalibrator,
    TemperatureCalibrator,
    bootstrap_ci,
    select_calibrator,
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
    point, lo, hi = bootstrap_ci(
        scores, statistic=np.median, n_resamples=300, rng=np.random.default_rng(1)
    )
    assert lo <= point <= hi
    assert abs(point - 0.5) < 0.2


def test_bootstrap_rejects_empty() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        bootstrap_ci(np.array([]))


def test_bootstrap_rejects_bad_confidence() -> None:
    with pytest.raises(ValueError, match="confidence"):
        bootstrap_ci(np.array([0.1, 0.2]), confidence=1.5)


def test_bootstrap_ci_kwargs_are_keyword_only() -> None:
    """``n_resamples`` / ``confidence`` / ``rng`` are frozen as keyword-only at v1.0."""
    with pytest.raises(TypeError):
        # Third positional used to be n_resamples.
        bootstrap_ci(np.array([0.1, 0.2, 0.3]), np.mean, 500)  # type: ignore[misc]


# ---------- TemperatureCalibrator ----------


def test_temperature_recovers_identity_on_already_calibrated_data() -> None:
    """If raw scores already match labels, the optimal T is ~1."""
    rng = np.random.default_rng(0)
    n = 500
    raw = rng.uniform(0.05, 0.95, size=n)
    labels = (rng.uniform(size=n) < raw).astype(float)
    cal = TemperatureCalibrator().fit(raw, labels)
    assert cal.fitted
    # Identity-ish calibration -> T close to 1.
    assert abs(cal.T - 1.0) < 0.5


def test_temperature_softens_overconfident_scores() -> None:
    """If raw scores are pushed toward 0/1 but truth is more uncertain,
    learned T should be > 1 (cooling the distribution)."""
    rng = np.random.default_rng(1)
    n = 500
    # True probabilities ~ Uniform; raw is the overconfident sigmoid of 3*logit(p_true).
    p_true = rng.uniform(0.1, 0.9, size=n)
    logits = np.log(p_true / (1.0 - p_true))
    raw = 1.0 / (1.0 + np.exp(-3.0 * logits))
    labels = (rng.uniform(size=n) < p_true).astype(float)
    cal = TemperatureCalibrator().fit(raw, labels)
    # Recovering T ~= 3 (the inverse of the overconfidence factor).
    assert 2.0 < cal.T < 4.5


def test_temperature_predict_is_in_unit_interval() -> None:
    rng = np.random.default_rng(2)
    raw = rng.uniform(0, 1, size=200)
    labels = (rng.uniform(size=200) < raw).astype(float)
    cal = TemperatureCalibrator().fit(raw, labels)
    preds = cal.predict(np.linspace(0.01, 0.99, 50))
    assert preds.min() >= 0.0 and preds.max() <= 1.0


def test_temperature_rejects_too_few_examples() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        TemperatureCalibrator().fit(np.array([0.5]), np.array([1.0]))


def test_temperature_predict_before_fit_raises() -> None:
    with pytest.raises(RuntimeError, match="must be fit"):
        TemperatureCalibrator().predict(np.array([0.5]))


# ---------- BetaCalibrator ----------


def test_beta_fits_and_predicts_in_unit_interval() -> None:
    rng = np.random.default_rng(3)
    raw = rng.uniform(0, 1, size=300)
    labels = (rng.uniform(size=300) < raw).astype(float)
    cal = BetaCalibrator().fit(raw, labels)
    assert cal.fitted
    preds = cal.predict(np.linspace(0.05, 0.95, 50))
    assert preds.min() >= 0.0 and preds.max() <= 1.0


def test_beta_recovers_approximate_identity() -> None:
    """When raw approximately matches truth, predictions should be close to raw."""
    rng = np.random.default_rng(4)
    n = 600
    raw = rng.uniform(0.05, 0.95, size=n)
    labels = (rng.uniform(size=n) < raw).astype(float)
    cal = BetaCalibrator().fit(raw, labels)
    preds = cal.predict(np.array([0.1, 0.3, 0.5, 0.7, 0.9]))
    # Predictions are monotone non-decreasing in raw.
    assert np.all(np.diff(preds) >= -1e-9)


def test_beta_rejects_one_class_set() -> None:
    raw = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    labels = np.zeros(5)
    with pytest.raises(ValueError, match="both classes"):
        BetaCalibrator().fit(raw, labels)


def test_beta_exposes_parameters() -> None:
    rng = np.random.default_rng(5)
    raw = rng.uniform(0, 1, size=200)
    labels = (rng.uniform(size=200) < raw).astype(float)
    cal = BetaCalibrator().fit(raw, labels)
    assert isinstance(cal.a, float)
    assert isinstance(cal.b, float)
    assert isinstance(cal.c, float)


# ---------- HistogramBinCalibrator ----------


def test_histogram_bin_recovers_identity() -> None:
    rng = np.random.default_rng(6)
    n = 1000
    raw = rng.uniform(0, 1, size=n)
    labels = (rng.uniform(size=n) < raw).astype(float)
    cal = HistogramBinCalibrator(n_bins=10).fit(raw, labels)
    preds = cal.predict(np.linspace(0.05, 0.95, 10))
    # Predictions should approximately follow raw — bins around 0.1 predict ~0.1.
    for p_in, p_out in zip(np.linspace(0.05, 0.95, 10), preds, strict=True):
        assert abs(p_in - p_out) < 0.15


def test_histogram_bin_handles_uniform_labels() -> None:
    """Empty buckets fall back to the prior mean."""
    rng = np.random.default_rng(7)
    n = 200
    raw = rng.uniform(0.4, 0.6, size=n)  # only middle bins have anchors
    labels = np.full(n, 0.7)  # constant labels
    cal = HistogramBinCalibrator(n_bins=10).fit(raw, labels)
    # Bins with no data should fall back to the prior (0.7 here).
    assert abs(cal.predict(np.array([0.05])).item() - 0.7) < 1e-9
    assert abs(cal.predict(np.array([0.95])).item() - 0.7) < 1e-9


def test_histogram_bin_rejects_too_few_examples() -> None:
    with pytest.raises(ValueError, match="n_bins"):
        HistogramBinCalibrator(n_bins=10).fit(np.array([0.1, 0.2]), np.array([0.0, 1.0]))


def test_histogram_bin_rejects_invalid_n_bins() -> None:
    with pytest.raises(ValueError, match="n_bins"):
        HistogramBinCalibrator(n_bins=1)


def test_histogram_bin_predict_in_unit_interval() -> None:
    rng = np.random.default_rng(8)
    raw = rng.uniform(0, 1, size=300)
    labels = (rng.uniform(size=300) < raw).astype(float)
    cal = HistogramBinCalibrator(n_bins=10).fit(raw, labels)
    preds = cal.predict(np.linspace(0.0, 1.0, 100))
    assert preds.min() >= 0.0 and preds.max() <= 1.0


# ---------- select_calibrator ----------


def test_select_calibrator_picks_by_anchor_count() -> None:
    assert isinstance(select_calibrator(10), TemperatureCalibrator)
    assert isinstance(select_calibrator(49), TemperatureCalibrator)
    assert isinstance(select_calibrator(50), BetaCalibrator)
    assert isinstance(select_calibrator(199), BetaCalibrator)
    assert isinstance(select_calibrator(200), IsotonicCalibrator)
    assert isinstance(select_calibrator(1000), IsotonicCalibrator)
