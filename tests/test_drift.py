"""Tests for the drift module."""

from __future__ import annotations

import numpy as np
import pytest

from judgekit.drift import DriftMonitor, kl_divergence, ks_test, psi, wasserstein


def test_kl_zero_when_distributions_identical() -> None:
    rng = np.random.default_rng(0)
    a = rng.uniform(0, 1, size=500)
    # Same array → identical histograms.
    assert kl_divergence(a, a) < 1e-6


def test_kl_positive_when_shifted() -> None:
    rng = np.random.default_rng(1)
    a = rng.uniform(0.0, 0.5, size=500)
    b = rng.uniform(0.5, 1.0, size=500)
    assert kl_divergence(a, b) > 0.5  # very different bins


def test_psi_symmetric_in_inputs() -> None:
    rng = np.random.default_rng(2)
    a = rng.uniform(0.0, 0.6, size=500)
    b = rng.uniform(0.3, 1.0, size=500)
    # PSI is symmetric by construction.
    assert abs(psi(a, b) - psi(b, a)) < 1e-9


def test_psi_thresholds_observed() -> None:
    rng = np.random.default_rng(3)
    ref = rng.normal(0.5, 0.1, size=1000)
    same = rng.normal(0.5, 0.1, size=1000)
    drifted = rng.normal(0.85, 0.05, size=1000)
    assert psi(ref, same) < 0.10  # stable
    assert psi(ref, drifted) > 0.25  # material drift


def test_drift_monitor_stable_verdict() -> None:
    rng = np.random.default_rng(4)
    ref = rng.normal(0.5, 0.1, size=500)
    sample = rng.normal(0.5, 0.1, size=500)
    monitor = DriftMonitor(reference_scores=ref)
    status = monitor.check(sample)
    assert status.verdict == "stable"
    assert status.is_drifted is False


def test_drift_monitor_flags_material_drift() -> None:
    rng = np.random.default_rng(5)
    ref = rng.normal(0.5, 0.1, size=500)
    sample = rng.normal(0.85, 0.05, size=500)
    monitor = DriftMonitor(reference_scores=ref)
    status = monitor.check(sample)
    assert status.verdict == "drifted"
    assert status.is_drifted is True


def test_drift_monitor_rejects_tiny_reference() -> None:
    with pytest.raises(ValueError, match=">=10 samples"):
        DriftMonitor(reference_scores=np.array([0.1, 0.2]))


def test_drift_monitor_rejects_empty_sample() -> None:
    monitor = DriftMonitor(reference_scores=np.linspace(0, 1, 50))
    with pytest.raises(ValueError, match="non-empty"):
        monitor.check(np.array([]))


# ---------- ks_test ----------


def test_ks_identical_samples_have_high_pvalue() -> None:
    rng = np.random.default_rng(20)
    a = rng.uniform(0, 1, size=500)
    stat, p = ks_test(a, a)
    assert stat < 1e-9
    assert p > 0.99


def test_ks_shifted_samples_have_tiny_pvalue() -> None:
    rng = np.random.default_rng(21)
    a = rng.normal(0.3, 0.05, size=500)
    b = rng.normal(0.7, 0.05, size=500)
    stat, p = ks_test(a, b)
    assert stat > 0.9
    assert p < 1e-50


def test_ks_matches_scipy_reference() -> None:
    from scipy.stats import ks_2samp

    rng = np.random.default_rng(22)
    a = rng.uniform(0, 1, size=200)
    b = rng.uniform(0.1, 0.9, size=200)
    ours = ks_test(a, b)
    ref = ks_2samp(a, b)
    assert abs(ours[0] - float(ref.statistic)) < 1e-12
    assert abs(ours[1] - float(ref.pvalue)) < 1e-12


def test_ks_rejects_empty() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        ks_test(np.array([]), np.array([0.5, 0.6]))


# ---------- wasserstein ----------


def test_wasserstein_zero_on_identical() -> None:
    rng = np.random.default_rng(30)
    a = rng.uniform(0, 1, size=500)
    assert wasserstein(a, a) < 1e-12


def test_wasserstein_increases_with_shift() -> None:
    rng = np.random.default_rng(31)
    a = rng.normal(0.3, 0.05, size=500)
    b = rng.normal(0.7, 0.05, size=500)
    d_far = wasserstein(a, b)
    c = rng.normal(0.32, 0.05, size=500)
    d_near = wasserstein(a, c)
    assert d_far > d_near
    # ~0.4 mean shift -> wasserstein should be close to 0.4 here.
    assert 0.30 < d_far < 0.50


def test_wasserstein_matches_scipy_reference() -> None:
    from scipy.stats import wasserstein_distance as ref_w

    rng = np.random.default_rng(32)
    a = rng.uniform(0, 1, size=200)
    b = rng.uniform(0.2, 0.8, size=200)
    ours = wasserstein(a, b)
    ref = float(ref_w(a, b))
    assert abs(ours - ref) < 1e-12


def test_wasserstein_rejects_empty() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        wasserstein(np.array([0.1]), np.array([]))


# ---------- DriftMonitor extended methods ----------


def test_drift_monitor_ks_method() -> None:
    rng = np.random.default_rng(40)
    ref = rng.normal(0.5, 0.05, size=500)
    sample = rng.normal(0.5, 0.05, size=500)
    monitor = DriftMonitor(reference_scores=ref, method="ks")
    status = monitor.check(sample)
    assert status.ks_statistic is not None
    assert status.ks_pvalue is not None
    assert status.wasserstein_distance is None
    assert status.verdict == "stable"


def test_drift_monitor_ks_method_flags_drift() -> None:
    rng = np.random.default_rng(41)
    ref = rng.normal(0.3, 0.04, size=500)
    sample = rng.normal(0.7, 0.04, size=500)
    monitor = DriftMonitor(reference_scores=ref, method="ks")
    status = monitor.check(sample)
    assert status.verdict == "drifted"
    assert status.ks_pvalue is not None
    assert status.ks_pvalue < 1e-50


def test_drift_monitor_wasserstein_method() -> None:
    rng = np.random.default_rng(42)
    ref = rng.normal(0.3, 0.05, size=500)
    sample = rng.normal(0.7, 0.05, size=500)
    monitor = DriftMonitor(reference_scores=ref, method="wasserstein", wasserstein_threshold=0.1)
    status = monitor.check(sample)
    assert status.verdict == "drifted"
    assert status.wasserstein_distance is not None
    assert status.wasserstein_distance > 0.1


def test_drift_monitor_all_method_combines_signals() -> None:
    rng = np.random.default_rng(43)
    ref = rng.normal(0.5, 0.05, size=500)
    sample = rng.normal(0.5, 0.05, size=500)
    monitor = DriftMonitor(reference_scores=ref, method="all")
    status = monitor.check(sample)
    # all should populate every field.
    assert status.ks_statistic is not None
    assert status.ks_pvalue is not None
    assert status.wasserstein_distance is not None
    assert status.verdict == "stable"


def test_drift_monitor_all_method_flags_when_either_drifts() -> None:
    rng = np.random.default_rng(44)
    ref = rng.normal(0.3, 0.04, size=500)
    sample = rng.normal(0.7, 0.04, size=500)
    monitor = DriftMonitor(reference_scores=ref, method="all")
    status = monitor.check(sample)
    assert status.verdict == "drifted"


def test_drift_monitor_psi_method_skips_extras() -> None:
    rng = np.random.default_rng(45)
    ref = rng.normal(0.5, 0.05, size=500)
    sample = rng.normal(0.5, 0.05, size=500)
    monitor = DriftMonitor(reference_scores=ref, method="psi")
    status = monitor.check(sample)
    # psi method does NOT populate KS / Wasserstein.
    assert status.ks_statistic is None
    assert status.ks_pvalue is None
    assert status.wasserstein_distance is None
