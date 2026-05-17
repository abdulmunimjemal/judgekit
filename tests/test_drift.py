"""Tests for the drift module."""

from __future__ import annotations

import numpy as np
import pytest

from judgekit.drift import DriftMonitor, kl_divergence, psi


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
