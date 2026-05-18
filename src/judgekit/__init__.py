"""judgekit — calibrate, monitor, and refuse to ship miscalibrated LLM judges."""

from judgekit.calibration import (
    BetaCalibrator,
    Calibrator,
    HistogramBinCalibrator,
    IsotonicCalibrator,
    PlattCalibrator,
    TemperatureCalibrator,
    bootstrap_ci,
    select_calibrator,
)
from judgekit.drift import DriftMonitor, DriftStatus, kl_divergence, ks_test, psi, wasserstein
from judgekit.harness import CalibrationStaleError, EvalResult, JudgeHarness
from judgekit.judge import CalibrationSet, Judge, JudgeOutput, LabeledExample

__version__ = "0.1.0"

__all__ = [
    "BetaCalibrator",
    "CalibrationSet",
    "CalibrationStaleError",
    "Calibrator",
    "DriftMonitor",
    "DriftStatus",
    "EvalResult",
    "HistogramBinCalibrator",
    "IsotonicCalibrator",
    "Judge",
    "JudgeHarness",
    "JudgeOutput",
    "LabeledExample",
    "PlattCalibrator",
    "TemperatureCalibrator",
    "bootstrap_ci",
    "kl_divergence",
    "ks_test",
    "psi",
    "select_calibrator",
    "wasserstein",
]
