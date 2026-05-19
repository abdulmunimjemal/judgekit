"""Tests for JudgeHarness.save / .load and the persistence module."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from judgekit import (
    CalibrationSet,
    JudgeHarness,
    JudgeOutput,
    LabeledExample,
    PlattCalibrator,
    StateFormatError,
    TemperatureCalibrator,
    load_metadata,
)


class _SeededJudge:
    """Mock judge: noisy version of the underlying label."""

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


def _make_harness(n: int = 200, seed: int = 0) -> JudgeHarness:
    judge = _SeededJudge(noise=0.05, seed=seed)
    rng = np.random.default_rng(seed + 1)
    examples = []
    for i in range(n):
        label = float(rng.uniform(0, 1))
        item = f"cal-{i}"
        judge.register(item, label)
        examples.append(LabeledExample(item=item, label=label))
    return JudgeHarness(judge=judge, calibration_set=CalibrationSet(examples=examples)).fit()


def test_save_writes_expected_files(tmp_path: Path) -> None:
    harness = _make_harness()
    target = tmp_path / "judge.judgekit"
    harness.save(target)
    assert (target / "state.json").is_file()
    assert (target / "calibrator.pkl").is_file()
    assert (target / "drift_reference.npy").is_file()


def test_save_metadata_round_trips(tmp_path: Path) -> None:
    harness = _make_harness()
    target = tmp_path / "judge.judgekit"
    harness.save(target)
    meta = load_metadata(target)
    assert meta.calibrator_class == "IsotonicCalibrator"
    assert 0.0 < meta.confidence < 1.0
    assert meta.n_calibration_anchors == len(harness.calibration_set)
    assert meta.drift_method == "psi"


def test_load_returns_fitted_harness(tmp_path: Path) -> None:
    harness = _make_harness()
    target = tmp_path / "judge.judgekit"
    harness.save(target)

    # Re-attach a fresh judge — same distribution as the original.
    judge = _SeededJudge(noise=0.05, seed=99)
    eval_rng = np.random.default_rng(2)
    eval_items = []
    for i in range(80):
        label = float(eval_rng.uniform(0, 1))
        item = f"eval-{i}"
        judge.register(item, label)
        eval_items.append(item)

    loaded = JudgeHarness.load(target, judge=judge)
    assert loaded.fitted is True

    # evaluate() should work and the calibrator should not have been re-fit.
    result = loaded.evaluate(eval_items, rng=eval_rng)
    assert result.n == len(eval_items)
    assert 0.0 <= result.point_estimate <= 1.0


def test_load_preserves_calibrator_predictions(tmp_path: Path) -> None:
    """Loaded calibrator must produce identical predictions to the original."""
    harness = _make_harness()
    probe = np.linspace(0.01, 0.99, 50)
    before = harness.calibrator.predict(probe)

    target = tmp_path / "judge.judgekit"
    harness.save(target)

    judge = _SeededJudge(noise=0.05, seed=99)
    loaded = JudgeHarness.load(target, judge=judge)
    after = loaded.calibrator.predict(probe)
    assert np.allclose(before, after, atol=1e-12)


def test_load_preserves_drift_baseline(tmp_path: Path) -> None:
    """Loaded drift monitor must compare against the same reference."""
    harness = _make_harness()
    target = tmp_path / "judge.judgekit"
    harness.save(target)

    judge = _SeededJudge(noise=0.05, seed=99)
    loaded = JudgeHarness.load(target, judge=judge)

    assert loaded._drift_monitor is not None
    assert np.allclose(loaded._drift_monitor.reference, harness._drift_monitor.reference)  # type: ignore[union-attr]


def test_load_works_with_platt(tmp_path: Path) -> None:
    judge = _SeededJudge(noise=0.05, seed=10)
    rng = np.random.default_rng(11)
    examples = []
    for i in range(160):
        label = float(rng.uniform(0, 1))
        item = f"cal-{i}"
        judge.register(item, label)
        examples.append(LabeledExample(item=item, label=label))
    harness = JudgeHarness(
        judge=judge, calibration_set=CalibrationSet(examples=examples), calibrator=PlattCalibrator()
    ).fit()

    target = tmp_path / "judge.judgekit"
    harness.save(target)
    meta = load_metadata(target)
    assert meta.calibrator_class == "PlattCalibrator"

    loaded = JudgeHarness.load(target, judge=judge)
    probe = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    assert np.allclose(harness.calibrator.predict(probe), loaded.calibrator.predict(probe))


def test_load_works_with_temperature(tmp_path: Path) -> None:
    judge = _SeededJudge(noise=0.05, seed=20)
    rng = np.random.default_rng(21)
    examples = []
    for i in range(40):
        label = float(rng.uniform(0, 1))
        item = f"cal-{i}"
        judge.register(item, label)
        examples.append(LabeledExample(item=item, label=label))
    harness = JudgeHarness(
        judge=judge,
        calibration_set=CalibrationSet(examples=examples),
        calibrator=TemperatureCalibrator(),
    ).fit()

    target = tmp_path / "judge.judgekit"
    harness.save(target)
    loaded = JudgeHarness.load(target, judge=judge)
    assert isinstance(loaded.calibrator, TemperatureCalibrator)
    probe = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    assert np.allclose(harness.calibrator.predict(probe), loaded.calibrator.predict(probe))


def test_save_unfitted_raises(tmp_path: Path) -> None:
    judge = _SeededJudge(seed=30)
    examples = []
    rng = np.random.default_rng(31)
    for i in range(20):
        label = float(rng.uniform(0, 1))
        item = f"cal-{i}"
        judge.register(item, label)
        examples.append(LabeledExample(item=item, label=label))
    harness = JudgeHarness(judge=judge, calibration_set=CalibrationSet(examples=examples))
    with pytest.raises(RuntimeError, match="fitted"):
        harness.save(tmp_path / "x.judgekit")


def test_load_rejects_unknown_format_version(tmp_path: Path) -> None:
    """An on-disk state with a major version we don't understand must error."""
    harness = _make_harness()
    target = tmp_path / "judge.judgekit"
    harness.save(target)

    # Bump the on-disk major to 99.
    state_file = target / "state.json"
    import json

    raw = json.loads(state_file.read_text())
    raw["format_version"] = "99.0"
    state_file.write_text(json.dumps(raw))

    with pytest.raises(StateFormatError, match="major version"):
        load_metadata(target)


def test_save_is_idempotent_overwrite(tmp_path: Path) -> None:
    """Saving twice to the same path overwrites cleanly."""
    harness = _make_harness()
    target = tmp_path / "judge.judgekit"
    harness.save(target)
    first_mtime = (target / "state.json").stat().st_mtime

    import time

    time.sleep(0.05)
    harness.save(target)
    second_mtime = (target / "state.json").stat().st_mtime
    assert second_mtime >= first_mtime


# ---------- Restricted unpickler ----------


def test_restricted_unpickler_refuses_os_system(tmp_path: Path) -> None:
    """Crafted calibrator.pkl that references os.system must be refused."""
    harness = _make_harness()
    target = tmp_path / "judge.judgekit"
    harness.save(target)

    # Build a malicious pickle that pulls in os.system. We don't actually
    # call it — we just verify the unpickler refuses to even resolve the
    # reference. We hand-write the GLOBAL opcode (`c<module>\n<name>\n.`)
    # so test setup itself never imports or executes the gadget.
    payload = b"cos\nsystem\n."
    (target / "calibrator.pkl").write_bytes(payload)

    judge = _SeededJudge(seed=42)
    with pytest.raises(StateFormatError, match=r"RestrictedUnpickler refused"):
        JudgeHarness.load(target, judge=judge)


def test_allow_unsafe_pickle_bypasses_restriction(tmp_path: Path) -> None:
    """Setting allow_unsafe_pickle=True falls back to the stdlib unpickler.

    We verify the path is wired up: the same regular pickle that loads fine
    via the restricted path also loads via the unrestricted path. A true
    bypass test (loading a class the restriction would refuse) is intentionally
    not exercised here — we don't want to make it convenient to forge.
    """
    harness = _make_harness()
    target = tmp_path / "judge.judgekit"
    harness.save(target)

    judge = _SeededJudge(seed=99)
    loaded = JudgeHarness.load(target, judge=judge, allow_unsafe_pickle=True)
    assert loaded.fitted


def test_state_schema_v1_1_fields_round_trip(tmp_path: Path) -> None:
    """Saved state.json must include the v1.1 schema additions."""
    import json as _json

    harness = _make_harness()
    target = tmp_path / "v11.judgekit"
    harness.save(target)
    raw = _json.loads((target / "state.json").read_text())

    assert raw["format_version"].startswith("1.")
    assert "drift_thresholds" in raw
    assert set(raw["drift_thresholds"]) >= {
        "psi_warn",
        "psi_fail",
        "ks_p_threshold",
        "wasserstein_threshold",
    }
    assert raw["harness_class"] == "JudgeHarness"
    assert raw["score_range"] == [0.0, 1.0]
    assert raw["drift_bins"] == 10
    assert "schema_extras" in raw
    assert "calibrator_params" in raw

    meta = load_metadata(target)
    assert meta.drift_bins == 10
    assert meta.harness_class == "JudgeHarness"
    assert meta.score_range == (0.0, 1.0)


def test_load_handles_legacy_1_0_state(tmp_path: Path) -> None:
    """A 1.0 state.json (without the new fields) must still load cleanly."""
    import json as _json

    harness = _make_harness()
    target = tmp_path / "legacy.judgekit"
    harness.save(target)

    raw = _json.loads((target / "state.json").read_text())
    # Strip the v1.1 additions and downgrade the version marker.
    for k in (
        "drift_thresholds",
        "calibrator_params",
        "harness_class",
        "score_range",
        "drift_bins",
        "schema_extras",
    ):
        raw.pop(k, None)
    raw["format_version"] = "1.0"
    (target / "state.json").write_text(_json.dumps(raw))

    judge = _SeededJudge(seed=42)
    loaded = JudgeHarness.load(target, judge=judge)
    assert loaded.fitted is True


def test_restricted_unpickler_accepts_all_calibrator_classes(tmp_path: Path) -> None:
    """All 5 calibrators must round-trip cleanly through the restricted unpickler."""
    import numpy as np

    from judgekit import (
        BetaCalibrator,
        HistogramBinCalibrator,
        IsotonicCalibrator,
        PlattCalibrator,
        TemperatureCalibrator,
    )

    for cal_cls in [
        PlattCalibrator,
        IsotonicCalibrator,
        TemperatureCalibrator,
        BetaCalibrator,
        HistogramBinCalibrator,
    ]:
        judge = _SeededJudge(seed=0)
        rng = np.random.default_rng(0)
        examples = []
        n = 220 if cal_cls is BetaCalibrator else 160
        for i in range(n):
            label = float(rng.uniform(0, 1))
            item = f"cal-{cal_cls.__name__}-{i}"
            judge.register(item, label)
            examples.append(LabeledExample(item=item, label=label))
        harness = JudgeHarness(
            judge=judge,
            calibration_set=CalibrationSet(examples=examples),
            calibrator=cal_cls(),
        ).fit()

        target = tmp_path / f"{cal_cls.__name__}.judgekit"
        harness.save(target)
        loaded = JudgeHarness.load(target, judge=judge)  # restricted by default
        assert isinstance(loaded.calibrator, cal_cls), (
            f"restricted unpickler rejected legitimate {cal_cls.__name__}"
        )
