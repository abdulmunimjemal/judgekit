"""Tests for the judgekit CLI."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from judgekit.cli import main


def _write_traces_jsonl(path: Path, items_scores: list[tuple[str, float]]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for item, score in items_scores:
            fh.write(json.dumps({"item": item, "score": score}) + "\n")


def _write_gold_jsonl(path: Path, items_labels: list[tuple[str, float]]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for item, label in items_labels:
            fh.write(json.dumps({"item": item, "label": label}) + "\n")


def _generate_aligned_traces_and_gold(
    tmp_path: Path, n: int = 60, seed: int = 0
) -> tuple[Path, Path]:
    rng = np.random.default_rng(seed)
    items_scores: list[tuple[str, float]] = []
    items_labels: list[tuple[str, float]] = []
    for i in range(n):
        item = f"item-{i}"
        label = float(rng.uniform(0, 1))
        score = float(np.clip(label + rng.normal(0, 0.05), 0.0, 1.0))
        items_scores.append((item, score))
        items_labels.append((item, label))
    traces_path = tmp_path / "traces.jsonl"
    gold_path = tmp_path / "gold.jsonl"
    _write_traces_jsonl(traces_path, items_scores)
    _write_gold_jsonl(gold_path, items_labels)
    return traces_path, gold_path


# ---------- version ----------


def test_cli_version(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["version"])
    out = capsys.readouterr().out
    assert code == 0
    assert "judgekit" in out
    assert "state format" in out


# ---------- calibrate ----------


def test_cli_calibrate_writes_state(tmp_path: Path) -> None:
    traces, gold = _generate_aligned_traces_and_gold(tmp_path, n=80)
    out = tmp_path / "judge.judgekit"
    code = main(["calibrate", "--traces", str(traces), "--gold", str(gold), "--out", str(out)])
    assert code == 0
    assert (out / "state.json").is_file()
    assert (out / "calibrator.pkl").is_file()
    assert (out / "drift_reference.npy").is_file()


def test_cli_calibrate_picks_auto_calibrator_for_small_set(tmp_path: Path) -> None:
    traces, gold = _generate_aligned_traces_and_gold(tmp_path, n=30)
    out = tmp_path / "judge.judgekit"
    code = main(["calibrate", "--traces", str(traces), "--gold", str(gold), "--out", str(out)])
    assert code == 0
    meta = json.loads((out / "state.json").read_text())
    # Auto picks Temperature for <50 anchors.
    assert meta["calibrator_class"] == "TemperatureCalibrator"


def test_cli_calibrate_explicit_calibrator(tmp_path: Path) -> None:
    traces, gold = _generate_aligned_traces_and_gold(tmp_path, n=60)
    out = tmp_path / "judge.judgekit"
    code = main(
        [
            "calibrate",
            "--traces",
            str(traces),
            "--gold",
            str(gold),
            "--out",
            str(out),
            "--calibrator",
            "platt",
        ]
    )
    assert code == 0
    meta = json.loads((out / "state.json").read_text())
    assert meta["calibrator_class"] == "PlattCalibrator"


def test_cli_calibrate_rejects_too_few_matched(tmp_path: Path) -> None:
    traces, gold = _generate_aligned_traces_and_gold(tmp_path, n=5)
    out = tmp_path / "judge.judgekit"
    with pytest.raises(SystemExit, match="10"):
        main(["calibrate", "--traces", str(traces), "--gold", str(gold), "--out", str(out)])


def test_cli_calibrate_rejects_bad_jsonl_fields(tmp_path: Path) -> None:
    traces = tmp_path / "bad.jsonl"
    traces.write_text(json.dumps({"item": "x", "wrong_field": 0.5}) + "\n")
    gold = tmp_path / "gold.jsonl"
    _write_gold_jsonl(gold, [("x", 1.0)])
    with pytest.raises(SystemExit, match="score"):
        main(
            [
                "calibrate",
                "--traces",
                str(traces),
                "--gold",
                str(gold),
                "--out",
                str(tmp_path / "x"),
            ]
        )


# ---------- report ----------


def test_cli_report_renders_html(tmp_path: Path) -> None:
    traces, gold = _generate_aligned_traces_and_gold(tmp_path, n=80)
    state = tmp_path / "judge.judgekit"
    main(["calibrate", "--traces", str(traces), "--gold", str(gold), "--out", str(state)])
    html = tmp_path / "report.html"
    code = main(["report", str(state), "--html", str(html)])
    assert code == 0
    assert html.is_file()
    assert "<!doctype html>" in html.read_text().lower()


def test_cli_report_custom_title(tmp_path: Path) -> None:
    traces, gold = _generate_aligned_traces_and_gold(tmp_path, n=80)
    state = tmp_path / "judge.judgekit"
    main(["calibrate", "--traces", str(traces), "--gold", str(gold), "--out", str(state)])
    html = tmp_path / "report.html"
    main(["report", str(state), "--html", str(html), "--title", "Run #42"])
    assert "Run #42" in html.read_text()


# ---------- audit ----------


def test_cli_audit_verbosity_emits_json(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    # Build a verbosity-biased trace: longer items get higher scores.
    items_scores = []
    for i in range(20):
        items_scores.append((f"item-{'x' * (i + 1)}", min(1.0, (i + 1) / 25.0)))
    traces = tmp_path / "judge.jsonl"
    _write_traces_jsonl(traces, items_scores)

    code = main(["audit-verbosity", "--judge-traces", str(traces)])
    out = capsys.readouterr().out
    parsed = json.loads(out)
    assert parsed["audit"] == "verbosity_bias"
    assert parsed["value"] > 0.5  # strong positive correlation
    # Exit code 2 signals "concerning" so CI can gate on it.
    assert code in (0, 2)


def test_cli_audit_verbosity_zero_exit_on_clean_judge(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rng = np.random.default_rng(0)
    items_scores = []
    for i in range(20):
        items_scores.append((f"item-{i}", float(rng.uniform(0.3, 0.7))))
    traces = tmp_path / "judge.jsonl"
    _write_traces_jsonl(traces, items_scores)
    code = main(["audit-verbosity", "--judge-traces", str(traces)])
    assert code == 0


def test_cli_audit_verbosity_rejects_too_few_records(tmp_path: Path) -> None:
    items_scores = [("x", 0.5), ("y", 0.6)]
    traces = tmp_path / "judge.jsonl"
    _write_traces_jsonl(traces, items_scores)
    with pytest.raises(SystemExit, match="3"):
        main(["audit-verbosity", "--judge-traces", str(traces)])


def test_cli_audit_legacy_emits_deprecation_warning(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The plain `audit` subcommand still works but warns it's deprecated."""
    rng = np.random.default_rng(0)
    items_scores = [(f"item-{i}", float(rng.uniform(0.3, 0.7))) for i in range(20)]
    traces = tmp_path / "judge.jsonl"
    _write_traces_jsonl(traces, items_scores)
    code = main(["audit", "--judge-traces", str(traces)])
    err = capsys.readouterr().err
    assert "DEPRECATION" in err
    assert "audit-verbosity" in err
    assert code == 0


# ---------- general ----------


def test_cli_no_args_prints_help(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc_info:
        main([])
    assert exc_info.value.code != 0
    err = capsys.readouterr().err
    assert "command" in err.lower() or "usage" in err.lower()


def test_cli_invalid_subcommand(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        main(["nonexistent"])
