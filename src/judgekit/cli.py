"""Command-line interface for judgekit.

``judgekit`` is the single entry-point. Subcommands:

- ``judgekit version`` — print library + state-format version info.
- ``judgekit calibrate --traces FILE --gold FILE --out DIR`` — read raw
  judge scores from a JSONL traces file and gold labels from a JSONL
  gold file, fit a harness, save it to a state directory. Useful for
  one-shot calibration without writing Python.
- ``judgekit report STATE --html PATH`` — load a saved state and emit
  an HTML report at PATH.
- ``judgekit audit --judge-traces FILE`` — run verbosity-bias audit on
  a pointwise judge given a JSONL of (item, score) records. (Position
  and format audits require live judge access, so they're not exposed
  via CLI.)

JSONL formats:

- traces / gold (one record per line): ``{"item": "...", "score": 0.7}``
  or ``{"item": "...", "label": 1.0}``.

The CLI is deliberately small: it covers the most common one-shot
workflows. Anything more sophisticated belongs in a Python script using
the public API.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any


def _read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise SystemExit(f"{path}:{line_no}: invalid JSON: {e.msg}") from e


def _version() -> str:
    from judgekit import __version__
    from judgekit.persistence import STATE_FORMAT_VERSION

    return f"judgekit {__version__} (state format v{STATE_FORMAT_VERSION})"


def _cmd_version(args: argparse.Namespace) -> int:
    print(_version())
    return 0


def _cmd_calibrate(args: argparse.Namespace) -> int:
    """Fit a harness from a JSONL traces file + gold-labels file.

    The traces file provides ``(item, score)`` pairs from running the
    judge. The gold file provides ``(item, label)`` ground-truth pairs.
    Items are matched on the ``item`` field. Unmatched items are dropped
    with a stderr warning.
    """

    from judgekit.calibration import (
        BetaCalibrator,
        Calibrator,
        HistogramBinCalibrator,
        IsotonicCalibrator,
        PlattCalibrator,
        TemperatureCalibrator,
        select_calibrator,
    )
    from judgekit.harness import JudgeHarness
    from judgekit.judge import CalibrationSet, JudgeOutput, LabeledExample

    traces_path = Path(args.traces)
    gold_path = Path(args.gold)
    out_path = Path(args.out)

    traces: dict[str, float] = {}
    for rec in _read_jsonl(traces_path):
        if "item" not in rec or "score" not in rec:
            raise SystemExit(f"{traces_path}: every record needs `item` and `score` fields")
        traces[str(rec["item"])] = float(rec["score"])

    gold: dict[str, float] = {}
    for rec in _read_jsonl(gold_path):
        if "item" not in rec or "label" not in rec:
            raise SystemExit(f"{gold_path}: every record needs `item` and `label` fields")
        gold[str(rec["item"])] = float(rec["label"])

    matched = sorted(set(traces) & set(gold))
    if len(matched) < 10:
        raise SystemExit(
            f"only {len(matched)} items matched between traces and gold; need >=10 for calibration"
        )

    dropped = (set(traces) | set(gold)) - set(matched)
    if dropped:
        print(
            f"WARNING: dropping {len(dropped)} items present in only one file",
            file=sys.stderr,
        )

    examples = [LabeledExample(item=item, label=gold[item]) for item in matched]
    calset = CalibrationSet(examples=examples)

    if args.calibrator == "auto":
        calibrator: Calibrator | None = None  # JudgeHarness will use default
        # But we want CLI to log which it would pick, so resolve manually.
        calibrator = select_calibrator(len(matched))
    else:
        calibrator = {
            "platt": PlattCalibrator(),
            "isotonic": IsotonicCalibrator(),
            "temperature": TemperatureCalibrator(),
            "beta": BetaCalibrator(),
            "histogram": HistogramBinCalibrator(),
        }[args.calibrator]

    class _TraceJudge:
        """Returns the pre-computed score for each item; no live judge needed."""

        def __call__(self, item: str) -> JudgeOutput:
            return JudgeOutput(score=traces[item])

    harness = JudgeHarness(
        judge=_TraceJudge(),
        calibration_set=calset,
        calibrator=calibrator,
    ).fit()
    harness.save(out_path)
    print(
        f"calibrated {len(matched)} items with {type(harness.calibrator).__name__}",
        file=sys.stderr,
    )
    print(out_path)
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    from judgekit.harness import JudgeHarness
    from judgekit.judge import JudgeOutput
    from judgekit.persistence import load_metadata

    state_path = Path(args.state)
    html_path = Path(args.html)

    # We need a Judge to attach for load(); the report doesn't actually
    # call it (we only re-render the baseline). Use a stub.
    class _StubJudge:
        def __call__(self, item: str) -> JudgeOutput:
            raise RuntimeError("CLI report stub judge should never be called")

    metadata = load_metadata(state_path)
    harness = JudgeHarness.load(state_path, judge=_StubJudge())
    harness.report(html_path, title=args.title or f"judgekit report for {state_path.name}")
    print(
        f"rendered {html_path} from {state_path} "
        f"(calibrator={metadata.calibrator_class}, n_anchors={metadata.n_calibration_anchors})",
        file=sys.stderr,
    )
    print(html_path)
    return 0


def _cmd_audit(args: argparse.Namespace) -> int:
    """Run a verbosity-bias audit from a JSONL traces file."""
    from judgekit.bias import verbosity_bias
    from judgekit.judge import JudgeOutput

    traces_path = Path(args.judge_traces)

    traces: dict[str, float] = {}
    for rec in _read_jsonl(traces_path):
        if "item" not in rec or "score" not in rec:
            raise SystemExit(f"{traces_path}: every record needs `item` and `score` fields")
        traces[str(rec["item"])] = float(rec["score"])

    if len(traces) < 3:
        raise SystemExit("verbosity_bias needs at least 3 records")

    class _TraceJudge:
        def __call__(self, item: str) -> JudgeOutput:
            return JudgeOutput(score=traces[item])

    items = list(traces.keys())
    report = verbosity_bias(_TraceJudge(), items)
    out = {
        "audit": report.audit,
        "value": report.value,
        "verdict": report.verdict,
        "threshold_warn": report.threshold_warn,
        "threshold_fail": report.threshold_fail,
        "detail": report.detail,
    }
    print(json.dumps(out, indent=2))
    return 0 if not report.is_concerning() else 2


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="judgekit",
        description="Calibrate, monitor, and refuse to ship miscalibrated LLM judges.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("version", help="Print version info.").set_defaults(func=_cmd_version)

    calibrate = sub.add_parser(
        "calibrate",
        help="Fit a harness from JSONL traces + gold labels and save to a state directory.",
    )
    calibrate.add_argument("--traces", required=True, help="JSONL of {item, score} records.")
    calibrate.add_argument("--gold", required=True, help="JSONL of {item, label} records.")
    calibrate.add_argument("--out", required=True, help="Output state directory.")
    calibrate.add_argument(
        "--calibrator",
        choices=["auto", "platt", "isotonic", "temperature", "beta", "histogram"],
        default="auto",
        help="Calibrator to use (default: auto = pick based on anchor count).",
    )
    calibrate.set_defaults(func=_cmd_calibrate)

    report = sub.add_parser(
        "report",
        help="Render a self-contained HTML report from a saved state directory.",
    )
    report.add_argument("state", help="Saved-state directory path.")
    report.add_argument("--html", required=True, help="Output HTML file path.")
    report.add_argument("--title", default=None, help="Optional report title.")
    report.set_defaults(func=_cmd_report)

    audit = sub.add_parser(
        "audit",
        help="Run a verbosity-bias audit on a pointwise judge's traces JSONL.",
    )
    audit.add_argument("--judge-traces", required=True, help="JSONL of {item, score} records.")
    audit.set_defaults(func=_cmd_audit)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    func = args.func
    return int(func(args) or 0)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
