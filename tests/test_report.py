"""Tests for the HTML report module."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from judgekit import (
    CalibrationSet,
    JudgeHarness,
    JudgeOutput,
    LabeledExample,
)


class _SeededJudge:
    def __init__(self, noise: float = 0.05, seed: int = 0) -> None:
        self._rng = np.random.default_rng(seed)
        self._noise = noise
        self._mapping: dict[str, float] = {}

    def register(self, item: str, true: float) -> None:
        self._mapping[item] = true

    def __call__(self, item: str) -> JudgeOutput:
        true = self._mapping[item]
        return JudgeOutput(score=float(np.clip(true + self._rng.normal(0, self._noise), 0.0, 1.0)))


def _make_fitted_harness(n: int = 200, seed: int = 0) -> JudgeHarness:
    judge = _SeededJudge(noise=0.05, seed=seed)
    rng = np.random.default_rng(seed + 1)
    examples = []
    for i in range(n):
        label = float(rng.uniform(0, 1))
        item = f"cal-{i}"
        judge.register(item, label)
        examples.append(LabeledExample(item=item, label=label))
    return JudgeHarness(judge=judge, calibration_set=CalibrationSet(examples=examples)).fit()


def test_report_writes_html_file(tmp_path: Path) -> None:
    harness = _make_fitted_harness()
    out = tmp_path / "report.html"
    harness.report(out, title="Test eval")
    assert out.is_file()
    content = out.read_text()
    assert "<!doctype html>" in content.lower()
    assert "Test eval" in content
    # Plotly was inlined (we asked for include_plotlyjs="inline").
    assert "plotly" in content.lower()


def test_report_self_contained_no_cdn(tmp_path: Path) -> None:
    """The HTML must not pull plotly from a CDN — fully portable.

    We check for <script src=...> tags pointing to CDNs, not raw string
    matches: plotly's inline bundle happens to contain the string
    "cdn.plot.ly" as a default config value, but it's never fetched.
    """
    import re

    harness = _make_fitted_harness()
    out = tmp_path / "report.html"
    harness.report(out)
    content = out.read_text()
    cdn_script_srcs = re.findall(
        r'<script[^>]+src=["\']([^"\']*(cdn\.plot\.ly|cdnjs\.cloudflare\.com)[^"\']*)["\']',
        content,
    )
    assert cdn_script_srcs == [], f"CDN script sources detected: {cdn_script_srcs}"


def test_report_with_eval_result_shows_current_run(tmp_path: Path) -> None:
    harness = _make_fitted_harness()
    # Build an eval set from the same distribution.
    judge = harness.judge
    eval_rng = np.random.default_rng(99)
    eval_items = []
    for i in range(60):
        label = float(eval_rng.uniform(0, 1))
        item = f"eval-{i}"
        judge.register(item, label)
        eval_items.append(item)
    result = harness.evaluate(eval_items, rng=eval_rng)
    out = tmp_path / "report.html"
    harness.report(out, result=result, title="With current run")
    content = out.read_text()
    assert "With current run" in content
    # Point estimate string should appear in the meta block.
    expected = f"{result.point_estimate:.4f}"
    assert expected in content


def test_report_creates_parent_directories(tmp_path: Path) -> None:
    harness = _make_fitted_harness()
    out = tmp_path / "nested" / "path" / "report.html"
    harness.report(out)
    assert out.is_file()


def test_report_requires_fitted_harness(tmp_path: Path) -> None:
    judge = _SeededJudge(seed=0)
    rng = np.random.default_rng(0)
    examples = []
    for i in range(20):
        label = float(rng.uniform(0, 1))
        item = f"cal-{i}"
        judge.register(item, label)
        examples.append(LabeledExample(item=item, label=label))
    harness = JudgeHarness(judge=judge, calibration_set=CalibrationSet(examples=examples))
    with pytest.raises(RuntimeError, match="fitted"):
        harness.report(tmp_path / "x.html")


def test_report_baseline_only_when_no_result(tmp_path: Path) -> None:
    harness = _make_fitted_harness()
    out = tmp_path / "baseline.html"
    harness.report(out)
    content = out.read_text()
    # No current-run point estimate yet -> em-dash placeholder.
    assert "—" in content


def test_report_escapes_title_html(tmp_path: Path) -> None:
    """Jinja2 autoescape must HTML-escape the title (and other interpolated
    strings). Verified by trying an XSS-flavored title and asserting the
    raw ``<script>`` tag does not survive to the output."""
    harness = _make_fitted_harness()
    out = tmp_path / "xss.html"
    harness.report(out, title="<script>alert('xss')</script>")
    content = out.read_text()
    # The raw script tag must be HTML-escaped in the title field.
    assert "<script>alert('xss')</script>" not in content
    # Escaped form should be present somewhere (in the <title> or <h1>).
    assert "&lt;script&gt;" in content


def test_report_extra_missing_raises_friendly_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If `pip install judgekit[report]` wasn't run, .report() must point users
    at the fix instead of leaking a bare ModuleNotFoundError."""
    import builtins

    real_import = builtins.__import__

    def fake_import(name: str, *args, **kwargs):  # type: ignore[no-untyped-def]
        if name in ("jinja2", "plotly", "plotly.graph_objects", "plotly.io"):
            raise ImportError(f"No module named '{name}'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    harness = _make_fitted_harness()
    out = tmp_path / "report.html"
    with pytest.raises(ImportError, match=r"\[report\] extra"):
        harness.report(out)
