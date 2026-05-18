"""Tests for the bias audit module."""

from __future__ import annotations

import numpy as np
import pytest

from judgekit.bias import (
    BiasReport,
    format_sensitivity,
    position_bias,
    verbosity_bias,
)
from judgekit.judge import JudgeOutput
from judgekit.pairwise import PairwiseOutcome, PairwiseVerdict

# ---------- Mock judges ----------


class _PositionBiasedPairwise:
    """Always picks slot 1."""

    def __call__(self, a: str, b: str) -> PairwiseVerdict:
        return PairwiseVerdict(outcome=PairwiseOutcome.A_WINS)


class _UnbiasedPairwise:
    """Verdict depends only on content (longer string wins)."""

    def __call__(self, a: str, b: str) -> PairwiseVerdict:
        if len(a) > len(b):
            return PairwiseVerdict(outcome=PairwiseOutcome.A_WINS)
        if len(b) > len(a):
            return PairwiseVerdict(outcome=PairwiseOutcome.B_WINS)
        return PairwiseVerdict(outcome=PairwiseOutcome.TIE)


class _VerbosityBiasedJudge:
    """Pointwise judge whose score is proportional to length."""

    def __call__(self, item: str) -> JudgeOutput:
        return JudgeOutput(score=min(1.0, len(item) / 200.0))


class _ContentJudge:
    """Pointwise judge that ignores length and returns a content-based score."""

    def __init__(self) -> None:
        self._rng = np.random.default_rng(0)

    def __call__(self, item: str) -> JudgeOutput:
        # Score depends only on a hash of the underlying tokens (length-free).
        return JudgeOutput(score=float(self._rng.uniform(0.3, 0.7)))


class _FormatSensitiveJudge:
    """Pointwise judge that scores higher on markdown-wrapped content."""

    def __call__(self, item: str) -> JudgeOutput:
        if item.startswith(">"):
            return JudgeOutput(score=0.9)
        return JudgeOutput(score=0.3)


class _FormatStableJudge:
    """Pointwise judge that returns the same score regardless of formatting."""

    def __call__(self, item: str) -> JudgeOutput:
        return JudgeOutput(score=0.5)


# ---------- BiasReport ----------


def test_bias_report_is_concerning_threshold() -> None:
    r = BiasReport(
        audit="x",
        value=0.5,
        threshold_warn=0.1,
        threshold_fail=0.3,
        verdict="concerning",
        detail={},
    )
    assert r.is_concerning() is True


def test_bias_report_clean_is_not_concerning() -> None:
    r = BiasReport(
        audit="x", value=0.0, threshold_warn=0.1, threshold_fail=0.3, verdict="clean", detail={}
    )
    assert r.is_concerning() is False


# ---------- position_bias ----------


def test_position_bias_flags_slot1_preferring_judge() -> None:
    judge = _PositionBiasedPairwise()
    pairs = [(f"a{i}", f"b{i}") for i in range(20)]
    report = position_bias(judge, pairs)
    # Every pair flips when reversed -> swap rate = 1.0 = concerning.
    assert report.value == pytest.approx(1.0)
    assert report.verdict == "concerning"
    assert report.is_concerning() is True


def test_position_bias_clean_on_content_judge() -> None:
    judge = _UnbiasedPairwise()
    # Length-based: A is longer in every pair, judge always picks A.
    pairs = [("longer text " + "x" * i, "short") for i in range(20)]
    report = position_bias(judge, pairs)
    assert report.value == pytest.approx(0.0)
    assert report.verdict == "clean"
    assert report.is_concerning() is False


def test_position_bias_rejects_empty_pairs() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        position_bias(_UnbiasedPairwise(), [])


def test_position_bias_threshold_overrides() -> None:
    judge = _PositionBiasedPairwise()
    pairs = [(f"a{i}", f"b{i}") for i in range(10)]
    # With a sky-high threshold, even a 100% swap rate is "clean".
    report = position_bias(judge, pairs, threshold_warn=2.0, threshold_fail=3.0)
    assert report.verdict == "clean"


# ---------- verbosity_bias ----------


def test_verbosity_bias_flags_length_proportional_judge() -> None:
    judge = _VerbosityBiasedJudge()
    items = [f"item-{'x' * i}" for i in range(10, 200, 5)]
    report = verbosity_bias(judge, items)
    # Score is strictly increasing with length -> correlation ~1.
    assert report.value > 0.9
    assert report.verdict == "concerning"


def test_verbosity_bias_clean_on_content_judge() -> None:
    judge = _ContentJudge()
    items = [f"item-{'x' * i}" for i in range(10, 200, 5)]
    report = verbosity_bias(judge, items)
    # Random content scores -> correlation near zero.
    assert abs(report.value) < 0.30
    assert report.verdict == "clean"


def test_verbosity_bias_rejects_too_few_items() -> None:
    with pytest.raises(ValueError, match="at least 3 items"):
        verbosity_bias(_VerbosityBiasedJudge(), ["a", "b"])


def test_verbosity_bias_handles_constant_input_gracefully() -> None:
    # All same-length items -> no variance in lengths -> correlation undefined.
    judge = _VerbosityBiasedJudge()
    items = ["same length" for _ in range(20)]
    report = verbosity_bias(judge, items)
    assert report.value == 0.0
    assert report.verdict == "clean"


# ---------- format_sensitivity ----------


def test_format_sensitivity_flags_format_biased_judge() -> None:
    judge = _FormatSensitiveJudge()
    items = ["item one", "item two", "item three"]
    report = format_sensitivity(judge, items)
    # Plain vs markdown_quote vs json scores differ widely -> high std.
    assert report.value > 0.20
    assert report.verdict == "concerning"


def test_format_sensitivity_clean_on_stable_judge() -> None:
    judge = _FormatStableJudge()
    items = ["item one", "item two", "item three"]
    report = format_sensitivity(judge, items)
    assert report.value == pytest.approx(0.0)
    assert report.verdict == "clean"


def test_format_sensitivity_custom_formats() -> None:
    judge = _FormatStableJudge()
    items = ["x", "y"]
    custom = {"a": "[A] {content}", "b": "[B] {content}"}
    report = format_sensitivity(judge, items, formats=custom)
    assert report.detail["n_formats"] == 2.0


def test_format_sensitivity_rejects_too_few_formats() -> None:
    with pytest.raises(ValueError, match="at least 2 formats"):
        format_sensitivity(_FormatStableJudge(), ["x"], formats={"only": "{content}"})


def test_format_sensitivity_rejects_template_without_content() -> None:
    with pytest.raises(ValueError, match="content"):
        format_sensitivity(
            _FormatStableJudge(),
            ["x"],
            formats={"a": "no placeholder", "b": "also no"},
        )


def test_format_sensitivity_rejects_empty_items() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        format_sensitivity(_FormatStableJudge(), [])
