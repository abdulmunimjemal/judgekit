"""Tests for the pairwise module."""

from __future__ import annotations

import numpy as np
import pytest

from judgekit.pairwise import (
    PairwiseHarness,
    PairwiseJudge,
    PairwiseOutcome,
    PairwiseResult,
    PairwiseVerdict,
)


class _DeterministicJudge:
    """Pairwise judge that returns a configured verdict per pair."""

    def __init__(self, default: PairwiseOutcome = PairwiseOutcome.A_WINS) -> None:
        self.default = default
        self._table: dict[tuple[str, str], PairwiseOutcome] = {}

    def set(self, a: str, b: str, outcome: PairwiseOutcome) -> None:
        self._table[(a, b)] = outcome

    def __call__(self, a: str, b: str) -> PairwiseVerdict:
        outcome = self._table.get((a, b), self.default)
        return PairwiseVerdict(outcome=outcome)


class _PositionBiasedJudge:
    """Always prefers whichever option is in the first slot."""

    def __call__(self, a: str, b: str) -> PairwiseVerdict:
        return PairwiseVerdict(outcome=PairwiseOutcome.A_WINS)


class _LengthPreferringJudge:
    """Prefers the longer of the two strings; ties on equal length."""

    def __call__(self, a: str, b: str) -> PairwiseVerdict:
        if len(a) > len(b):
            return PairwiseVerdict(outcome=PairwiseOutcome.A_WINS)
        if len(b) > len(a):
            return PairwiseVerdict(outcome=PairwiseOutcome.B_WINS)
        return PairwiseVerdict(outcome=PairwiseOutcome.TIE)


# ---------- Protocol / dataclasses ----------


def test_pairwise_judge_protocol_runtime_checkable() -> None:
    assert isinstance(_DeterministicJudge(), PairwiseJudge)


def test_pairwise_verdict_is_frozen() -> None:
    v = PairwiseVerdict(outcome=PairwiseOutcome.TIE)
    with pytest.raises((AttributeError, Exception)):
        v.outcome = PairwiseOutcome.A_WINS  # type: ignore[misc]


# ---------- Win rate basics ----------


def test_pairwise_result_perfect_a_wins() -> None:
    judge = _DeterministicJudge(default=PairwiseOutcome.A_WINS)
    harness = PairwiseHarness(judge)
    pairs = [(f"a{i}", f"b{i}") for i in range(30)]
    result = harness.evaluate(pairs, position_bias_correct=False)
    assert result.n_pairs == 30
    assert result.n_a_wins == 30
    assert result.n_b_wins == 0
    assert result.n_ties == 0
    assert result.win_rate_a == pytest.approx(1.0)


def test_pairwise_result_perfect_b_wins() -> None:
    judge = _DeterministicJudge(default=PairwiseOutcome.B_WINS)
    harness = PairwiseHarness(judge)
    pairs = [(f"a{i}", f"b{i}") for i in range(30)]
    result = harness.evaluate(pairs, position_bias_correct=False)
    assert result.win_rate_a == pytest.approx(0.0)
    assert result.n_b_wins == 30


def test_pairwise_result_all_ties_gives_half() -> None:
    judge = _DeterministicJudge(default=PairwiseOutcome.TIE)
    harness = PairwiseHarness(judge)
    pairs = [(f"a{i}", f"b{i}") for i in range(20)]
    result = harness.evaluate(pairs, position_bias_correct=False)
    assert result.win_rate_a == pytest.approx(0.5)
    assert result.n_ties == 20


def test_pairwise_mixed_outcomes() -> None:
    judge = _DeterministicJudge()
    pairs = [(f"a{i}", f"b{i}") for i in range(10)]
    # 6 A_WINS, 2 B_WINS, 2 TIE
    for i in range(6):
        judge.set(f"a{i}", f"b{i}", PairwiseOutcome.A_WINS)
    judge.set("a6", "b6", PairwiseOutcome.B_WINS)
    judge.set("a7", "b7", PairwiseOutcome.B_WINS)
    judge.set("a8", "b8", PairwiseOutcome.TIE)
    judge.set("a9", "b9", PairwiseOutcome.TIE)

    harness = PairwiseHarness(judge)
    result = harness.evaluate(pairs, position_bias_correct=False)
    assert result.n_a_wins == 6
    assert result.n_b_wins == 2
    assert result.n_ties == 2
    # Win rate = (6 + 0.5 * 2) / 10 = 0.7
    assert result.win_rate_a == pytest.approx(0.7)


# ---------- Position bias correction ----------


def test_position_bias_correction_neutralizes_first_slot_preference() -> None:
    """Judge always picks A. With swap correction, win rate collapses to 0.5."""
    judge = _PositionBiasedJudge()
    harness = PairwiseHarness(judge)
    pairs = [(f"a{i}", f"b{i}") for i in range(40)]

    uncorrected = harness.evaluate(pairs, position_bias_correct=False)
    assert uncorrected.win_rate_a == pytest.approx(1.0)
    assert uncorrected.position_bias is None
    assert uncorrected.position_bias_corrected is False

    corrected = harness.evaluate(pairs, position_bias_correct=True)
    # When the judge always picks slot-1, the bias-corrected score is 0.5.
    assert corrected.win_rate_a == pytest.approx(0.5)
    assert corrected.position_bias == pytest.approx(1.0)
    assert corrected.position_bias_corrected is True


def test_position_bias_correction_preserves_real_signal() -> None:
    """Length-preferring judge is position-invariant; bias = 0 and win rate
    reflects the true 'A is longer' rate."""
    judge = _LengthPreferringJudge()
    harness = PairwiseHarness(judge)
    # In 7 pairs A is longer, in 3 B is longer.
    pairs = []
    for i in range(7):
        pairs.append(("very long string for A " + "x" * i, "short B"))
    for i in range(3):
        pairs.append(("short A", "very long string for B " + "x" * i))

    result = harness.evaluate(pairs, position_bias_correct=True)
    assert result.position_bias == pytest.approx(0.0)
    assert result.win_rate_a == pytest.approx(0.7)


# ---------- CI behavior ----------


def test_pairwise_ci_brackets_point_estimate() -> None:
    judge = _LengthPreferringJudge()
    harness = PairwiseHarness(judge)
    pairs = []
    for i in range(50):
        pairs.append(("x" * (i + 1), "y" * (40 - i)))
    result = harness.evaluate(pairs, position_bias_correct=False, rng=np.random.default_rng(0))
    lo, hi = result.win_rate_a_ci
    assert lo <= result.win_rate_a <= hi
    assert 0.0 <= lo <= hi <= 1.0


def test_pairwise_custom_confidence() -> None:
    judge = _LengthPreferringJudge()
    harness = PairwiseHarness(judge, confidence=0.80)
    pairs = [("x" * (i + 1), "y" * (10 - i)) for i in range(10)]
    result = harness.evaluate(pairs, position_bias_correct=False, rng=np.random.default_rng(0))
    assert result.confidence == 0.80


# ---------- Edge cases ----------


def test_pairwise_rejects_empty_pairs() -> None:
    harness = PairwiseHarness(_DeterministicJudge())
    with pytest.raises(ValueError, match="non-empty"):
        harness.evaluate([])


def test_pairwise_rejects_invalid_confidence() -> None:
    with pytest.raises(ValueError, match="confidence"):
        PairwiseHarness(_DeterministicJudge(), confidence=1.5)


def test_per_pair_scores_match_n_pairs() -> None:
    harness = PairwiseHarness(_LengthPreferringJudge())
    pairs = [("aa", "b"), ("c", "dd"), ("ee", "ff")]
    result = harness.evaluate(pairs, position_bias_correct=False)
    assert result.per_pair_scores.shape == (3,)
    assert isinstance(result, PairwiseResult)
