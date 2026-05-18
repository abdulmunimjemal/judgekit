"""Pairwise judges: A-vs-B preference comparisons.

Pairwise eval is the dominant paradigm for LLM judges that compare two
candidate responses against a prompt — Chatbot Arena, MT-Bench, AlpacaEval,
and most internal A/B regressions.

This module ships:

- :class:`PairwiseOutcome` — the discrete verdict (A_WINS, B_WINS, TIE).
- :class:`PairwiseVerdict` — one call result, including optional confidence.
- :class:`PairwiseJudge` Protocol — any callable ``(a, b) -> PairwiseVerdict``.
- :class:`PairwiseResult` — aggregated win-rate with bootstrap CI and
  optional position-bias diagnostic.
- :class:`PairwiseHarness` — runs a judge over a list of pairs, optionally
  swapping each pair to correct for position bias.

Aggregation uses the standard score: ``win_rate_a = (#A_wins + 0.5 * #ties) /
n``. With position-bias correction enabled, each pair is judged twice — once
as (a, b) and once as (b, a) — and the two scores are averaged. The
correction halves the influence of any side-preference bias the judge
might have.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol, runtime_checkable

import numpy as np

from judgekit.calibration import bootstrap_ci


class PairwiseOutcome(str, Enum):
    """Discrete verdict for one A-vs-B comparison."""

    A_WINS = "A_wins"
    B_WINS = "B_wins"
    TIE = "tie"


@dataclass(frozen=True)
class PairwiseVerdict:
    """One pairwise judge output.

    ``outcome`` is the discrete verdict. ``confidence`` is optional —
    judges that produce a probability for "A is better" should expose
    it here (in ``[0.0, 1.0]``) so downstream calibration / aggregation
    can use it. ``raw`` is an opaque payload for the judge's full output
    (e.g. the LLM's natural-language reasoning).
    """

    outcome: PairwiseOutcome
    confidence: float | None = None
    raw: object | None = None


@runtime_checkable
class PairwiseJudge(Protocol):
    """Anything callable that returns a :class:`PairwiseVerdict` for a pair."""

    def __call__(self, a: str, b: str) -> PairwiseVerdict: ...


@dataclass
class PairwiseResult:
    """Aggregated pairwise result over a batch of pairs.

    ``win_rate_a`` is the standard "A is better" score over the batch:
    ``(#A_wins + 0.5 * #ties) / n``.

    With position-bias correction, ``position_bias`` is the swap rate —
    the fraction of pairs where the judge's verdict flipped when the
    pair order was reversed. 0.0 = perfectly position-invariant; 1.0 =
    judge always picks whichever item is first.
    """

    win_rate_a: float
    win_rate_a_ci: tuple[float, float]
    n_pairs: int
    n_a_wins: int
    n_b_wins: int
    n_ties: int
    confidence: float
    position_bias_corrected: bool
    position_bias: float | None = None
    per_pair_scores: np.ndarray = field(default_factory=lambda: np.asarray([]))

    def __repr__(self) -> str:  # pragma: no cover — repr only
        lo, hi = self.win_rate_a_ci
        bias_str = (
            f", position_bias={self.position_bias:.3f}" if self.position_bias is not None else ""
        )
        return (
            f"PairwiseResult(n={self.n_pairs}, win_rate_a={self.win_rate_a:.4f}, "
            f"CI{int(self.confidence * 100)}=[{lo:.4f}, {hi:.4f}]"
            f"{bias_str})"
        )


def _outcome_to_score(outcome: PairwiseOutcome) -> float:
    """Map a verdict to a numeric A-preference score in [0, 1]."""
    if outcome == PairwiseOutcome.A_WINS:
        return 1.0
    if outcome == PairwiseOutcome.B_WINS:
        return 0.0
    return 0.5


class PairwiseHarness:
    """Runs a pairwise judge across a batch of (a, b) pairs.

    Usage::

        harness = PairwiseHarness(my_judge)
        result = harness.evaluate(pairs)
        print(result.win_rate_a, result.win_rate_a_ci)

    Set ``position_bias_correct=True`` (the default) to judge each pair in
    both orderings and average the two scores. This halves any position
    bias the judge has at the cost of doubling judge calls.
    """

    def __init__(
        self,
        judge: PairwiseJudge,
        confidence: float = 0.95,
    ) -> None:
        if not 0.0 < confidence < 1.0:
            raise ValueError("confidence must be in (0, 1)")
        self.judge = judge
        self.confidence = confidence

    def evaluate(
        self,
        pairs: list[tuple[str, str]],
        *,
        position_bias_correct: bool = True,
        rng: np.random.Generator | None = None,
        n_resamples: int = 1000,
    ) -> PairwiseResult:
        if not pairs:
            raise ValueError("pairs must be non-empty")

        forward = np.empty(len(pairs), dtype=float)
        reverse = np.empty(len(pairs), dtype=float) if position_bias_correct else None
        a_wins = 0
        b_wins = 0
        ties = 0

        for i, (a, b) in enumerate(pairs):
            v_ab = self.judge(a, b)
            forward[i] = _outcome_to_score(v_ab.outcome)
            if v_ab.outcome == PairwiseOutcome.A_WINS:
                a_wins += 1
            elif v_ab.outcome == PairwiseOutcome.B_WINS:
                b_wins += 1
            else:
                ties += 1

            if reverse is not None:
                v_ba = self.judge(b, a)
                # Reverse call: B is in position 1, A is in position 2.
                # Convert verdict back to A's perspective: 1 - score.
                reverse[i] = 1.0 - _outcome_to_score(v_ba.outcome)

        if reverse is not None:
            per_pair = (forward + reverse) / 2.0
            # Position bias: fraction of pairs where the verdicts disagreed.
            position_bias: float | None = float(np.mean(forward != reverse))
        else:
            per_pair = forward
            position_bias = None

        point, lo, hi = bootstrap_ci(
            per_pair,
            n_resamples=n_resamples,
            confidence=self.confidence,
            rng=rng,
        )

        return PairwiseResult(
            win_rate_a=point,
            win_rate_a_ci=(lo, hi),
            n_pairs=len(pairs),
            n_a_wins=a_wins,
            n_b_wins=b_wins,
            n_ties=ties,
            confidence=self.confidence,
            position_bias_corrected=position_bias_correct,
            position_bias=position_bias,
            per_pair_scores=per_pair,
        )
