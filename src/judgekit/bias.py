"""Bias audits for LLM judges.

Three audits ship here, each probes a well-documented LLM-as-judge
failure mode:

- :func:`position_bias` — pairwise judges that prefer whichever option
  is in slot 1 (or slot 2) regardless of content. Swap-rate test.
- :func:`verbosity_bias` — pointwise judges that reward longer outputs
  even when length is irrelevant. Correlation test.
- :func:`format_sensitivity` — pointwise judges whose scores depend on
  output format (markdown vs JSON vs plain text) even when content is
  identical. Variance-across-formats test.

Each returns a :class:`BiasReport` with the raw metric and a verdict
field driven by published thresholds. Run :class:`BiasReport.is_concerning`
to get a simple bool you can gate releases on.

Math references:
- Zheng et al. (2023) "Judging LLM-as-a-Judge with MT-Bench and Chatbot
  Arena" — original position-bias documentation (~25-75% swap rates on
  GPT-4 in 2023).
- Feldhus et al. (2026) "Judge Circuits" — format-sensitivity finding
  that judges have fragile format-specific terminal branches.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from judgekit.judge import Judge
from judgekit.pairwise import PairwiseJudge, PairwiseOutcome


@dataclass
class BiasReport:
    """One bias-audit result.

    ``value`` is the raw metric (swap rate, correlation, std dev across
    formats — depending on which audit). ``threshold_warn`` and
    ``threshold_fail`` come from published norms. ``verdict`` is one of
    ``"clean"`` / ``"watch"`` / ``"concerning"``.
    """

    audit: str
    value: float
    threshold_warn: float
    threshold_fail: float
    verdict: str
    detail: dict[str, float]

    def is_concerning(self) -> bool:
        return self.verdict == "concerning"

    def _verdict_from_value(self) -> str:  # pragma: no cover — internal
        if abs(self.value) >= self.threshold_fail:
            return "concerning"
        if abs(self.value) >= self.threshold_warn:
            return "watch"
        return "clean"


def _verdict_for(value: float, warn: float, fail: float) -> str:
    v = abs(value)
    if v >= fail:
        return "concerning"
    if v >= warn:
        return "watch"
    return "clean"


def position_bias(
    judge: PairwiseJudge,
    pairs: list[tuple[str, str]],
    *,
    threshold_warn: float = 0.10,
    threshold_fail: float = 0.25,
) -> BiasReport:
    """Audit pairwise judges for position bias.

    Runs each pair as ``(a, b)`` AND ``(b, a)``. If the judge is
    unbiased, the two verdicts should agree on a winner — A still beats
    B regardless of which slot it's in. The audit metric is the
    **swap rate**: fraction of pairs where the A-preference score
    differed between the two orderings.

    Returns a :class:`BiasReport` with ``value`` = swap rate in [0, 1].
    Thresholds default to 0.10 (watch) and 0.25 (concerning) following
    the Zheng et al. 2023 MT-Bench findings.
    """
    if not pairs:
        raise ValueError("pairs must be non-empty")

    n_swap = 0
    n_total = 0
    for a, b in pairs:
        v_ab = judge(a, b).outcome
        v_ba = judge(b, a).outcome
        # Map each verdict to an A-preference score: A_WINS=1, TIE=0.5,
        # B_WINS=0 — both directions.
        score_ab = _outcome_to_a_score(v_ab)
        score_ba_inverted = 1.0 - _outcome_to_a_score(v_ba)
        if score_ab != score_ba_inverted:
            n_swap += 1
        n_total += 1

    rate = n_swap / n_total
    verdict = _verdict_for(rate, threshold_warn, threshold_fail)
    return BiasReport(
        audit="position_bias",
        value=rate,
        threshold_warn=threshold_warn,
        threshold_fail=threshold_fail,
        verdict=verdict,
        detail={"n_swap": float(n_swap), "n_total": float(n_total)},
    )


def _outcome_to_a_score(outcome: PairwiseOutcome) -> float:
    if outcome == PairwiseOutcome.A_WINS:
        return 1.0
    if outcome == PairwiseOutcome.B_WINS:
        return 0.0
    return 0.5


def verbosity_bias(
    judge: Judge,
    items: list[str],
    *,
    threshold_warn: float = 0.30,
    threshold_fail: float = 0.60,
) -> BiasReport:
    """Audit pointwise judges for verbosity bias.

    Runs the judge on each item, then computes the Pearson correlation
    between the item's character length and the judge's score. A large
    positive correlation means "the judge thinks longer responses are
    better" regardless of content; large negative means "shorter is
    better."

    Returns a :class:`BiasReport` with ``value`` = correlation in [-1, 1].
    Thresholds default to ±0.30 (watch) and ±0.60 (concerning).
    """
    if len(items) < 3:
        raise ValueError("verbosity_bias needs at least 3 items to compute a correlation")

    lengths = np.asarray([len(item) for item in items], dtype=float)
    scores = np.asarray([judge(item).score for item in items], dtype=float)

    if np.std(lengths) < 1e-12 or np.std(scores) < 1e-12:
        # No variance in lengths or scores -> correlation is undefined.
        # That's a sign of a degenerate input; report clean (no bias detected).
        return BiasReport(
            audit="verbosity_bias",
            value=0.0,
            threshold_warn=threshold_warn,
            threshold_fail=threshold_fail,
            verdict="clean",
            detail={"n": float(len(items)), "note_no_variance": 1.0},
        )

    correlation = float(np.corrcoef(lengths, scores)[0, 1])
    verdict = _verdict_for(correlation, threshold_warn, threshold_fail)
    return BiasReport(
        audit="verbosity_bias",
        value=correlation,
        threshold_warn=threshold_warn,
        threshold_fail=threshold_fail,
        verdict=verdict,
        detail={"n": float(len(items))},
    )


def format_sensitivity(
    judge: Judge,
    items: list[str],
    *,
    formats: dict[str, str] | None = None,
    threshold_warn: float = 0.10,
    threshold_fail: float = 0.20,
) -> BiasReport:
    """Audit pointwise judges for format sensitivity.

    For each item, render it through each of the configured ``formats``
    (e.g. wrap in markdown code-fence, wrap in JSON, leave plain). Score
    each variant and compute the standard deviation of scores per item.
    The audit metric is the **mean per-item standard deviation across
    formats** — a stable judge should produce nearly identical scores
    regardless of formatting wrapper.

    ``formats`` is a mapping of ``name -> wrapper format string``. Each
    format string must contain ``{content}``; ``content`` will be the
    raw item text. Default formats wrap as plain, markdown-blockquote,
    and JSON.

    Returns a :class:`BiasReport` with ``value`` = mean per-item score
    std. Thresholds default to 0.10 (watch) and 0.20 (concerning) on
    a [0, 1] score scale.
    """
    if formats is None:
        formats = {
            "plain": "{content}",
            "markdown_quote": "> {content}",
            "json": '{{"text": "{content}"}}',
        }
    if len(formats) < 2:
        raise ValueError("format_sensitivity needs at least 2 formats")
    if not items:
        raise ValueError("items must be non-empty")

    per_item_std = np.empty(len(items), dtype=float)
    for i, item in enumerate(items):
        format_scores = np.empty(len(formats), dtype=float)
        for j, (_name, template) in enumerate(formats.items()):
            if "{content}" not in template:
                raise ValueError(
                    f"every format template must contain '{{content}}'; got {template!r}"
                )
            rendered = template.format(content=item)
            format_scores[j] = judge(rendered).score
        per_item_std[i] = float(np.std(format_scores))

    mean_std = float(np.mean(per_item_std))
    verdict = _verdict_for(mean_std, threshold_warn, threshold_fail)
    return BiasReport(
        audit="format_sensitivity",
        value=mean_std,
        threshold_warn=threshold_warn,
        threshold_fail=threshold_fail,
        verdict=verdict,
        detail={
            "n_items": float(len(items)),
            "n_formats": float(len(formats)),
            "max_per_item_std": float(per_item_std.max()),
        },
    )
