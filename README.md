# judgekit

> Calibrate, monitor, and refuse to ship miscalibrated LLM judges.

LLM-as-judge has become the default eval method, but raw judge scores are uncalibrated proxies. They drift across judge-model versions, prompt-mix changes, and time — and most teams find out *after* a release moves the headline metric outside its previous confidence interval.

**judgekit** is a small, opinionated Python library that wraps any judge with three things:

1. **A calibrator** (Platt or isotonic) fit against a small human-labeled anchor set, so the scores you publish are estimated ground-truth probabilities — not raw judge logits.
2. **A drift monitor** (PSI + KL) over the score distribution, so an upstream judge-model update can't silently invalidate your calibration.
3. **A bootstrap CI** around your headline metric, so a non-overlapping CI is the rule for "this change is real."

When drift exceeds the configured threshold, `JudgeHarness.evaluate()` **raises** rather than returning a number you'd treat as comparable to last week's. That refusal is the point.

---

## Install

```bash
pip install judgekit
```

Python 3.10+. Dependencies: `numpy`, `scipy`, `scikit-learn`.

Optional extras:

```bash
pip install 'judgekit[report]'   # HTML eval reports (plotly + jinja2)
pip install 'judgekit[all]'      # everything optional
```

---

## 30-second quickstart

```python
from judgekit import (
    CalibrationSet,
    JudgeHarness,
    JudgeOutput,
    LabeledExample,
)

# 1. Your judge — anything callable that returns a JudgeOutput.
def my_judge(item: str) -> JudgeOutput:
    raw = call_llm_as_judge(item)          # your code, returns float in [0, 1]
    return JudgeOutput(score=raw)

# 2. A small human-labeled calibration set (50–500 items is typical).
calset = CalibrationSet(examples=[
    LabeledExample(item="response A …", label=1.0),
    LabeledExample(item="response B …", label=0.0),
    # …
])

# 3. Fit once.
harness = JudgeHarness(judge=my_judge, calibration_set=calset).fit()

# 4. Use in CI before publishing eval numbers.
result = harness.evaluate(my_eval_items)
print(result.point_estimate, result.confidence_interval)
# 0.7423, (0.6981, 0.7864)
```

**Strict mode.** By default `JudgeHarness` is `strict=True`: `evaluate()` raises `CalibrationStaleError` when the score distribution has drifted (PSI ≥ 0.25) since fit-time. This is on purpose — silent drift is what we're guarding against.

**Heads up for small toy data.** PSI is noisy under ~100 calibration items, so the quickstart can trip the strict-mode guard on small toys. While you're learning the library, pass `JudgeHarness(..., strict=False)` and inspect `result.drift` yourself; for real eval pipelines keep `strict=True`.

```python
# Learning / exploration mode — warning, not refusal.
harness = JudgeHarness(judge=my_judge, calibration_set=calset, strict=False).fit()
result = harness.evaluate(my_eval_items)
if result.drift.is_drifted:
    print(f"⚠ judge drift (PSI={result.drift.psi:.3f}) — recalibrate before shipping")
```

If your judge-model rolled overnight and the score distribution shifted, the strict-mode path raises `CalibrationStaleError` — you re-fit on a fresh calibration set before reporting numbers. No silent regressions.

---

## API surface

| Module | What's in it |
|---|---|
| `judgekit.judge` | `Judge` protocol, `JudgeOutput`, `LabeledExample`, `CalibrationSet` |
| `judgekit.calibration` | `Calibrator` ABC + `PlattCalibrator`, `IsotonicCalibrator`, `TemperatureCalibrator`, `BetaCalibrator`, `HistogramBinCalibrator`, `select_calibrator`, `bootstrap_ci` |
| `judgekit.drift` | `DriftMonitor`, `DriftStatus`, `kl_divergence`, `psi`, `ks_test`, `wasserstein` |
| `judgekit.agreement` | `cohens_kappa`, `fleiss_kappa`, `krippendorff_alpha` |
| `judgekit.pairwise` | `PairwiseJudge` protocol, `PairwiseVerdict`, `PairwiseOutcome`, `PairwiseHarness`, `PairwiseResult` |
| `judgekit.bias` | `position_bias`, `verbosity_bias`, `format_sensitivity`, `BiasReport` |
| `judgekit.persistence` | `load_harness`, `load_metadata`, `StateMetadata`, `StateFormatError` |
| `judgekit.harness` | `JudgeHarness`, `EvalResult`, `CalibrationStaleError` |
| `judgekit.report` | HTML report (requires `[report]` extra) |
| `judgekit.cli` | `judgekit` console script |

Default calibrator is `IsotonicCalibrator` (flexible, non-parametric). For smaller calibration sets, use `select_calibrator(n_anchors)` — it returns `TemperatureCalibrator` for <50 anchors, `BetaCalibrator` for 50–200, `IsotonicCalibrator` for ≥200.

---

## Choosing a calibrator

| | **Platt (logistic)** | **Isotonic** |
|---|---|---|
| Form | Single-feature sigmoid | Monotone step function |
| Min data | ~50 items | ~200+ items for stable fit |
| Bias | Smooth, may underfit non-sigmoid curves | Can overfit on small sets |
| Use when | Calibration set is small, relationship is roughly sigmoid | You have ≥200 anchors and want flexibility |

Both expose the same `fit(raw_scores, labels) -> Calibrator` and `predict(raw_scores) -> np.ndarray` interface.

---

## Drift thresholds

Defaults follow the standard PSI rules-of-thumb from credit-risk monitoring:

| PSI | Verdict |
|---|---|
| < 0.10 | `stable` — calibration trusted |
| 0.10 – 0.25 | `watch` — re-calibrate soon |
| > 0.25 | `drifted` — `evaluate()` raises in strict mode |

Override per-domain via `JudgeHarness(psi_warn=…, psi_fail=…)`.

---

## Why this exists

Hamel Husain, [*LLM Evals FAQ* (2026)](https://hamel.dev/blog/posts/evals-faq/):

> "The single most impactful investment for AI teams isn't a fancy evaluation dashboard — it's building a customized interface that lets anyone examine what their AI is actually doing."

OpenCompass [#2392 (2026)](https://github.com/open-compass/opencompass/issues/2392):

> "Raw judge scores are uncalibrated proxies that can invert rankings (proxy-goodhart), drift across time/domains/prompt mixes, and produce misleading confidence intervals."

There are great frameworks for *running* LLM judges (Phoenix, Langfuse, OpenCompass, DeepEval). None of them ship a primitive that says "this score is calibrated, here's the CI, and here's when I refuse to ship." That's the gap judgekit fills.

---

## Design choices

- **No async.** Judges that need to be parallelized should be parallelized at the caller's level. Keeping judgekit synchronous makes the math obvious and the failure modes legible.
- **Refuse-by-default.** `strict=True` is the default for `JudgeHarness`. If you want soft warnings, pass `strict=False` and inspect `result.drift` yourself — but the library's stance is that publishing eval results past a known drift threshold is worse than not publishing them.
- **Model-agnostic.** Anything callable is a judge. LLM, fine-tuned classifier, rubric, human. The library doesn't know or care.
- **Small surface.** Four modules. Two calibrators. Two drift measures. One harness. Resist scope creep.

---

## License

Apache-2.0.

## Status

`0.1.0` — alpha. API may change before `1.0`. Issues, PRs, and design pushback welcome.
