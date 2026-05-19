# Changelog

All notable changes to `judgekit` are documented here. Format adapted from [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `judgekit.JudgekitError` base exception. Every error judgekit raises now inherits from it, so a single `except JudgekitError` catches the entire library. `CalibrationStaleError` and `StateFormatError` continue to inherit from `RuntimeError` transitively, so existing `except RuntimeError` blocks still work.
- `judgekit.ConfidenceInterval` NamedTuple. `EvalResult.confidence_interval` and `PairwiseResult.win_rate_a_ci` now return this type, which exposes `.lower` / `.upper` accessors while staying tuple-compatible — `lo, hi = result.confidence_interval` keeps working.
- State format `1.1`: `state.json` now records `drift_thresholds`, `calibrator_params`, `harness_class`, `score_range`, `drift_bins`, and a forward-compat `schema_extras` slot. Loading a `1.0` state still works — the new fields default sensibly when missing.

### Changed

- **API freeze (breaking, intentional, v1.0-only window).** Every argument past the second positional one on the four public constructors / helpers is now keyword-only: `JudgeHarness(judge, calibration_set, *, calibrator=..., confidence=..., psi_warn=..., psi_fail=..., strict=...)`, `DriftMonitor(reference_scores, *, bins=..., psi_warn=..., psi_fail=..., method=..., ks_p_threshold=..., wasserstein_threshold=...)`, `PairwiseHarness(judge, *, confidence=...)`, and `bootstrap_ci(scores, statistic=..., *, n_resamples=..., confidence=..., rng=...)`. This freezes the call shape for v1.x so we can add parameters later without shifting positional meanings.

## [1.0.0rc1] — 2026-05-19

First public release candidate. Public-API contract: every symbol re-exported from `judgekit.__init__.__all__`, every CLI subcommand and flag listed in `judgekit --help`, and every field in `state.json` will retain its name, signature, and semantics through all v1.x releases. Additions are allowed; removals or renames require v2.0.

### Added

- 5 calibrators: `PlattCalibrator`, `IsotonicCalibrator`, `TemperatureCalibrator` (Guo et al. 2017), `BetaCalibrator` (Kull et al. 2017), `HistogramBinCalibrator` (Naeini et al. 2015). `select_calibrator(n_anchors)` picks a default for the calibration-set size.
- 4 drift measures: `kl_divergence`, `psi`, `ks_test` (wraps `scipy.stats.ks_2samp`), `wasserstein` (wraps `scipy.stats.wasserstein_distance`). `DriftMonitor` supports `method="psi"|"ks"|"wasserstein"|"all"`.
- 3 inter-rater agreement metrics: `cohens_kappa`, `fleiss_kappa`, `krippendorff_alpha` (nominal / ordinal / interval / ratio). Each validated against the reference library (sklearn / statsmodels / `krippendorff`) to 1e-9 tolerance.
- 3 bias audits: `position_bias`, `verbosity_bias`, `format_sensitivity` with `BiasReport` dataclass and `.is_concerning()` thresholds from published norms.
- Pairwise eval primitives: `PairwiseJudge` Protocol, `PairwiseVerdict`, `PairwiseOutcome`, `PairwiseHarness` with bootstrap win-rate CIs and position-bias-corrected aggregation.
- Persistence: `JudgeHarness.save(path)` and `JudgeHarness.load(path, judge)` with a versioned state format. Predictions round-trip exactly (verified to 1e-12 across all five calibrators).
- HTML eval report: `JudgeHarness.report(path)` renders a self-contained HTML file with plotly inlined (no CDN). Contains reliability diagram, score distribution overlay, and a PSI drift gauge.
- Command-line interface: `judgekit calibrate`, `judgekit report`, `judgekit audit`, `judgekit version`.
- PEP 561 `py.typed` marker — downstream `mypy` / `pyright` users now read our type annotations.
- Apache-2.0 LICENSE file shipped in both sdist and wheel.
- `judgekit[all]` extra that installs every optional integration in one go.

### Changed

- Runtime dependencies pinned with major-version ceilings: `numpy>=1.26,<3`, `scipy>=1.11,<2`, `scikit-learn>=1.4,<2`. Prevents silent breakage on future major bumps.
- Optional `[report]` extra also pinned: `plotly>=5.20,<7`, `jinja2>=3.1,<4`.
- `Development Status` classifier promoted from `3 - Alpha` to `4 - Beta` pending v1.0.0 final.
- Version is now single-sourced from `src/judgekit/__init__.py.__version__` via `hatchling`'s dynamic-version hook.

## [0.1.0] — 2026-05-17

Initial scaffold (private). Not published.

### Added

- `judgekit.judge`: `Judge` Protocol, `JudgeOutput`, `LabeledExample`, `CalibrationSet` dataclasses.
- `judgekit.calibration`: `Calibrator` ABC, `PlattCalibrator` (sklearn LogisticRegression), `IsotonicCalibrator` (sklearn IsotonicRegression), `bootstrap_ci()` percentile-based confidence intervals.
- `judgekit.drift`: `kl_divergence()`, `psi()`, `DriftMonitor`, `DriftStatus`. PSI thresholds default to the credit-risk literature (warn ≥ 0.10, fail ≥ 0.25).
- `judgekit.harness`: `JudgeHarness` orchestrator; `EvalResult` dataclass; `CalibrationStaleError` raised in strict mode when distribution drift breaches the fail threshold.
- 25 unit tests, ruff-clean, validated against sklearn/numpy/scipy.
- Apache-2.0 license; Python 3.10+ supported.

[Unreleased]: https://github.com/abdulmunimjemal/judgekit/compare/v1.0.0rc1...HEAD
[1.0.0rc1]: https://github.com/abdulmunimjemal/judgekit/releases/tag/v1.0.0rc1
[0.1.0]: https://github.com/abdulmunimjemal/judgekit/releases/tag/v0.1.0
