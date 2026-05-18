# Changelog

All notable changes to `judgekit` are documented here. Format adapted from [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- GitHub Actions CI pipeline: `ruff check`, `ruff format --check`, `mypy --strict`, `pytest` matrix across Python 3.10/3.11/3.12/3.13 × Ubuntu/macOS; coverage gate at 85% (will tighten to 90% before v1.0).
- GitHub Actions release workflow: Trusted Publishing to PyPI on tag, auto-generated GitHub release notes.
- `CONTRIBUTING.md` — quality bar, statistical-primitive submission requirements, design constraints.
- Issue + pull request templates under `.github/`.

## [0.1.0] — 2026-05-17

Initial scaffold (private). Not published.

### Added

- `judgekit.judge`: `Judge` Protocol, `JudgeOutput`, `LabeledExample`, `CalibrationSet` dataclasses.
- `judgekit.calibration`: `Calibrator` ABC, `PlattCalibrator` (sklearn LogisticRegression), `IsotonicCalibrator` (sklearn IsotonicRegression), `bootstrap_ci()` percentile-based confidence intervals.
- `judgekit.drift`: `kl_divergence()`, `psi()`, `DriftMonitor`, `DriftStatus`. PSI thresholds default to the credit-risk literature (warn ≥ 0.10, fail ≥ 0.25).
- `judgekit.harness`: `JudgeHarness` orchestrator; `EvalResult` dataclass; `CalibrationStaleError` raised in strict mode when distribution drift breaches the fail threshold.
- 25 unit tests, ruff-clean, validated against sklearn/numpy/scipy.
- Apache-2.0 license; Python 3.10+ supported.

[Unreleased]: https://github.com/abdulmunimj/judgekit/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/abdulmunimj/judgekit/releases/tag/v0.1.0
