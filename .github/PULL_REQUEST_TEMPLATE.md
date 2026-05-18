<!--
Thanks for the PR. Before submitting, run locally and make sure all pass:

    pytest --cov=src/judgekit --cov-fail-under=85
    ruff check src tests
    ruff format --check src tests
    mypy --strict src/judgekit
-->

## Summary

<!-- One sentence on what changes and why. -->

## Type

- [ ] Bug fix (no API change)
- [ ] New statistical primitive (calibrator / drift measure / agreement / bias audit)
- [ ] New integration (Phoenix / Langfuse / new platform)
- [ ] Documentation / examples
- [ ] CI / tooling
- [ ] Other:

## Statistical primitives only — equivalence check

<!-- Delete this section if not adding a new primitive. -->

- Reference implementation cited: <!-- e.g., `statsmodels.stats.inter_rater.fleiss_kappa` -->
- Equivalence test fixtures (at least 3): `tests/test_<module>.py::test_*`
- Tolerance: <= 1e-9 on all fixtures (CI will fail otherwise).

## Tests

- [ ] Added or extended tests covering the change.
- [ ] All existing tests still pass locally.
- [ ] Tests use seeded RNG (`numpy.random.default_rng(seed)`).

## Checklist

- [ ] CHANGELOG.md updated under `## [Unreleased]`.
- [ ] Public API (`__all__`) updated if a new export was added.
- [ ] No changes to v1.x public API surface (or deprecation cycle started).
- [ ] Docstrings explain *why*, not *what*.
- [ ] Linked the issue this closes (`Fixes #N`) or the discussion this implements.
