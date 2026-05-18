# Contributing to judgekit

Thanks for thinking about contributing — `judgekit` is a small, focused library and we want to keep it that way. Read this whole page before opening a non-trivial PR.

## What this library is (and isn't)

**Is:** the calibration + drift-monitoring + refuse-by-default layer for LLM judges. Statistically grounded, batteries included, opinionated.

**Isn't:** a general eval framework, an observability platform, or a model server. Use Langfuse, Phoenix, Promptfoo, DeepEval, etc. for those — and pair them with `judgekit` for the reliability layer.

If your PR pushes us toward "more eval framework" or "more dashboard," it'll probably be declined. Issues first.

## Getting set up

```bash
git clone https://github.com/abdulmunimj/judgekit.git
cd judgekit
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

Python 3.10+ required. macOS and Linux supported; Windows is best-effort.

Optional integration deps:
- `pip install -e ".[phoenix]"` — work on the Phoenix integration
- `pip install -e ".[langfuse]"` — work on the Langfuse integration
- `pip install -e ".[report]"` — work on HTML report rendering
- `pip install -e ".[all]"` — everything

## Quality bar (enforced in CI)

Before opening a PR:

```bash
pytest --cov=src/judgekit --cov-fail-under=85
ruff check src tests
ruff format --check src tests
mypy --strict src/judgekit
```

All four must pass. CI runs the same on Python 3.10/3.11/3.12/3.13 across Ubuntu and macOS. If your platform passes but the matrix fails, it's still a fail.

## What we'll review fast vs. slow

**Fast review (1-3 days):**
- Bug fixes with a regression test
- New statistical primitives that ship a reference-implementation equivalence test (sklearn / scipy / statsmodels / netcal)
- Documentation improvements
- Integration adapters for new eval platforms (Braintrust, Helicone, OpenLLMetry, etc.)

**Slow review or likely decline:**
- New eval-framework features that overlap with Promptfoo / DeepEval / etc.
- Dashboards beyond the existing HTML report
- Async refactors of the core (sync-only is a deliberate design choice for v1.x)
- Anything that changes the public API of v1.x without a deprecation cycle

## Adding a new statistical primitive

This is the most common contribution. Required pieces:

1. **A reference implementation** — what does scipy / sklearn / statsmodels / netcal / `krippendorff` do? Cite it in a comment.
2. **Three equivalence-test fixtures** — your implementation must match the reference to within `1e-9` on each. Tests live in `tests/test_<module>.py`.
3. **A short statistical-references docs entry** — paper citation, primary use case, edge cases. `docs/statistical-references.md`.
4. **An entry in CHANGELOG.md** under `## Unreleased`.

We will not merge a new calibrator or drift measure without these four.

## Style

- `ruff format` is the formatter. Run it. CI fails otherwise.
- `ruff check` rules are configured in `pyproject.toml`.
- Docstrings: short, with the "why" not the "what". Don't comment what `numpy.histogram` does; comment why you chose 10 bins.
- `from __future__ import annotations` at the top of every module — keeps type hints lazy.
- Public API in `__all__`. New modules add their exports to `src/judgekit/__init__.py`.
- Tests use `numpy.random.default_rng(seed)` for any randomness. Seeded tests only.

## Commit messages

Conventional Commits format:

```
type(scope): short imperative subject

Optional body with the why.
```

Examples seen in `git log`:

- `feat(calibration): add Beta calibrator`
- `fix(drift): handle empty reference window`
- `docs: clarify refuse-by-default semantics`

`type` ∈ `feat | fix | docs | refactor | perf | test | chore | build | ci | revert`.

## Versioning

Semantic versioning. v1.x is API-stable: no breaking changes. Anything that would break a v1.0 user goes through a deprecation cycle and lands in v2.0 at earliest.

If you need to add a new primitive without breaking existing API, the recipe is: add a new optional field with a default, never reorder positional args, never change return types.

## License

Apache-2.0. By submitting a PR you agree to license your contribution under the same terms.

## Be honest in PR descriptions

If your fix is approximate, say so. If your benchmark is small, say so. If you're not sure your statistical primitive is correct, say so — we'll help. Don't oversell.

## Where to ask questions

GitHub Discussions for design questions and "how do I use this". GitHub Issues for bugs, feature requests, and tracked work.
