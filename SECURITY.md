# Security policy

`judgekit` is a library for calibrating, monitoring, and reporting on LLM
judges. It writes a small amount of state to disk and renders HTML
reports. This document describes its threat model, what we defend
against by default, and what we explicitly do not.

## Reporting a vulnerability

Please send a private email to **abdulmunimjemal@gmail.com** with a
description of the issue, a minimal reproducer, and the version (or
commit SHA) you tested against. Please do not open a public GitHub
issue for vulnerability reports.

We aim to respond within **7 days** and ship a fix or mitigation within
**30 days** for confirmed vulnerabilities, faster for criticals.

## Threat model

| Surface | What we defend by default | What we don't |
|---|---|---|
| `JudgeHarness.load(path, judge)` / `judgekit.persistence.load_harness` | Loads `calibrator.pkl` through a restricted unpickler that only accepts classes from `judgekit.*`, `sklearn.*`, `scipy.*`, `numpy.*`, `joblib.*`, and a small allow-list of `builtins` / `collections`. Anything else (e.g. `os.system`, `subprocess.Popen`, `eval`) raises `StateFormatError`. | Files where the attacker is already in the allow-list namespace (e.g. a malicious sklearn subclass). Pickle is fundamentally not a sandbox; we narrow the attack surface, we don't eliminate it. **Do not load state files from sources you don't trust** — restricted-unpickle is a defence-in-depth, not a sandbox. |
| `JudgeHarness.report(path)` / HTML report | Jinja2 template renders with `autoescape=True`. The `title` parameter, calibrator class name, drift verdict, and all other interpolated strings are HTML-escaped automatically before they reach the file. Only the three plotly fragments are explicitly marked as `Markup` and rendered raw. | Output to a directory you didn't intend (the library writes wherever `path` points). Wrap calls in your own filesystem boundary if that matters. |
| `judgekit calibrate --traces F --gold F --out DIR` | Refuses if `traces` and `gold` don't have at least 10 matched items; refuses if records lack the required JSON fields. | Path traversal on `--out` — the CLI passes the path through `pathlib.Path` directly. If you invoke `judgekit` with untrusted CLI args, restrict the working directory (e.g. via container / chroot / sandbox). |
| `_read_jsonl(path)` | Catches `json.JSONDecodeError` and surfaces the line number. | DoS via very large files / deeply nested JSON: we don't bound parse size. If you accept user-supplied JSONL as a service, gate file size at the boundary. |
| `numpy / scipy / scikit-learn` imports | Major-version ceilings (`numpy<3`, `scipy<2`, `scikit-learn<2`) so a future major bump can't silently change behaviour. CVE-tracked via `pip-audit` in CI. | C-extension supply-chain compromise of these packages — they include compiled binaries; pin transitive deps in your production lockfile. |

## What this library is *not*

- A sandbox for executing other people's code.
- A way to safely deserialize arbitrary user uploads. Calibrator state files are intended to flow inside *your own* trusted pipeline (CI artifact → eval gate; teammate share inside a trust boundary). If you accept calibrator state files from external sources, treat them as you would any pickled artifact: restricted-unpickle is a meaningful narrowing of the attack surface, but it is not a sandbox.

## Bypass options (use with care)

- `JudgeHarness.load(path, judge, allow_unsafe_pickle=True)` skips the restricted unpickler. Pass this only when you produced the file yourself in a trusted context AND your calibrator pickle references classes outside the allow-list (e.g. a custom calibrator subclass from a third-party library you trust). Audit before flipping it.

## Out of scope for v1.x

These are tracked as v2.0 candidates rather than v1.x security work:

- Pickle-free serialization of `PlattCalibrator` and `IsotonicCalibrator` (would let us drop `pickle` from the dependency surface entirely). `TemperatureCalibrator`, `BetaCalibrator`, and `HistogramBinCalibrator` only need primitive state + numpy arrays — those are already cheap to migrate. The blocker is sklearn's `LogisticRegression` and `IsotonicRegression`, which have no stable JSON serialization.
- Signature / checksum verification on state-file bundles (à la Sigstore for ML models).
- Sandboxed CLI execution profile.

## Past advisories

None yet. This file will list every CVE-class issue we ship a fix for, with affected versions and the upgrade path.
