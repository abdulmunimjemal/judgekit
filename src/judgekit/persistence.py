"""Save / load fitted JudgeHarness state to disk.

A judgekit "state" file is a directory containing:

- ``state.json`` — versioned metadata (judgekit version, calibrator class,
  drift baseline summary, configuration, fit timestamp).
- ``calibrator.pkl`` — pickled sklearn / numpy state for the fitted
  calibrator. Pickle is used here because sklearn's `IsotonicRegression`
  and `LogisticRegression` don't have a stable JSON serialization;
  `joblib` would be a heavier dep for the same result.
- ``drift_reference.npy`` — the raw reference scores the drift monitor
  was built from. Plain numpy binary so reload doesn't need our pickle.

The judge itself is NOT persisted. Callers re-attach a live judge at
load time; this is by design — most judges are LLM API clients with
secrets and connection state we shouldn't try to serialize.

A judgekit state file with major-version newer than the running judgekit
is refused. Minor-version newer is allowed but warnings can be added.

Security
--------
Plain ``pickle.load`` is a known arbitrary-code-execution surface — any
pickle stream can call any callable on import. judgekit ships a
:class:`RestrictedUnpickler` that only allows classes coming from
``judgekit``, ``sklearn``, ``scipy``, ``numpy``, and a small allow-list
of built-ins / collections. Anything else raises
:class:`StateFormatError` immediately. This is enforced by default in
:func:`load_harness`; pass ``allow_unsafe_pickle=True`` only when
loading a file you produced yourself in a trusted context (e.g. CI
artifact from your own pipeline) and want to bypass the allow-list.
See :doc:`SECURITY.md </SECURITY>` for the threat model.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from judgekit.calibration import (
    BetaCalibrator,
    HistogramBinCalibrator,
    IsotonicCalibrator,
    PlattCalibrator,
    TemperatureCalibrator,
)

if TYPE_CHECKING:
    from judgekit.calibration import Calibrator
    from judgekit.drift import DriftMethod
    from judgekit.harness import JudgeHarness
    from judgekit.judge import Judge

# Bump the MAJOR component when the on-disk format changes incompatibly.
# MINOR component: added optional fields that older loaders should ignore.
STATE_FORMAT_VERSION = "1.0"


# Allow-list for the restricted unpickler. We accept classes whose fully
# qualified module name starts with any of these prefixes, plus a small
# set of explicit (module, name) pairs from `builtins` / `collections`.
# Everything else is refused.
_ALLOWED_MODULE_PREFIXES: tuple[str, ...] = (
    "judgekit.",
    "sklearn.",
    "scipy.",
    "numpy.",
    "numpy",  # bare "numpy" (e.g. numpy.ndarray's __module__ in some versions)
    "joblib.",
)

_ALLOWED_BUILTINS: frozenset[tuple[str, str]] = frozenset(
    {
        ("builtins", "object"),
        ("builtins", "list"),
        ("builtins", "tuple"),
        ("builtins", "dict"),
        ("builtins", "set"),
        ("builtins", "frozenset"),
        ("builtins", "int"),
        ("builtins", "float"),
        ("builtins", "bool"),
        ("builtins", "str"),
        ("builtins", "bytes"),
        ("builtins", "bytearray"),
        ("builtins", "complex"),
        ("builtins", "NoneType"),
        ("builtins", "type"),
        ("builtins", "slice"),
        ("builtins", "range"),
        ("collections", "OrderedDict"),
        ("collections", "defaultdict"),
        ("collections", "deque"),
        ("collections", "Counter"),
        ("copyreg", "_reconstructor"),
        ("copyreg", "__newobj__"),
        ("copyreg", "__newobj_ex__"),
    }
)


class RestrictedUnpickler(pickle.Unpickler):
    """A :mod:`pickle` Unpickler that refuses classes outside the allow-list.

    Pickle streams can reference *any* callable, including ``os.system``
    and other gadget primitives. This subclass overrides
    :meth:`pickle.Unpickler.find_class` and accepts a class reference
    only if its module starts with one of :data:`_ALLOWED_MODULE_PREFIXES`
    OR the ``(module, name)`` pair is in :data:`_ALLOWED_BUILTINS`.

    Anything else raises :class:`StateFormatError` with a message that
    names the offending reference, so a calling user can see exactly
    what was rejected.
    """

    def find_class(self, module: str, name: str) -> Any:
        if (module, name) in _ALLOWED_BUILTINS:
            return super().find_class(module, name)
        for prefix in _ALLOWED_MODULE_PREFIXES:
            if module == prefix.rstrip(".") or module.startswith(prefix):
                return super().find_class(module, name)
        raise StateFormatError(
            f"RestrictedUnpickler refused to load {module}.{name}. "
            "Only judgekit / sklearn / scipy / numpy / joblib / "
            "safe-builtins are allowed in calibrator.pkl. If you trust "
            "this file and need to load it anyway, pass "
            "allow_unsafe_pickle=True to load_harness()."
        )


def _safe_pickle_load(fh: object, *, allow_unsafe: bool = False) -> Any:
    """Load a pickle stream, defaulting to the restricted unpickler."""
    if allow_unsafe:
        return pickle.load(fh)  # type: ignore[arg-type]
    return RestrictedUnpickler(fh).load()  # type: ignore[arg-type]


_CALIBRATOR_REGISTRY: dict[str, type[Calibrator]] = {
    "PlattCalibrator": PlattCalibrator,
    "IsotonicCalibrator": IsotonicCalibrator,
    "TemperatureCalibrator": TemperatureCalibrator,
    "BetaCalibrator": BetaCalibrator,
    "HistogramBinCalibrator": HistogramBinCalibrator,
}


@dataclass(frozen=True)
class StateMetadata:
    """Decoded `state.json` contents.

    Exposed for tooling (CLI, report, integrations) so callers can
    inspect a saved state without unpickling.
    """

    format_version: str
    judgekit_version: str
    calibrator_class: str
    confidence: float
    psi_warn: float
    psi_fail: float
    strict: bool
    drift_method: str
    fitted_at: str
    n_calibration_anchors: int


class StateFormatError(RuntimeError):
    """Raised when a saved state file has an incompatible major version."""


def _atomic_write_text(path: Path, content: str) -> None:
    """Write `content` to `path` via a temp file + rename (crash-safe)."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content)
    tmp.replace(path)


def save_harness(harness: JudgeHarness, path: str | Path) -> None:
    """Persist a fitted `JudgeHarness` to a directory.

    Creates the directory if it doesn't exist. Overwrites existing files
    atomically (temp-write + rename).
    """
    if not harness.fitted:
        raise RuntimeError("JudgeHarness must be fitted before save_harness()")
    assert harness._drift_monitor is not None  # for type checkers
    assert harness._calibration_raw is not None

    from judgekit import __version__ as judgekit_version

    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)

    metadata = {
        "format_version": STATE_FORMAT_VERSION,
        "judgekit_version": judgekit_version,
        "calibrator_class": type(harness.calibrator).__name__,
        "confidence": harness.confidence,
        "psi_warn": harness.psi_warn,
        "psi_fail": harness.psi_fail,
        "strict": harness.strict,
        "drift_method": harness._drift_monitor.method,
        "fitted_at": datetime.now(timezone.utc).isoformat(),
        "n_calibration_anchors": len(harness.calibration_set),
    }
    _atomic_write_text(target / "state.json", json.dumps(metadata, indent=2))

    # Pickle the fitted calibrator. Use HIGHEST_PROTOCOL for compactness;
    # protocol 5 is supported by every Python we target (>=3.10).
    cal_bytes = pickle.dumps(harness.calibrator, protocol=pickle.HIGHEST_PROTOCOL)
    tmp = target / "calibrator.pkl.tmp"
    tmp.write_bytes(cal_bytes)
    tmp.replace(target / "calibrator.pkl")

    # Reference scores -> numpy binary so reload is library-only.
    np.save(target / "drift_reference.npy", harness._calibration_raw)


def load_metadata(path: str | Path) -> StateMetadata:
    """Read state metadata without unpickling the calibrator."""
    source = Path(path)
    raw_text = (source / "state.json").read_text()
    raw: dict[str, Any] = json.loads(raw_text)
    _check_format_version(raw.get("format_version", ""))
    return StateMetadata(
        format_version=raw["format_version"],
        judgekit_version=raw["judgekit_version"],
        calibrator_class=raw["calibrator_class"],
        confidence=raw["confidence"],
        psi_warn=raw["psi_warn"],
        psi_fail=raw["psi_fail"],
        strict=raw["strict"],
        drift_method=raw["drift_method"],
        fitted_at=raw["fitted_at"],
        n_calibration_anchors=raw["n_calibration_anchors"],
    )


def load_harness(
    path: str | Path,
    judge: Judge,
    *,
    allow_unsafe_pickle: bool = False,
) -> JudgeHarness:
    """Restore a fitted `JudgeHarness` from disk and reattach a live judge.

    The `judge` argument is NOT validated against the original — we don't
    persist judge identity. Caller is responsible for re-attaching the
    right judge. If you want to detect drift between the original judge
    and the new one, run `harness.evaluate(...)` on a held-out set and
    inspect the drift status.

    By default the calibrator pickle is loaded via :class:`RestrictedUnpickler`,
    which refuses any class outside the judgekit / sklearn / scipy /
    numpy / safe-builtins allow-list. Pass ``allow_unsafe_pickle=True``
    only when loading a file you produced yourself in a trusted context
    and you need to bypass the allow-list (e.g. a third-party calibrator
    subclass). See ``SECURITY.md`` for the threat model.
    """
    # Import here to avoid the import cycle (harness imports drift, drift
    # is independent; we import harness lazily so persistence can be
    # imported during harness init without a cycle).
    from judgekit.drift import DriftMonitor
    from judgekit.harness import JudgeHarness
    from judgekit.judge import CalibrationSet

    metadata = load_metadata(path)
    source = Path(path)

    with (source / "calibrator.pkl").open("rb") as fh:
        loaded_calibrator: Calibrator = _safe_pickle_load(fh, allow_unsafe=allow_unsafe_pickle)
    if type(loaded_calibrator).__name__ != metadata.calibrator_class:
        raise StateFormatError(
            f"calibrator.pkl contains {type(loaded_calibrator).__name__} but "
            f"state.json declares {metadata.calibrator_class}"
        )

    reference: np.ndarray = np.asarray(np.load(source / "drift_reference.npy"), dtype=float)

    drift_method: DriftMethod = metadata.drift_method  # type: ignore[assignment]

    # The harness needs a calibration_set; we don't persist items+labels,
    # so we reconstruct a placeholder with the right anchor count. The
    # calibrator state itself carries everything needed for predict().
    placeholder_examples = []
    # Calibration set must be non-empty AND >= 10 to satisfy harness invariants.
    n = max(10, metadata.n_calibration_anchors)
    from judgekit.judge import LabeledExample

    for i in range(n):
        placeholder_examples.append(LabeledExample(item=f"_loaded_{i}", label=0.5))

    harness = JudgeHarness(
        judge=judge,
        calibration_set=CalibrationSet(examples=placeholder_examples),
        calibrator=loaded_calibrator,
        confidence=metadata.confidence,
        psi_warn=metadata.psi_warn,
        psi_fail=metadata.psi_fail,
        strict=metadata.strict,
    )
    harness._drift_monitor = DriftMonitor(
        reference_scores=reference,
        psi_warn=metadata.psi_warn,
        psi_fail=metadata.psi_fail,
        method=drift_method,
    )
    harness._calibration_raw = reference
    harness._fitted = True
    return harness


def _check_format_version(version: str) -> None:
    """Refuse loads where the on-disk major version doesn't match."""
    if not version:
        raise StateFormatError("state.json has no format_version field; cannot load.")
    try:
        on_disk_major = int(version.split(".")[0])
    except ValueError as e:
        raise StateFormatError(f"unparseable format_version {version!r}") from e
    current_major = int(STATE_FORMAT_VERSION.split(".")[0])
    if on_disk_major != current_major:
        raise StateFormatError(
            f"saved state is format v{version}; this judgekit understands "
            f"v{STATE_FORMAT_VERSION} (major version {current_major} only). "
            "Upgrade judgekit or migrate the state file."
        )
