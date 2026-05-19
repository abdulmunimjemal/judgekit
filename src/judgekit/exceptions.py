"""Public exception hierarchy for judgekit.

Every error judgekit raises inherits from :class:`JudgekitError` so callers
can write a single ``except JudgekitError`` block at a boundary without
having to enumerate the specific subclasses they care about.

The class hierarchy is:

    RuntimeError
    └── JudgekitError              # catch-all
        ├── CalibrationStaleError  # raised by JudgeHarness when drift >= fail
        └── StateFormatError       # raised by load_harness on bad/old state

``CalibrationStaleError`` and ``StateFormatError`` remain re-exported
from :mod:`judgekit.harness` and :mod:`judgekit.persistence` respectively
for backwards-compatible imports.
"""

from __future__ import annotations


class JudgekitError(RuntimeError):
    """Base class for every error raised by judgekit.

    Subclassing :class:`RuntimeError` (not :class:`Exception`) preserves
    the pre-1.0 behaviour where ``CalibrationStaleError`` and
    ``StateFormatError`` were direct ``RuntimeError`` subclasses — code
    that does ``except RuntimeError`` continues to work.
    """
