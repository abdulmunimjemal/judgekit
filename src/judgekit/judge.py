"""Judge protocol and the data shapes that flow through judgekit."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class JudgeOutput:
    """A single judge verdict on one input.

    `score` is a continuous raw judge output in [0.0, 1.0]. We deliberately
    don't model categorical labels here — if your judge is categorical, map
    to probability of the positive class before passing it in. Calibration
    only makes sense on a continuous score.
    """

    score: float
    raw: object | None = None  # opaque payload from the underlying judge


@runtime_checkable
class Judge(Protocol):
    """A judge is anything callable that returns a JudgeOutput for one input.

    judgekit is deliberately model-agnostic. A judge can be:
      - an LLM call wrapped in a function
      - a fine-tuned classifier
      - a heuristic rubric
      - a human (sync API only — async humans need a queue)

    What matters is that you can call it many times and get comparable scores.
    """

    def __call__(self, item: str) -> JudgeOutput: ...


@dataclass(frozen=True)
class LabeledExample:
    """A human-labeled calibration anchor.

    `label` is the ground-truth probability in [0.0, 1.0]. For binary tasks
    use 0.0 / 1.0. For graded tasks, use the rater-mean (or majority vote).
    """

    item: str
    label: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.label <= 1.0:
            raise ValueError(f"label must be in [0, 1], got {self.label}")


@dataclass
class CalibrationSet:
    """A small human-labeled set the judge is calibrated against.

    Typical size: 50 to 500 items. Smaller than your eval set, large enough to
    fit a Platt or isotonic calibrator with usable confidence intervals.
    """

    examples: list[LabeledExample] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.examples)

    def items(self) -> list[str]:
        return [e.item for e in self.examples]

    def labels(self) -> list[float]:
        return [e.label for e in self.examples]
