"""Certainty-gated abstention — the frame family's first use as a CONTROL surface.

Every other consumer of `FrameLibrary` measures something. This one changes a
decision: it converts an `Answer` the model is not confident about into
`Abstain(reason="low_certainty")` so the caller escalates.

WHAT LICENSES THIS (`experiments/certainty_abstain.py`, SmolLM2-1.7B, n=200 over 5
BBH tasks):

  certainty post-answer   AUC 0.653   permutation p = 0.003
  margin (cheap baseline) AUC 0.544   permutation p = 0.312   -- not significant
  accuracy at 50% coverage: 0.570 vs 0.445 base, +0.120 over margin

and the confound that had to be ruled out first — pooling tasks with base rates
from .30 to .65 means a signal encoding WHICH TASK would score above chance without
being confidence at all. It survives: mean WITHIN-task AUC 0.616 against a pooled
0.653, 4/5 tasks above chance, and between-task variance is BELOW within-task
(ratio 0.71), so the coordinate is not a task detector.

Two design facts that matter:

  - Read AFTER the candidate answer is appended, not before. Pre-answer scored
    AUC 0.560 (p=0.133, not significant); post-answer 0.653. The useful question is
    "is the model unsure about THIS answer", not "is this a hard prompt".
  - The prior was failure. `MetacognitiveProbe`, trained DIRECTLY on the model's own
    success, got AUC ~0.50 at all 25 layers. The certainty frame works where that
    did not, and the difference is that it was never fit on success — it is an
    external semantic direction that transfers.

HONEST SCOPE: AUC 0.653 is a modest gate, not a dramatic one, measured on ONE model
over five tasks. It is opt-in for that reason. End-to-end BBH impact has NOT been
measured — abstaining costs coverage, and whether that trade pays depends on what
the caller does with an abstention.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CertaintyGate:
    """Turns a certainty-frame coordinate into an abstain decision.

    `threshold` is in the frame's own normalized units (`Frame.project_norm`), so it
    is comparable across frames but NOT across models — calibrate per backbone.
    """

    threshold: float
    layer: int
    direction: np.ndarray
    mean: np.ndarray
    scale: np.ndarray
    coord_scale: float = 1.0
    model_fingerprint: str = ""
    # provenance, so a shipped gate can be audited without re-deriving it
    auc: float | None = None
    n_calibration: int = 0

    @classmethod
    def from_frame(cls, frame, threshold: float, fingerprint: str = "",
                   auc: float | None = None, n: int = 0) -> "CertaintyGate":
        ax = frame.axis
        return cls(threshold=threshold, layer=ax.layer, direction=ax.direction,
                   mean=ax.mean, scale=ax.scale,
                   coord_scale=getattr(frame, "coord_scale", 1.0) or 1.0,
                   model_fingerprint=fingerprint, auc=auc, n_calibration=n)

    def coord(self, hidden: np.ndarray) -> float:
        """Certainty coordinate of a hidden state, in the frame's normalized units."""
        z = (np.asarray(hidden, float) - self.mean) / self.scale
        return float(z @ self.direction) / (self.coord_scale or 1.0)

    def should_abstain(self, hidden: np.ndarray) -> bool:
        return self.coord(hidden) < self.threshold

    # -- calibration --
    @staticmethod
    def pick_threshold(coords, correct=None, *, coverage: float = 0.8) -> float:
        """Threshold that keeps `coverage` of examples — a quantile, not a magic number.

        Coverage-based rather than accuracy-based on purpose: picking the threshold
        that maximises held-out accuracy would fit the calibration set, and the
        caller usually has a coverage budget anyway. `correct` is accepted only so
        callers can record the achieved AUC alongside.
        """
        c = np.asarray(coords, float)
        if not 0.0 < coverage <= 1.0:
            raise ValueError("coverage must be in (0, 1]")
        return float(np.quantile(c, 1.0 - coverage))

    def to_dict(self) -> dict:
        return {"threshold": self.threshold, "layer": int(self.layer),
                "direction": [round(float(x), 6) for x in self.direction],
                "mean": [round(float(x), 6) for x in self.mean],
                "scale": [round(float(x), 6) for x in self.scale],
                "coord_scale": self.coord_scale,
                "model_fingerprint": self.model_fingerprint,
                "auc": self.auc, "n_calibration": self.n_calibration}

    @classmethod
    def from_dict(cls, d: dict) -> "CertaintyGate":
        return cls(threshold=float(d["threshold"]), layer=int(d["layer"]),
                   direction=np.asarray(d["direction"], float),
                   mean=np.asarray(d["mean"], float),
                   scale=np.asarray(d["scale"], float),
                   coord_scale=float(d.get("coord_scale", 1.0)),
                   model_fingerprint=d.get("model_fingerprint", ""),
                   auc=d.get("auc"), n_calibration=int(d.get("n_calibration", 0)))


def accuracy_at_coverage(coords, correct, coverage: float) -> float:
    """Accuracy over the most-certain `coverage` fraction. For reporting a gate's
    trade-off honestly: a gate is only worth shipping if this beats the base rate by
    more than the coverage it costs is worth to the caller."""
    c = np.asarray(coords, float)
    y = np.asarray(correct, int)
    k = max(1, int(len(c) * coverage))
    keep = np.argsort(-c)[:k]
    return float(y[keep].mean())
