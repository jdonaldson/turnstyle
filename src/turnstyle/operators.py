"""Semantic operators on frame coordinates — detection and symbolic application.

The frame-operators experiments (experiments/frame_operators{,_negation,_replicate,
_hedge_detector}.py, 2026-07-07) established:

  * Negation is NOT linearly recoverable from frame geometry: at the adjective
    token it is a layer-gated overwrite (shared hedge vector + partial erasure,
    trough ~L14); frames decodable early (age@L2, size@L3) are negation-BLIND,
    and the model does not compose double negation ("not unkind" lands with
    "not kind"). So consumers must read the BARE word's coordinate and apply
    negation SYMBOLICALLY.
  * Detection must be a probe, not a neg-regex: negation has an open
    paraphrastic tail (hardly / far from / by no means — all score in the
    "not" band on a detector that never saw them) and "not only" is a regex
    false-positive by construction. The validated detector is projection onto
    the HEDGE VECTOR v_neg = mean(state(not w) − state(w)) at the trough
    layer: AUC 0.964 on held-out templates and surface forms, 6/6
    pre-registered predictions.

Two primitives:

  NegationDetector — the hedge-vector probe. `score(state)` in sd units of the
    bare distribution; `detect(state)` -> True / False / None with an ABSTAIN
    band (downtoners and "not only" sit between the clear bands; a binary
    threshold would lie). Fit with `fit_negation_detector`.

  OperatorTable — the measured per-(frame, operator) affine signatures
    c(op w) = slope * c(w) + intercept. `apply` composes operators in code —
    which, unlike the model, gets double negation right by construction.

Both are pure-numpy after fitting and serialize to plain dicts (the
ModelProfile linear-params discipline: no pickled callables).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import json

import numpy as np

# Predicative templates for fitting (a modifier slot directly before the
# adjective — the attributive frame templates like "It is a {w} object."
# cannot host "not"). Structural scaffolding, not content vocabulary.
# MIXED constructions per frame (statement / question / past): single-format
# fitting shifts the score bands on unseen constructions (question-form
# prompts scored ~+0.5-1.5 higher against a statement-only baseline — the
# option-boundary probe lesson again: mixed-format training is the unlock).
PREDICATIVE_TEMPLATES = {
    "opinion": ["It is {m}{w}.", "Is it {m}{w}?", "It was {m}{w}."],
    "size": ["It is {m}{w}.", "Is it {m}{w}?", "It was {m}{w}."],
    "warmth": ["They are {m}{w}.", "Is he {m}{w}?", "He was {m}{w}."],
    "certainty": ["It is {m}{w} that it happened.",
                  "Is it {m}{w} that it happened?",
                  "It was {m}{w} that they left."],
}
DEFAULT_FIT_FRAMES = list(PREDICATIVE_TEMPLATES)
DEFAULT_LAYER = 14          # negation-application trough (SmolLM2 sweep)

# Fallback bands (sd units of the bare distribution) — used only when a
# detector is constructed without calibration data. `fit_negation_detector`
# DERIVES both thresholds from its own two calibration distributions
# (bare quantile + class-mean midpoint): absolute scores shift by ~1 sd
# across sentence constructions, so hand-set constants from one
# construction misplace the band on another (observed: predicative-fit
# constants sat on top of the question-form negated cluster).
Z_NEGATED = 3.5
Z_CLEAR = 2.0


@dataclass
class NegationDetector:
    """Hedge-vector negation probe: score = (state - mu0) . v / s0."""

    layer: int
    direction: np.ndarray          # unit hedge vector (H,)
    mu0: np.ndarray                # mean bare state (H,)
    s0: float                      # sd of bare projections
    z_negated: float = Z_NEGATED
    z_clear: float = Z_CLEAR
    meta: dict = field(default_factory=dict)

    def score(self, state: np.ndarray) -> float:
        """Hedge projection of a (H,) state at `self.layer`, in bare-sd units."""
        return float((np.asarray(state, float) - self.mu0)
                     @ self.direction / self.s0)

    def detect(self, state: np.ndarray) -> bool | None:
        """True = negated, False = clear, None = abstain (mid band)."""
        z = self.score(state)
        if z >= self.z_negated:
            return True
        if z < self.z_clear:
            return False
        return None

    def score_word(self, model, tokenizer, device, sent: str, word: str) -> float:
        """Convenience: forward `sent`, score at the last subword of `word`."""
        return self.score(_word_state(model, tokenizer, device,
                                      sent, word, self.layer))

    def detect_word(self, model, tokenizer, device, sent: str,
                    word: str) -> bool | None:
        z = self.score_word(model, tokenizer, device, sent, word)
        if z >= self.z_negated:
            return True
        if z < self.z_clear:
            return False
        return None

    def to_dict(self) -> dict:
        return {"layer": self.layer,
                "direction": [round(float(x), 6) for x in self.direction],
                "mu0": [round(float(x), 6) for x in self.mu0],
                "s0": round(float(self.s0), 6),
                "z_negated": self.z_negated, "z_clear": self.z_clear,
                "meta": self.meta}

    @classmethod
    def from_dict(cls, d: dict) -> "NegationDetector":
        return cls(layer=int(d["layer"]),
                   direction=np.asarray(d["direction"], float),
                   mu0=np.asarray(d["mu0"], float), s0=float(d["s0"]),
                   z_negated=float(d.get("z_negated", Z_NEGATED)),
                   z_clear=float(d.get("z_clear", Z_CLEAR)),
                   meta=d.get("meta", {}))


@dataclass
class OperatorTable:
    """Measured affine operator signatures per frame: c(op w) = b*c(w) + a.

    `table[frame][op] = {"slope": b, "intercept": a, "r2": ...}`. Application
    is symbolic — composition (`ops` outermost-last, surface order) is exact
    in code even where the model itself collapses (double negation)."""

    table: dict = field(default_factory=dict)

    def apply(self, frame: str, op: str, coord: float) -> float:
        e = self.table[frame][op]
        return e["slope"] * coord + e["intercept"]

    def apply_chain(self, frame: str, ops: list, coord: float) -> float:
        """Apply modifiers in surface order ("not very X" -> [very, not]
        innermost-first)."""
        c = coord
        for op in ops:
            c = self.apply(frame, op, c)
        return c

    def to_dict(self) -> dict:
        return {"table": self.table}

    @classmethod
    def from_dict(cls, d: dict) -> "OperatorTable":
        return cls(table=d["table"])

    @classmethod
    def from_results_json(cls, path) -> "OperatorTable":
        """Load from experiments/results/frame_operators.json (the measured
        sweep)."""
        d = json.loads(Path(path).read_text())
        table = {f: dict(row["ops"]) for f, row in d["frames"].items()}
        return cls(table=table)


# ── model-touching: fitting ────────────────────────────────────────────────────

def _word_state(model, tokenizer, device, sent: str, word: str,
                layer: int) -> np.ndarray:
    """(H,) hidden state at the last subword of `word` in `sent` at `layer`."""
    import torch

    cs = sent.rfind(word)
    ce = cs + len(word)
    enc = tokenizer(sent, return_offsets_mapping=True, return_tensors="pt")
    offs = enc.pop("offset_mapping")[0].tolist()
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        hs = model(**enc, output_hidden_states=True).hidden_states
    idxs = [k for k, (s, e) in enumerate(offs) if e > cs and s < ce] or [-1]
    return hs[layer][0, idxs[-1], :].float().cpu().numpy()


def fit_negation_detector(model, tokenizer, device, library, *,
                          frames: list | None = None,
                          layer: int = DEFAULT_LAYER) -> NegationDetector:
    """Calibrate the hedge vector from bare/"not" pairs over frame lexicons.

    One forward per (template, word, condition) — ~380 forwards for the
    default four frames x three constructions. The library supplies the
    lexicons; PREDICATIVE_TEMPLATES the sentence scaffolding."""
    frames = frames or [f for f in DEFAULT_FIT_FRAMES if f in library.frames]
    deltas, bares = [], []
    for fname in frames:
        for tmpl in PREDICATIVE_TEMPLATES[fname]:
            for w in library.frames[fname].data:
                sb = _word_state(model, tokenizer, device,
                                 tmpl.format(m="", w=w), w, layer)
                sn = _word_state(model, tokenizer, device,
                                 tmpl.format(m="not ", w=w), w, layer)
                deltas.append(sn - sb)
                bares.append(sb)
    v = np.mean(deltas, 0)
    v /= np.linalg.norm(v) + 1e-12
    bares = np.array(bares)
    mu0 = bares.mean(0)
    bare_z = (bares - mu0) @ v
    s0 = float(bare_z.std()) + 1e-9
    bare_z = bare_z / s0
    neg_z = bare_z + np.asarray(deltas) @ v / s0    # negated = bare + delta
    # Thresholds from the calibration distributions — PRECISION-FIRST. The
    # mid band between the lines is abstain territory by design: downtoners,
    # "not only" (an also-like focus particle: it drops to control level
    # under a contrastive direction), and cross-construction negations all
    # live there, and a false verdict in either direction flips answers
    # while an abstain only costs a fallback. A class-mean midpoint was
    # tried and false-fired on "not only" 40/63 — do not lower these bands
    # without re-running the frame_hedge_detector battery.
    z_clear = float(np.quantile(bare_z, 0.975))
    z_negated = max(float(np.quantile(neg_z, 0.25)), z_clear + 2.0)
    return NegationDetector(
        layer=layer, direction=v, mu0=mu0, s0=s0,
        z_negated=z_negated, z_clear=z_clear,
        meta={"fit_frames": frames, "n_pairs": len(deltas),
              "bare_z_mean": round(float(bare_z.mean()), 3),
              "neg_z_mean": round(float(neg_z.mean()), 3)})


# ── fingerprint-addressed store (user cache, alongside the frame libraries) ───

def _store_path(model_or_fp) -> Path:
    from turnstyle.frame_library import _fingerprint, _USER_FRAMES
    return _USER_FRAMES / f"{_fingerprint(model_or_fp)}.negation.json"


def save_detector(det: NegationDetector, model_or_fp) -> Path:
    p = _store_path(model_or_fp)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(det.to_dict()))
    return p


def load_detector(model_or_fp) -> NegationDetector | None:
    p = _store_path(model_or_fp)
    if not p.exists():
        return None
    return NegationDetector.from_dict(json.loads(p.read_text()))


__all__ = ["NegationDetector", "OperatorTable", "fit_negation_detector",
           "save_detector", "load_detector",
           "PREDICATIVE_TEMPLATES", "DEFAULT_LAYER", "Z_NEGATED", "Z_CLEAR"]
