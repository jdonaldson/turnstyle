"""Adjective-polarity — a tested, model-level primitive.

Polarity is whether an adjective names the HIGH or LOW end of its scalar axis
(tallest/oldest/widest/cheapest...). If a backbone encodes it as a linear
direction, that direction is **semantic, not lexical**: it transfers across
vocabulary and across languages (validated on SmolLM2 — an English-trained probe
poles Spanish/French/German adjectives at ~98%, and generalizes to held-out
*axes* at ~92%). So any ordering/comparison task can lean on it instead of a
hardcoded English adjective list — *where the model has it*.

It is NOT safe to assume. The capability is per-model and per-axis (SmolLM2 has
it cleanly for size/price/speed/... but its age axis collapses — "old" reads as
high-magnitude). So this module *tests* for the capability and reports where it
ships, rather than trusting it blindly.

    probe = detect_polarity(model, tok, device)     # calibrate once per backbone
    probe.capability.ship                           # bool — does this model have it?
    probe.capability.per_axis                       # {axis: leave-one-axis-out acc}
    probe.pole(hidden_vec)                           # +1 HIGH / -1 LOW

The fitted probe serializes to the same linear-params form ModelProfile already
uses (no pickled sklearn) and is stored as a model-level profile slot.

`pole()`/`predict()` are pure-numpy (standardize → sign(w·z + b)); only
`detect_polarity` needs a model + sklearn.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

HIGH, LOW = +1, -1   # the two poles of a scalar axis

# ── the labeled lexicon (training seed; used only to fit the direction) ───────
# axis -> list of (positive, comparative, superlative, pole)  pole +1 = HIGH end
POLARITY_LEXICON: dict[str, list[tuple[str, str, str, int]]] = {
    "size": [("big", "bigger", "biggest", +1), ("small", "smaller", "smallest", -1),
             ("large", "larger", "largest", +1), ("tall", "taller", "tallest", +1),
             ("short", "shorter", "shortest", -1), ("long", "longer", "longest", +1),
             ("wide", "wider", "widest", +1), ("narrow", "narrower", "narrowest", -1),
             ("deep", "deeper", "deepest", +1), ("shallow", "shallower", "shallowest", -1)],
    "weight": [("heavy", "heavier", "heaviest", +1), ("light", "lighter", "lightest", -1)],
    "speed": [("fast", "faster", "fastest", +1), ("slow", "slower", "slowest", -1)],
    "temp": [("hot", "hotter", "hottest", +1), ("cold", "colder", "coldest", -1),
             ("warm", "warmer", "warmest", +1)],
    "value": [("expensive", "more expensive", "most expensive", +1),
              ("cheap", "cheaper", "cheapest", -1), ("rich", "richer", "richest", +1),
              ("poor", "poorer", "poorest", -1)],
    "quality": [("good", "better", "best", +1), ("bad", "worse", "worst", -1)],
    "intensity": [("strong", "stronger", "strongest", +1), ("weak", "weaker", "weakest", -1),
                  ("bright", "brighter", "brightest", +1), ("dark", "darker", "darkest", -1),
                  ("loud", "louder", "loudest", +1), ("quiet", "quieter", "quietest", -1)],
    "age": [("new", "newer", "newest", +1), ("old", "older", "oldest", -1),
            ("young", "younger", "youngest", +1)],
    "mood": [("happy", "happier", "happiest", +1), ("sad", "sadder", "saddest", -1)],
}

# templates spanning positive / comparative / superlative morphology, so the
# probe sees the forms it must classify at runtime (newer, oldest, most expensive)
_TEMPLATES = [
    ("pos", "In comparison, this {n} is very {form}."),
    ("cmp", "The first {n} is {form} than the second {n}."),
    ("sup", "Of the group, this {n} is the {form}."),
]
_NOUNS = ["object", "item", "one"]

SHIP_GATE = 0.75            # leave-one-axis-out threshold to call the primitive present


# ── capability report ────────────────────────────────────────────────────────

@dataclass
class PolarityCapability:
    layer: int
    loo_axis: float                       # mean leave-one-axis-out accuracy
    per_axis: dict                        # axis -> held-out accuracy
    ship: bool

    def axes_shipping(self, gate: float = SHIP_GATE) -> list[str]:
        return [a for a, v in self.per_axis.items() if v >= gate]


# ── the probe (pure-numpy inference) ─────────────────────────────────────────

@dataclass
class PolarityProbe:
    layer: int
    mean: np.ndarray                      # (H,)
    scale: np.ndarray                     # (H,)
    coef: np.ndarray                      # (H,)  weight for the HIGH class
    intercept: float
    capability: Optional[PolarityCapability] = None
    lexicon_axes: list = field(default_factory=list)

    def score(self, vec: np.ndarray) -> float:
        z = (np.asarray(vec, dtype=float) - self.mean) / self.scale
        return float(z @ self.coef + self.intercept)

    def pole(self, vec: np.ndarray) -> int:
        """+1 HIGH end / -1 LOW end for one adjective-token hidden state."""
        return HIGH if self.score(vec) > 0 else LOW

    def confidence(self, vec: np.ndarray) -> float:
        return 1.0 / (1.0 + np.exp(-abs(self.score(vec))))

    # -- serialization (ModelProfile linear-params form, no pickled sklearn) --
    def to_dict(self) -> dict:
        cap = self.capability
        return {
            "kind": "polarity",
            "layer": int(self.layer),
            "scaler": {"mean": self.mean.tolist(), "scale": self.scale.tolist()},
            "classifier": {"coef": self.coef.tolist(),
                           "intercept": float(self.intercept)},
            "lexicon_axes": list(self.lexicon_axes),
            "capability": None if cap is None else {
                "layer": cap.layer, "loo_axis": cap.loo_axis,
                "per_axis": cap.per_axis, "ship": cap.ship,
            },
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PolarityProbe":
        cap = d.get("capability")
        return cls(
            layer=d["layer"],
            mean=np.asarray(d["scaler"]["mean"], dtype=float),
            scale=np.asarray(d["scaler"]["scale"], dtype=float),
            coef=np.asarray(d["classifier"]["coef"], dtype=float),
            intercept=float(d["classifier"]["intercept"]),
            lexicon_axes=d.get("lexicon_axes", []),
            capability=None if cap is None else PolarityCapability(
                layer=cap["layer"], loo_axis=cap["loo_axis"],
                per_axis=cap["per_axis"], ship=cap["ship"]),
        )


# ── calibration (needs a model) ──────────────────────────────────────────────

def _collect(model, tok, device):
    """One forward pass per lexicon sentence; return (acts[N,L+1,H], y, axes)."""
    import torch

    acts, poles, axes = [], [], []
    for axis, words in POLARITY_LEXICON.items():
        for pos, cmp_, sup, pole in words:
            forms = {"pos": pos, "cmp": cmp_, "sup": sup}
            for kind, tmpl in _TEMPLATES:
                form = forms[kind]
                for n in _NOUNS:
                    sent = tmpl.format(n=n, form=form)
                    content = form.split()[-1]
                    cs = sent.rfind(content)
                    ce = cs + len(content)
                    enc = tok(sent, return_offsets_mapping=True, return_tensors="pt")
                    offs = enc.pop("offset_mapping")[0].tolist()
                    enc = {k: v.to(device) for k, v in enc.items()}
                    with torch.no_grad():
                        out = model(**enc, output_hidden_states=True)
                    hs = torch.stack(out.hidden_states, 0)[:, 0].float().cpu().numpy()
                    tk = next((k for k, (s, e) in enumerate(offs)
                               if e > cs and s < ce), None)
                    if tk is None:
                        continue
                    acts.append(hs[:, tk, :])
                    poles.append(1 if pole > 0 else 0)
                    axes.append(axis)
    return np.stack(acts), np.array(poles), np.array(axes)


def _fit_linear(X, y, C=0.5):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(X)
    clf = LogisticRegression(max_iter=2000, C=C).fit(sc.transform(X), y)
    return sc, clf


def detect_polarity(model, tok, device, layer: Optional[int] = None,
                    gate: float = SHIP_GATE) -> PolarityProbe:
    """Calibrate the polarity primitive on `model` and report where it ships.

    Collects lexicon activations, picks the layer with the best leave-one-AXIS-out
    accuracy (the honest 'does this model have the primitive' metric), and returns
    a fitted `PolarityProbe` carrying a per-axis capability map.
    """
    A, y, axes = _collect(model, tok, device)
    uniq_axes = list(POLARITY_LEXICON)
    nL = A.shape[1]
    layers = range(nL) if layer is None else [layer]

    best = None
    for L in layers:
        X = A[:, L, :]
        per = {}
        for ax in uniq_axes:
            te = axes == ax
            if len(set(y[~te])) < 2 or not te.any():
                continue
            sc, clf = _fit_linear(X[~te], y[~te])
            per[ax] = float(clf.score(sc.transform(X[te]), y[te]))
        loo = float(np.mean(list(per.values()))) if per else 0.0
        if best is None or loo > best[0]:
            best = (loo, L, per)

    assert best is not None, "no layers to calibrate over"
    loo, L, per = best
    cap = PolarityCapability(layer=L, loo_axis=loo, per_axis=per,
                             ship=loo >= gate)
    sc, clf = _fit_linear(A[:, L, :], y)
    return PolarityProbe(
        layer=L,
        mean=sc.mean_.astype(float),
        scale=sc.scale_.astype(float),
        coef=clf.coef_[0].astype(float),
        intercept=float(clf.intercept_[0]),
        capability=cap,
        lexicon_axes=uniq_axes,
    )
