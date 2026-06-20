"""SemanticFrame — supervised signed semantic axes, the turnstyle→dyf seam.

`experiments/concept_geometry.py` showed that a model's RAW activation geometry is
surface/language-dominated: dyf (or any unsupervised method) partitions the dominant
variance, which is *which language*, not *which concept*. The semantic structure is
real but low-variance — it only surfaces under supervision (the polarity probe found
it because it was *given* the labels).

A `BipolarAxis` is that supervision as a reusable object: a learned **signed**
direction with two named poles (hot↔cold, expensive↔cheap), fit contrastively from
labeled pole words. `project(vec)` returns a signed scalar — positive toward the high
pole. A `SemanticFrame` stacks axes and maps an activation to a **semantic-coordinate
vector** (one signed scalar per axis). That projected space is what you hand to dyf:
instead of structuring raw surface-dominated activations, dyf structures meaning.

  frame = fit_semantic_frame(model, tok, dev, {
      "temperature": (["hot","warm"], ["cold","cool"]),
      "value":       (["expensive","rich"], ["cheap","poor"]),
  })
  coords = frame.project_words(model, tok, dev, ["caliente","frío","teuer"])  # (3, 2)
  # → dyf.build_dyf_tree(coords)  : a tree over MEANING, not surface

`project()`/`coordinates()` are pure-numpy; only fitting needs a model.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# neutral scalar carrier; pole is a lexical property so context is unneeded
_TEMPLATE = "In comparison, this object is very {w}."


# ── one signed axis ──────────────────────────────────────────────────────────

@dataclass
class BipolarAxis:
    name: str
    low_label: str
    high_label: str
    layer: int
    mean: np.ndarray            # (H,) feature standardization
    scale: np.ndarray           # (H,)
    direction: np.ndarray       # (H,) unit vector, points toward the high pole
    center: float               # projection of the high/low midpoint (the boundary)

    def project(self, vec: np.ndarray) -> float:
        """Signed position on the axis: > 0 toward high pole, < 0 toward low."""
        z = (np.asarray(vec, dtype=float) - self.mean) / self.scale
        return float(z @ self.direction - self.center)

    def pole(self, vec: np.ndarray) -> str:
        return self.high_label if self.project(vec) > 0 else self.low_label

    def to_dict(self) -> dict:
        return {"name": self.name, "low_label": self.low_label,
                "high_label": self.high_label, "layer": int(self.layer),
                "mean": self.mean.tolist(), "scale": self.scale.tolist(),
                "direction": self.direction.tolist(), "center": float(self.center)}

    @classmethod
    def from_dict(cls, d: dict) -> "BipolarAxis":
        return cls(name=d["name"], low_label=d["low_label"], high_label=d["high_label"],
                   layer=d["layer"], mean=np.asarray(d["mean"], float),
                   scale=np.asarray(d["scale"], float),
                   direction=np.asarray(d["direction"], float), center=float(d["center"]))


def fit_axis_from_vectors(name, low_label, high_label, layer,
                          high_vecs: np.ndarray, low_vecs: np.ndarray) -> BipolarAxis:
    """Contrastive fit (numpy only): standardize, take the mean high−low direction,
    center on the midpoint. No sklearn — the direction IS the difference of class
    means, which is the optimal contrastive axis under isotropic noise."""
    allv = np.vstack([high_vecs, low_vecs]).astype(float)
    mean = allv.mean(0)
    scale = allv.std(0) + 1e-6
    zh = (high_vecs - mean) / scale
    zl = (low_vecs - mean) / scale
    d = zh.mean(0) - zl.mean(0)
    d = d / (np.linalg.norm(d) + 1e-12)
    center = 0.5 * (zh.mean(0) + zl.mean(0)) @ d
    return BipolarAxis(name, low_label, high_label, layer, mean, scale, d, float(center))


# ── a stack of axes = a semantic coordinate system ───────────────────────────

@dataclass
class SemanticFrame:
    axes: list                       # list[BipolarAxis]
    layer: int

    @property
    def names(self) -> list:
        return [a.name for a in self.axes]

    def coordinates(self, vec: np.ndarray) -> np.ndarray:
        """Map one activation to its signed semantic coordinate (one per axis)."""
        return np.array([a.project(vec) for a in self.axes])

    def project_matrix(self, vecs: np.ndarray) -> np.ndarray:
        """(N, H) activations → (N, n_axes) semantic coordinates (for dyf)."""
        return np.vstack([self.coordinates(v) for v in vecs])

    def project_words(self, model, tokenizer, device, words,
                      template: str = _TEMPLATE) -> np.ndarray:
        """Read each word's activation and project to semantic coordinates."""
        vecs = _word_vectors(model, tokenizer, device, words, self.layer, template)
        return self.project_matrix(vecs)

    def to_dict(self) -> dict:
        return {"layer": int(self.layer), "axes": [a.to_dict() for a in self.axes]}

    @classmethod
    def from_dict(cls, d: dict) -> "SemanticFrame":
        return cls(axes=[BipolarAxis.from_dict(a) for a in d["axes"]], layer=d["layer"])


# ── fitting (needs a model) ──────────────────────────────────────────────────

def _word_vectors(model, tokenizer, device, words, layer, template) -> np.ndarray:
    """Last-token hidden state of each word at `layer`, in a neutral template."""
    import torch

    out = []
    for w in words:
        sent = template.format(w=w)
        content = w.split()[-1]
        cs = sent.rfind(content)
        ce = cs + len(content)
        enc = tokenizer(sent, return_offsets_mapping=True, return_tensors="pt")
        offs = enc.pop("offset_mapping")[0].tolist()
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            res = model(**enc, output_hidden_states=True)
        h = res.hidden_states[layer][0]
        tk = next((k for k, (s, e) in enumerate(offs) if e > cs and s < ce), None)
        out.append(h[tk].float().cpu().numpy())
    return np.vstack(out)


def fit_semantic_frame(model, tokenizer, device, axis_specs: dict, layer: int = 14,
                       template: str = _TEMPLATE, surface_suppress: int = 3,
                       orthogonalize: bool = False) -> SemanticFrame:
    """Fit one BipolarAxis per spec. `axis_specs` maps name → (high_words, low_words).

    All axes share ONE standardization (a common feature space) so their directions
    are comparable. `project()` is unchanged regardless of the options below — they
    only reshape the fitted directions.

    `surface_suppress=K` (default 3) projects each axis direction **orthogonal to the
    top-K principal components** of the training activations. Surface/language is the
    dominant variance ([[concept_geometry_dyf_measurement]]); removing it sharpens
    BOTH cross-lingual transfer AND meaning-clustering. Validated on SmolLM2 @L11
    (scalar template): raw axes already cluster by meaning (axis-NMI 0.44 ≫ lang 0.02,
    transfer 0.96); K=3 lifts both (transfer 0.98, axis-NMI 0.48). K too large
    (≥30) deletes the signal. The earlier "orthogonal-vs-transfer tension" was a
    neutral-template artifact — it dissolves in the in-context scalar template.

    `orthogonalize=True` additionally QR-decorrelates the axes from each other (no
    measured benefit in the scalar template; kept for the dyf-clustering use)."""
    names = list(axis_specs)
    hv = {n: _word_vectors(model, tokenizer, device, axis_specs[n][0], layer, template)
          for n in names}
    lv = {n: _word_vectors(model, tokenizer, device, axis_specs[n][1], layer, template)
          for n in names}

    allv = np.vstack([v for n in names for v in (hv[n], lv[n])]).astype(float)
    mean = allv.mean(0)
    scale = allv.std(0) + 1e-6

    def z(m):
        return (m - mean) / scale

    raw_dirs = np.vstack([z(hv[n]).mean(0) - z(lv[n]).mean(0) for n in names])

    if surface_suppress > 0:
        zc = z(allv)
        _, _, Vt = np.linalg.svd(zc - zc.mean(0), full_matrices=False)
        surf = Vt[:surface_suppress]                # top-K surface directions
        raw_dirs = raw_dirs - (raw_dirs @ surf.T) @ surf

    if orthogonalize:
        Q, _ = np.linalg.qr(raw_dirs.T)             # (H, k) orthonormal
        dirs = Q.T[:len(names)]
        for i, n in enumerate(names):               # point toward each high pole
            if (z(hv[n]).mean(0) - z(lv[n]).mean(0)) @ dirs[i] < 0:
                dirs[i] = -dirs[i]
    else:
        dirs = raw_dirs / (np.linalg.norm(raw_dirs, axis=1, keepdims=True) + 1e-12)

    axes = []
    for i, n in enumerate(names):
        d = dirs[i]
        center = 0.5 * (z(hv[n]).mean(0) + z(lv[n]).mean(0)) @ d
        axes.append(BipolarAxis(n, f"-{n}", f"+{n}", layer, mean, scale, d, float(center)))
    return SemanticFrame(axes=axes, layer=layer)


def axis_from_polarity_probe(probe, name: str = "polarity",
                             low_label: str = "low", high_label: str = "high") -> BipolarAxis:
    """Bridge: turn a fitted PolarityProbe (one global HIGH/LOW direction) into a
    BipolarAxis, so the existing calibration drops straight into a SemanticFrame."""
    d = np.asarray(probe.coef, float)
    d = d / (np.linalg.norm(d) + 1e-12)
    # PolarityProbe scores z@coef + intercept; express the boundary as a center on d
    center = -float(probe.intercept) / (np.linalg.norm(probe.coef) + 1e-12)
    return BipolarAxis(name, low_label, high_label, probe.layer,
                       np.asarray(probe.mean, float), np.asarray(probe.scale, float),
                       d, center)
