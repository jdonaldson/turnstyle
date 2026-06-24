"""FrameLibrary — a registry of named semantic frames (conceptual axes) for a model.

The experiments in `experiments/{frame_family_matrix,time_frame,size_frame,
ordering_frames}.py` showed SmolLM2 carries a family of recoverable, mutually-orthogonal
low-D conceptual frames — essentially the rungs of the English adjective-ordering
hierarchy (opinion/size/age/shape/color/origin/material) plus number and time. This turns
that finding into a reusable measurement primitive: fit a scalar `BipolarAxis` per frame
from labeled (word → value) data, then `project` any word onto every frame at once.

  lib = FrameLibrary().fit_canonical(model, tok, dev)   # opinion/size/age/.../number/time
  lib.project_word("enormous", model, tok, dev)         # {'size': 2.6, 'opinion': 0.1, ...}
  lib.orthogonality(model, tok, dev, layer=8)           # cross-frame |cos| matrix
  lib.save("frames.json"); FrameLibrary.load("frames.json")

Design notes:
- Built on `semantic_frame.BipolarAxis` (serializable, pure-numpy `project`).
- LAST-subword token readout by default — the cross-lingual audit
  ([[semantic_frame_family]]) showed first-token understates multi-token words.
- Fitting is pure numpy (dual/kernel ridge — n words << hidden dim, so the n×n solve is
  cheap); no sklearn dependency. Only collection needs the model.
- Each frame keeps its own best layer (max held-out recoverability); `project_word` runs
  one forward per distinct template and reads each frame's layer.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from turnstyle.semantic_frame import BipolarAxis

_DEFAULT_TEMPLATE = "It is a {w} object."

# fingerprint-addressed library store (mirrors turnstyle.profile's two-tier loader):
# a user cache overlays a bundled base, per-frame (user-fit frames win, bundled fills gaps).
_USER_FRAMES = Path(
    os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache"))
) / "turnstyle" / "frames"
_BUNDLED_FRAMES = Path(__file__).parent / "data" / "frames"


# ── numpy ridge (dual form: n words << H, so an n×n solve is cheap) ────────────

def _ridge_dir(Z: np.ndarray, y: np.ndarray, alpha: float = 10.0) -> np.ndarray:
    """Unit ridge-regression direction for y on standardized features Z (N,H)."""
    yc = y - y.mean()
    K = Z @ Z.T
    a = np.linalg.solve(K + alpha * np.eye(len(y)), yc)
    w = Z.T @ a
    n = np.linalg.norm(w)
    return w / (n + 1e-12)


def _cv_r(X: np.ndarray, y: np.ndarray, k: int = 5, alpha: float = 10.0,
          seed: int = 0) -> float:
    """Held-out k-fold Pearson r (shuffled folds, per-fold standardization, dual ridge)."""
    n = len(y)
    if n < k + 1:
        return float("nan")
    idx = np.random.default_rng(seed).permutation(n)
    folds = np.array_split(idx, k)
    pred = np.zeros(n)
    for i in range(k):
        te = folds[i]
        tr = np.concatenate([folds[j] for j in range(k) if j != i])
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
        Z, Zt = (X[tr] - mu) / sd, (X[te] - mu) / sd
        yb = y[tr].mean()
        a = np.linalg.solve(Z @ Z.T + alpha * np.eye(len(tr)), y[tr] - yb)
        pred[te] = Zt @ (Z.T @ a) + yb
    if np.std(pred) < 1e-9:
        return 0.0
    return float(np.corrcoef(pred, y)[0, 1])


def _superlative_roots(word: str) -> set:
    """Candidate base forms of a (possibly superlative) adjective — structural
    morphology only (closed-class -est/-iest, consonant-doubling), no vocabulary.
    e.g. biggest→{big}, heaviest→{heavy}, smallest→{small}, oldest→{old}."""
    w = word.lower().strip()
    out = {w}
    if w.endswith("iest"):
        out.add(w[:-4] + "y")
    if w.endswith("est"):
        out.add(w[:-3])
        if len(w) > 4 and w[-4] == w[-5]:          # biggest → big (undo doubling)
            out.add(w[:-4])
    return out


def _axis_from_scalar(name, X, y, layer, low_label, high_label) -> BipolarAxis:
    """Fit a BipolarAxis from graded (word→value) data at one layer (X = (N,H) acts)."""
    mean, scale = X.mean(0), X.std(0) + 1e-6
    Z = (X - mean) / scale
    d = _ridge_dir(Z, y)
    if (Z @ d) @ (y - y.mean()) < 0:          # point toward high value
        d = -d
    center = float((Z @ d).mean())            # 0 at the average word
    return BipolarAxis(name, low_label, high_label, layer, mean, scale, d, center)


# ── one named frame ───────────────────────────────────────────────────────────

@dataclass
class Frame:
    axis: BipolarAxis
    template: str = _DEFAULT_TEMPLATE
    pool: str = "last"
    cv_r: float | None = None
    data: dict | None = None        # training word→value, kept for refit/orthogonality
    coord_scale: float = 1.0        # std of training-word projections (cross-frame compare)

    @property
    def name(self) -> str:
        return self.axis.name

    @property
    def layer(self) -> int:
        return self.axis.layer

    def project(self, vec) -> float:
        return self.axis.project(vec)

    def project_norm(self, vec) -> float:
        """Projection in units of the frame's own spread — comparable across frames."""
        return self.axis.project(vec) / (self.coord_scale or 1.0)

    def to_dict(self) -> dict:
        ax = self.axis.to_dict()
        for k in ("mean", "scale", "direction"):           # trim float bloat (6 dp is
            ax[k] = [round(float(x), 6) for x in ax[k]]     # ample for unit dirs / z-stats)
        ax["center"] = round(float(ax["center"]), 6)
        return {"axis": ax, "template": self.template, "pool": self.pool,
                "cv_r": self.cv_r, "data": self.data, "coord_scale": self.coord_scale}

    @classmethod
    def from_dict(cls, d: dict) -> "Frame":
        return cls(axis=BipolarAxis.from_dict(d["axis"]), template=d["template"],
                   pool=d.get("pool", "last"), cv_r=d.get("cv_r"), data=d.get("data"),
                   coord_scale=d.get("coord_scale", 1.0))


# ── model-touching collection (last-subword across all layers) ────────────────

def _collect(model, tokenizer, device, words, template, pool="last") -> dict:
    """{word: (n_layers+1, H)} — pool the word's subword span (last/first/mean)."""
    import torch

    out = {}
    for w in words:
        sent = template.format(w=w)
        cs = sent.rfind(w); ce = cs + len(w)
        enc = tokenizer(sent, return_offsets_mapping=True, return_tensors="pt")
        offs = enc.pop("offset_mapping")[0].tolist()
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            hs = model(**enc, output_hidden_states=True).hidden_states
        stk = torch.stack(hs, 0)[:, 0, :, :]                # (L+1, T, H)
        idxs = [k for k, (s, e) in enumerate(offs) if e > cs and s < ce] or [-1]
        sel = {"last": stk[:, idxs[-1], :], "first": stk[:, idxs[0], :],
               "mean": stk[:, idxs, :].mean(1)}[pool]
        out[w] = sel.float().cpu().numpy()
    return out


# ── the library ───────────────────────────────────────────────────────────────

@dataclass
class FrameLibrary:
    frames: dict = field(default_factory=dict)        # name -> Frame
    fingerprint: str = ""                             # model fingerprint (set on save)
    model_id: str = ""                                # human identity for inspection

    # -- introspection --
    @property
    def names(self) -> list:
        return list(self.frames)

    def __len__(self):
        return len(self.frames)

    def __contains__(self, name):
        return name in self.frames

    def add(self, frame: Frame) -> "FrameLibrary":
        self.frames[frame.name] = frame
        return self

    def recoverability(self) -> dict:
        return {n: f.cv_r for n, f in self.frames.items()}

    # -- fitting --
    def fit_scalar(self, name, model, tokenizer, device, data: dict, *,
                   layer: int | None = None, template: str = _DEFAULT_TEMPLATE,
                   pool: str = "last", low_label: str | None = None,
                   high_label: str | None = None) -> Frame:
        """Fit a scalar frame from {word: value}. If layer is None, sweep all layers and
        keep the one with the best held-out CV r. Returns the Frame (also registered)."""
        words = list(data)
        y = np.asarray([data[w] for w in words], float)
        acts = _collect(model, tokenizer, device, words, template, pool)
        n_layers = acts[words[0]].shape[0]
        low_label = low_label or min(data, key=data.get)
        high_label = high_label or max(data, key=data.get)

        layers = [layer] if layer is not None else range(n_layers)
        best = None
        for L in layers:
            X = np.array([acts[w][L] for w in words])
            r = _cv_r(X, y)
            if best is None or (r == r and r > best[0]):   # r==r skips NaN
                best = (r, L, X)
        r, L, X = best
        axis = _axis_from_scalar(name, X, y, L, low_label, high_label)
        projs = np.array([axis.project(X[i]) for i in range(len(X))])
        coord_scale = float(projs.std()) or 1.0
        frame = Frame(axis=axis, template=template, pool=pool, cv_r=r, data=data,
                      coord_scale=coord_scale)
        self.add(frame)
        return frame

    def fit_canonical(self, model, tokenizer, device, which=None) -> "FrameLibrary":
        """Fit the standard frame family (adjective-ordering rungs + number + time)."""
        for name, spec in CANONICAL_FRAMES.items():
            if which is not None and name not in which:
                continue
            self.fit_scalar(name, model, tokenizer, device, spec["data"],
                            template=spec.get("template", _DEFAULT_TEMPLATE))
        return self

    # -- projection --
    def project_word(self, word, model, tokenizer, device) -> dict:
        """Coordinate of `word` on every frame. One forward per distinct template."""
        coords = {}
        by_tmpl: dict[tuple, list] = {}
        for n, f in self.frames.items():
            by_tmpl.setdefault((f.template, f.pool), []).append(n)
        for (tmpl, pool), names in by_tmpl.items():
            acts = _collect(model, tokenizer, device, [word], tmpl, pool)[word]
            for n in names:
                f = self.frames[n]
                coords[n] = f.project(acts[f.layer])
        return coords

    def coordinates(self, word, model, tokenizer, device) -> np.ndarray:
        c = self.project_word(word, model, tokenizer, device)
        return np.array([c[n] for n in self.names])

    # -- frame-as-column: rank candidates / route an attribute to its frame --
    def rank(self, candidates, frame_name, model, tokenizer, device, *,
             descending: bool = True):
        """Project candidates onto one frame → [(candidate, coord)] sorted. The frame
        SYNTHESIZES a numeric column the data lacks (e.g. order colors by warmth)."""
        f = self.frames[frame_name]
        acts = _collect(model, tokenizer, device, list(candidates), f.template, f.pool)
        scored = [(c, f.project(acts[c][f.layer])) for c in candidates]
        return sorted(scored, key=lambda kv: kv[1], reverse=descending)

    def route(self, word, model=None, tokenizer=None, device=None, *,
              margin: float = 1.5):
        """Which frame does an attribute word belong to, and at which pole?
        Returns (frame_name, sign) — sign=+1 if the word names the HIGH pole.

        Primary: MEMBERSHIP — if the word's (superlative-stripped) root is one of a
        frame's defining words, route there (reliable; a frame IS its word-set).
        Fallback (best-effort, only if a model is given): projection onto every frame,
        most-extreme in own-spread units, must clear the runner-up by `margin`. Pure
        projection is noisy (e.g. 'big'→shape), so it's the fallback, not the primary."""
        roots = _superlative_roots(word)
        for n, f in self.frames.items():
            if not f.data:
                continue
            for r in roots:
                if r in f.data:
                    return (n, 1 if f.data[r] >= 0 else -1)
        if model is None:
            return None
        coords = self.project_word(word, model, tokenizer, device)
        norm = sorted(((n, coords[n] / (self.frames[n].coord_scale or 1.0))
                       for n in coords), key=lambda kv: -abs(kv[1]))
        if not norm:
            return None
        top, v = norm[0]
        if len(norm) > 1 and abs(v) < margin * abs(norm[1][1]):
            return None
        return (top, 1 if v > 0 else -1)

    # -- orthogonality (refit each frame's direction at a common layer) --
    def orthogonality(self, model, tokenizer, device, layer: int = 8):
        """Re-fit each frame's direction at a shared layer (shared standardization) and
        return (names, |cos| matrix). Needs each frame's training `data`."""
        usable = [n for n, f in self.frames.items() if f.data]
        acts = {n: _collect(model, tokenizer, device, list(self.frames[n].data),
                            self.frames[n].template, self.frames[n].pool)
                for n in usable}
        allX = np.concatenate([np.array([acts[n][w][layer] for w in self.frames[n].data])
                               for n in usable])
        mu, sd = allX.mean(0), allX.std(0) + 1e-6
        dirs = {}
        for n in usable:
            f = self.frames[n]
            X = (np.array([acts[n][w][layer] for w in f.data]) - mu) / sd
            y = np.asarray(list(f.data.values()), float)
            dirs[n] = _ridge_dir(X, y)
        M = np.array([[abs(float(dirs[a] @ dirs[b])) for b in usable] for a in usable])
        return usable, M

    # -- persistence --
    def to_dict(self) -> dict:
        return {"fingerprint": self.fingerprint, "model_id": self.model_id,
                "frames": {n: f.to_dict() for n, f in self.frames.items()}}

    @classmethod
    def from_dict(cls, d: dict) -> "FrameLibrary":
        return cls(frames={n: Frame.from_dict(fd) for n, fd in d["frames"].items()},
                   fingerprint=d.get("fingerprint", ""), model_id=d.get("model_id", ""))

    def save(self, path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict()))
        return p

    @classmethod
    def load(cls, path) -> "FrameLibrary":
        return cls.from_dict(json.loads(Path(path).read_text()))

    # -- compact binary store (.npz, float32; ~3x smaller than JSON, numpy-only) --
    def save_npz(self, path) -> Path:
        """Dense arrays as compressed float32; scalar metadata as a JSON blob. The
        primary on-disk format (JSON save/load stays for human inspection)."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        arrays, meta = {}, {"fingerprint": self.fingerprint, "model_id": self.model_id,
                            "frames": {}}
        for name, f in self.frames.items():
            ax = f.axis
            arrays[f"{name}|mean"] = ax.mean.astype(np.float32)
            arrays[f"{name}|scale"] = ax.scale.astype(np.float32)
            arrays[f"{name}|direction"] = ax.direction.astype(np.float32)
            meta["frames"][name] = {
                "low_label": ax.low_label, "high_label": ax.high_label,
                "layer": int(ax.layer), "center": float(ax.center),
                "template": f.template, "pool": f.pool, "cv_r": f.cv_r, "data": f.data,
                "coord_scale": f.coord_scale}
        arrays["__meta__"] = np.frombuffer(json.dumps(meta).encode(), dtype=np.uint8)
        np.savez_compressed(p, **arrays)
        return p if p.suffix == ".npz" else Path(str(p) + ".npz")

    @classmethod
    def load_npz(cls, path) -> "FrameLibrary":
        z = np.load(Path(path), allow_pickle=False)            # no pickle: safe to ship
        meta = json.loads(bytes(z["__meta__"]).decode())
        frames = {}
        for name, m in meta["frames"].items():
            ax = BipolarAxis(name, m["low_label"], m["high_label"], m["layer"],
                             np.asarray(z[f"{name}|mean"], float),
                             np.asarray(z[f"{name}|scale"], float),
                             np.asarray(z[f"{name}|direction"], float), float(m["center"]))
            frames[name] = Frame(ax, template=m["template"], pool=m["pool"],
                                 cv_r=m["cv_r"], data=m["data"],
                                 coord_scale=m.get("coord_scale", 1.0))
        return cls(frames=frames, fingerprint=meta.get("fingerprint", ""),
                   model_id=meta.get("model_id", ""))


# ── fingerprint-addressed store (two-tier: bundled base + user overlay) ────────

def _fingerprint(model_or_fp) -> str:
    if isinstance(model_or_fp, str):
        return model_or_fp
    from turnstyle.profile import model_fingerprint
    return model_fingerprint(model_or_fp)


def save_library(lib: FrameLibrary, model_or_fp, model_id: str = "") -> Path:
    """Write a library to the user cache (compact .npz), addressed by fingerprint."""
    lib.fingerprint = _fingerprint(model_or_fp)
    if model_id:
        lib.model_id = model_id
    return lib.save_npz(_USER_FRAMES / f"{lib.fingerprint}.npz")


def _load_tier(base: Path, fp: str) -> "FrameLibrary | None":
    """Load <fp>.npz (primary) or <fp>.json (fallback) from one store tier."""
    npz, js = base / f"{fp}.npz", base / f"{fp}.json"
    if npz.exists():
        return FrameLibrary.load_npz(npz)
    if js.exists():
        return FrameLibrary.load(js)
    return None


def load_library(model_or_fp) -> "FrameLibrary | None":
    """Load the library matching this model's fingerprint: bundled base, user-cache
    overlay (user-fit frames win per-name, bundled fills gaps). None if neither exists."""
    fp = _fingerprint(model_or_fp)
    tiers = {}
    for name, base in (("bundled", _BUNDLED_FRAMES), ("user", _USER_FRAMES)):
        t = _load_tier(base, fp)
        if t is not None:
            tiers[name] = t
    if "bundled" in tiers and "user" in tiers:
        merged = FrameLibrary(frames=dict(tiers["bundled"].frames), fingerprint=fp,
                              model_id=tiers["user"].model_id or tiers["bundled"].model_id)
        merged.frames.update(tiers["user"].frames)            # user wins per-frame
        return merged
    return tiers.get("user") or tiers.get("bundled")


# ── canonical family: the adjective-ordering rungs + number + time ────────────
# Validated in experiments/ordering_frames.py (rungs) and frame_family_matrix.py /
# time_frame.py (number, time). Scalars are ordinal within-frame gradients.

CANONICAL_FRAMES = {
    "opinion": {"data": {"terrible": -3, "horrible": -3, "awful": -3, "bad": -2,
        "nasty": -2, "poor": -1, "mediocre": 0, "decent": 1, "good": 2, "great": 2,
        "lovely": 3, "wonderful": 3, "excellent": 3, "delightful": 3}},
    "size": {"data": {"tiny": -3, "minuscule": -3, "small": -2, "little": -2,
        "modest": -1, "average": 0, "large": 1, "big": 1, "huge": 2, "enormous": 3,
        "gigantic": 3, "massive": 3}},
    "age": {"data": {"newborn": -3, "new": -2, "young": -2, "fresh": -2, "recent": -1,
        "modern": -1, "mature": 1, "old": 2, "aged": 2, "elderly": 2, "ancient": 3,
        "antique": 3}},
    "shape": {"data": {"round": 0, "circular": 0, "spherical": 0, "oval": 1, "curved": 1,
        "square": 2, "rectangular": 2, "boxy": 2, "flat": 2, "long": 3, "thin": 3,
        "narrow": 3, "elongated": 3}},
    "space": {"data": {"local": 0, "domestic": 1, "native": 1, "regional": 2,
        "national": 3, "foreign": 4, "distant": 5, "remote": 5, "exotic": 5,
        "faraway": 6, "alien": 7, "cosmic": 9}},
    # material as NATURALNESS (natural↔synthetic) — the axis SmolLM2 actually encodes
    # cleanly + causally (recov 0.93, steer Δ+13). The earlier hardness ordinal was a
    # poor 1D summary (recov 0.73, steer Δ+5) — see experiments/material_investigate.py.
    "material": {"template": "It is made of {w}.", "data": {
        "cotton": 1, "wool": 1, "wood": 1, "stone": 1, "leather": 1, "silk": 1,
        "linen": 1, "clay": 1, "plastic": 0, "nylon": 0, "polyester": 0, "acrylic": 0,
        "vinyl": 0, "rubber": 0, "foam": 0}},
    "number": {"template": "It is a {w}.", "data": {"one": 0.0, "two": 0.301,
        "three": 0.477, "four": 0.602, "five": 0.699, "six": 0.778, "seven": 0.845,
        "eight": 0.903, "nine": 0.954, "ten": 1.0, "twenty": 1.301, "fifty": 1.699,
        "hundred": 2.0, "thousand": 3.0, "million": 6.0, "billion": 9.0}},
    "time": {"template": "It lasted a {w}.", "data": {"millisecond": -3.0, "second": 0.0,
        "minute": 1.78, "hour": 3.56, "day": 4.94, "week": 5.78, "month": 6.42,
        "year": 7.5, "decade": 8.5, "century": 9.5, "millennium": 10.5}},
}


__all__ = ["Frame", "FrameLibrary", "CANONICAL_FRAMES",
           "save_library", "load_library"]
