"""Per-model calibration profiles — hash a backbone, store its tuned configs, apply later.

A `ModelProfile` is everything turnstyle learns about *one* backbone: which layer +
probe solves each task, the ship/no-ship verdict, and (later) per-task extraction
templates. It is keyed by an **activation-sensitive fingerprint** of the model
(weights + dtype + topology), so a fine-tune or a quantized copy gets its own profile
instead of silently inheriting another model's layers.

Pipeline:
    profile = build_profile(model, tok, device, {task: (examples, target_fn)})
    save_profile(profile)                       # → ~/.cache/turnstyle/profiles/<fp>.json
    ...
    profile = load_profile(model)               # fingerprint match → ready to apply
    art = profile.get_probe("snarks")           # ProbeArtifact, predicts on new prompts

The fingerprint folds in `CALIBRATION_VERSION`, so when the probe/finder logic changes
old profiles are ignored and recalibrated rather than applied stale.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np

# Bump when the probe/finder/sweep logic changes in a way that invalidates fitted
# artifacts. Folded into the fingerprint → stale profiles miss the lookup.
CALIBRATION_VERSION = 1

_USER_CACHE = Path(
    os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache"))
) / "turnstyle" / "profiles"
_BUNDLED = Path(__file__).parent / "data" / "profiles"


# ── fingerprint ───────────────────────────────────────────────────────────────

def model_fingerprint(model) -> str:
    """Activation-sensitive hash of a model: config + dtype + full topology +
    a strided byte-sample of a few tensors. Deterministic across loads; sensitive
    to fine-tunes and quantization without hashing gigabytes. Includes
    CALIBRATION_VERSION so a calibration-logic change invalidates old profiles."""
    import torch

    h = hashlib.sha256()
    h.update(f"calv{CALIBRATION_VERSION}".encode())
    c = model.config
    h.update(
        f"{getattr(c, 'model_type', '?')}|{getattr(c, 'hidden_size', '?')}|"
        f"{getattr(c, 'num_hidden_layers', '?')}|{getattr(c, 'vocab_size', '?')}".encode()
    )
    h.update(str(getattr(model, "dtype", "?")).encode())

    sd = model.state_dict()
    keys = sorted(sd)
    h.update("|".join(f"{k}:{tuple(sd[k].shape)}" for k in keys).encode())  # topology
    # cheap weight sample: ~8 tensors, strided bytes
    for k in keys[:: max(1, len(keys) // 8)]:
        t = sd[k].detach().to("cpu").reshape(-1)[::10007].to(torch.float32)
        h.update(k.encode())
        h.update(t.numpy().tobytes())
    return h.hexdigest()[:16]


# ── portable probe (re)serialization ────────────────────────────────────────────

def _probe_to_dict(artifact, finder_name: str) -> dict:
    """Extract a ProbeArtifact's fitted linear params into a JSON-able dict.
    The finder is stored by name (resolved against DEFAULT_FINDERS at load)."""
    sc, clf = artifact.scaler, artifact.classifier
    return {
        "finder": finder_name,
        "layer": int(artifact.layer),
        "mode": artifact.mode,
        "classes": list(artifact.classes),
        "answer_format": artifact.answer_format,
        "scaler": {"mean": sc.mean_.tolist(), "scale": sc.scale_.tolist()},
        "classifier": {
            "coef": clf.coef_.tolist(),
            "intercept": clf.intercept_.tolist(),
            "classes_": clf.classes_.tolist(),
        },
    }


def _probe_from_dict(d: dict):
    """Rebuild a predict-capable ProbeArtifact from stored params. Reconstructs
    real sklearn objects (exact predict semantics) and resolves the finder by name."""
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from turnstyle.autoprobe import ProbeArtifact, DEFAULT_FINDERS

    sc = StandardScaler()
    sc.mean_ = np.asarray(d["scaler"]["mean"], dtype=float)
    sc.scale_ = np.asarray(d["scaler"]["scale"], dtype=float)
    sc.var_ = sc.scale_ ** 2
    sc.n_features_in_ = sc.mean_.shape[0]

    clf = LogisticRegression()
    clf.coef_ = np.asarray(d["classifier"]["coef"], dtype=float)
    clf.intercept_ = np.asarray(d["classifier"]["intercept"], dtype=float)
    clf.classes_ = np.asarray(d["classifier"]["classes_"])
    clf.n_features_in_ = clf.coef_.shape[1]

    finder = DEFAULT_FINDERS.get(d["finder"])
    if finder is None:
        raise KeyError(f"finder {d['finder']!r} not in DEFAULT_FINDERS")
    return ProbeArtifact(
        finder=finder, layer=d["layer"], mode=d["mode"], classes=d["classes"],
        scaler=sc, classifier=clf, answer_format=d["answer_format"],
    )


# ── the profile ─────────────────────────────────────────────────────────────────

@dataclass
class ModelProfile:
    """Calibrated configs for one backbone, keyed by its fingerprint."""
    fingerprint: str
    model_id: str                       # human identity (for inspecting the cache)
    calibration_version: int = CALIBRATION_VERSION
    created: str = ""
    components: dict = field(default_factory=dict)   # task -> serialized probe dict
    extraction: dict = field(default_factory=dict)   # task -> extraction profile (future)
    support: dict = field(default_factory=dict)      # task -> {method, accuracy, shipped}
    polarity: dict = field(default_factory=dict)     # model-level adjective-polarity probe

    def get_probe(self, task: str):
        """Reconstruct the fitted ProbeArtifact for a task, or None if not shipped."""
        d = self.components.get(task)
        return _probe_from_dict(d) if d is not None else None

    def get_polarity(self):
        """Reconstruct the adjective-polarity primitive, or None if not calibrated.

        Model-level (not per-task): a linear polarity direction + per-axis capability
        map. See turnstyle.polarity / [[probe_locality_symbolic_global]]."""
        if not self.polarity:
            return None
        from turnstyle.polarity import PolarityProbe
        return PolarityProbe.from_dict(self.polarity)

    def set_polarity(self, probe) -> None:
        """Store a fitted PolarityProbe + a support entry recording where it ships."""
        self.polarity = probe.to_dict()
        cap = probe.capability
        self.support["_polarity"] = {
            "method": "polarity", "layer": probe.layer,
            "loo_axis": None if cap is None else cap.loo_axis,
            "shipped": bool(cap and cap.ship),
            "axes": None if cap is None else cap.axes_shipping(),
        }

    def set_probe(self, task: str, artifact, finder_name: str, accuracy=None) -> None:
        """Store a fitted ProbeArtifact for a task (serialized) + a support entry."""
        self.components[task] = _probe_to_dict(artifact, finder_name)
        self.support[task] = {
            "method": "probe", "finder": finder_name, "layer": int(artifact.layer),
            "accuracy": accuracy, "shipped": True,
        }

    def to_dict(self) -> dict:
        return {
            "fingerprint": self.fingerprint, "model_id": self.model_id,
            "calibration_version": self.calibration_version, "created": self.created,
            "components": self.components, "extraction": self.extraction,
            "support": self.support, "polarity": self.polarity,
        }

    def save(self, path: Path | str) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=1))
        return path

    @classmethod
    def load(cls, path: Path | str) -> "ModelProfile":
        d = json.loads(Path(path).read_text())
        return cls(**d)


# ── two-tier, fingerprint-addressed load/save ────────────────────────────────────

def _merge_profiles(base: ModelProfile, overlay: ModelProfile) -> ModelProfile:
    """Overlay (user cache) wins per key; base (bundled) fills the gaps. Lets a
    shipped model-level capability (e.g. polarity) reach a user who has only
    calibrated some task probes of their own."""
    return ModelProfile(
        fingerprint=overlay.fingerprint, model_id=overlay.model_id,
        calibration_version=overlay.calibration_version, created=overlay.created,
        components={**base.components, **overlay.components},
        extraction={**base.extraction, **overlay.extraction},
        support={**base.support, **overlay.support},
        polarity=overlay.polarity or base.polarity,
    )


def load_profile(model) -> Optional[ModelProfile]:
    """Find the profile matching this model's fingerprint. The bundled profile is
    the base; a user-cache profile (if present) overlays it — user-tuned entries
    win, bundled capabilities (e.g. polarity) fill gaps the user hasn't calibrated.
    Returns None (→ dispatch abstains) if neither tier matches."""
    fp = model_fingerprint(model)
    tiers = {}
    for name, d in (("bundled", _BUNDLED), ("user", _USER_CACHE)):
        p = d / f"{fp}.json"
        if p.exists():
            prof = ModelProfile.load(p)
            if prof.calibration_version == CALIBRATION_VERSION:
                tiers[name] = prof
    if "bundled" in tiers and "user" in tiers:
        return _merge_profiles(tiers["bundled"], tiers["user"])
    return tiers.get("user") or tiers.get("bundled")


def save_profile(profile: ModelProfile) -> Path:
    """Write a profile to the user cache, content-addressed by fingerprint."""
    return profile.save(_USER_CACHE / f"{profile.fingerprint}.json")


# ── the builder: fold autoprobe in as the first populator ────────────────────────

def build_profile(
    model, tokenizer, device,
    tasks: dict[str, tuple[list, Callable]],
    model_id: str = "",
    verbose: bool = False,
    with_polarity: bool = False,
) -> ModelProfile:
    """Calibrate a model over a set of probe tasks. `tasks` maps task name →
    (examples, target_fn). Each task runs through autoprobe's layer×finder×mode
    sweep + ship gate; shipped probes land in `components`, every task gets a
    `support` entry (so non-shipped tasks are explicit, not silently missing).

    `with_polarity=True` also calibrates the model-level adjective-polarity
    primitive (turnstyle.polarity.detect_polarity) and records where it ships."""
    from turnstyle.autoprobe import autoprobe

    prof = ModelProfile(
        fingerprint=model_fingerprint(model),
        model_id=model_id or getattr(model.config, "_name_or_path", "") or "unknown",
        created=time.strftime("%Y-%m-%d"),
    )
    if with_polarity:
        from turnstyle.polarity import detect_polarity
        pp = detect_polarity(model, tokenizer, device)
        prof.set_polarity(pp)
        if verbose:
            cap = pp.capability
            print(f"[_polarity] {'SHIP' if cap and cap.ship else 'no-ship'} "
                  f"@L{pp.layer} loo_axis={cap.loo_axis:.3f} axes={cap.axes_shipping()}")
    for task, (examples, target_fn) in tasks.items():
        res = autoprobe(
            examples=examples, target_fn=target_fn,
            model=model, tokenizer=tokenizer, device=device, verbose=verbose,
        )
        chosen_layer = res.chosen[2] if res.chosen else None
        cv = res.chosen[3] if res.chosen else None
        if res.ship and res.fitted is not None and res.chosen is not None:
            prof.set_probe(task, res.fitted, res.chosen[0], accuracy=cv)
        else:
            prof.support[task] = {
                "method": "probe", "layer": chosen_layer, "accuracy": cv,
                "shipped": False, "reason": res.reason,
            }
        if verbose:
            print(f"[{task}] {'SHIP' if prof.support[task]['shipped'] else 'no-ship'} — {res.reason}")
    return prof
