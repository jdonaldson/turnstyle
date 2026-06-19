"""Adjective-polarity capability detector — a model-level primitive test.

Polarity (does an adjective denote the HIGH or LOW end of its scalar axis:
tallest/oldest/widest/cheapest...) is a candidate task-agnostic primitive: if a
model encodes it as a linear direction, it transfers across vocabulary AND across
languages (the direction is semantic, not lexical), and any ordering/comparison
task can lean on it instead of a hardcoded adjective list.

But it is NOT safe to assume — pole_generalize.py showed SmolLM2 has it for
size/price but fails on age ("old" reads as high-magnitude). So we TEST for it
per model and report where it ships.

Capability metric = leave-one-AXIS-out: train the polarity direction on all
axes but one, test on the held-out axis. This is the honest generalization test
(leave-one-word-out leaks via same-axis neighbors). A model "has the primitive"
if held-out axes transfer well; per-axis scores say where to trust it.

Reuses the activations cached by pole_generalize.py (no model run needed).

Usage:  python experiments/polarity_capability.py
"""
from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, "experiments")
sys.path.insert(0, "src")
import pole_harness as H
import pole_generalize as G

# adjectives grouped by scalar axis; pole = "more of the attribute" = HIGH (+1)
AXES = {
    "size":      [("big", +1), ("small", -1), ("large", +1), ("tall", +1),
                  ("short", -1), ("long", +1), ("wide", +1), ("narrow", -1),
                  ("deep", +1), ("shallow", -1)],
    "weight":    [("heavy", +1), ("light", -1)],
    "speed":     [("fast", +1), ("slow", -1)],
    "temp":      [("hot", +1), ("cold", -1), ("warm", +1)],
    "value":     [("expensive", +1), ("cheap", -1), ("rich", +1), ("poor", -1)],
    "quality":   [("good", +1), ("bad", -1)],
    "intensity": [("strong", +1), ("weak", -1), ("bright", +1), ("dark", -1),
                  ("loud", +1), ("quiet", -1)],
    "age":       [("new", +1), ("old", -1), ("young", +1)],
    "mood":      [("happy", +1), ("sad", -1)],
}
ROOT_AXIS = {r: ax for ax, ws in AXES.items() for r, _ in ws}


def load():
    d = np.load(G.CACHE, allow_pickle=True)
    A = d["acts"].astype(np.float32)
    y = (d["poles"] == H.HIGH).astype(int)
    roots = np.array([str(r) for r in d["roots"]])
    axis = np.array([ROOT_AXIS.get(r, "?") for r in roots])
    return A, y, roots, axis


def report():
    A, y, roots, axis = load()
    nL = A.shape[1]
    axes = list(AXES)
    print(f"axes={len(axes)}  words={len(set(roots))}  occ={len(y)}")

    # leave-one-axis-out capability score per layer
    print(f"\n{'L':>3} {'LOaxis':>7}   capability (held-out-axis pole acc)")
    best = (-1.0, -1)
    layer_axis = {}
    for L in range(nL):
        X = A[:, L, :]
        accs = {}
        for ax in axes:
            te = axis == ax
            if len(set(y[~te])) < 2:
                continue
            sc, clf = G._fit(X[~te], y[~te])
            accs[ax] = clf.score(sc.transform(X[te]), y[te])
        m = float(np.mean(list(accs.values())))
        layer_axis[L] = accs
        if m > best[0]:
            best = (m, L)
        print(f"{L:>3} {m:>7.3f}")
    Lb = best[1]
    print(f"\nbest leave-one-axis-out: L{Lb}  capability={best[0]:.3f}  "
          f"→ SHIP={'yes' if best[0] >= 0.75 else 'no'} (gate 0.75)")

    # per-axis transfer at the best layer — where the primitive is trustworthy
    print(f"\nper-axis transfer @L{Lb} (train on other axes → predict this axis):")
    for ax, acc in sorted(layer_axis[Lb].items(), key=lambda kv: -kv[1]):
        bar = "█" * round(acc * 20)
        flag = "" if acc >= 0.75 else "   ← unreliable on this model"
        print(f"  {ax:10s} {acc:5.2f} {bar}{flag}")

    # antonym-opposition: do antonym pairs land on OPPOSITE poles? (the real test)
    print(f"\nantonym opposition @L{Lb} (held-out-axis, both ends correct):")
    X = A[:, Lb, :]
    for ax in axes:
        te = axis == ax
        sc, clf = G._fit(X[~te], y[~te])
        pred = clf.predict(sc.transform(X[te]))
        truth = y[te]
        hi_ok = pred[truth == 1].mean() if (truth == 1).any() else float("nan")
        lo_ok = (1 - pred[truth == 0]).mean() if (truth == 0).any() else float("nan")
        ok = hi_ok > 0.6 and lo_ok > 0.6
        print(f"  {ax:10s} HIGH-end={hi_ok:.2f} LOW-end={lo_ok:.2f}  "
              f"{'opposed ✓' if ok else 'COLLAPSED ✗'}")


if __name__ == "__main__":
    report()
