"""The depth question, asked without an argmax.

The retraction was specifically about MAX-over-layers: best-of-31 is a large
comparison surface and adjacent-layer similarity gives it a high null. Two depth
questions do not need a max, and both have clean permutation nulls:

  LEXICAL-vs-COMPUTED   r(L0) vs r(L_final). Two PREDETERMINED points, no
                        selection at all. Is the frame already in the token
                        embedding, or does the stack build it?

  TREND                 Spearman(layer index, r) over all layers. One degree of
                        freedom, uses every point instead of the luckiest one, so
                        it is far better powered than a max. Does recoverability
                        RISE with depth?

Both are z-scored against label permutations, the same null that killed `spread`.
Reported per (frame, model) and pooled, with the cross-model consistency check that
matters: a real property of a CONCEPT should reproduce across backbones, whereas
noise scatters.

    uv run python experiments/depth_v3_lexical_vs_computed.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from turnstyle.frame_library import _cv_r  # noqa: E402
from localization_v2 import load_frames  # noqa: E402

RESULTS = Path(__file__).parent / "results"
N_PERM = 30
MODELS = [
    ("HuggingFaceTB/SmolLM2-360M-Instruct", "float32"),
    ("Qwen/Qwen2.5-1.5B-Instruct", "float32"),
    ("HuggingFaceTB/SmolLM2-1.7B-Instruct", "float32"),
    ("google/gemma-3n-E2B-it", "float32"),
    ("microsoft/Phi-4-mini-instruct", "bfloat16"),
    ("mistralai/Mistral-7B-Instruct-v0.3", "bfloat16"),
]


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    if ra.std() < 1e-9 or rb.std() < 1e-9:
        return 0.0
    return float(np.corrcoef(ra, rb)[0, 1])


def measures(c):
    """Both statistics, neither involving a max over layers."""
    layers = np.arange(len(c), dtype=float)
    return {"gain": float(c[-1] - c[0]), "trend": spearman(layers, c)}


def main():
    rng = np.random.default_rng(0)
    per_frame = {}
    print(f"{'model':12s} {'frame':14s} {'r(L0)':>7s} {'r(end)':>7s} {'gain':>7s} "
          f"{'gain_z':>7s} {'trend':>7s} {'trend_z':>8s}")
    for mid, dtype in MODELS:
        frames = load_frames(mid, dtype)
        tag = mid.split("/")[-1][:12]
        for name in sorted(frames):
            words, X, y = frames[name]

            def curve(yy, _X=X):
                return np.nan_to_num(
                    np.array([_cv_r(Xi, yy) for Xi in _X], float), nan=0.0)

            real = measures(curve(y))
            null = [measures(curve(rng.permutation(y))) for _ in range(N_PERM)]
            z = {}
            for k in ("gain", "trend"):
                v = np.array([n[k] for n in null])
                z[k] = float((real[k] - v.mean()) / (v.std() + 1e-9))
            c = curve(y)
            per_frame.setdefault(name, []).append(
                {"model": tag, "r0": float(c[0]), "rend": float(c[-1]), **real,
                 "gain_z": z["gain"], "trend_z": z["trend"]})
            print(f"{tag:12s} {name:14s} {c[0]:>7.3f} {c[-1]:>7.3f} "
                  f"{real['gain']:>+7.3f} {z['gain']:>+7.1f} {real['trend']:>+7.3f} "
                  f"{z['trend']:>+8.1f}", flush=True)

    print("\n" + "=" * 88)
    print("POOLED — a real property of a CONCEPT should reproduce across backbones")
    print("=" * 88)
    print(f"  {'frame':14s} {'mean r(L0)':>11s} {'mean gain':>10s} {'models w/ gain_z>=2':>21s} "
          f"{'mean trend':>11s}   class")
    rows = []
    for name, recs in sorted(per_frame.items(),
                             key=lambda kv: np.mean([r["gain"] for r in kv[1]])):
        r0 = np.mean([r["r0"] for r in recs])
        g = np.mean([r["gain"] for r in recs])
        ng = sum(1 for r in recs if r["gain_z"] >= 2)
        tr = np.mean([r["trend"] for r in recs])
        cls = ("COMPUTED — built by the stack" if ng >= len(recs) - 1 and g > 0.05 else
               "lexical — already in the embedding" if ng == 0 and abs(g) < 0.05 else
               "mixed")
        rows.append((name, r0, g, ng, tr, cls))
        print(f"  {name:14s} {r0:>11.3f} {g:>+10.3f} {f'{ng}/{len(recs)}':>21s} "
              f"{tr:>+11.3f}   {cls}")

    comp = [r[0] for r in rows if r[5].startswith("COMPUTED")]
    lex = [r[0] for r in rows if r[5].startswith("lexical")]
    print(f"\n  COMPUTED (needs the stack, consistent across models): {comp or 'none'}")
    print(f"  LEXICAL  (present at the embedding):                  {lex or 'none'}")
    print(f"  mixed / inconsistent:                                 "
          f"{[r[0] for r in rows if r[5] == 'mixed'] or 'none'}")

    out = RESULTS / "depth_v3.json"
    out.write_text(json.dumps({"n_perm": N_PERM, "per_frame": per_frame,
                               "computed": comp, "lexical": lex}, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
