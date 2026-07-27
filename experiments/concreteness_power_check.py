"""Is `concreteness` localizing for real, or because it has the biggest lexicon?

Across six backbones, only 5/84 (frame, model) tests clear the localization gate,
and 3 of the 5 are `concreteness` (P(repeat>=3 | random scatter) = .048). Strip it
and the remainder sits exactly at the null band edge. So the whole cross-model
signal rests on this one frame.

Competing explanations:
  REAL   concreteness genuinely has depth structure the others lack
  POWER  concreteness has n=80 words, >2x the next largest, so it is simply the
         frame most able to clear a 2-sigma double gate

Discriminator: subsample it to a TYPICAL lexicon size and re-test. Real signal
should degrade gracefully and often survive; a pure power artifact should vanish at
matched n.

Counter-evidence already on the table, which is why this is worth running rather
than assuming: animacy (n=37), certainty (22) and competence (19) are the next
best-powered frames and hit 0/6 between them. Pure power should have produced
occasional hits there too.

    uv run python experiments/concreteness_power_check.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from turnstyle.frame_library import _cv_r  # noqa: E402
from localization_v2 import load_frames, stats_of, N_PERM, Z_GATE  # noqa: E402

RESULTS = Path(__file__).parent / "results"
# the three models where concreteness cleared the gate, plus one where it did not
MODELS = [
    ("mistralai/Mistral-7B-Instruct-v0.3", "bfloat16", True),
    ("Qwen/Qwen2.5-1.5B-Instruct", "float32", True),
    ("HuggingFaceTB/SmolLM2-360M-Instruct", "float32", True),
    ("HuggingFaceTB/SmolLM2-1.7B-Instruct", "float32", False),
]
SUB_N = 20          # matched to a typical frame lexicon
N_SUB = 8           # independent subsamples per model


def gate(X, y, rng):
    def curve(yy):
        return np.nan_to_num(np.array([_cv_r(Xi, yy) for Xi in X], float), nan=0.0)
    real = stats_of(curve(y))
    null = [stats_of(curve(rng.permutation(y))) for _ in range(N_PERM)]
    z = {}
    for k in ("peak_r", "shape_r2"):
        v = np.array([n[k] for n in null])
        z[k] = float((real[k] - v.mean()) / (v.std() + 1e-9))
    return z, bool(z["shape_r2"] >= Z_GATE and z["peak_r"] >= Z_GATE)


def main():
    rng = np.random.default_rng(0)
    out = {}
    print(f"subsampling concreteness 80 -> {SUB_N} words, {N_SUB} draws per model, "
          f"gate = shape_z & peak_z >= {Z_GATE}\n")
    print(f"  {'model':26s} {'full n=80':>10s} {'subsampled hits':>17s} "
          f"{'median shape_z':>15s}")
    for mid, dtype, hit_full in MODELS:
        frames = load_frames(mid, dtype)
        words, X, y = frames["concreteness"]
        n = len(words)
        hits, zs = 0, []
        for s in range(N_SUB):
            idx = rng.choice(n, size=SUB_N, replace=False)
            Xs = [Xi[idx] for Xi in X]
            z, ok = gate(Xs, y[idx], np.random.default_rng(100 + s))
            hits += ok
            zs.append(z["shape_r2"])
        tag = mid.split("/")[-1]
        out[tag] = {"full_hit": hit_full, "sub_hits": hits, "n_sub": N_SUB,
                    "median_shape_z": float(np.median(zs)),
                    "shape_z": [float(v) for v in zs]}
        print(f"  {tag:26s} {'HIT' if hit_full else 'miss':>10s} "
              f"{f'{hits}/{N_SUB}':>17s} {np.median(zs):>15.2f}", flush=True)

    hitm = [v for k, v in out.items() if v["full_hit"]]
    sub_rate = sum(v["sub_hits"] for v in hitm) / (len(hitm) * N_SUB)
    print(f"\n  among models where the FULL lexicon localizes, subsampled hit rate "
          f"= {sub_rate:.2f}")
    if sub_rate >= 0.5:
        print("  => REAL: survives at matched n, so it is not a lexicon-size artifact.")
    elif sub_rate <= 0.15:
        print("  => POWER: vanishes at matched n. concreteness cleared the gate because")
        print("     it has 80 words, not because it has depth structure the others lack.")
        print("     With it removed the six-model result is 2/78 -- pure noise.")
    else:
        print("  => AMBIGUOUS at this n; the frame is underpowered once matched and the")
        print("     cross-model claim should not be leaned on either way.")

    p = RESULTS / "concreteness_power_check.json"
    p.write_text(json.dumps({"sub_n": SUB_N, "n_sub": N_SUB,
                             "sub_rate_where_full_hit": sub_rate,
                             "models": out}, indent=2))
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
