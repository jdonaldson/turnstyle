"""Is Phi-4-mini's frame flatness a MODEL property or a bf16 artifact?

Phi-4-mini was the outlier in the six-backbone depth audit: 1/14 frames with a
locatable peak, against 5/14 for Mistral-7B on the SAME bf16 dtype
([[peak_layer_not_load_bearing]]).  Mistral argues against precision being the
cause, but does not clear it -- Phi and Mistral are different models, so the
comparison is not controlled.

This IS the controlled version: same model, same lexicons, same estimator, dtype
as the only varied factor.

The decisive statistic is not the verdict count (a threshold at spread=0.20 can
flip on noise) but the per-frame CORRELATION between the two CV-r curves.  If the
curves agree at r ~ 0.99, precision is irrelevant and Phi's flatness is real; if
they diverge, the bf16 run was measuring its own rounding.

    uv run python experiments/depth_stability.py --model microsoft/Phi-4-mini-instruct \\
        --dtype float32 --out experiments/results/phi4_fp32_probe.json
    uv run python experiments/phi4_dtype_control.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

RESULTS = Path(__file__).parent / "results"
BF16 = RESULTS / "depth_stability_Phi-4-mini-instruct.json"
FP32 = RESULTS / "phi4_fp32_probe.json"


def verdict(spread, w2):
    if spread < 0.20:
        return "FLAT"
    return "soft" if w2 > 6 else "real"


def main():
    for p in (BF16, FP32):
        if not p.exists():
            raise SystemExit(f"missing {p} — see the docstring for how to produce it")
    a = json.loads(BF16.read_text())
    b = json.loads(FP32.read_text())
    print(f"bf16: {a['model']} ({a.get('dtype')})   fp32: {b['model']} ({b.get('dtype')})\n")

    print("=" * 84)
    print("PER-FRAME: does precision change the curve?")
    print("=" * 84)
    print(f"  {'frame':14s} {'bf16':>22s} {'fp32':>22s} {'curve r':>9s} {'max|dr|':>8s}")
    rows = []
    for f in sorted(a["frames"]):
        if f not in b["frames"]:
            continue
        fa, fb = a["frames"][f], b["frames"][f]
        ca = np.array([np.nan if x is None else x for x in fa["curve"]], float)
        cb = np.array([np.nan if x is None else x for x in fb["curve"]], float)
        ca[ca < -1] = np.nan
        cb[cb < -1] = np.nan
        ok = ~np.isnan(ca) & ~np.isnan(cb)
        cr = float(np.corrcoef(ca[ok], cb[ok])[0, 1]) if ok.sum() > 2 else float("nan")
        mad = float(np.nanmax(np.abs(ca - cb)))
        va = verdict(fa["spread"], fa["within_02"])
        vb = verdict(fb["spread"], fb["within_02"])
        rows.append((f, va, vb, cr, mad, fa, fb))
        sa = f"{va:4s} L{fa['layer']:<2d} sp={fa['spread']:.2f}"
        sb = f"{vb:4s} L{fb['layer']:<2d} sp={fb['spread']:.2f}"
        flip = "" if va == vb else "   <- VERDICT CHANGED"
        print(f"  {f:14s} {sa:>22s} {sb:>22s} {cr:>9.3f} {mad:>8.3f}{flip}")

    print("\n" + "=" * 84)
    print("VERDICT")
    print("=" * 84)
    n_a = sum(r[1] == "real" for r in rows)
    n_b = sum(r[2] == "real" for r in rows)
    flips = [r[0] for r in rows if r[1] != r[2]]
    crs = np.array([r[3] for r in rows], float)
    mads = np.array([r[4] for r in rows], float)
    print(f"  citable (real peak):   bf16 {n_a}/{len(rows)}   fp32 {n_b}/{len(rows)}")
    print(f"  verdict changes:       {len(flips)}/{len(rows)}  {flips}")
    print(f"  curve correlation:     mean {np.nanmean(crs):.4f}  min {np.nanmin(crs):.4f}")
    print(f"  max |delta r|:         mean {np.nanmean(mads):.4f}  max {np.nanmax(mads):.4f}")
    print()
    if np.nanmean(crs) > 0.98 and np.nanmean(mads) < 0.05:
        print("  => PRECISION IS NOT THE CAUSE. The curves are the same shape in both")
        print("     dtypes, so Phi-4-mini's flatness is a property of the MODEL.")
        print("     bf16 is safe for this measurement on the other backbones too.")
    elif np.nanmean(crs) < 0.90 or np.nanmean(mads) > 0.15:
        print("  => PRECISION MATTERS. The bf16 curves are not measuring the same thing;")
        print("     every bf16 row in the six-backbone audit (Phi, Mistral) needs rerunning")
        print("     in fp32 before its verdicts can be cited.")
    else:
        print("  => AMBIGUOUS. Curves broadly agree but not tightly; report both and do")
        print("     not lean on Phi's verdict counts either way.")


if __name__ == "__main__":
    main()
