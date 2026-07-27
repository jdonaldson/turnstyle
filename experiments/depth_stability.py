"""Is the frame depth gradient built on real peaks, or on argmax noise?

[[frame_sweep_patterns]] reports a lexical-early / relational-late gradient read off
each frame's best layer.  That pattern has never been checked for STABILITY: an
argmax over a flat CV-r curve is noise wearing a layer number
([[peak_layer_not_load_bearing]], found on Gemma 3n where concreteness varies only
0.083 across the whole stack).

Re-runs the canonical family on any backbone capturing the FULL per-layer curve, and
reports for each frame:

  peak       argmax layer (what the gradient was built from)
  <=.01/.02  how many layers sit within that of the max  -- the plateau
  spread     max - min over L1..L_top (excludes the L0 embedding)
  depth%     peak as a FRACTION of stack depth -- the only cross-model comparable
             form, since a 24-layer and a 30-layer model do not share layer indices

    uv run python experiments/depth_stability.py
    uv run python experiments/depth_stability.py --model Qwen/Qwen2.5-1.5B-Instruct

Run OUTSIDE the sandbox (torch misreports MPS as unavailable inside it).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from turnstyle.frame_library import (  # noqa: E402
    CANONICAL_FRAMES, _collect, _cv_r, _DEFAULT_TEMPLATE,
)

RESULTS = Path(__file__).parent / "results"


def stability(curve: list[float]) -> dict:
    c = np.array(curve, float)
    c[c < -1] = np.nan
    mx = float(np.nanmax(c))
    peak = int(np.nanargmax(c))
    n = len(c)
    return {
        "r": mx,
        "layer": peak,
        "depth_frac": peak / (n - 1),
        "within_01": int(np.nansum(c >= mx - 0.01)),
        "within_02": int(np.nansum(c >= mx - 0.02)),
        "spread": float(np.nanmax(c[1:]) - np.nanmin(c[1:])),
        "curve": [None if x != x else round(float(x), 4) for x in c],
    }


def verdict(s: dict) -> str:
    if s["spread"] < 0.20:
        return "FLAT — peak is noise"
    if s["within_02"] > 6:
        return "soft — wide plateau"
    return "real"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="HuggingFaceTB/SmolLM2-1.7B-Instruct")
    ap.add_argument("--frames", default="")
    # fp32 keeps every backbone on the same footing, but a 7B in fp32 is ~28GB.
    # bf16 (not fp16) is the safe downgrade: same exponent range, so none of the
    # deep-layer overflow NaNs that bit the steering work.
    ap.add_argument("--dtype", default="float32", choices=("float32", "bfloat16"))
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=getattr(torch, args.dtype)).to(dev).eval()
    print(f"{args.model}  device={dev}  dtype={args.dtype}", flush=True)

    specs = dict(CANONICAL_FRAMES)
    if args.frames:
        want = [f.strip() for f in args.frames.split(",")]
        specs = {k: v for k, v in specs.items() if k in want}

    out = {}
    for name, spec in specs.items():
        data = spec["data"]
        words = list(data)
        y = np.asarray([data[w] for w in words], float)
        acts = _collect(model, tok, dev, words,
                        spec.get("template", _DEFAULT_TEMPLATE), "last")
        n_layers = acts[words[0]].shape[0]
        curve = [_cv_r(np.array([acts[w][L] for w in words]), y) for L in range(n_layers)]
        curve = [(-9.0 if c != c else c) for c in curve]
        s = stability(curve)
        out[name] = s
        print(f"  {name:14s} r={s['r']:+.3f} @L{s['layer']:<2d} "
              f"({s['depth_frac']:.0%})  plateau≤.02={s['within_02']:<2d} "
              f"spread={s['spread']:.3f}  {verdict(s)}", flush=True)

    print(f"\n{'=' * 78}\nDEPTH GRADIENT — only 'real' rows may be cited\n{'=' * 78}")
    print(f"  {'frame':14s} {'peak':>5s} {'depth%':>7s} {'≤.01':>5s} {'≤.02':>5s} "
          f"{'spread':>8s}   verdict")
    for name, s in sorted(out.items(), key=lambda kv: kv[1]["depth_frac"]):
        print(f"  {name:14s} {'L' + str(s['layer']):>5s} {s['depth_frac']:>6.0%} "
              f"{s['within_01']:>5d} {s['within_02']:>5d} {s['spread']:>8.3f}   {verdict(s)}")

    real = [n for n, s in out.items() if verdict(s) == "real"]
    flat = [n for n, s in out.items() if s["spread"] < 0.20]
    print(f"\n  citable (real peak): {len(real)}/{len(out)} — {sorted(real)}")
    print(f"  FLAT (peak is noise): {len(flat)}/{len(out)} — {sorted(flat)}")

    tag = args.model.split("/")[-1]
    p = RESULTS / f"depth_stability_{tag}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"model": args.model, "dtype": args.dtype,
                             "n_layers": len(out[next(iter(out))]["curve"]),
                             "frames": out}, indent=2))
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
