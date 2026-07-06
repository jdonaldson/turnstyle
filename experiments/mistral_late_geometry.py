"""
mistral_late_geometry.py — what geometric structure ASSEMBLES in the late stack?

Motivated by the weekday ring on Mistral-7B: linear through L24, bent into a
near-perfect circle by L30 — while the MONTH ring peaks mid-stack (L9) and
degrades after L21. Is late-stack ring assembly a one-off, or does other
structure form there? Two parts:

A. RING TESTS on more cyclic domains (time_ring's line-vs-circle machinery):
     clock hours (mod 12 arithmetic is real), compass 8-points (allocentric
     frame was clean mid-stack in SmolLM2), musical keys (pitch class is
     genuinely modular), zodiac (calendar-attached), seasons (P=4, weak n).
   Report the winner per layer band — specifically whether circularity EMERGES
   late (weekday pattern) vs mid (month pattern) vs never.

B. SCALAR CURVES for all 14 canonical frames: full per-layer CV curves (the
   library fit only saved peaks). Classify each frame: EARLY-flat, MID, or
   LATE-ASSEMBLED (peak in last third AND rising ≥0.08 from the mid-stack
   plateau — 'peaks late' alone can't distinguish assembly from persistence).

Run:  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python \
        experiments/mistral_late_geometry.py [model_id]
Writes experiments/results/mistral_late_geometry.json.
"""
from __future__ import annotations
import json, os, sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from time_ring import collect_last as ring_collect, analyze
from ordering_frames import cv_r
from freq_audit_family import collect_last_t
from turnstyle.frame_library import CANONICAL_FRAMES

RING_SETS = {
    "clock_hours": (["one", "two", "three", "four", "five", "six", "seven",
                     "eight", "nine", "ten", "eleven", "twelve"], 12,
                    "It happened at {w} o'clock."),
    "compass8": (["north", "northeast", "east", "southeast", "south",
                  "southwest", "west", "northwest"], 8,
                 "They headed {w}."),
    "musical_keys": (["A", "B", "C", "D", "E", "F", "G"], 7,
                     "The song was in the key of {w}."),
    "zodiac": (["Aries", "Taurus", "Gemini", "Cancer", "Leo", "Virgo", "Libra",
                "Scorpio", "Sagittarius", "Capricorn", "Aquarius", "Pisces"], 12,
               "They were born under {w}."),
    "seasons": (["spring", "summer", "autumn", "winter"], 4,
                "It happened in {w}."),
}


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    mid = sys.argv[1] if len(sys.argv) > 1 else "mistralai/Mistral-7B-Instruct-v0.3"
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(mid)
    mdl = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.float16).to(dev)
    print(f"model {mid} on {dev}", flush=True)

    results = {"model": mid, "rings": {}, "frames": {}}

    # ── A: ring tests ──────────────────────────────────────────────────────────
    for name, (words, P, tmpl) in RING_SETS.items():
        acts = ring_collect(mdl, tok, dev, words, tmpl)
        n_layers = acts[words[0]].shape[0]
        rows = {}
        best_circ = (-9, -1)
        print(f"\n=== ring: {name} (P={P}) ===   L: r_lin  r_circ  winner", flush=True)
        for L in range(0, n_layers, 3):
            r_lin, r_circ, wrap_rank, n_pairs, adj = analyze(acts, words, P, L)
            rows[L] = {"lin": round(float(r_lin), 3), "circ": round(float(r_circ), 3),
                       "wrap_rank": f"{wrap_rank}/{n_pairs}", "nn_adj": round(adj, 2)}
            if r_circ > best_circ[0]: best_circ = (float(r_circ), L)
            win = "CIRC" if r_circ > r_lin else "lin"
            print(f"  L{L:2d}: {r_lin:+.2f}  {r_circ:+.2f}  {win}  "
                  f"wrap {wrap_rank}/{n_pairs}  adj {adj:.2f}", flush=True)
        results["rings"][name] = {"by_layer": rows,
                                  "best_circ": {"r": round(best_circ[0], 3),
                                                "layer": best_circ[1]}}

    # ── B: full scalar curves for the canonical frames ─────────────────────────
    print("\n=== frame curves: EARLY-flat / MID / LATE-ASSEMBLED ===", flush=True)
    for fname, spec in CANONICAL_FRAMES.items():
        words = list(spec["data"])
        y = np.array([spec["data"][w] for w in words], dtype=float)
        tmpl = spec.get("template", "It is a {w} object.")
        acts = collect_last_t(mdl, tok, dev, words, tmpl)
        n_layers = acts[words[0]].shape[0]
        curve = [round(float(cv_r(np.array([acts[w][L] for w in words]), y)), 3)
                 for L in range(n_layers)]
        peak_L = int(np.argmax(curve)); peak_r = curve[peak_L]
        third = n_layers // 3
        mid_plateau = max(curve[third:2 * third])          # best mid-stack value
        late_peak = max(curve[2 * third:])
        assembled = peak_L >= 2 * third and (late_peak - mid_plateau) >= 0.08
        cls = ("LATE-ASSEMBLED" if assembled
               else "late-flat" if peak_L >= 2 * third
               else "mid" if peak_L >= third else "early")
        results["frames"][fname] = {"curve": curve, "peak": {"r": peak_r, "layer": peak_L},
                                    "mid_plateau": mid_plateau, "class": cls}
        print(f"  {fname:13s} peak {peak_r:+.2f}@L{peak_L:<3d} mid-plateau "
              f"{mid_plateau:+.2f}  -> {cls}", flush=True)

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "results", "mistral_late_geometry.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
