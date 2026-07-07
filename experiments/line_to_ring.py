"""
line_to_ring.py — does the calendar LINE bend into a RING with scale?

Endpoints known: SmolLM2-1.7B = ordinal line at every layer (time is a line;
Dec->Jan seam failure), Mistral-7B = months circular from ~L4, hollow in the
readout plane mid-stack (ring_denoise). The transition is unobserved.

Two hypotheses with different signatures across the 0.36B -> 7B ladder:
  BENDING   intermediate models show intermediate SEAM RATIOS while linear
            RSA stays high — a horseshoe gradually closing.
  DE NOVO   the seam stays wide (line-like) and snaps shut only at the
            scale where the ring appears.

Metrics per model x layer x category (months n=12, weekdays n=7), on
K=20-template-averaged last-subword states (ring_denoise recipe):
  lin / circ   Spearman RSA vs ordinal-line and circle distance models
  seam         d(last, first) / mean d(adjacent)  — the closure metric
               (open line >> 1; closed ring ~ 1). Operationalizes the
               documented SmolLM2 Dec->Jan seam.
  pc2 H1r/z    hollowness in the top-2-PC plane (calibrated spectrometer)

Ladder: SmolLM2-360M, Qwen2.5-1.5B, SmolLM2-1.7B, Phi-4-mini (3.8B),
Mistral-7B — all cached locally from the hub-tier frame work.

Run:  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python \
        experiments/line_to_ring.py
Writes experiments/results/line_to_ring.json.
"""
from __future__ import annotations
import json, os, sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from homology_resweep import spectro, pdist
from ring_denoise import CATS, TEMPLATES, collect_multi, std, spearman
from time_ring import models, upper

MODELS = [
    ("HuggingFaceTB/SmolLM2-360M-Instruct", 0.36),
    ("Qwen/Qwen2.5-1.5B-Instruct", 1.5),
    ("HuggingFaceTB/SmolLM2-1.7B-Instruct", 1.7),
    ("microsoft/Phi-4-mini-instruct", 3.8),
    ("mistralai/Mistral-7B-Instruct-v0.3", 7.0),
]
RINGS = {"month": CATS["month"], "weekday": CATS["weekday"]}


def seam_ratio(D):
    """d(last,first) / mean d(adjacent-in-order). Order = list order."""
    n = len(D)
    adj = np.mean([D[i, i + 1] for i in range(n - 1)])
    return float(D[n - 1, 0] / (adj + 1e-12))


def top_pcs(X, k):
    Xc = X - X.mean(0)
    U, S, _ = np.linalg.svd(Xc, full_matrices=False)
    return (U[:, :k] * S[:k]).astype(np.float32)


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    results: dict = {"K": len(TEMPLATES), "models": {}}
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "results", "line_to_ring.json")

    for mid, bparams in MODELS:
        print(f"\n########## {mid} ({bparams}B) ##########", flush=True)
        tok = AutoTokenizer.from_pretrained(mid)
        mdl = AutoModelForCausalLM.from_pretrained(
            mid, dtype=torch.float16).to(dev)
        mres = {"params_b": bparams, "cats": {}}
        for cat, words in RINGS.items():
            acts = collect_multi(mdl, tok, dev, words, TEMPLATES)
            n_layers = acts[words[0]].shape[1]   # [T, L, H]: layers axis 1
            P = len(words)
            lin_m, circ_m = models(P)
            lu, cu = upper(lin_m), upper(circ_m)
            rng = np.random.default_rng(0)
            print(f"\n=== {cat} (n={P}) ===", flush=True)
            print("  L |  lin  circ | seam seam2 | pc2 H1r/z", flush=True)
            rows = {}
            for L in range(n_layers):
                Xa = std(np.array([acts[w][:, L].mean(0) for w in words],
                                  dtype=np.float32))
                D = pdist(Xa)
                Du = upper(D)
                lin, circ = spearman(Du, lu), spearman(Du, cu)
                seam = seam_ratio(D)
                # in-plane seam: full-space distances concentrate (ratio
                # pinned ~1.1-1.4 for every model); closure is only readable
                # in the plane the ring lives in
                P2 = top_pcs(Xa, 2)
                seam2 = seam_ratio(pdist(P2))
                r2 = spectro(P2, rng)
                rows[L] = {"lin": round(lin, 2), "circ": round(circ, 2),
                           "seam": round(seam, 2), "seam_pc2": round(seam2, 2),
                           "pc2_H1r": r2["H1_ratio"], "pc2_z": r2["z1"],
                           "pc2_cls": r2["cls"]}
                print(f" {L:2d} | {lin:+.2f} {circ:+.2f} | {seam:4.2f} "
                      f"{seam2:5.2f} | {r2['H1_ratio']:4.2f}/"
                      f"{str(r2['z1']):>5s} {r2['cls']}", flush=True)
            # summary: best (min) seam, best circ, best hollow layer
            # seam is only meaningful where circular order exists: report
            # seam_pc2 at the max-circ layer, not min over all layers
            Lc = max(rows, key=lambda L: rows[L]["circ"])
            best_h1 = max(rows.values(), key=lambda r: r["pc2_H1r"])
            print(f"  SUMMARY {cat}: @max-circ L{Lc} "
                  f"circ={rows[Lc]['circ']:+.2f} seam_pc2="
                  f"{rows[Lc]['seam_pc2']:.2f} | "
                  f"max_pc2_H1r={best_h1['pc2_H1r']:.2f}", flush=True)
            mres["cats"][cat] = rows
        results["models"][mid] = mres
        with open(out, "w") as f:
            json.dump(results, f, indent=2)
        del mdl
        if dev == "mps":
            torch.mps.empty_cache()

    print("\n\n===== LADDER SUMMARY (@max-circ layer per model) =====")
    print("model                         B | cat     circ seam_pc2 max_H1r")
    for mid, mres in results["models"].items():
        for cat, rows in mres["cats"].items():
            Lc = max(rows, key=lambda L: rows[L]["circ"])
            mh = max(r["pc2_H1r"] for r in rows.values())
            print(f"{mid.split('/')[-1]:28s} {mres['params_b']:3.1f} | "
                  f"{cat:7s} {rows[Lc]['circ']:+.2f} "
                  f"{rows[Lc]['seam_pc2']:8.2f} {mh:6.2f}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
