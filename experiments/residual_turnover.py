"""Why are Phi-4-mini's frame curves FLAT? Does its residual stream turn over less?

Phi-4-mini is the consistent oddball ([[crosslingual_bottleneck_universal]]):
1/14 citable frame peaks against Qwen's 6/14, on a frame lexicon its tokenizer
fragments identically (1.11 vs 1.11 tok/word).  So the flatness is not a
tokenization artifact and needs a different explanation.

Phi is flat-HIGH, not flat-low: worst-layer CV r 0.691 against a peak of 0.915 --
frames are readable at EVERY depth.  That is exactly what you would see if the
residual stream barely rotates through the stack: a feature written early is still
present late, so no layer is privileged and every readout works about equally.

Measures turnover directly, with no lexicon and no probe:

  adjacent   mean cos(h_L, h_L+1)  -- how much one layer changes the state
  to_final   cos(h_L, h_last)      -- how early the final state is determined
  from_input cos(h_L, h_0)         -- how long the embedding persists

Prediction if turnover explains flatness: Phi has the HIGHEST adjacent cosine and
the earliest-rising to_final curve, and these should rank-order the six models the
same way frame localization does (Mistral most localized / lowest cosine, Phi least
localized / highest cosine).

Note this is a WEAKER claim than it looks: turnover and localization would be
correlated, not causally linked, and n=6 cannot separate them. A null here kills
the hypothesis cleanly; a positive is suggestive only.

    uv run python experiments/residual_turnover.py

Run OUTSIDE the sandbox (torch misreports MPS as unavailable inside it).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

RESULTS = Path(__file__).parent / "results"

# Neutral, varied sentences -- turnover should be a property of the stack, not of
# one prompt, so average over several.
SENTS = [
    "The enormous ancient stone bridge collapsed last winter.",
    "She quietly placed the small red book on the table.",
    "Scientists announced the discovery of a new species yesterday.",
    "It is undeniable that the situation has become complicated.",
    "The heavy iron anvil sank immediately into the soft mud.",
]

MODELS = [
    ("mistralai/Mistral-7B-Instruct-v0.3", "bfloat16"),
    ("microsoft/Phi-4-mini-instruct", "bfloat16"),
    ("Qwen/Qwen2.5-1.5B-Instruct", "float32"),
    ("HuggingFaceTB/SmolLM2-1.7B-Instruct", "float32"),
    ("HuggingFaceTB/SmolLM2-360M-Instruct", "float32"),
]

# frame localization (mean spread) from experiments/results/depth_stability_*.json
FRAME_LOC = {
    "Mistral-7B-Instruct-v0.3": 0.469,
    "Qwen2.5-1.5B-Instruct": 0.339,
    "SmolLM2-360M-Instruct": 0.275,
    "SmolLM2-1.7B-Instruct": 0.249,
    "Phi-4-mini-instruct": 0.221,
}


def cos(a, b):
    a, b = a.astype(np.float64), b.astype(np.float64)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def turnover(model, tok, dev):
    adj, tofin, frominp = [], [], []
    for s in SENTS:
        enc = tok(s, return_tensors="pt").to(dev)
        with torch.no_grad():
            hs = model(**enc, output_hidden_states=True).hidden_states
        v = [h[0, -1].float().cpu().numpy() for h in hs]
        adj.append([cos(v[i], v[i + 1]) for i in range(len(v) - 1)])
        tofin.append([cos(v[i], v[-1]) for i in range(len(v))])
        frominp.append([cos(v[i], v[0]) for i in range(len(v))])
    return (np.mean(adj, 0), np.mean(tofin, 0), np.mean(frominp, 0))


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    out = {}
    print(f"  {'model':26s} {'frame loc':>9s} {'adj cos':>8s} {'half-way to final':>18s} "
          f"{'input persists to':>18s}")
    for mid, dt in MODELS:
        tok = AutoTokenizer.from_pretrained(mid)
        m = AutoModelForCausalLM.from_pretrained(
            mid, dtype=getattr(torch, dt)).to(dev).eval()
        a, tf, fi = turnover(m, tok, dev)
        tag = mid.split("/")[-1]
        n = len(tf)
        # depth at which the state first reaches cos 0.5 with the final state, and
        # the depth at which it last stays above cos 0.5 with the embedding
        half = next((i / (n - 1) for i in range(n) if tf[i] >= 0.5), 1.0)
        keep = max((i / (n - 1) for i in range(n) if fi[i] >= 0.5), default=0.0)
        out[tag] = {"adjacent": a.tolist(), "to_final": tf.tolist(),
                    "from_input": fi.tolist(), "adj_mean": float(np.mean(a)),
                    "half_to_final": half, "input_persists_to": keep,
                    "frame_loc": FRAME_LOC.get(tag)}
        print(f"  {tag:26s} {FRAME_LOC.get(tag, float('nan')):>9.3f} "
              f"{np.mean(a):>8.3f} {half:>17.0%} {keep:>17.0%}", flush=True)
        del m

    tags = [t for t in out if out[t]["frame_loc"] is not None]
    fl = np.array([out[t]["frame_loc"] for t in tags])
    am = np.array([out[t]["adj_mean"] for t in tags])
    hf = np.array([out[t]["half_to_final"] for t in tags])
    print(f"\n  corr(frame localization, adjacent cos)   = {np.corrcoef(fl, am)[0, 1]:+.3f}")
    print(f"  corr(frame localization, half-to-final)  = {np.corrcoef(fl, hf)[0, 1]:+.3f}")
    print("\n  prediction was: LOW localization <-> HIGH adjacent cos, i.e. a")
    print("  strongly NEGATIVE first correlation. Phi should sit at the high-cos end.")
    rank = sorted(tags, key=lambda t: -out[t]["adj_mean"])
    print(f"  adjacent-cos ranking (highest first): {rank}")

    p = RESULTS / "residual_turnover.json"
    p.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
