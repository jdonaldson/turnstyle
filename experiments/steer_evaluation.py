"""Steering test: is the Evaluation axis CAUSAL, or only decodable?

A direction can decode a property without controlling it. This adds a diff-in-means
Evaluation steering vector to the residual stream at one layer during generation and
checks whether output sentiment moves MONOTONICALLY with the coefficient α — scored by
an INDEPENDENT sentiment lexicon (not the eval axis, to avoid circularity).

If sentiment tracks α, the axis is causal and "semantic programming" gets a write
instruction. If generations stay flat (or break into noise), it's decodable-not-steerable.

Usage:  python experiments/steer_evaluation.py [--layer L] [--scale S]
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

sys.path.insert(0, "experiments")
sys.path.insert(0, "src")
import pole_probe as PP
from osgood_epa import EPA

# independent sentiment lexicon (distinct from the eval pole words used to fit v)
POS = set("great wonderful happy love joy excellent amazing delightful perfect best "
          "fantastic enjoyed lovely brilliant warm fun cheerful gorgeous superb thrilled "
          "glad grateful peaceful hopeful bright wonder excitement excited beautiful good "
          "nice friendly welcoming charming cozy comfortable success successful enjoy "
          "pleasant vibrant thriving wonderful-ly".split())
NEG = set("terrible awful sad hate horrible disgusting miserable pain worst boring "
          "disappointing angry fear cruel dull broken sick tired gloomy bleak grim "
          "afraid lonely hopeless dark ruined war zone corrupt incompetent brutal worse "
          "losing lost screw nightmare violence dangerous crime poverty filthy dirty "
          "depressing toxic hostile struggling failing".split())

PROMPTS = [
    "Last weekend I went to the new cafe downtown, and",
    "My coworker told me about the meeting, and honestly it was",
    "When I think about how the year is going, I feel",
    "The neighborhood I live in is",
]


def sentiment(text):
    ws = [w.strip(".,!?;:\"'").lower() for w in text.split()]
    p = sum(w in POS for w in ws); n = sum(w in NEG for w in ws)
    return (p - n) / max(1, len(ws)), p, n


def main(layer, scale):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(PP.MODEL_ID)
    mdl = AutoModelForCausalLM.from_pretrained(
        PP.MODEL_ID, dtype=torch.float16).to(dev).eval()

    # diff-in-means Evaluation steering vector at `layer` (adjective token)
    def adj_vec(w):
        sent = f"It is very {w}."
        cs = sent.rfind(w); ce = cs + len(w)
        enc = tok(sent, return_offsets_mapping=True, return_tensors="pt")
        offs = enc.pop("offset_mapping")[0].tolist()
        enc = {k: v.to(dev) for k, v in enc.items()}
        with torch.no_grad():
            out = mdl(**enc, output_hidden_states=True)
        tk = next((k for k, (s, e) in enumerate(offs) if e > cs and s < ce), -1)
        return out.hidden_states[layer][0, tk]

    pos = torch.stack([adj_vec(w) for w in EPA["evaluation"]["en"]["hi"]]).mean(0)
    neg = torch.stack([adj_vec(w) for w in EPA["evaluation"]["en"]["lo"]]).mean(0)
    v = (pos - neg)
    v = v / v.norm()                      # unit direction; α carries the magnitude

    layers = mdl.model.layers
    state = {"alpha": 0.0}

    def hook(module, inp, out):
        if state["alpha"] == 0.0:
            return out
        h = out[0] if isinstance(out, tuple) else out
        # norm-relative: α is a fraction of each token's residual magnitude
        h = h + state["alpha"] * h.norm(dim=-1, keepdim=True) * v.to(h.dtype)
        return (h,) + out[1:] if isinstance(out, tuple) else h

    handle = layers[layer].register_forward_hook(hook)

    alphas = [-scale, -scale / 2, 0.0, scale / 2, scale]
    print(f"steering Evaluation @L{layer}, |v|=unit, scale={scale}\n")
    try:
        for pr in PROMPTS:
            print(f"PROMPT: {pr}")
            for a in alphas:
                state["alpha"] = a
                enc = tok(pr, return_tensors="pt").to(dev)
                with torch.no_grad():
                    g = mdl.generate(**enc, max_new_tokens=32, do_sample=False,
                                     repetition_penalty=1.3, pad_token_id=tok.eos_token_id)
                txt = tok.decode(g[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
                s, p, n = sentiment(txt)
                print(f"  α={a:+5.1f}  sent={s:+.3f} (+{p}/-{n})  {txt.strip()[:90]}")
            print()
    finally:
        handle.remove()

    # monotonicity summary across prompts
    print("mean sentiment by α (should increase with α if causal):")
    for a in alphas:
        state["alpha"] = a
        scores = []
        for pr in PROMPTS:
            enc = tok(pr, return_tensors="pt").to(dev)
            handle = layers[layer].register_forward_hook(hook)
            with torch.no_grad():
                g = mdl.generate(**enc, max_new_tokens=32, do_sample=False,
                                 repetition_penalty=1.3, pad_token_id=tok.eos_token_id)
            handle.remove()
            scores.append(sentiment(tok.decode(g[0][enc["input_ids"].shape[1]:], skip_special_tokens=True))[0])
        print(f"  α={a:+5.1f}  mean_sent={np.mean(scores):+.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=13)
    ap.add_argument("--scale", type=float, default=10.0)
    args = ap.parse_args()
    main(args.layer, args.scale)
