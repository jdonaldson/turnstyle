"""Write-side: are the FrameLibrary frames CAUSAL, not just decodable?

For each frame, build a diff-in-means steering vector from its OWN defining high/low
words (median split), add alpha * |h| * v to the residual at a layer during a forward
pass, and read a SENSITIVE causal metric: the next-token LOGIT-DIFFERENCE between an
INDEPENDENT set of high-pole vs low-pole words (disjoint from the steering words, so
non-circular). If logit-diff rises monotonically with alpha, the frame is a write
instruction. We SWEEP the steering layer — the layer where a property is most decodable
(recoverability peak) is not necessarily where it's most steerable.

opinion is the positive control (Evaluation is known causal, steer_evaluation.py).

Usage: python experiments/steer_frames.py [--frames size,age,opinion] [--scale 10]
"""
from __future__ import annotations

import argparse
import sys
import numpy as np

sys.path.insert(0, "experiments")
from turnstyle.frame_library import load_library

# independent high/low word sets — VERIFIED DISJOINT from CANONICAL_FRAMES training
# words (so the logit-diff metric isn't circular with the steering vector).
LEX = {
    "size": ("vast immense towering colossal giant hulking mammoth bulky".split(),
             "microscopic miniature petite dwarf minute puny diminutive teeny".split()),
    "age": ("vintage archaic timeworn venerable bygone olden decrepit prehistoric".split(),
            "youthful juvenile nascent fledgling infant budding".split()),
    "opinion": ("happy joyful fantastic amazing brilliant superb marvelous splendid".split(),
                "sad miserable dreadful gloomy grim bleak unpleasant disappointing".split()),
    "material": ("hard rigid solid dense sturdy tough firm".split(),
                 "squishy supple downy pliable feathery cushy spongy plush".split()),
}
PROMPTS = ["The object is very", "It felt quite", "Honestly it was rather",
           "I would call it"]
STEER_LAYERS = [3, 6, 9, 12, 15, 18]


def _first_ids(tok, words):
    out = []
    for w in words:
        ids = tok.encode(" " + w, add_special_tokens=False)
        if ids:
            out.append(ids[0])
    return out


def collect(mdl, tok, dev, words, template, layer):
    import torch
    vecs = []
    for w in words:
        sent = template.format(w=w)
        cs = sent.rfind(w); ce = cs + len(w)
        enc = tok(sent, return_offsets_mapping=True, return_tensors="pt")
        offs = enc.pop("offset_mapping")[0].tolist()
        enc = {k: v.to(dev) for k, v in enc.items()}
        with torch.no_grad():
            out = mdl(**enc, output_hidden_states=True)
        idxs = [k for k, (s, e) in enumerate(offs) if e > cs and s < ce] or [-1]
        vecs.append(out.hidden_states[layer][0, idxs[-1]])
    return torch.stack(vecs).mean(0)


def main(frames, scale):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    mid = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(mid)
    mdl = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.float16).to(dev).eval()
    lib = load_library(mdl)

    state = {"alpha": 0.0, "v": None}

    def hook(module, inp, out):
        if state["alpha"] == 0.0 or state["v"] is None:
            return out
        h = out[0] if isinstance(out, tuple) else out
        hf = h.float()                              # fp32 math: fp16 overflows at depth
        hf = hf + state["alpha"] * hf.norm(dim=-1, keepdim=True) * state["v"].float()
        h = hf.to(h.dtype)
        return (h,) + out[1:] if isinstance(out, tuple) else h

    alphas = [-scale, -scale / 2, 0.0, scale / 2, scale]
    enc = [tok(p, return_tensors="pt").to(dev) for p in PROMPTS]

    for fname in frames:
        f = lib.frames[fname]
        vals = sorted(f.data.values())
        med = vals[len(vals) // 2]
        hi = [w for w, v in f.data.items() if v > med]
        lo = [w for w, v in f.data.items() if v < med]
        pos_ids, neg_ids = _first_ids(tok, LEX[fname][0]), _first_ids(tok, LEX[fname][1])
        print(f"\n===== FRAME {fname}  (steer-layer sweep; logit-diff hi-lo, independent) =====")
        best = None
        for L in STEER_LAYERS:
            v = collect(mdl, tok, dev, hi, f.template, L) - \
                collect(mdl, tok, dev, lo, f.template, L)
            state["v"] = v / v.norm()
            handle = mdl.model.layers[L].register_forward_hook(hook)
            diffs = {}
            for a in alphas:
                state["alpha"] = a
                ds = []
                for e in enc:
                    with torch.no_grad():
                        lg = mdl(**e).logits[0, -1].float()
                    ds.append(float(lg[pos_ids].mean() - lg[neg_ids].mean()))
                diffs[a] = float(np.mean(ds))
            handle.remove()
            state["alpha"] = 0.0
            order = [diffs[a] for a in alphas]
            if any(np.isnan(order)):
                print(f"  L{L:<2d}  NaN (fp16 overflow at this depth/scale) — skipped")
                continue
            r = float(np.corrcoef(alphas, order)[0, 1])     # causal score: logitdiff~alpha
            delta = diffs[alphas[-1]] - diffs[alphas[0]]
            print(f"  L{L:<2d}  logitdiff by alpha: " +
                  " ".join(f"{diffs[a]:+5.2f}" for a in alphas) +
                  f"   delta={delta:+.2f}  r(alpha)={r:+.2f}")
            if best is None or r > best[0]:
                best = (r, L, delta)
        if best is not None:
            print(f"  -> best L{best[1]}: r(alpha)={best[0]:+.2f} delta={best[2]:+.2f}  "
                  f"{'CAUSAL' if best[0] > 0.9 else 'weak/none'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", default="opinion,size,age,material")  # adjective frames
    # (number/time omitted: the adjective-eliciting prompts don't fit count/duration words)
    ap.add_argument("--scale", type=float, default=10.0)
    a = ap.parse_args()
    main(a.frames.split(","), a.scale)
