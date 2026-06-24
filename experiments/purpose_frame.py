"""Is 'purpose' (the last adjective-ordering rung) a frame — and what KIND?

purpose is categorical-functional (cooking/racing/hunting/sleeping/cleaning/gardening),
not a scalar with a high/low pole, so the bipolar-axis machinery may not fit. Tests:
  (a) categorical recoverability — can a multiclass nearest-centroid probe separate the
      purpose categories? (5-fold CV acc vs chance 1/K), layer sweep.
  (b) effective dimensionality of the category centroids (participation ratio) — bipolar
      frames are ~1D; if purpose needs many dims it's categorical, not a single axis.
  (c) a candidate bipolar SUB-axis (active↔passive purpose): recoverability + causal
      steering (logit-diff, disjoint metric) — does purpose have a steerable scalar at
      all, or is that just the Activity affect axis in disguise?
"""
from __future__ import annotations

import sys
import numpy as np

sys.path.insert(0, "experiments")

CATS = {
    "cooking": "cooking baking roasting frying boiling grilling".split(),
    "racing": "racing sprinting drifting rallying speeding overtaking".split(),
    "hunting": "hunting tracking stalking trapping shooting fishing".split(),
    "sleeping": "sleeping napping resting dozing slumbering relaxing".split(),
    "cleaning": "cleaning washing scrubbing mopping dusting wiping".split(),
    "gardening": "gardening planting weeding pruning watering digging".split(),
}
TMPL = "It is used for {w}."
MODEL = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
# (c) active vs passive purpose sub-axis
ACTIVE = "racing sprinting hunting climbing fighting jumping".split()
PASSIVE = "sleeping resting napping storing waiting sitting".split()
# disjoint independent metric for the steer
ACT_HI = "running chasing leaping racing-fast vigorous".split()
ACT_LO = "lying idle still motionless dormant".split()
PROMPTS = ["The activity was", "It mostly involves", "All day it kept"]


def first_ids(tok, ws):
    return [i[0] for w in ws if (i := tok.encode(" " + w, add_special_tokens=False))]


def collect(mdl, tok, dev, words, layer=None):
    import torch
    out = []
    for w in words:
        sent = TMPL.format(w=w); cs = sent.rfind(w); ce = cs + len(w)
        enc = tok(sent, return_offsets_mapping=True, return_tensors="pt")
        offs = enc.pop("offset_mapping")[0].tolist()
        enc = {k: v.to(dev) for k, v in enc.items()}
        with torch.no_grad():
            hs = mdl(**enc, output_hidden_states=True).hidden_states
        idxs = [k for k, (s, e) in enumerate(offs) if e > cs and s < ce] or [-1]
        st = torch.stack(hs, 0)[:, 0, idxs[-1], :]
        out.append(st if layer is None else st[layer])
    return out


def multiclass_cv(Xc, labels, k=5):
    """nearest-centroid leave-fold-out accuracy."""
    import numpy as np
    n = len(labels); idx = np.random.default_rng(0).permutation(n)
    folds = np.array_split(idx, k); correct = 0
    for i in range(k):
        te = set(folds[i].tolist()); tr = [j for j in range(n) if j not in te]
        cents = {}
        for c in set(labels):
            rows = [Xc[j] for j in tr if labels[j] == c]
            if rows:
                cents[c] = np.mean(rows, 0)
        for j in folds[i]:
            pred = min(cents, key=lambda c: np.linalg.norm(Xc[j] - cents[c]))
            correct += (pred == labels[j])
    return correct / n


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL)
    mdl = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16).to(dev).eval()
    nL = mdl.config.num_hidden_layers
    layers = sorted({max(1, int(d * nL)) for d in (0.12, 0.25, 0.4, 0.5, 0.65, 0.75)})

    words = [w for c in CATS for w in CATS[c]]
    labels = [c for c in CATS for _ in CATS[c]]
    allL = collect(mdl, tok, dev, words)                     # (L+1,H) per word
    chance = 1.0 / len(CATS)

    print(f"(a) categorical recoverability (nearest-centroid 5-fold; chance={chance:.2f})")
    best = (0, 0)
    for L in layers:
        X = np.array([allL[i][L].float().cpu().numpy() for i in range(len(words))])
        acc = multiclass_cv(X, labels)
        if acc > best[0]:
            best = (acc, L)
        print(f"    L{L:<2d} acc={acc:.2f}")
    print(f"    -> best {best[0]:.2f} @L{best[1]}  ({best[0]/chance:.1f}x chance)")

    # (b) effective dimensionality of the 6 category centroids at best layer
    Xb = np.array([allL[i][best[1]].float().cpu().numpy() for i in range(len(words))])
    cents = np.array([Xb[[j for j in range(len(words)) if labels[j] == c]].mean(0)
                      for c in CATS])
    cc = cents - cents.mean(0)
    lam = np.linalg.svd(cc, compute_uv=False) ** 2
    pr = float(lam.sum() ** 2 / (lam ** 2).sum())
    print(f"\n(b) effective dim of {len(CATS)} category centroids: PR={pr:.2f} "
          f"(bipolar frame ~1; categorical >>1)")

    # (c) active vs passive purpose sub-axis: recoverability + steering
    print("\n(c) active<->passive purpose sub-axis")
    state = {"alpha": 0.0, "v": None}

    def hook(m, i, o):
        if state["alpha"] == 0.0 or state["v"] is None:
            return o
        h = o[0] if isinstance(o, tuple) else o
        hf = h.float() + state["alpha"] * h.float().norm(dim=-1, keepdim=True) * state["v"].float()
        h = hf.to(h.dtype)
        return (h,) + o[1:] if isinstance(o, tuple) else h

    pos_ids, neg_ids = first_ids(tok, ACT_HI), first_ids(tok, ACT_LO)
    enc = [tok(p, return_tensors="pt").to(dev) for p in PROMPTS]
    alphas = [-4, -2, 0.0, 2, 4]
    best_steer = None
    for L in layers:
        hi = torch.stack(collect(mdl, tok, dev, ACTIVE, L)).mean(0)
        lo = torch.stack(collect(mdl, tok, dev, PASSIVE, L)).mean(0)
        state["v"] = (hi - lo) / (hi - lo).norm()
        handle = mdl.model.layers[L].register_forward_hook(hook)
        diffs = []
        for a in alphas:
            state["alpha"] = a
            ds = []
            for e in enc:
                with torch.no_grad():
                    lg = mdl(**e).logits[0, -1].float()
                ds.append(float(lg[pos_ids].mean() - lg[neg_ids].mean()))
            diffs.append(float(np.mean(ds)))
        handle.remove(); state["alpha"] = 0.0
        if any(np.isnan(diffs)):
            continue
        r = float(np.corrcoef(alphas, diffs)[0, 1]); delta = diffs[-1] - diffs[0]
        if best_steer is None or delta > best_steer[0]:
            best_steer = (delta, r, L)
    print(f"    steer active-vs-passive: best delta={best_steer[0]:+.2f} "
          f"r={best_steer[1]:+.2f} @L{best_steer[2]}")


if __name__ == "__main__":
    main()
