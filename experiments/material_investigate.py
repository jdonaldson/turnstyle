"""Why is the material frame weakly steerable on SmolLM2 (vs strong on Phi)?

Tests whether the weakness is the SCALAR FRAMING (maybe SmolLM2 encodes material along a
different sub-axis than 'hardness') or material being decodable-but-not-causal. Three
framings — hardness, naturalness (natural vs synthetic), density (heavy vs light) — each
scored for RECOVERABILITY (shuffled CV r) and STEERABILITY (logit-diff causal r + delta,
non-circular independent metric). If one framing steers cleanly, the canonical
material=hardness scalar was just the wrong axis for this model.
"""
from __future__ import annotations

import sys
import numpy as np

sys.path.insert(0, "experiments")

FRAMINGS = {
    "hardness": {
        "data": {"soft": 0, "woolen": 0, "fluffy": 0, "papery": 1, "wooden": 2,
                 "plastic": 2, "glassy": 3, "ceramic": 3, "stony": 4, "metallic": 5,
                 "iron": 5, "golden": 5},
        "tmpl": "It is a {w} object.",
        "hi": "hard rigid solid dense sturdy tough firm".split(),
        "lo": "squishy supple downy pliable feathery cushy spongy".split()},
    "naturalness": {
        "data": {"cotton": 1, "woolen": 1, "wooden": 1, "stone": 1, "leather": 1,
                 "silk": 1, "linen": 1, "plastic": 0, "nylon": 0, "polyester": 0,
                 "acrylic": 0, "vinyl": 0, "rubber": 0, "synthetic": 0},
        "tmpl": "It is made of {w}.",
        "hi": "natural organic earthy".split(),
        "lo": "artificial manmade fake".split()},
    "density": {
        "data": {"foam": 0, "feather": 0, "paper": 0, "cork": 0, "balsa": 0,
                 "lead": 1, "steel": 1, "gold": 1, "iron": 1, "granite": 1,
                 "concrete": 1},
        "tmpl": "It is made of {w}.",
        "hi": "heavy dense leaden weighty hefty".split(),
        "lo": "light airy weightless featherlight".split()},
}
PROMPTS = ["The object is very", "It felt quite", "Honestly it was rather", "I would call it"]
MODEL = "HuggingFaceTB/SmolLM2-1.7B-Instruct"


def _first_ids(tok, words):
    return [ids[0] for w in words if (ids := tok.encode(" " + w, add_special_tokens=False))]


def collect(mdl, tok, dev, words, template, layer=None):
    import torch
    out = []
    for w in words:
        sent = template.format(w=w)
        cs = sent.rfind(w); ce = cs + len(w)
        enc = tok(sent, return_offsets_mapping=True, return_tensors="pt")
        offs = enc.pop("offset_mapping")[0].tolist()
        enc = {k: v.to(dev) for k, v in enc.items()}
        with torch.no_grad():
            hs = mdl(**enc, output_hidden_states=True).hidden_states
        idxs = [k for k, (s, e) in enumerate(offs) if e > cs and s < ce] or [-1]
        st = torch.stack(hs, 0)[:, 0, idxs[-1], :]
        out.append(st if layer is None else st[layer])
    return out


def cv_r(X, y):
    from sklearn.linear_model import RidgeCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_predict, KFold
    est = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(0, 5, 11)))
    pred = cross_val_predict(est, X, y, cv=KFold(5, shuffle=True, random_state=0))
    return float(np.corrcoef(pred, y)[0, 1])


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL)
    mdl = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16).to(dev).eval()
    n_layers = mdl.config.num_hidden_layers
    layers = sorted({max(1, int(d * n_layers)) for d in (0.12, 0.25, 0.4, 0.5, 0.65, 0.75)})

    state = {"alpha": 0.0, "v": None}

    def hook(module, inp, out):
        if state["alpha"] == 0.0 or state["v"] is None:
            return out
        h = out[0] if isinstance(out, tuple) else out
        hf = h.float() + state["alpha"] * h.float().norm(dim=-1, keepdim=True) * state["v"].float()
        h = hf.to(h.dtype)
        return (h,) + out[1:] if isinstance(out, tuple) else h

    alphas = [-4, -2, 0.0, 2, 4]
    enc = [tok(p, return_tensors="pt").to(dev) for p in PROMPTS]

    for name, spec in FRAMINGS.items():
        words = list(spec["data"]); y = np.array([spec["data"][w] for w in words], float)
        allL = collect(mdl, tok, dev, words, spec["tmpl"])         # (L+1,H) per word
        # recoverability
        recov = max(cv_r(np.array([allL[i][L].float().cpu().numpy() for i in range(len(words))]), y)
                    for L in layers)
        # steerability (mean split — works for binary 0/1 framings too)
        thr = float(y.mean())
        hi = [w for w, v in spec["data"].items() if v > thr]
        lo = [w for w, v in spec["data"].items() if v < thr]
        pos_ids, neg_ids = _first_ids(tok, spec["hi"]), _first_ids(tok, spec["lo"])
        best = None
        for L in layers:
            v = (torch.stack(collect(mdl, tok, dev, hi, spec["tmpl"], L)).mean(0) -
                 torch.stack(collect(mdl, tok, dev, lo, spec["tmpl"], L)).mean(0))
            state["v"] = v / v.norm()
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
            if best is None or delta > best[0]:
                best = (delta, r, L)
        print(f"{name:12s} recoverability={recov:+.3f}   steer: "
              f"best delta={best[0]:+.2f} r={best[1]:+.2f} @L{best[2]}")


if __name__ == "__main__":
    main()
