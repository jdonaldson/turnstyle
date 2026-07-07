"""
ring_denoise.py — are the calendar/hue rings HOLLOW, or only ORDERED?

homology_resweep (calibrated instrument) found ZERO topologically nontrivial
clusters in single-template Mistral-7B word states — months max H1r=0.23 vs
the 0.5 weak floor and the ~2.0 synthetic-circle benchmark. Yet time_ring
finds months parametrically CIRCULAR (0.81@L9). Hypothesis: the ring lives in
a low-D readout plane, and single-template lexical noise off that plane fills
the hole — circular ORDER without Rips-HOLLOWNESS.

Test: average last-subword states over K=20 diverse mention-templates
(noise shrinks ~sqrt(K); the ring plane doesn't). Per layer, per category
(months n=12, weekdays n=7, hue terms n=10):
  - H1 ratio + z on single-template states (control = resweep condition)
  - H1 ratio + z on template-AVERAGED states (the denoise condition)
  - H1 ratio on the averaged states projected to their top-2 PCs
    (maximal denoise: if no hole even in 2D, there is no ring shape at all)
  - circular-order RSA (Spearman vs the P-point circle model) for continuity
    with time_ring's parametric result.

Run:  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python \
        experiments/ring_denoise.py [model_id]
Writes experiments/results/ring_denoise.json.
"""
from __future__ import annotations
import json, os, sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from homology_resweep import spectro, pdist
from time_ring import models, upper

CATS = {
    "month": ["January", "February", "March", "April", "May", "June", "July",
              "August", "September", "October", "November", "December"],
    "weekday": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
                "Saturday", "Sunday"],
    "hue": ["red", "orange", "yellow", "green", "cyan", "blue", "violet",
            "purple", "magenta", "pink"],
}
TEMPLATES = [
    "They talked about {w}.",
    "The report mentioned {w}.",
    "She wrote {w} on the board.",
    "Everyone kept discussing {w}.",
    "He circled {w} on the list.",
    "A note about {w} was left behind.",
    "The word {w} appeared in the article.",
    "Someone whispered {w} during the meeting.",
    "The label said {w}.",
    "I remember {w} clearly.",
    "The answer was {w}.",
    "It all came back to {w}.",
    "The card had {w} printed on it.",
    "Nobody expected {w}.",
    "The teacher underlined {w}.",
    "Their favorite was {w}.",
    "The sign read {w}.",
    "We settled on {w}.",
    "The quiz asked about {w}.",
    "That reminded her of {w}.",
]


def collect_multi(model, tok, device, words, templates):
    """{w: [T, L, H]} last-subword states across templates."""
    import torch
    out = {}
    for w in words:
        per_t = []
        for tmpl in templates:
            sent = tmpl.format(w=w)
            cs = sent.rfind(w); ce = cs + len(w)
            enc = tok(sent, return_offsets_mapping=True, return_tensors="pt")
            offs = enc.pop("offset_mapping")[0].tolist()
            enc = {k: v.to(device) for k, v in enc.items()}
            with torch.no_grad():
                hs = model(**enc, output_hidden_states=True).hidden_states
            stk = torch.stack(hs, 0)[:, 0, :, :]
            idxs = [k for k, (s, e) in enumerate(offs)
                    if e > cs and s < ce] or [-1]
            per_t.append(stk[:, idxs[-1], :].float().cpu().numpy())
        out[w] = np.stack(per_t, 0)
    return out


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    return float((ra @ rb) / (np.linalg.norm(ra) * np.linalg.norm(rb) + 1e-12))


def std(X):
    return (X - X.mean(0)) / (X.std(0) + 1e-6)


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    mid = sys.argv[1] if len(sys.argv) > 1 else "mistralai/Mistral-7B-Instruct-v0.3"
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(mid)
    mdl = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.float16).to(dev)

    results: dict = {"model": mid, "K": len(TEMPLATES), "cats": {}}
    for cat, words in CATS.items():
        acts = collect_multi(mdl, tok, dev, words, TEMPLATES)
        n_layers = acts[words[0]].shape[1]
        P = len(words)
        _, circ = models(P)
        cu = upper(circ)
        rng = np.random.default_rng(0)
        print(f"\n=== {cat} (n={P}, K={len(TEMPLATES)}) ===", flush=True)
        print("  L | single H1r/z | avg H1r/z | pc2 H1r/z | circRSA(single"
              " -> avg)", flush=True)
        rows = {}
        for L in range(n_layers):
            X1 = std(np.array([acts[w][0, L] for w in words],
                              dtype=np.float32))
            Xa = std(np.array([acts[w][:, L].mean(0) for w in words],
                              dtype=np.float32))
            r1 = spectro(X1, rng)
            ra = spectro(Xa, rng)
            Xc = Xa - Xa.mean(0)
            U, S, _ = np.linalg.svd(Xc, full_matrices=False)
            X2 = (U[:, :2] * S[:2]).astype(np.float32)
            r2 = spectro(X2, rng)
            rsa1 = spearman(upper(pdist(X1)), cu)
            rsaa = spearman(upper(pdist(Xa)), cu)
            rows[L] = {"single": r1, "avg": ra, "pc2": r2,
                       "circ_rsa_single": round(rsa1, 2),
                       "circ_rsa_avg": round(rsaa, 2)}
            print(f" {L:2d} | {r1['H1_ratio']:4.2f}/{str(r1['z1']):>5s} | "
                  f"{ra['H1_ratio']:4.2f}/{str(ra['z1']):>5s} | "
                  f"{r2['H1_ratio']:4.2f}/{str(r2['z1']):>5s} | "
                  f"{rsa1:+.2f} -> {rsaa:+.2f}", flush=True)
        results["cats"][cat] = rows

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "results", "ring_denoise.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
