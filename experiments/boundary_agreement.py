"""Is the 'agreement' signal genuine stance, or just lexical similarity?

boundary_test.py found agreement (does B's stance match A's) reads at 1.00 in English.
But agree-pairs skew lexically similar, so it might be reading similarity, not stance.
This decorrelates them with a 2x2 of cells:

  valence words across 8 LEXICAL FIELDS, each an antonym pair (good/bad, honest/
  dishonest, brave/cowardly, ...). For a pair (wA, wB):
    agree     = same valence sign
    similar   = same lexical field (antonyms OR same word)

  cells:  same-word (agree+similar)  | antonym (DISAGREE+similar)  ← the killer cell
          cross-field-agree (agree+dissimilar) | cross-field-disagree (disagree+dissim.)

If the agreement axis is high on the ANTONYM cell (lexically close, stance-opposite)
and on cross-field-agree (lexically far, stance-same), it is genuine stance. A pure
lexical-similarity baseline (cosine of the two adjectives) must FAIL on exactly those
cells. The contrast settles it.

English only — the confound is not about language. Usage: [--collect]
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, "experiments")
sys.path.insert(0, "src")
import pole_probe as PP

CACHE = "experiments/data/boundary_agreement_acts.npz"

# field -> (positive, negative); same field = lexically similar (antonyms)
FIELDS = {
    "moral": ("good", "bad"), "honesty": ("honest", "dishonest"),
    "courage": ("brave", "cowardly"), "beauty": ("beautiful", "ugly"),
    "generosity": ("generous", "greedy"), "health": ("healthy", "sick"),
    "safety": ("safe", "dangerous"), "taste": ("delicious", "disgusting"),
}
WORDS = []   # (word, valence, field)
for f, (p, n) in FIELDS.items():
    WORDS.append((p, 1, f)); WORDS.append((n, -1, f))

PAIR_TMPL = "A: It is {a}. B: It is {b}."
ADJ_TMPL = "It is very {w}."


def collect():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(PP.MODEL_ID)
    mdl = AutoModelForCausalLM.from_pretrained(
        PP.MODEL_ID, dtype=torch.float16).to(dev).eval()

    def last_act(sent):
        enc = {k: v.to(dev) for k, v in tok(sent, return_tensors="pt").items()}
        with torch.no_grad():
            out = mdl(**enc, output_hidden_states=True)
        return np.stack([h[0, -1].float().cpu().numpy() for h in out.hidden_states]).astype(np.float16)

    def word_act(w):
        sent = ADJ_TMPL.format(w=w)
        cs = sent.rfind(w); ce = cs + len(w)
        enc = tok(sent, return_offsets_mapping=True, return_tensors="pt")
        offs = enc.pop("offset_mapping")[0].tolist()
        enc = {k: v.to(dev) for k, v in enc.items()}
        with torch.no_grad():
            out = mdl(**enc, output_hidden_states=True)
        tk = next((k for k, (s, e) in enumerate(offs) if e > cs and s < ce), len(offs) - 1)
        return np.stack([h[0, tk].float().cpu().numpy() for h in out.hidden_states]).astype(np.float16)

    # pair activations (end of B) + per-pair metadata
    pacts, agree, similar, cell = [], [], [], []
    for wa, va, fa in WORDS:
        for wb, vb, fb in WORDS:
            pacts.append(last_act(PAIR_TMPL.format(a=wa, b=wb)))
            ag = int(va == vb); sm = int(fa == fb)
            agree.append(ag); similar.append(sm)
            if fa == fb and wa == wb:
                cell.append("same_word")
            elif fa == fb:
                cell.append("antonym")
            elif ag:
                cell.append("xfield_agree")
            else:
                cell.append("xfield_disagree")
    # adjective activations (for the similarity baseline)
    wacts = {w: word_act(w) for w, _, _ in WORDS}
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    np.savez_compressed(
        CACHE, pacts=np.stack(pacts), agree=np.array(agree),
        similar=np.array(similar), cell=np.array(cell),
        wkeys=np.array(list(wacts)), wacts=np.stack(list(wacts.values())),
        pair_a=np.array([wa for wa, _, _ in WORDS for _ in WORDS]),
        pair_b=np.array([wb for _ in WORDS for wb, _, _ in WORDS]))
    print(f"saved {len(pacts)} pairs → {CACHE}")


def analyze():
    d = np.load(CACHE, allow_pickle=True)
    nL = d["pacts"].shape[1]
    agree = d["agree"]; cell = d["cell"]
    pa = d["pair_a"]; pb = d["pair_b"]
    wk = {w: i for i, w in enumerate(d["wkeys"].tolist())}
    cells = ["same_word", "antonym", "xfield_agree", "xfield_disagree"]
    print(f"N={len(agree)} pairs   cells: " +
          ", ".join(f"{c}={int((cell==c).sum())}" for c in cells))
    print("\nAGREEMENT AXIS (fit on a random half, eval per cell on the other half):")
    print(f"{'L':>3} {'overall':>8} | " + " ".join(f"{c[:9]:>11}" for c in cells))
    best = (0.0, -1)
    rng = np.random.RandomState(0)
    tr = rng.rand(len(agree)) < 0.5
    for L in range(nL):
        X = d["pacts"][:, L].astype(np.float32)
        mu = X[tr].mean(0); sd = X[tr].std(0) + 1e-6
        Z = (X - mu) / sd
        hi = Z[tr & (agree == 1)].mean(0); lo = Z[tr & (agree == 0)].mean(0)
        dd = (hi - lo) / (np.linalg.norm(hi - lo) + 1e-12)
        c = 0.5 * (hi + lo) @ dd
        pred = (Z @ dd - c) > 0
        te = ~tr
        ov = (pred[te] == (agree[te] == 1)).mean()
        cellacc = []
        for cc in cells:
            m = te & (cell == cc)
            cellacc.append((pred[m] == (agree[m] == 1)).mean() if m.any() else float("nan"))
        if ov > best[0]:
            best = (ov, L)
        print(f"{L:>3} {ov:>8.2f} | " + " ".join(f"{a:>11.2f}" for a in cellacc))

    # similarity baseline at the best layer: predict agree if cos(adjA,adjB) high
    L = best[1]
    W = d["wacts"][:, L].astype(np.float32)
    Wn = W / (np.linalg.norm(W, axis=1, keepdims=True) + 1e-12)
    sim = np.array([float(Wn[wk[a]] @ Wn[wk[b]]) for a, b in zip(pa, pb)])
    thr = np.median(sim)
    simpred = sim > thr
    print(f"\nLEXICAL-SIMILARITY BASELINE @L{L} (predict agree if cos>median):")
    print(f"  overall {np.mean(simpred == (agree == 1)):.2f} | " +
          " ".join(f"{c[:9]}={np.mean(simpred[cell==c]==(agree[cell==c]==1)):.2f}" for c in cells))
    print(f"\nagreement axis best overall L{best[1]}={best[0]:.2f}")
    print("genuine stance ⇒ axis HIGH on antonym + xfield_agree where similarity FAILS")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--collect", action="store_true")
    args = ap.parse_args()
    if args.collect or not os.path.exists(CACHE):
        collect()
    analyze()
