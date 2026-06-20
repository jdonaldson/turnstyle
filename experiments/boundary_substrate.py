"""Falsifier for the upgraded thesis: do relations INHERIT substrate cleanliness?

Agreement-on-Evaluation was genuine stance (boundary_agreement.py, antonym cell 1.00).
Thesis: that worked because Evaluation is the model's cleanest substrate. Prediction:
the SAME agreement computation on NOISIER substrates degrades — and specifically the
antonym cell (the genuine-relational test) tracks how cleanly the substrate's pole is
read.

Three substrates, decreasing cleanliness (from osgood_epa): Evaluation > Activity >
Potency. For each: (a) pole-axis held-out LOO sign = substrate cleanliness; (b) the
agreement 2x2 (8 antonym fields, fit axis, per-cell) = relational quality. If the
antonym-cell accuracy tracks pole cleanliness, the thesis holds.

Usage:  python experiments/boundary_substrate.py [--collect]
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, "experiments")
sys.path.insert(0, "src")
import pole_probe as PP

CACHE = "experiments/data/boundary_substrate_acts.npz"

# substrate -> {field: (high_word, low_word)}  (high = +1 pole)
SUBSTRATES = {
    "evaluation": {"moral": ("good", "bad"), "honesty": ("honest", "dishonest"),
                   "courage": ("brave", "cowardly"), "beauty": ("beautiful", "ugly"),
                   "generosity": ("generous", "greedy"), "health": ("healthy", "sick"),
                   "safety": ("safe", "dangerous"), "taste": ("delicious", "disgusting")},
    "activity":   {"speed": ("fast", "slow"), "engagement": ("active", "passive"),
                   "liveliness": ("lively", "dull"), "pace": ("quick", "sluggish"),
                   "energy": ("energetic", "lethargic"), "busyness": ("busy", "idle"),
                   "motion": ("dynamic", "static"), "agility": ("agile", "clumsy")},
    "potency":    {"strength": ("strong", "weak"), "size": ("big", "small"),
                   "weight": ("heavy", "light"), "hardness": ("hard", "soft"),
                   "magnitude": ("large", "tiny"), "thickness": ("thick", "thin"),
                   "depth": ("deep", "shallow"), "toughness": ("tough", "fragile")},
}
PAIR_TMPL = "A: It is {a}. B: It is {b}."
ADJ_TMPL = "It is very {w}."


def _words(sub):
    out = []
    for f, (hi, lo) in SUBSTRATES[sub].items():
        out.append((hi, 1, f)); out.append((lo, -1, f))
    return out


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

    save = {}
    for sub in SUBSTRATES:
        ws = _words(sub)
        pacts, agree, cell, pa, pb = [], [], [], [], []
        for wa, va, fa in ws:
            for wb, vb, fb in ws:
                pacts.append(last_act(PAIR_TMPL.format(a=wa, b=wb)))
                ag = int(va == vb)
                agree.append(ag); pa.append(wa); pb.append(wb)
                cell.append("same_word" if (fa == fb and wa == wb)
                            else "antonym" if fa == fb
                            else "xfield_agree" if ag else "xfield_disagree")
        wkeys = [w for w, _, _ in ws]
        save[f"{sub}_pacts"] = np.stack(pacts)
        save[f"{sub}_agree"] = np.array(agree)
        save[f"{sub}_cell"] = np.array(cell)
        save[f"{sub}_wkeys"] = np.array(wkeys)
        save[f"{sub}_pole"] = np.array([v for _, v, _ in ws])
        save[f"{sub}_wacts"] = np.stack([word_act(w) for w in wkeys])
        print(f"  {sub}: {len(pacts)} pairs", flush=True)
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    np.savez_compressed(CACHE, **save)
    print(f"saved → {CACHE}")


def _axis(hi, lo):
    d = hi.mean(0) - lo.mean(0)
    d = d / (np.linalg.norm(d) + 1e-12)
    return d, 0.5 * (hi.mean(0) + lo.mean(0)) @ d


def analyze():
    d = np.load(CACHE, allow_pickle=True)
    nL = d["evaluation_pacts"].shape[1]
    rows = []
    for sub in SUBSTRATES:
        W = d[f"{sub}_wacts"].astype(np.float32); pole = d[f"{sub}_pole"]
        P = d[f"{sub}_pacts"]; agree = d[f"{sub}_agree"]; cell = d[f"{sub}_cell"]
        # pick the best agreement layer for this substrate by overall held-out
        rng = np.random.RandomState(0); tr = rng.rand(len(agree)) < 0.5
        best = (0.0, -1)
        for L in range(nL):
            X = P[:, L].astype(np.float32)
            mu = X[tr].mean(0); sd = X[tr].std(0) + 1e-6
            Z = (X - mu) / sd
            ax, c = _axis(Z[tr & (agree == 1)], Z[tr & (agree == 0)])
            ov = (((Z[~tr] @ ax - c) > 0) == (agree[~tr] == 1)).mean()
            if ov > best[0]:
                best = (ov, L)
        L = best[1]
        # substrate cleanliness: pole-axis held-out LOO sign at L
        Wl = W[:, L]; mu = Wl.mean(0); sd = Wl.std(0) + 1e-6; Zw = (Wl - mu) / sd
        ok = 0
        for i in range(len(Zw)):
            keep = np.arange(len(Zw)) != i
            ax, c = _axis(Zw[keep & (pole == 1)], Zw[keep & (pole == -1)])
            ok += int((Zw[i] @ ax - c > 0) == (pole[i] > 0))
        clean = ok / len(Zw)
        # agreement axis per-cell at L
        X = P[:, L].astype(np.float32); mu = X[tr].mean(0); sd = X[tr].std(0) + 1e-6
        Z = (X - mu) / sd
        ax, c = _axis(Z[tr & (agree == 1)], Z[tr & (agree == 0)])
        pred = (Z @ ax - c) > 0; te = ~tr
        anto = te & (cell == "antonym")
        antocc = (pred[anto] == (agree[anto] == 1)).mean() if anto.any() else float("nan")
        rows.append((sub, L, clean, best[0], antocc))
    print(f"{'substrate':12} {'L':>3} {'pole-clean':>10} {'agr-overall':>11} {'antonym-cell':>12}")
    for sub, L, clean, ov, anto in rows:
        print(f"{sub:12} {L:>3} {clean:>10.2f} {ov:>11.2f} {anto:>12.2f}")
    print("\nthesis holds if antonym-cell (genuine relational) tracks pole-clean "
          "(substrate quality): Evaluation > Activity > Potency")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--collect", action="store_true")
    args = ap.parse_args()
    if args.collect or not os.path.exists(CACHE):
        collect()
    analyze()
