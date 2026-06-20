"""Cross-lingual agreement falsifier: does relational transfer track substrate cleanliness?

This is the one place the substrate-cleanliness thesis can bite — cross-lingual transfer
is where pole cleanliness genuinely varies (Activity 0.87 > Evaluation 0.70 > Potency 0.67).

For each substrate (Evaluation / Activity / Potency): build agreement pairs
"A: It is {wA}. B: It is {wB}." (agree = same pole) in EN/ES/FR/DE from the Osgood
multilingual pole words; fit the agreement axis on ENGLISH; measure:
  pole-xl  — substrate cross-lingual pole sign (from osgood cache)
  agr-EN   — English held-out agreement (sanity)
  agr-xl   — does the English-fit agreement axis read agreement in es/fr/de?

Prediction (thesis): agr-xl tracks pole-xl across substrates (Activity > Eval > Potency).
If it doesn't, the substrate-cleanliness thesis is dead even where cleanliness varies.

Usage:  python experiments/boundary_xling_agreement.py [--collect]
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, "experiments")
sys.path.insert(0, "src")
import pole_probe as PP
import osgood_epa as OE

CACHE = "experiments/data/xling_agreement_acts.npz"
SUBS = ["evaluation", "activity", "potency"]
LANGS = ["en", "es", "fr", "de"]
PAIR_TMPL = {"en": "A: It is {a}. B: It is {b}.", "es": "A: Es {a}. B: Es {b}.",
             "fr": "A: C'est {a}. B: C'est {b}.", "de": "A: Es ist {a}. B: Es ist {b}."}


def _words(sub, lg):
    return [(w, 1) for w in OE.EPA[sub][lg]["hi"][:4]] + \
           [(w, -1) for w in OE.EPA[sub][lg]["lo"][:4]]


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

    save = {}
    for sub in SUBS:
        acts, agree, lang = [], [], []
        for lg in LANGS:
            ws = _words(sub, lg)
            for wa, sa in ws:
                for wb, sb in ws:
                    acts.append(last_act(PAIR_TMPL[lg].format(a=wa, b=wb)))
                    agree.append(int(sa == sb)); lang.append(lg)
        save[f"{sub}_acts"] = np.stack(acts)
        save[f"{sub}_agree"] = np.array(agree)
        save[f"{sub}_lang"] = np.array(lang)
        print(f"  {sub}: {len(acts)} pairs", flush=True)
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    np.savez_compressed(CACHE, **save)
    print(f"saved → {CACHE}")


def _axis(hi, lo):
    d = hi.mean(0) - lo.mean(0); d = d / (np.linalg.norm(d) + 1e-12)
    return d, 0.5 * (hi.mean(0) + lo.mean(0)) @ d


def _pole_xl(sub, L):
    """substrate cross-lingual pole sign from the osgood cache, at layer L."""
    o = np.load(OE.CACHE, allow_pickle=True)
    A = o["acts"].astype(np.float32)[:, L, :]; fac = o["fac"]; lang = o["lang"]; pole = o["pole"]
    m = fac == sub; en = m & (lang == "en")
    mu = A[en].mean(0); sd = A[en].std(0) + 1e-6; Z = (A - mu) / sd
    d, c = _axis(Z[en & (pole == 1)], Z[en & (pole == -1)])
    non = m & (lang != "en")
    return (((Z[non] @ d - c) > 0) == (pole[non] > 0)).mean()


def analyze():
    dd = np.load(CACHE, allow_pickle=True)
    nL = dd["evaluation_acts"].shape[1]
    print(f"{'substrate':11} {'L':>3} {'pole-xl':>8} {'agr-EN':>7} {'agr-xl':>7}")
    rows = []
    for sub in SUBS:
        A = dd[f"{sub}_acts"]; agree = dd[f"{sub}_agree"]; lang = dd[f"{sub}_lang"]
        en = lang == "en"
        rng = np.random.RandomState(0); tr = (rng.rand(len(agree)) < 0.5) & en
        best = (0.0, -1)
        for L in range(nL):
            X = A[:, L].astype(np.float32); mu = X[tr].mean(0); sd = X[tr].std(0) + 1e-6
            Z = (X - mu) / sd
            ax, c = _axis(Z[tr & (agree == 1)], Z[tr & (agree == 0)])
            teEN = en & ~tr
            enacc = (((Z[teEN] @ ax - c) > 0) == (agree[teEN] == 1)).mean()
            if enacc > best[0]:
                best = (enacc, L)
        L = best[1]
        X = A[:, L].astype(np.float32); mu = X[tr].mean(0); sd = X[tr].std(0) + 1e-6
        Z = (X - mu) / sd
        ax, c = _axis(Z[en & (agree == 1)], Z[en & (agree == 0)])  # fit on all EN
        non = ~en
        xl = (((Z[non] @ ax - c) > 0) == (agree[non] == 1)).mean()
        polexl = _pole_xl(sub, L)
        rows.append((sub, L, polexl, best[0], xl))
        print(f"{sub:11} {L:>3} {polexl:>8.2f} {best[0]:>7.2f} {xl:>7.2f}")
    print("\nthesis holds if agr-xl ranks with pole-xl across substrates.")
    order_pole = [r[0] for r in sorted(rows, key=lambda r: -r[2])]
    order_agr = [r[0] for r in sorted(rows, key=lambda r: -r[4])]
    print(f"  pole-xl order: {order_pole}")
    print(f"  agr-xl  order: {order_agr}")
    print(f"  {'MATCH (thesis supported)' if order_pole == order_agr else 'MISMATCH (thesis not supported)'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--collect", action="store_true")
    args = ap.parse_args()
    if args.collect or not os.path.exists(CACHE):
        collect()
    analyze()
