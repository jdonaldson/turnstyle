"""Boundary test for 'semantic programming': where does bipolar-axis capture break?

Prediction (from the denotation→connotation degradation): a SCALAR pragmatic property
emerges as a clean axis; a RELATIONAL one does not. Two probes, same gate:

  CERTAINTY (scalar, predicted PASS) — epistemic adverbs (definitely … maybe). Fit a
    BipolarAxis from extremes; held-out sign (EN LOO) + cross-lingual sign (es/fr/de).

  AGREEMENT (relational, predicted BREAK) — "A: It is {wA}. B: It is {wB}." labeled by
    whether B's stance MATCHES A's. Each B word appears in both agree and disagree pairs
    (paired with matching/opposing A), so B alone carries NO signal — only the relation
    A↔B solves it. Read the activation at the end of B. Fit an axis from agree vs
    disagree; held-out (EN split) + cross-lingual accuracy.

The contrast IS the result: certainty high, agreement near chance ⇒ the technique reaches
scalar pragmatics but stops at relational composition (the predicted wall). If agreement
ALSO passes, the prediction is falsified — the wall is farther out.

Usage:  python experiments/boundary_test.py [--collect]
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, "experiments")
sys.path.insert(0, "src")
import pole_probe as PP
from osgood_epa import EPA   # reuse the evaluation pole words (4 langs)

CACHE = "experiments/data/boundary_test_acts.npz"

# ── scalar: certainty (epistemic adverbs) ────────────────────────────────────
CERTAIN = {
    "en": {"hi": ["definitely", "certainly", "surely", "undoubtedly", "clearly", "obviously"],
           "lo": ["maybe", "perhaps", "possibly", "probably", "supposedly", "allegedly"]},
    "es": {"hi": ["definitivamente", "ciertamente", "seguramente", "indudablemente", "claramente", "obviamente"],
           "lo": ["quizás", "tal vez", "posiblemente", "probablemente", "supuestamente", "presuntamente"]},
    "fr": {"hi": ["définitivement", "certainement", "sûrement", "indubitablement", "clairement", "évidemment"],
           "lo": ["peut-être", "possiblement", "probablement", "prétendument", "vraisemblablement", "soi-disant"]},
    "de": {"hi": ["definitiv", "sicherlich", "bestimmt", "zweifellos", "eindeutig", "offensichtlich"],
           "lo": ["vielleicht", "womöglich", "möglicherweise", "wahrscheinlich", "angeblich", "vermutlich"]},
}
CERT_TMPL = {"en": "{w}, the answer is yes.", "es": "{w}, la respuesta es sí.",
             "fr": "{w}, la réponse est oui.", "de": "{w}, die Antwort ist ja."}

# ── relational: agreement, built from the evaluation pole words ───────────────
AGREE_TMPL = {"en": "A: It is {a}. B: It is {b}.", "es": "A: Es {a}. B: Es {b}.",
              "fr": "A: C'est {a}. B: C'est {b}.", "de": "A: Es ist {a}. B: Es ist {b}."}
LANGS = ["en", "es", "fr", "de"]


def _eval_words(lg):
    return [(w, 1) for w in EPA["evaluation"][lg]["hi"][:4]] + \
           [(w, -1) for w in EPA["evaluation"][lg]["lo"][:4]]


def _last_token_act(mdl, tok, dev, sent):
    import torch
    enc = tok(sent, return_tensors="pt")
    enc = {k: v.to(dev) for k, v in enc.items()}
    with torch.no_grad():
        out = mdl(**enc, output_hidden_states=True)
    return np.stack([h[0, -1].float().cpu().numpy() for h in out.hidden_states])


def _word_token_act(mdl, tok, dev, sent, word):
    import torch
    content = word.split()[-1]
    cs = sent.rfind(content); ce = cs + len(content)
    enc = tok(sent, return_offsets_mapping=True, return_tensors="pt")
    offs = enc.pop("offset_mapping")[0].tolist()
    enc = {k: v.to(dev) for k, v in enc.items()}
    with torch.no_grad():
        out = mdl(**enc, output_hidden_states=True)
    tk = next((k for k, (s, e) in enumerate(offs) if e > cs and s < ce), len(offs) - 1)
    return np.stack([h[0, tk].float().cpu().numpy() for h in out.hidden_states])


def collect():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(PP.MODEL_ID)
    mdl = AutoModelForCausalLM.from_pretrained(
        PP.MODEL_ID, dtype=torch.float16).to(dev).eval()

    # certainty
    cA, cPole, cLang = [], [], []
    for lg, poles in CERTAIN.items():
        for pol, words in poles.items():
            for w in words:
                cA.append(_word_token_act(mdl, tok, dev, CERT_TMPL[lg].format(w=w), w).astype(np.float16))
                cPole.append(1 if pol == "hi" else -1); cLang.append(lg)
    # agreement
    gA, gLab, gLang = [], [], []
    for lg in LANGS:
        ws = _eval_words(lg)
        for wa, sa in ws:
            for wb, sb in ws:
                sent = AGREE_TMPL[lg].format(a=wa, b=wb)
                gA.append(_last_token_act(mdl, tok, dev, sent).astype(np.float16))
                gLab.append(1 if sa == sb else 0); gLang.append(lg)
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    np.savez_compressed(
        CACHE,
        cert_acts=np.stack(cA), cert_pole=np.array(cPole), cert_lang=np.array(cLang),
        agr_acts=np.stack(gA), agr_lab=np.array(gLab), agr_lang=np.array(gLang))
    print(f"saved certainty={len(cA)} agreement={len(gA)} → {CACHE}")


def _axis(Ztr_hi, Ztr_lo):
    d = Ztr_hi.mean(0) - Ztr_lo.mean(0)
    d = d / (np.linalg.norm(d) + 1e-12)
    c = 0.5 * (Ztr_hi.mean(0) + Ztr_lo.mean(0)) @ d
    return d, c


def analyze():
    d = np.load(CACHE, allow_pickle=True)
    nL = d["cert_acts"].shape[1]
    print(f"{'L':>3} | {'CERT held':>9} {'CERT xl':>8} | {'AGR held':>9} {'AGR xl':>7}")
    bestC = bestG = (0.0, -1)
    for L in range(nL):
        # ── certainty ──
        cA = d["cert_acts"][:, L].astype(np.float32); cp = d["cert_pole"]; cl = d["cert_lang"]
        en = cl == "en"
        mu = cA[en].mean(0); sd = cA[en].std(0) + 1e-6
        Z = (cA - mu) / sd
        # held-out LOO sign (EN)
        idxs = np.where(en)[0]; ok = 0
        for i in idxs:
            keep = en & (np.arange(len(cA)) != i)
            dd, cc = _axis(Z[keep & (cp == 1)], Z[keep & (cp == -1)])
            ok += int((Z[i] @ dd - cc > 0) == (cp[i] > 0))
        c_held = ok / len(idxs)
        dd, cc = _axis(Z[en & (cp == 1)], Z[en & (cp == -1)])
        non = ~en
        c_xl = (((Z[non] @ dd - cc) > 0) == (cp[non] > 0)).mean()

        # ── agreement ──
        gA = d["agr_acts"][:, L].astype(np.float32); gy = d["agr_lab"]; gl = d["agr_lang"]
        en = gl == "en"
        mu = gA[en].mean(0); sd = gA[en].std(0) + 1e-6
        Z = (gA - mu) / sd
        # held-out: 50/50 split of EN pairs (deterministic)
        eidx = np.where(en)[0]
        half = eidx[::2]; other = eidx[1::2]
        dd, cc = _axis(Z[half][gy[half] == 1], Z[half][gy[half] == 0])
        g_held = (((Z[other] @ dd - cc) > 0) == (gy[other] == 1)).mean()
        dd, cc = _axis(Z[en][gy[en] == 1], Z[en][gy[en] == 0])
        non = ~en
        g_xl = (((Z[non] @ dd - cc) > 0) == (gy[non] == 1)).mean()

        if c_xl > bestC[0]:
            bestC = (c_xl, L)
        if g_xl > bestG[0]:
            bestG = (g_xl, L)
        print(f"{L:>3} | {c_held:>9.2f} {c_xl:>8.2f} | {g_held:>9.2f} {g_xl:>7.2f}")
    print(f"\nCERTAINTY (scalar)   best x-ling L{bestC[1]} = {bestC[0]:.2f}")
    print(f"AGREEMENT (relational) best x-ling L{bestG[1]} = {bestG[0]:.2f}   (chance 0.50)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--collect", action="store_true")
    args = ap.parse_args()
    if args.collect or not os.path.exists(CACHE):
        collect()
    analyze()
