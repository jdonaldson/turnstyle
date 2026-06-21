"""Phase 3 — is the 4th axis a real AFFECT dimension or a lexical confound?

Phase 1 found a 4th anchor axis (novelty/unpredictability) that was independent of
EPA and encodable. The danger: "novelty" (unpredictable/surprising/novel vs
predictable/familiar/expected) may just track lexical **familiarity / frequency /
concreteness** — a surface property, not affect. This decides between:

  H1 (real 4th affect axis): the model's novelty projection tracks human SURPRISE
     (NRC EmoLex), stays correlated after controlling for lexical confounds, is
     independent of V/A/D, and adds incremental prediction of surprise beyond V/A/D.
  H2 (lexical confound): it's explained by concreteness / log-frequency / familiarity
     and its surprise correlation vanishes once those are controlled.

Data (gitignored; fetch URLs in epa_external_validation.py + below):
  Warriner V/A/D; Brysbaert concreteness + SUBTLEX freq; Glasgow familiarity (FAM_M);
  NRC EmoLex surprise/anticipation (binary).
    brysbaert: cltl/python-for-text-analysis .../Concreteness_ratings_Brysbaert_et_al_BRM.txt
    glasgow:   mllewis/IATLANG/master/data/study1a/raw/GlasgowNorms.csv
    emolex:    andyreagan/sentidict .../NRC-emotion-lexicon-wordlevel-alphabetized-v0.92.txt(.gz)

    python -m experiments.epa_fourth_axis [--layer 11]
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_auc_score

MODEL = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
TEMPLATES = ["It is {w}.", "They felt {w}."]
LEX = "experiments/data/lexicons"
NOVELTY = (["unpredictable", "surprising", "novel", "unexpected", "strange", "sudden"],
           ["predictable", "familiar", "expected", "ordinary", "routine", "usual"])


def load_warriner():
    out = {}
    with open(f"{LEX}/Warriner.csv", newline="") as f:
        for row in csv.DictReader(f):
            w = row["Word"].strip().lower()
            if w and " " not in w:
                try:
                    out[w] = (float(row["V.Mean.Sum"]), float(row["A.Mean.Sum"]),
                              float(row["D.Mean.Sum"]))
                except (ValueError, KeyError):
                    pass
    return out


def load_brysbaert():
    out = {}
    with open(f"{LEX}/brysbaert_conc.txt", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            w = row["Word"].strip().lower()
            try:
                out[w] = (float(row["Conc.M"]), math.log1p(float(row["SUBTLEX"])))
            except (ValueError, KeyError):
                pass
    return out


def load_glasgow():
    out = {}
    with open(f"{LEX}/glasgow.csv", newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            w = (row.get("word") or "").strip().lower()
            try:
                out[w] = float(row["FAM_M"])
            except (ValueError, KeyError, TypeError):
                pass
    return out


def load_emolex():
    """word -> {'surprise':0/1, 'anticipation':0/1}; vocab = all words seen."""
    surprise, anticip, vocab = {}, {}, set()
    with open(f"{LEX}/emolex.txt") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 3:
                continue
            w, emo, val = parts[0].strip().lower(), parts[1], parts[2]
            vocab.add(w)
            if emo == "surprise":
                surprise[w] = int(val)
            elif emo == "anticipation":
                anticip[w] = int(val)
    return surprise, anticip, vocab


def collect_acts(model, tok, device, words):
    acts = {}
    for i, w in enumerate(words):
        per = []
        for tmpl in TEMPLATES:
            sent = tmpl.format(w=w)
            cs = sent.rfind(w); ce = cs + len(w)
            enc = tok(sent, return_offsets_mapping=True, return_tensors="pt")
            offs = enc.pop("offset_mapping")[0].tolist()
            enc = {k: v.to(device) for k, v in enc.items()}
            with torch.no_grad():
                hs = model(**enc, output_hidden_states=True).hidden_states
            tk = next((k for k, (s, e) in enumerate(offs) if e > cs and s < ce), -1)
            per.append(torch.stack(hs, 0)[:, 0, tk, :].float().cpu().numpy())
        acts[w] = np.mean(per, 0)
        if (i + 1) % 150 == 0:
            print(f"  acts {i+1}/{len(words)}", flush=True)
    return acts


def novelty_direction(actsM, hi_idx, lo_idx, k_surf=3):
    """fit the novelty BipolarAxis direction in standardized space (surface-suppressed)."""
    mean = actsM.mean(0); scale = actsM.std(0) + 1e-6
    Z = (actsM - mean) / scale
    d = Z[hi_idx].mean(0) - Z[lo_idx].mean(0)
    if k_surf:
        _, _, Vt = np.linalg.svd(Z - Z.mean(0), full_matrices=False)
        surf = Vt[:k_surf]
        d = d - (d @ surf.T) @ surf
    d = d / (np.linalg.norm(d) + 1e-12)
    return mean, scale, d


def r(a, b):
    return float(np.corrcoef(a, b)[0, 1])


def ols_r2(X, y):
    Xd = np.column_stack([np.ones(len(y)), X])
    beta, *_ = np.linalg.lstsq(Xd, y, rcond=None)
    pred = Xd @ beta
    return 1.0 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2)


def residual(y, X):
    Xd = np.column_stack([np.ones(len(y)), X])
    beta, *_ = np.linalg.lstsq(Xd, y, rcond=None)
    return y - Xd @ beta


def partial_corr(a, b, controls):
    return r(residual(a, controls), residual(b, controls))


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--layer", type=int, default=11)
    p.add_argument("--max-words", type=int, default=1200)
    p.add_argument("--output", default="experiments/data/epa_fourth_axis.json")
    args = p.parse_args(argv)

    war, brys, glas = load_warriner(), load_brysbaert(), load_glasgow()
    surprise, anticip, emo_vocab = load_emolex()
    common = sorted(set(war) & set(brys) & emo_vocab)
    print(f"Warriner={len(war)} Brysbaert={len(brys)} EmoLex={len(emo_vocab)} "
          f"Glasgow={len(glas)} common={len(common)}")
    if len(common) > args.max_words:
        common = common[:: max(1, len(common) // args.max_words)][: args.max_words]
        print(f"  capped to {len(common)} (even stride)")

    words = sorted(set(common) | set(NOVELTY[0]) | set(NOVELTY[1]))
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL)
    mdl = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16).to(device)
    print(f"device={device}  collecting acts for {len(words)} words…", flush=True)
    acts = collect_acts(mdl, tok, device, words)

    L = args.layer
    M = np.vstack([acts[w][L] for w in common])
    nov_words = NOVELTY[0] + NOVELTY[1]
    novM = np.vstack([acts[w][L] for w in nov_words])
    # fit novelty dir on novelty anchors' standardization, project common words
    mean, scale, d = novelty_direction(novM, list(range(len(NOVELTY[0]))),
                                       list(range(len(NOVELTY[0]), len(nov_words))))
    nov = ((M - mean) / scale) @ d                       # novelty projection per word

    V = np.array([war[w][0] for w in common]); A = np.array([war[w][1] for w in common])
    D = np.array([war[w][2] for w in common])
    conc = np.array([brys[w][0] for w in common]); freq = np.array([brys[w][1] for w in common])
    surp = np.array([surprise.get(w, 0) for w in common], float)
    ant = np.array([anticip.get(w, 0) for w in common], float)
    has_fam = [w in glas for w in common]
    fam = np.array([glas.get(w, np.nan) for w in common])

    print(f"\nL{L}  common={len(common)}  surprise base-rate={surp.mean():.2f}  "
          f"familiarity coverage={sum(has_fam)}")

    print("\n=== novelty-axis correlations (what does the 4th axis track?) ===")
    cors = {"surprise": r(nov, surp), "anticipation": r(nov, ant),
            "concreteness": r(nov, conc), "log-frequency": r(nov, freq),
            "valence": r(nov, V), "arousal": r(nov, A), "dominance": r(nov, D)}
    if sum(has_fam) > 30:
        fm = np.array(has_fam)
        cors["familiarity"] = r(nov[fm], fam[fm])
    for k, v in sorted(cors.items(), key=lambda kv: -abs(kv[1])):
        print(f"  {k:14s} r={v:+.3f}")

    print("\n=== H1 vs H2 tests ===")
    lex = np.column_stack([conc, freq])
    nov_lex_r2 = ols_r2(lex, nov)
    print(f"  novelty ~ lexical(conc,freq)         R²={nov_lex_r2:.3f}  "
          f"(high → 4th axis is largely lexical)")
    pc_surp_lex = partial_corr(nov, surp, lex)
    print(f"  partial r(novelty, surprise | lexical)  ={pc_surp_lex:+.3f}  "
          f"(survives → affect signal beyond lexical)")
    pc_surp_vad = partial_corr(nov, surp, np.column_stack([V, A, D]))
    print(f"  partial r(novelty, surprise | V,A,D)    ={pc_surp_vad:+.3f}  "
          f"(survives → independent of the 3 affect axes)")

    # incremental validity: predict human surprise, held-out CV R²
    base = np.column_stack([V, A, D])
    r2_vad = ols_cv_r2(base, surp)
    r2_vad_nov = ols_cv_r2(np.column_stack([base, nov]), surp)
    r2_vad_lex = ols_cv_r2(np.column_stack([base, conc, freq]), surp)
    print(f"\n  surprise ~ V,A,D            CV R²={r2_vad:.3f}")
    print(f"  surprise ~ V,A,D + novelty  CV R²={r2_vad_nov:.3f}  (Δ={r2_vad_nov-r2_vad:+.3f})")
    print(f"  surprise ~ V,A,D + lexical  CV R²={r2_vad_lex:.3f}  (Δ={r2_vad_lex-r2_vad:+.3f})")

    # is a surprise signal present, and is it BEYOND the 3 affect axes / lexical?
    auc = decode_auc(M, surp)
    auc_vad = decode_auc(np.column_stack([V, A, D]), surp)
    auc_lex = decode_auc(np.column_stack([conc, freq]), surp)
    print(f"\n  decode human surprise — 5-fold AUC (0.5=chance):")
    print(f"    from activations        = {auc:.3f}")
    print(f"    from V,A,D (3 axes)     = {auc_vad:.3f}")
    print(f"    from lexical(conc,freq) = {auc_lex:.3f}")
    print(f"    → acts carry surprise info BEYOND the 3 axes if {auc:.2f} >> {auc_vad:.2f}")

    # three-way verdict
    if nov_lex_r2 > 0.15:
        verdict = "H2: 4th axis is largely a LEXICAL confound"
    elif auc - auc_vad > 0.05:
        verdict = ("H1-partial: a surprise signal exists in acts BEYOND the 3 axes "
                   "(not lexical), but the simple novelty-anchor axis doesn't capture it")
    else:
        verdict = ("weak: 4th axis is NOT lexical, but no clear independent affect "
                   "dimension beyond the 3 axes at this probe/base-rate")
    print(f"\nVERDICT: {verdict}")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({"layer": L, "n": len(common), "correlations": cors,
                   "novelty_lexical_r2": nov_lex_r2,
                   "partial_surprise_given_lexical": pc_surp_lex,
                   "partial_surprise_given_vad": pc_surp_vad,
                   "cv_r2": {"vad": r2_vad, "vad_novelty": r2_vad_nov, "vad_lexical": r2_vad_lex},
                   "surprise_decode_auc": auc, "verdict": verdict}, f, indent=1)
    print(f"\nwrote {args.output}")


def ols_cv_r2(X, y, folds=5):
    from sklearn.linear_model import LinearRegression
    pred = cross_val_predict(make_pipeline(StandardScaler(), LinearRegression()), X, y, cv=folds)
    return 1.0 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2)


def decode_auc(X, y):
    if len(set(y)) < 2:
        return float("nan")
    est = make_pipeline(StandardScaler(),
                        LogisticRegressionCV(Cs=8, max_iter=2000, scoring="roc_auc"))
    proba = cross_val_predict(est, X, y, cv=5, method="predict_proba")[:, 1]
    return float(roc_auc_score(y, proba))


if __name__ == "__main__":
    main()
