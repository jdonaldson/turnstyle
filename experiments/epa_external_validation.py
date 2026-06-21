"""Phase 2 — anti-circular external validation of the affect axes.

Phase 1 fit AND scored axes on theory-defined anchor words (circular). Here we
test against INDEPENDENT human ratings:

  - Warriner et al. 2013 (~14k words): Valence / Arousal / Dominance  (the VAD/PAD axes)
  - Affect Control Theory, usfullsurveyor2015 (Heise): Evaluation / Potency / Activity

Method: for each affect factor, train a ridge probe from SmolLM2 activations →
the *human* rating, report **held-out 5-fold CV** Pearson r (and R²). No anchor
words, no circularity — the model either linearly predicts human affect or it
doesn't. Everything is evaluated on the SAME common word set (ACT ∩ Warriner) so
the decisive **Potency (ACT) vs Dominance (Warriner)** third-axis race is fair.

Also reports the human-side inter-factor correlations (is Potency even the same
construct as Dominance?) and the model-decoded cross-framework correspondence
(E↔V, Activity↔Arousal, Potency↔Dominance).

    python -m experiments.epa_external_validation
    python -m experiments.epa_external_validation --layers 8,11,14 --max-words 600

DATA (gitignored under experiments/data/lexicons/ — re-fetch with):
  Warriner V/A/D:
    curl -L -o experiments/data/lexicons/Warriner.csv \
      https://raw.githubusercontent.com/JULIELab/XANEW/master/Ratings_Warriner_et_al.csv
  ACT E/P/A (usfullsurveyor2015, ahcombs/actdata, main branch):
    for k in mods identities behaviors; do curl -L -o experiments/data/lexicons/act_$k.csv \
      https://raw.githubusercontent.com/ahcombs/actdata/main/data-raw/dicts/means_only/usfullsurveyor2015_$k.csv; done
"""
from __future__ import annotations

import argparse
import csv
import json
import os

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_predict

MODEL = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
TEMPLATES = ["It is {w}.", "They felt {w}."]
LEX = "experiments/data/lexicons"
ALPHAS = np.logspace(0, 5, 11)


def load_warriner():
    """word -> (V, A, D) mean ratings (1–9 scale)."""
    out = {}
    with open(f"{LEX}/Warriner.csv", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            w = row["Word"].strip().lower()
            if not w or " " in w:
                continue
            try:
                out[w] = (float(row["V.Mean.Sum"]), float(row["A.Mean.Sum"]),
                          float(row["D.Mean.Sum"]))
            except (ValueError, KeyError):
                pass
    return out


def load_act():
    """word -> (E, P, A) mean ratings (~ -4.3..4.3). mods + identities, single-word."""
    out = {}
    for kind in ("act_mods.csv", "act_identities.csv"):
        with open(f"{LEX}/{kind}", newline="") as f:
            for row in csv.DictReader(f):
                t = row["term"].strip().lower()
                if not t or "_" in t or " " in t:
                    continue
                try:
                    epa = (float(row["E"]), float(row["P"]), float(row["A"]))
                except (ValueError, KeyError):
                    continue
                out[t] = tuple(np.mean([out[t], epa], 0)) if t in out else epa
    return out


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
        acts[w] = np.mean(per, 0)            # (L+1, H)
        if (i + 1) % 100 == 0:
            print(f"  acts {i+1}/{len(words)}", flush=True)
    return acts


def cv_scores(X, y):
    """held-out 5-fold CV: Pearson r and R² of out-of-fold predictions."""
    est = make_pipeline(StandardScaler(), RidgeCV(alphas=ALPHAS))
    pred = cross_val_predict(est, X, y, cv=5)
    r = float(np.corrcoef(pred, y)[0, 1])
    ss_res = float(np.sum((y - pred) ** 2)); ss_tot = float(np.sum((y - y.mean()) ** 2))
    return r, 1.0 - ss_res / ss_tot


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--layers", help="comma-separated (default: all)")
    p.add_argument("--max-words", type=int, default=1000)
    p.add_argument("--output", default="experiments/data/epa_external_validation.json")
    args = p.parse_args(argv)

    war, act = load_warriner(), load_act()
    common = sorted(set(war) & set(act))
    print(f"Warriner={len(war)}  ACT={len(act)}  common={len(common)}")
    if len(common) > args.max_words:                 # even stride keeps it unbiased
        common = common[:: max(1, len(common) // args.max_words)][: args.max_words]
        print(f"  capped to {len(common)} (even stride)")

    # human-side factor matrix on the common words: V,A_war,D, E,P,A_act
    FAC = ["V", "A_war", "D", "E", "P", "A_act"]
    H = np.array([[*war[w], *act[w]] for w in common])  # cols: V,A_war,D,E,P,A_act
    y = {f: H[:, i] for i, f in enumerate(FAC)}

    print("\n=== HUMAN inter-factor |r| (is Potency == Dominance?) ===")
    C = np.corrcoef(H.T)
    print("        " + "  ".join(f"{f:>6}" for f in FAC))
    for i, f in enumerate(FAC):
        print(f"{f:7s} " + "  ".join(f"{C[i,j]:+.2f}" for j in range(len(FAC))))

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL)
    mdl = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16).to(device)
    print(f"\ndevice={device}  collecting acts for {len(common)} words…", flush=True)
    acts = collect_acts(mdl, tok, device, common)
    n_layers = acts[common[0]].shape[0]
    layers = ([int(x) for x in args.layers.split(",")] if args.layers
              else list(range(n_layers)))

    report = {}
    for L in layers:
        X = np.vstack([acts[w][L] for w in common])
        scores = {f: cv_scores(X, y[f]) for f in FAC}
        report[L] = {f: {"r": round(scores[f][0], 3), "r2": round(scores[f][1], 3)}
                     for f in FAC}
        print(f"L{L:2d} CV r: " + "  ".join(f"{f}={scores[f][0]:.2f}" for f in FAC),
              flush=True)

    # summary: best CV r per factor, and the third-axis race
    print("\n=== EXTERNAL DECODABILITY (best layer, held-out CV r) ===")
    best = {}
    for f in FAC:
        bl = max(layers, key=lambda L: report[L][f]["r"])
        best[f] = (bl, report[bl][f]["r"], report[bl][f]["r2"])
        print(f"  {f:7s} r={best[f][1]:.3f} (R²={best[f][2]:.2f}) @L{best[f][0]}")

    print("\n=== THIRD-AXIS RACE (anti-circular) ===")
    print(f"  Potency (ACT)     best r={best['P'][1]:.3f} @L{best['P'][0]}")
    print(f"  Dominance (Warr.) best r={best['D'][1]:.3f} @L{best['D'][0]}")
    winner = "Potency" if best["P"][1] > best["D"][1] else "Dominance"
    print(f"  → model encodes {winner} better against human ratings")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({"common_n": len(common), "human_corr": C.tolist(),
                   "factors": FAC, "by_layer": report,
                   "best": {k: list(v) for k, v in best.items()}}, f, indent=1)
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
