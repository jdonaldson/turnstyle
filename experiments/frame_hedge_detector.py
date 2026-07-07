"""
frame_hedge_detector.py — is the shared negation delta (the "hedge vector")
a keyword-free NEGATION DETECTOR?

frame_operators_negation.py found negation adds a shared direction to the
phrase state (cross-frame mean-delta cos 0.67 @L14). If that direction is a
negation SIGNATURE, projecting a phrase's adjective-token state onto it
detects negation with no keyword list — the principled replacement for
neg-regex (which the DAG work already showed loses 0/9 vs 9/9 on
paraphrastic negation).

Detector: v_neg = mean over (frame, word) of state(not w) - state(w) at
L14, template set A. Score(phrase) = (state - bare_mean) . v_neg / ||v_neg||,
in sd units of the bare distribution. NO thresholds fit on held-out types.

Pre-registered predictions (held-out phrase types, template set B):
  H1 syntactic transfer: "not {w}" in NEW templates scores HIGH.
  H2 paraphrastic negation NEVER SEEN by the detector scores HIGH:
     "hardly {w}", "far from {w}", "by no means {w}".
  H3 intensifiers/controls score LOW: "very {w}", "extremely {w}",
     "also {w}".
  H4 THE REGEX-IMPOSSIBLE CASE: "not only {w}" (positive polarity,
     contains literal 'not') scores LOW — regex false-positives here by
     construction.
  H5 downtoners sit between: "somewhat {w}", "slightly {w}".
  H6 morphological un- scores LOW/weak (lexical antonym, not contextual
     overwrite — needs no operator handling, and the detector should
     agree).
Report per-type mean score + AUC(negated types vs control types), and the
regex baseline's confusion on the same items.

Run:  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python \
        experiments/frame_hedge_detector.py
Writes experiments/results/frame_hedge_detector.json.
"""
from __future__ import annotations
import json, os, sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from frame_operators import TEMPLATES, UN_PAIRS, MODEL
from frame_operators_replicate import TEMPLATES_B

L_DET = 14
TRAIN_FRAMES = ["opinion", "size", "warmth", "certainty"]

# held-out phrase types: (name, modifier, expected: neg/ctl/mid/morph)
HELD_OUT = [
    ("not_B", "not ", "neg"),                 # H1 syntactic, new template
    ("hardly", "hardly ", "neg"),             # H2 paraphrastic
    ("far_from", "far from ", "neg"),
    ("by_no_means", "by no means ", "neg"),
    ("very", "very ", "ctl"),                 # H3 controls
    ("extremely", "extremely ", "ctl"),
    ("also", "also ", "ctl"),
    ("not_only", "not only ", "ctl"),         # H4 regex-impossible
    ("somewhat", "somewhat ", "mid"),         # H5 downtoners
    ("slightly", "slightly ", "mid"),
]


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from turnstyle.frame_library import load_library

    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL)
    mdl = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.float16).to(dev)
    lib = load_library(mdl)
    frames = {n: lib.frames[n] for n in TRAIN_FRAMES}

    def collect(sent: str, w: str) -> np.ndarray:
        """L_DET hidden state at the last subword of w."""
        cs = sent.rfind(w); ce = cs + len(w)
        enc = tok(sent, return_offsets_mapping=True, return_tensors="pt")
        offs = enc.pop("offset_mapping")[0].tolist()
        enc = {k: v.to(dev) for k, v in enc.items()}
        with torch.no_grad():
            hs = mdl(**enc, output_hidden_states=True).hidden_states
        idxs = [k for k, (s, e) in enumerate(offs) if e > cs and s < ce] or [-1]
        return hs[L_DET][0, idxs[-1], :].float().cpu().numpy()

    # ── fit: v_neg from template set A, "not" only ───────────────────────────
    print("fitting v_neg (template A)...", flush=True)
    deltas, bares = [], []
    for fname, fr in frames.items():
        tmpl = TEMPLATES[fname]
        for w in fr.data:
            sb = collect(tmpl.format(m="", w=w), w)
            sn = collect(tmpl.format(m="not ", w=w), w)
            deltas.append(sn - sb)
            bares.append(sb)
    v = np.mean(deltas, 0)
    v /= np.linalg.norm(v) + 1e-12
    bares = np.array(bares)
    mu0 = bares.mean(0)
    proj_bare = (bares - mu0) @ v
    s0 = proj_bare.std() + 1e-9

    def score(sent: str, w: str) -> float:
        return float(((collect(sent, w) - mu0) @ v) / s0)

    # ── held-out scoring, template set B ─────────────────────────────────────
    print("scoring held-out phrase types (template B)...", flush=True)
    rows = {name: [] for name, _, _ in HELD_OUT}
    rows["bare_B"] = []
    for fname, fr in frames.items():
        tmpl = TEMPLATES_B[fname]
        for w in fr.data:
            rows["bare_B"].append(score(tmpl.format(m="", w=w), w))
            for name, mod, _ in HELD_OUT:
                rows[name].append(score(tmpl.format(m=mod, w=w), w))
        print(f"  {fname} done", flush=True)

    # morphological un- (H6)
    rows["un_morph"] = []
    for fname, pairs in UN_PAIRS.items():
        tmpl = TEMPLATES_B.get(fname, TEMPLATES[fname])
        for base, un in pairs:
            rows["un_morph"].append(score(tmpl.format(m="", w=un), un))

    kinds = {name: kind for name, _, kind in HELD_OUT}
    kinds.update({"bare_B": "ctl", "un_morph": "morph"})

    print("\n=== hedge-projection score by phrase type (sd units of bare-A) "
          "===")
    results = {"model": MODEL, "layer": L_DET, "types": {}}
    for name, vals in rows.items():
        a = np.array(vals)
        results["types"][name] = {"mean": round(float(a.mean()), 2),
                                  "sd": round(float(a.std()), 2),
                                  "n": len(a), "kind": kinds[name]}
        print(f"  {name:12s} [{kinds[name]:5s}]  {a.mean():+6.2f} "
              f"+/- {a.std():4.2f}  (n={len(a)})")

    # AUC: negated types vs control types (pooled items)
    neg = np.concatenate([rows[n] for n, k in kinds.items() if k == "neg"])
    ctl = np.concatenate([rows[n] for n, k in kinds.items() if k == "ctl"])
    auc = float(np.mean([[x > y for y in ctl] for x in neg]))
    print(f"\nAUC(neg > ctl) = {auc:.3f}   (n_neg={len(neg)}, "
          f"n_ctl={len(ctl)})")
    results["auc_neg_vs_ctl"] = round(auc, 3)

    # the regex baseline on the same items (surface contains a neg keyword)
    import re
    neg_re = re.compile(r"\b(not|no|never)\b|n't")
    print("\n=== regex baseline (surface match) confusion by type ===")
    reg = {}
    for name, mod, kind in HELD_OUT:
        hit = bool(neg_re.search(mod))
        want = kind == "neg"
        verdict = ("ok" if hit == want else
                   ("FALSE-POS" if hit else "MISS")) if kind != "mid" else "--"
        reg[name] = {"regex_fires": hit, "verdict": verdict}
        print(f"  {name:12s} regex={'fires' if hit else 'silent':6s}  "
              f"{verdict}")
    results["regex_baseline"] = reg

    outp = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "results", "frame_hedge_detector.json")
    with open(outp, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {outp}")


if __name__ == "__main__":
    main()
