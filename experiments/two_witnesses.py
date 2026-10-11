#!/usr/bin/env python3
"""Two-witness protocol on snarks: hidden-state probe (⊨) × SenticNet lexicon (⊢).

The experiment behind the "Two Witnesses for Sarcasm" post. Per example:
  - Witness ⊨: the L13 per-option probe's 5-fold CV *held-out* prediction
    (same formulation/folds as snarks_probe.py::probe_per_option_cv, but
    recording per-example predictions instead of collapsing to accuracy).
  - Witness ⊢: SenticNet-9 polarity-incongruity scorer (symbolic_witness.py),
    fit on NOTHING — task-external by construction.

Outputs the 2×2 (probe-correct × symbolic-correct), agreement-gated accuracy
and coverage, per-witness accuracy on the disagreement cells, error-correlation
(phi), and a symbolic-vs-length confound check. Per-example rows land in
experiments/results/two_witnesses_snarks.json.

Pre-registered decisions (do not tune on these labels):
  - symbolic score = polarity span (max_pos − min_neg), both signs required
  - symbolic tie → witness abstains (excluded from its accuracy, counted)
  - probe layer 13, LogReg C=0.1, StratifiedKFold(5, shuffle, seed=42)

Run:
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \\
        /Users/jdonaldson/Projects/swollm/.venv/bin/python \\
        experiments/two_witnesses.py [--senticnet PATH]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, "/Users/jdonaldson/Projects/swollm/src")

from snarks_probe import _OPTION_RE, collect_records, load_model  # noqa: E402
from swollm.bench.bbh import load_task  # noqa: E402
from symbolic_witness import load_senticnet, predict as symbolic_predict  # noqa: E402

LAYER = 13
# The SenticNet lexicon is licensed separately and is not committed; drop
# senticnet.py under experiments/data/senticnet/ (gitignored) or pass --senticnet.
SENTICNET_DEFAULT = str(Path(__file__).parent / "data" / "senticnet" / "senticnet.py")


def option_texts(text: str) -> dict[str, str] | None:
    opts = {m.group(1): m.group(2).strip() for m in _OPTION_RE.finditer(text)}
    return opts if {"A", "B"} <= opts.keys() else None


def probe_cv_predictions(records, layer=LAYER, n_splits=5, seed=42):
    """probe_per_option_cv, but returning per-example held-out predictions."""
    y_ex = np.array([1 if r["sarcastic"] == "A" else 0 for r in records])
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    preds = [None] * len(records)
    probs = [None] * len(records)
    for fold, (tr_idx, te_idx) in enumerate(skf.split(np.arange(len(records)), y_ex)):
        X_tr, y_tr = [], []
        for i in tr_idx:
            hA, hB = records[i]["per_layer"][layer]
            sarc = records[i]["sarcastic"]
            X_tr.append(hA); y_tr.append(1 if sarc == "A" else 0)
            X_tr.append(hB); y_tr.append(1 if sarc == "B" else 0)
        sc = StandardScaler().fit(np.array(X_tr))
        clf = LogisticRegression(max_iter=2000, C=0.1)
        clf.fit(sc.transform(np.array(X_tr)), np.array(y_tr))

        for i in te_idx:
            hA, hB = records[i]["per_layer"][layer]
            pa = clf.predict_proba(sc.transform(hA[None]))[0, 1]
            pb = clf.predict_proba(sc.transform(hB[None]))[0, 1]
            preds[i] = "A" if pa > pb else "B"
            probs[i] = (float(pa), float(pb))
    return preds, probs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--senticnet", default=SENTICNET_DEFAULT)
    args = ap.parse_args()

    lex = load_senticnet(args.senticnet)
    print(f"SenticNet lexicon: {len(lex)} polarity concepts", flush=True)

    model, tokenizer, device = load_model()
    examples = load_task("snarks")
    print(f"Loaded {len(examples)} snarks examples", flush=True)

    print("\nCollecting hidden states…", flush=True)
    records = collect_records(examples, model, tokenizer, device)
    # keep the raw text alongside each record (collect_records drops it)
    kept = []
    for ex in examples:
        if ex["target"].strip() in ("(A)", "(B)") and option_texts(ex["input"]):
            kept.append(ex)
    assert len(kept) == len(records), f"{len(kept)} kept vs {len(records)} records"

    print("\nWitness ⊨ (probe, 5-fold CV held-out)…", flush=True)
    probe_preds, probe_probs = probe_cv_predictions(records)

    print("Witness ⊢ (SenticNet incongruity)…", flush=True)
    rows = []
    for i, (ex, rec) in enumerate(zip(kept, records)):
        opts = option_texts(ex["input"])
        sym_pred, sym_conf, sa, sb = symbolic_predict(opts["A"], opts["B"], lex)
        rows.append({
            "idx": i,
            "option_a": opts["A"],
            "option_b": opts["B"],
            "gold": rec["sarcastic"],
            "probe": probe_preds[i],
            "probe_pa": probe_probs[i][0],
            "probe_pb": probe_probs[i][1],
            "symbolic": sym_pred,           # None = symbolic tie/abstain
            "symbolic_conf": sym_conf,
            "symbolic_score_a": sa,
            "symbolic_score_b": sb,
            "len_pick": "A" if rec["len_A"] >= rec["len_B"] else "B",
        })

    # ---- analysis ----
    n = len(rows)
    scored = [r for r in rows if r["symbolic"] is not None]
    ties = n - len(scored)

    probe_acc = np.mean([r["probe"] == r["gold"] for r in rows])
    sym_acc = np.mean([r["symbolic"] == r["gold"] for r in scored])
    len_acc = np.mean([r["len_pick"] == r["gold"] for r in rows])

    print(f"\n=== Witness accuracies (n={n}) ===")
    print(f"  ⊨ probe (CV held-out):      {probe_acc:.1%}")
    print(f"  ⊢ symbolic (n={len(scored)}, {ties} ties): {sym_acc:.1%}")
    print(f"  length-only reference:      {len_acc:.1%}")

    # 2×2 on examples where both witnesses committed
    cells = {"both": 0, "probe_only": 0, "sym_only": 0, "neither": 0}
    for r in scored:
        p_ok = r["probe"] == r["gold"]
        s_ok = r["symbolic"] == r["gold"]
        key = ("both" if p_ok and s_ok else
               "probe_only" if p_ok else
               "sym_only" if s_ok else "neither")
        cells[key] += 1
    print(f"\n=== 2×2 (n={len(scored)} both-committed) ===")
    print(f"                 symbolic ✓   symbolic ✗")
    print(f"  probe ✓        {cells['both']:>6}       {cells['probe_only']:>6}")
    print(f"  probe ✗        {cells['sym_only']:>6}       {cells['neither']:>6}")

    # error correlation (phi) between witnesses
    p_err = np.array([r["probe"] != r["gold"] for r in scored], dtype=float)
    s_err = np.array([r["symbolic"] != r["gold"] for r in scored], dtype=float)
    if p_err.std() > 0 and s_err.std() > 0:
        phi = float(np.corrcoef(p_err, s_err)[0, 1])
    else:
        phi = float("nan")
    print(f"\n  error correlation (phi): {phi:+.3f}"
          f"  ({'near-independent' if abs(phi) < 0.15 else 'correlated'})")

    # agreement gating
    agree = [r for r in scored if r["probe"] == r["symbolic"]]
    disagree = [r for r in scored if r["probe"] != r["symbolic"]]
    agree_acc = np.mean([r["probe"] == r["gold"] for r in agree]) if agree else float("nan")
    dis_probe = np.mean([r["probe"] == r["gold"] for r in disagree]) if disagree else float("nan")
    dis_sym = np.mean([r["symbolic"] == r["gold"] for r in disagree]) if disagree else float("nan")
    print(f"\n=== Agreement gating ===")
    print(f"  agree:    n={len(agree):>3} ({len(agree)/len(scored):.0%} coverage)  acc={agree_acc:.1%}")
    print(f"  disagree: n={len(disagree):>3}   probe acc={dis_probe:.1%}  symbolic acc={dis_sym:.1%}")

    # symbolic vs length confound
    sym_is_len = np.mean([r["symbolic"] == r["len_pick"] for r in scored])
    print(f"\n  symbolic picks the longer option {sym_is_len:.0%} of the time"
          f" (length itself: {len_acc:.1%} accurate)")

    out_path = Path(__file__).parent / "results" / "two_witnesses_snarks.json"
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps({
        "n": n, "ties": ties,
        "probe_acc": float(probe_acc), "symbolic_acc": float(sym_acc),
        "length_acc": float(len_acc), "phi": phi,
        "cells": cells,
        "agree_n": len(agree), "agree_acc": float(agree_acc),
        "disagree_n": len(disagree),
        "disagree_probe_acc": float(dis_probe), "disagree_symbolic_acc": float(dis_sym),
        "rows": rows,
    }, indent=1))
    print(f"\nPer-example rows → {out_path}", flush=True)


if __name__ == "__main__":
    main()
