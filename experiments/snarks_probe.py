#!/usr/bin/env python3
"""Snarks layer-sweep probe diagnostic.

Per probe-task playbook (memory: probe_task_playbook.md):
  1. Token of interest: last token of option (A) and last token of option (B)
  2. offset_mapping: not needed (option boundaries found via "(A)"/"(B)" markers)
  3. Cheap baselines first: majority class, length, identity-only at L0

Two probe formulations:
  A) Per-option binary: each option contributes one (h_last, is_sarcastic) row.
     Eval: pick option with higher P(sarcastic). N=356, balanced 50/50.
  B) Differential: per example, (h_A - h_B) → label is "A" or "B". N=178.

Reports 5-fold CV across all 25 layers.

Run:
    /Users/jdonaldson/Projects/swollm/.venv/bin/python \\
        experiments/snarks_probe.py
"""
from __future__ import annotations

import re
import sys

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, "/Users/jdonaldson/Projects/swollm/src")
from swollm.bench.bbh import load_task

MODEL_ID = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
TASK = "snarks"

# Match (A) and (B) options as char ranges
_OPTION_RE = re.compile(r"\(([AB])\)\s+(.+?)(?=\n\(|\Z)", flags=re.S)


def load_model():
    device = (
        "mps" if torch.backends.mps.is_available() else
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Loading {MODEL_ID} on {device}…", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    mdl = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.float16
    ).to(device).eval()
    return mdl, tok, device


def find_option_last_tokens(text: str, tokenizer):
    """Return {letter: last_token_idx} for options A and B.

    Uses offset_mapping to map char ranges → token indices.
    """
    encoded = tokenizer(text, return_offsets_mapping=True, add_special_tokens=True)
    offsets = encoded["offset_mapping"]
    result = {}
    for m in _OPTION_RE.finditer(text):
        letter = m.group(1)
        # Last non-whitespace char of option text
        opt_text = m.group(2).rstrip()
        opt_end_char = m.start(2) + len(opt_text)
        # Find token whose offset overlaps the last char (opt_end_char - 1)
        last_idx = None
        for tok_idx, (s, e) in enumerate(offsets):
            if s < opt_end_char and e >= opt_end_char and s != e:
                last_idx = tok_idx
        if last_idx is None:
            return None
        result[letter] = last_idx
    if "A" not in result or "B" not in result:
        return None
    return result, encoded


def collect_records(examples, model, tokenizer, device):
    """Forward each example, capture hidden states at A's and B's last tokens
    across all layers."""
    n_layers = model.config.num_hidden_layers + 1
    records = []
    skipped = 0

    for i, ex in enumerate(examples):
        text = ex["input"]
        target = ex["target"].strip()
        if target not in ("(A)", "(B)"):
            skipped += 1
            continue
        sarcastic = "A" if target == "(A)" else "B"

        out = find_option_last_tokens(text, tokenizer)
        if out is None:
            skipped += 1
            continue
        positions, encoded = out

        ids = {
            "input_ids": torch.tensor([encoded["input_ids"]]).to(device),
            "attention_mask": torch.tensor([encoded["attention_mask"]]).to(device),
        }
        with torch.no_grad():
            out_m = model(**ids, output_hidden_states=True)
        hidden = out_m.hidden_states  # tuple of (1, seq, dim) per layer

        per_layer = {}
        for layer_idx in range(n_layers):
            hA = hidden[layer_idx][0, positions["A"]].float().cpu().numpy()
            hB = hidden[layer_idx][0, positions["B"]].float().cpu().numpy()
            per_layer[layer_idx] = (hA, hB)

        # Length proxy baseline
        m_a = list(_OPTION_RE.finditer(text))
        len_A = len(m_a[0].group(2).strip())
        len_B = len(m_a[1].group(2).strip()) if len(m_a) > 1 else 0

        records.append({
            "sarcastic": sarcastic,
            "len_A": len_A,
            "len_B": len_B,
            "per_layer": per_layer,
            "n_layers": n_layers,
        })

        if (i + 1) % 25 == 0:
            print(f"  [{i+1}/{len(examples)}] records={len(records)}", flush=True)

    if skipped:
        print(f"  skipped {skipped} examples", flush=True)
    return records


def cheap_baselines(records):
    """Majority class, length-based, longer-is-sarcastic."""
    y = np.array([1 if r["sarcastic"] == "A" else 0 for r in records])
    n = len(y)
    majority = max(y.mean(), 1 - y.mean())

    # Length: pick longer option
    pred_longer_A = np.array([1 if r["len_A"] >= r["len_B"] else 0 for r in records])
    pred_shorter_A = 1 - pred_longer_A
    longer_acc = max((pred_longer_A == y).mean(), (pred_shorter_A == y).mean())

    return {
        "n": n,
        "majority": float(majority),
        "longer_or_shorter": float(longer_acc),
    }


def probe_per_option_cv(records, n_splits=5, seed=42):
    """Formulation A: per-option binary 'is sarcastic'.

    For each example, we contribute 2 rows: (h_A, A_is_sarcastic), (h_B, B_is_sarcastic).
    At eval time, predict_proba on both options, pick the one with higher P(sarcastic).
    Crucially: split by *example* so both rows of an example go to the same fold.
    """
    n_layers = records[0]["n_layers"]
    n_ex = len(records)
    examples_idx = np.arange(n_ex)
    y_ex = np.array([1 if r["sarcastic"] == "A" else 0 for r in records])

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = list(skf.split(examples_idx, y_ex))

    layer_acc = {}
    for layer_idx in range(n_layers):
        accs = []
        for tr_idx, te_idx in folds:
            X_tr = []
            y_tr = []
            for i in tr_idx:
                hA, hB = records[i]["per_layer"][layer_idx]
                sarc = records[i]["sarcastic"]
                X_tr.append(hA); y_tr.append(1 if sarc == "A" else 0)
                X_tr.append(hB); y_tr.append(1 if sarc == "B" else 0)
            X_tr = np.array(X_tr); y_tr = np.array(y_tr)

            sc = StandardScaler().fit(X_tr)
            clf = LogisticRegression(max_iter=2000, C=0.1)
            clf.fit(sc.transform(X_tr), y_tr)

            correct = 0
            for i in te_idx:
                hA, hB = records[i]["per_layer"][layer_idx]
                pa = clf.predict_proba(sc.transform(hA[None]))[0, 1]
                pb = clf.predict_proba(sc.transform(hB[None]))[0, 1]
                pred = "A" if pa > pb else "B"
                if pred == records[i]["sarcastic"]:
                    correct += 1
            accs.append(correct / len(te_idx))
        layer_acc[layer_idx] = float(np.mean(accs))
    return layer_acc


def probe_diff_cv(records, n_splits=5, seed=42):
    """Formulation B: (h_A - h_B) → 2-class which is sarcastic."""
    n_layers = records[0]["n_layers"]
    y = np.array([1 if r["sarcastic"] == "A" else 0 for r in records])

    layer_acc = {}
    for layer_idx in range(n_layers):
        X = np.array([
            r["per_layer"][layer_idx][0] - r["per_layer"][layer_idx][1]
            for r in records
        ])
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        accs = []
        for tr_idx, te_idx in skf.split(X, y):
            sc = StandardScaler().fit(X[tr_idx])
            clf = LogisticRegression(max_iter=2000, C=0.1)
            clf.fit(sc.transform(X[tr_idx]), y[tr_idx])
            preds = clf.predict(sc.transform(X[te_idx]))
            accs.append((preds == y[te_idx]).mean())
        layer_acc[layer_idx] = float(np.mean(accs))
    return layer_acc


def length_direction(records):
    """Which direction is the 61%? 'longer is sarcastic' or 'shorter is sarcastic'?"""
    y = np.array([1 if r["sarcastic"] == "A" else 0 for r in records])
    longer_A = np.array([1 if r["len_A"] >= r["len_B"] else 0 for r in records])
    acc_longer = (longer_A == y).mean()
    return "longer" if acc_longer >= 0.5 else "shorter", float(max(acc_longer, 1 - acc_longer))


def orthogonality_check(records, layer_idx, n_splits=5, seed=42):
    """Test whether probe signal is orthogonal to length signal.

    Computes:
      - probe-only at layer_idx (CV)
      - length-only via LogReg on (len_A, len_B, len_diff) (CV)
      - combined: concat(probe_features, length_features) (CV)
      - probe accuracy on length-correct vs length-wrong subsets
    """
    n_ex = len(records)
    y = np.array([1 if r["sarcastic"] == "A" else 0 for r in records])

    # Feature matrices for differential probe
    X_diff = np.array([
        r["per_layer"][layer_idx][0] - r["per_layer"][layer_idx][1]
        for r in records
    ])
    X_len = np.array([[r["len_A"], r["len_B"], r["len_A"] - r["len_B"]]
                      for r in records], dtype=np.float64)

    direction, _ = length_direction(records)
    longer_A = np.array([1 if r["len_A"] >= r["len_B"] else 0 for r in records])
    length_correct = (longer_A == y) if direction == "longer" else (1 - longer_A == y)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = list(skf.split(X_diff, y))

    def cv_acc(X):
        accs = []
        for tr, te in folds:
            sc = StandardScaler().fit(X[tr])
            clf = LogisticRegression(max_iter=2000, C=0.1)
            clf.fit(sc.transform(X[tr]), y[tr])
            accs.append((clf.predict(sc.transform(X[te])) == y[te]).mean())
        return float(np.mean(accs))

    probe_only = cv_acc(X_diff)
    length_only = cv_acc(X_len)
    combined = cv_acc(np.hstack([X_diff, X_len]))

    # Probe accuracy on length-correct vs length-wrong subsets
    # Use LOO-style: train on 4 folds, eval per-subset on held-out
    sub_probe_correct = []
    sub_probe_wrong = []
    for tr, te in folds:
        sc = StandardScaler().fit(X_diff[tr])
        clf = LogisticRegression(max_iter=2000, C=0.1)
        clf.fit(sc.transform(X_diff[tr]), y[tr])
        preds = clf.predict(sc.transform(X_diff[te]))
        for i, idx in enumerate(te):
            ok = preds[i] == y[idx]
            if length_correct[idx]:
                sub_probe_correct.append(ok)
            else:
                sub_probe_wrong.append(ok)

    return {
        "layer": layer_idx,
        "length_direction": direction,
        "probe_only_cv": probe_only,
        "length_only_cv": length_only,
        "combined_cv": combined,
        "probe_acc_when_length_correct": float(np.mean(sub_probe_correct)) if sub_probe_correct else float("nan"),
        "probe_acc_when_length_wrong": float(np.mean(sub_probe_wrong)) if sub_probe_wrong else float("nan"),
        "n_length_correct": len(sub_probe_correct),
        "n_length_wrong": len(sub_probe_wrong),
    }


def main():
    model, tokenizer, device = load_model()
    examples = load_task(TASK)
    print(f"Loaded {len(examples)} {TASK} examples", flush=True)

    print("\nCollecting hidden states at option-last tokens…", flush=True)
    records = collect_records(examples, model, tokenizer, device)
    print(f"Collected {len(records)} records", flush=True)

    base = cheap_baselines(records)
    direction, _ = length_direction(records)
    print(f"\n=== Cheap baselines ===")
    print(f"  N: {base['n']}")
    print(f"  Majority class:           {base['majority']:.1%}")
    print(f"  Length ({direction} is sarcastic): {base['longer_or_shorter']:.1%}")
    print(f"  3-shot baseline (memory): 46.3% (below chance!)")

    print("\n=== Per-option probe (formulation A) ===")
    print("  symmetric: each example contributes h_A & h_B; predict argmax P(sarc)")
    perop = probe_per_option_cv(records)
    for layer_idx, acc in perop.items():
        marker = ""
        if acc > 0.65: marker = " ◀"
        if acc > 0.75: marker = " ◀◀"
        print(f"  L{layer_idx:>2}  {acc:>6.1%}{marker}", flush=True)

    print("\n=== Differential probe (formulation B) ===")
    print("  (h_A - h_B) → 2-class which is sarcastic")
    diffp = probe_diff_cv(records)
    for layer_idx, acc in diffp.items():
        marker = ""
        if acc > 0.65: marker = " ◀"
        if acc > 0.75: marker = " ◀◀"
        print(f"  L{layer_idx:>2}  {acc:>6.1%}{marker}", flush=True)

    bestA = max(perop.items(), key=lambda kv: kv[1])
    bestB = max(diffp.items(), key=lambda kv: kv[1])
    print(f"\nBest per-option:    L{bestA[0]} = {bestA[1]:.1%}")
    print(f"Best differential:  L{bestB[0]} = {bestB[1]:.1%}")
    print(f"Baselines:  majority={base['majority']:.1%}  length={base['longer_or_shorter']:.1%}  3-shot=46.3%")

    print("\n=== Orthogonality check (best differential layer) ===")
    ortho = orthogonality_check(records, bestB[0])
    print(f"  Probe-only CV:                      {ortho['probe_only_cv']:.1%}")
    print(f"  Length-only LR CV:                  {ortho['length_only_cv']:.1%}")
    print(f"  Combined (probe + length) CV:       {ortho['combined_cv']:.1%}")
    print(f"  Δ from combining = {ortho['combined_cv'] - max(ortho['probe_only_cv'], ortho['length_only_cv']):+.1%}")
    print(f"  Probe acc when length-correct (n={ortho['n_length_correct']}):  "
          f"{ortho['probe_acc_when_length_correct']:.1%}")
    print(f"  Probe acc when length-wrong   (n={ortho['n_length_wrong']}):  "
          f"{ortho['probe_acc_when_length_wrong']:.1%}")
    print(f"  Interpretation:")
    print(f"    if probe-on-length-wrong is much lower, probe is rediscovering length")
    print(f"    if it stays high, probe captures something orthogonal")


if __name__ == "__main__":
    main()
