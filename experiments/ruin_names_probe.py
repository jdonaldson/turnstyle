#!/usr/bin/env python3
"""Ruin_names layer-sweep probe diagnostic.

Per probe-task playbook:
  1. Token of interest: last token of each option (4-way → 1-of-4).
  2. offset_mapping for char→token resolution.
  3. Cheap baselines critical: majority, length, *and per-option LM perplexity*.

The humorous option is always a real-English-word edit ("ruin man",
"the dork knight rises", "the shawshark redemption") while distractors
are typo-gibberish ("thetdark", "shawshanknredemption"). Per-option
LM perplexity is the most plausible confound — if it alone reaches 70%+,
the L-probe is likely rediscovering the LM signal.

Run:
    /Users/jdonaldson/Projects/swollm/.venv/bin/python \\
        experiments/ruin_names_probe.py
"""
from __future__ import annotations

import re
import sys

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, "/Users/jdonaldson/Projects/swollm/src")
from swollm.bench.bbh import load_task

MODEL_ID = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
TASK = "ruin_names"

_OPTION_RE = re.compile(r"\(([ABCD])\)\s+(.+?)(?=\n\(|\Z)", flags=re.S)
_NAME_RE = re.compile(r"name:\s*'([^']+)'")


def load_model():
    device = ("mps" if torch.backends.mps.is_available() else
              "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading {MODEL_ID} on {device}…", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    mdl = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.float16
    ).to(device).eval()
    return mdl, tok, device


def find_option_last_tokens(text: str, tokenizer):
    """{letter: last_token_idx} for options A/B/C/D using offset_mapping."""
    encoded = tokenizer(text, return_offsets_mapping=True, add_special_tokens=True)
    offsets = encoded["offset_mapping"]
    positions = {}
    for m in _OPTION_RE.finditer(text):
        letter = m.group(1)
        opt_text = m.group(2).rstrip()
        opt_end_char = m.start(2) + len(opt_text)
        last_idx = None
        for tok_idx, (s, e) in enumerate(offsets):
            if s < opt_end_char and e >= opt_end_char and s != e:
                last_idx = tok_idx
        if last_idx is None:
            return None
        positions[letter] = last_idx
    if set(positions) != {"A", "B", "C", "D"}:
        return None
    return positions, encoded


def per_option_perplexity(option_texts: list[str], model, tokenizer, device):
    """Mean NLL per option, computed from a standalone forward pass on each.

    Lower NLL = more "real English" = more likely the humorous edit.
    """
    nlls = []
    for text in option_texts:
        ids = tokenizer(text, return_tensors="pt").to(device)
        input_ids = ids["input_ids"]
        if input_ids.shape[1] < 2:
            nlls.append(float("nan"))
            continue
        with torch.no_grad():
            out = model(input_ids)
        # Per-token NLL of x_t given x_<t
        logits = out.logits[0, :-1].float()
        targets = input_ids[0, 1:]
        nll = F.cross_entropy(logits, targets, reduction="mean").item()
        nlls.append(nll)
    return nlls


def collect_records(examples, model, tokenizer, device):
    n_layers = model.config.num_hidden_layers + 1
    records = []
    skipped = 0

    for i, ex in enumerate(examples):
        text = ex["input"]
        target = ex["target"].strip()
        m = re.match(r"\(([ABCD])\)", target)
        if not m:
            skipped += 1; continue
        humorous = m.group(1)

        out = find_option_last_tokens(text, tokenizer)
        if out is None:
            skipped += 1; continue
        positions, encoded = out

        # Extract option texts for perplexity
        opt_texts = {}
        for mm in _OPTION_RE.finditer(text):
            opt_texts[mm.group(1)] = mm.group(2).strip()

        # Single forward pass for hidden states
        ids = {
            "input_ids": torch.tensor([encoded["input_ids"]]).to(device),
            "attention_mask": torch.tensor([encoded["attention_mask"]]).to(device),
        }
        with torch.no_grad():
            out_m = model(**ids, output_hidden_states=True)
        hidden = out_m.hidden_states

        per_layer = {}
        for layer_idx in range(n_layers):
            per_layer[layer_idx] = {
                letter: hidden[layer_idx][0, pos].float().cpu().numpy()
                for letter, pos in positions.items()
            }

        # Per-option standalone perplexity (4 small forward passes)
        nll = per_option_perplexity([opt_texts[k] for k in "ABCD"],
                                     model, tokenizer, device)
        nll_dict = dict(zip("ABCD", nll))

        # Length per option
        len_dict = {k: len(opt_texts[k]) for k in "ABCD"}

        records.append({
            "humorous": humorous,
            "per_layer": per_layer,
            "nll": nll_dict,
            "len": len_dict,
            "n_layers": n_layers,
        })

        if (i + 1) % 25 == 0:
            print(f"  [{i+1}/{len(examples)}] records={len(records)}", flush=True)

    if skipped:
        print(f"  skipped {skipped}", flush=True)
    return records


def cheap_baselines(records):
    """Majority class, longest option, min-perplexity (real-English-likeliest)."""
    n = len(records)
    targets = [r["humorous"] for r in records]
    from collections import Counter
    cnt = Counter(targets)
    majority = max(cnt.values()) / n

    longest_correct = sum(
        1 for r in records
        if max(r["len"], key=r["len"].get) == r["humorous"]
    )
    shortest_correct = sum(
        1 for r in records
        if min(r["len"], key=r["len"].get) == r["humorous"]
    )

    min_nll_correct = sum(
        1 for r in records
        if (lambda d: min(d, key=d.get))(r["nll"]) == r["humorous"]
    )
    max_nll_correct = sum(
        1 for r in records
        if (lambda d: max(d, key=d.get))(r["nll"]) == r["humorous"]
    )

    return {
        "n": n,
        "majority": majority,
        "longest": longest_correct / n,
        "shortest": shortest_correct / n,
        "min_perplexity": min_nll_correct / n,
        "max_perplexity": max_nll_correct / n,
        "class_dist": dict(cnt),
    }


def probe_per_option_cv(records, n_splits=5, seed=42):
    """Per-option binary 'is humorous?' (1 of 4 positive per example).

    Each example contributes 4 rows: (h_letter, is_humorous_letter).
    Eval: predict P(humorous) for all 4 options, argmax → letter.
    """
    n_layers = records[0]["n_layers"]
    n_ex = len(records)
    examples_idx = np.arange(n_ex)
    # Stratify on which letter is humorous (somewhat balanced anyway)
    y_class = np.array([ord(r["humorous"]) - ord("A") for r in records])

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = list(skf.split(examples_idx, y_class))

    layer_acc = {}
    for layer_idx in range(n_layers):
        accs = []
        for tr_idx, te_idx in folds:
            X_tr, y_tr = [], []
            for i in tr_idx:
                for letter in "ABCD":
                    X_tr.append(records[i]["per_layer"][layer_idx][letter])
                    y_tr.append(1 if letter == records[i]["humorous"] else 0)
            X_tr = np.array(X_tr); y_tr = np.array(y_tr)

            sc = StandardScaler().fit(X_tr)
            clf = LogisticRegression(max_iter=2000, C=0.1, class_weight="balanced")
            clf.fit(sc.transform(X_tr), y_tr)

            correct = 0
            for i in te_idx:
                scores = {}
                for letter in "ABCD":
                    h = records[i]["per_layer"][layer_idx][letter]
                    scores[letter] = clf.predict_proba(sc.transform(h[None]))[0, 1]
                pred = max(scores, key=scores.get)
                if pred == records[i]["humorous"]:
                    correct += 1
            accs.append(correct / len(te_idx))
        layer_acc[layer_idx] = float(np.mean(accs))
    return layer_acc


def orthogonality_check(records, layer_idx, n_splits=5, seed=42):
    """Test whether L-probe signal is orthogonal to perplexity.

    Three feature sets:
      a) hidden_state alone
      b) hidden_state + nll_per_option (per row: append NLL of that option)
      c) probe accuracy on perplexity-correct vs perplexity-wrong subsets
    """
    n_ex = len(records)
    y_class = np.array([ord(r["humorous"]) - ord("A") for r in records])

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = list(skf.split(np.arange(n_ex), y_class))

    # Sanitize NaN nll values: replace with global median across all options.
    all_nlls = [v for r in records for v in r["nll"].values()
                if not np.isnan(v)]
    median_nll = float(np.median(all_nlls)) if all_nlls else 0.0
    for r in records:
        for k, v in r["nll"].items():
            if np.isnan(v):
                r["nll"][k] = median_nll

    perp_pred = []
    for r in records:
        perp_pred.append(min(r["nll"], key=r["nll"].get))
    perp_correct = np.array([p == r["humorous"]
                             for p, r in zip(perp_pred, records)])

    def cv_accuracy(use_nll: bool):
        accs = []
        for tr_idx, te_idx in folds:
            X_tr, y_tr = [], []
            for i in tr_idx:
                for letter in "ABCD":
                    feat = records[i]["per_layer"][layer_idx][letter]
                    if use_nll:
                        feat = np.concatenate([feat, [records[i]["nll"][letter]]])
                    X_tr.append(feat)
                    y_tr.append(1 if letter == records[i]["humorous"] else 0)
            X_tr = np.array(X_tr); y_tr = np.array(y_tr)
            sc = StandardScaler().fit(X_tr)
            clf = LogisticRegression(max_iter=2000, C=0.1,
                                     class_weight="balanced")
            clf.fit(sc.transform(X_tr), y_tr)

            correct = 0
            for i in te_idx:
                scores = {}
                for letter in "ABCD":
                    feat = records[i]["per_layer"][layer_idx][letter]
                    if use_nll:
                        feat = np.concatenate([feat, [records[i]["nll"][letter]]])
                    scores[letter] = clf.predict_proba(sc.transform(feat[None]))[0, 1]
                pred = max(scores, key=scores.get)
                if pred == records[i]["humorous"]:
                    correct += 1
            accs.append(correct / len(te_idx))
        return float(np.mean(accs))

    # subset analysis using probe-only at this layer (re-do CV to track per-example)
    probe_correct_when_perp_right = []
    probe_correct_when_perp_wrong = []
    for tr_idx, te_idx in folds:
        X_tr, y_tr = [], []
        for i in tr_idx:
            for letter in "ABCD":
                X_tr.append(records[i]["per_layer"][layer_idx][letter])
                y_tr.append(1 if letter == records[i]["humorous"] else 0)
        X_tr = np.array(X_tr); y_tr = np.array(y_tr)
        sc = StandardScaler().fit(X_tr)
        clf = LogisticRegression(max_iter=2000, C=0.1, class_weight="balanced")
        clf.fit(sc.transform(X_tr), y_tr)
        for i in te_idx:
            scores = {}
            for letter in "ABCD":
                h = records[i]["per_layer"][layer_idx][letter]
                scores[letter] = clf.predict_proba(sc.transform(h[None]))[0, 1]
            pred = max(scores, key=scores.get)
            ok = pred == records[i]["humorous"]
            if perp_correct[i]:
                probe_correct_when_perp_right.append(ok)
            else:
                probe_correct_when_perp_wrong.append(ok)

    return {
        "layer": layer_idx,
        "probe_only": cv_accuracy(use_nll=False),
        "probe_plus_nll": cv_accuracy(use_nll=True),
        "probe_acc_perp_correct": float(np.mean(probe_correct_when_perp_right))
            if probe_correct_when_perp_right else float("nan"),
        "probe_acc_perp_wrong": float(np.mean(probe_correct_when_perp_wrong))
            if probe_correct_when_perp_wrong else float("nan"),
        "n_perp_correct": len(probe_correct_when_perp_right),
        "n_perp_wrong": len(probe_correct_when_perp_wrong),
    }


def main():
    model, tokenizer, device = load_model()
    examples = load_task(TASK)
    print(f"Loaded {len(examples)} {TASK} examples", flush=True)

    print("\nCollecting hidden states + per-option perplexity…", flush=True)
    records = collect_records(examples, model, tokenizer, device)
    print(f"Collected {len(records)} records", flush=True)

    base = cheap_baselines(records)
    print("\n=== Cheap baselines ===")
    print(f"  N: {base['n']}  class dist: {base['class_dist']}")
    print(f"  Majority class:                {base['majority']:.1%}")
    print(f"  Longest option:                {base['longest']:.1%}")
    print(f"  Shortest option:               {base['shortest']:.1%}")
    print(f"  Min-perplexity (real English): {base['min_perplexity']:.1%}")
    print(f"  Max-perplexity:                {base['max_perplexity']:.1%}")
    print(f"  3-shot baseline (memory):      28.3%")

    print("\n=== Per-option probe (4-way → argmax P(humorous)) ===")
    perop = probe_per_option_cv(records)
    for layer_idx, acc in perop.items():
        marker = ""
        if acc > 0.40: marker = " ◀"
        if acc > 0.55: marker = " ◀◀"
        print(f"  L{layer_idx:>2}  {acc:>6.1%}{marker}", flush=True)

    best = max(perop.items(), key=lambda kv: kv[1])
    print(f"\nBest probe layer: L{best[0]} = {best[1]:.1%}")
    print(f"vs cheap baselines: majority={base['majority']:.1%}  "
          f"min-perplexity={base['min_perplexity']:.1%}  "
          f"longest={base['longest']:.1%}")

    print("\n=== Orthogonality check vs perplexity (best layer) ===")
    ortho = orthogonality_check(records, best[0])
    print(f"  Probe-only CV:                  {ortho['probe_only']:.1%}")
    print(f"  Probe + NLL feature CV:         {ortho['probe_plus_nll']:.1%}")
    print(f"  Δ from adding NLL: {ortho['probe_plus_nll'] - ortho['probe_only']:+.1%}")
    print(f"  Probe acc when perplexity correct (n={ortho['n_perp_correct']}):  "
          f"{ortho['probe_acc_perp_correct']:.1%}")
    print(f"  Probe acc when perplexity wrong   (n={ortho['n_perp_wrong']}):  "
          f"{ortho['probe_acc_perp_wrong']:.1%}")
    print("  Interpretation:")
    print("    if probe-on-perp-wrong subset is much lower, probe rediscovers perplexity")
    print("    if it stays high, probe captures something orthogonal")


if __name__ == "__main__":
    main()
