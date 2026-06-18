#!/usr/bin/env python3
"""Salient_translation_error_detection layer-sweep probe diagnostic.

Per probe-task playbook:
  Token of interest: three hypotheses tested in parallel
    (a) last token of full prompt
    (b) last token of translation text (where source/translation
        comparison completes)
    (c) differential: h_translation_last - h_source_last
  Cheap baselines: majority class, identity-only (L0).

The task is 6-way: which category of error is in the translation —
Named Entities, Numerical Values, Modifiers, Negation, Facts, or
Dropped Content. The category positions in the option list shuffle
per example, so we predict the *category index* (0..5) by reading
the option mapping per example.

Run:
    /Users/jdonaldson/Projects/swollm/.venv/bin/python \\
        experiments/salient_translation_probe.py
"""
from __future__ import annotations

import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

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
TASK = "salient_translation_error_detection"

CATEGORIES = [
    "Named Entities",
    "Numerical Values",
    "Modifiers or Adjectives",
    "Negation or Antonyms",
    "Facts",
    "Dropped Content",
]
CAT_TO_IDX = {c: i for i, c in enumerate(CATEGORIES)}

_OPTION_RE = re.compile(r"\(([A-F])\)\s+(.+?)(?=\n\(|\Z)", flags=re.S)
_SOURCE_RE = re.compile(r"Source:\s*(.+?)\nTranslation:", flags=re.S)
_TRANS_RE = re.compile(r"Translation:\s*(.+?)\n", flags=re.S)


def load_model():
    device = ("mps" if torch.backends.mps.is_available() else
              "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading {MODEL_ID} on {device}…", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    mdl = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.float16
    ).to(device).eval()
    return mdl, tok, device


def find_char_position_token(offsets, end_char):
    """Return the token whose offset spans up through end_char."""
    last_idx = None
    for tok_idx, (s, e) in enumerate(offsets):
        if s < end_char and e >= end_char and s != e:
            last_idx = tok_idx
    return last_idx


def collect_records(examples, model, tokenizer, device):
    n_layers = model.config.num_hidden_layers + 1
    records = []
    skipped = 0

    for i, ex in enumerate(examples):
        text = ex["input"]
        target = ex["target"].strip()
        m_t = re.match(r"\(([A-F])\)", target)
        if not m_t:
            skipped += 1
            continue
        target_letter = m_t.group(1)

        # Map letter → category text by parsing the option block
        opts = list(_OPTION_RE.finditer(text))
        letter_to_cat = {m.group(1): m.group(2).strip() for m in opts}
        if target_letter not in letter_to_cat:
            skipped += 1
            continue
        target_cat = letter_to_cat[target_letter]
        if target_cat not in CAT_TO_IDX:
            skipped += 1
            continue
        category_idx = CAT_TO_IDX[target_cat]

        # Locate source / translation char ranges
        m_src = _SOURCE_RE.search(text)
        m_tr = _TRANS_RE.search(text)
        if not m_src or not m_tr:
            skipped += 1
            continue
        src_end_char = m_src.end(1)
        tr_end_char = m_tr.end(1)

        encoded = tokenizer(text, return_offsets_mapping=True,
                            add_special_tokens=True)
        offsets = encoded["offset_mapping"]
        src_tok = find_char_position_token(offsets, src_end_char)
        tr_tok = find_char_position_token(offsets, tr_end_char)
        if src_tok is None or tr_tok is None:
            skipped += 1
            continue

        ids = {
            "input_ids": torch.tensor([encoded["input_ids"]]).to(device),
            "attention_mask": torch.tensor([encoded["attention_mask"]]).to(device),
        }
        with torch.no_grad():
            out = model(**ids, output_hidden_states=True)
        hidden = out.hidden_states  # tuple per layer (1, seq, dim)
        seq_len = ids["input_ids"].shape[1]
        last_tok = seq_len - 1

        per_layer = {}
        for layer_idx in range(n_layers):
            h_last = hidden[layer_idx][0, last_tok].float().cpu().numpy()
            h_src = hidden[layer_idx][0, src_tok].float().cpu().numpy()
            h_tr = hidden[layer_idx][0, tr_tok].float().cpu().numpy()
            per_layer[layer_idx] = {
                "last": h_last,
                "src": h_src,
                "tr": h_tr,
                "diff": h_tr - h_src,
            }

        records.append({
            "category_idx": category_idx,
            "category": target_cat,
            "per_layer": per_layer,
            "n_layers": n_layers,
        })

        if (i + 1) % 25 == 0:
            print(f"  [{i+1}/{len(examples)}] records={len(records)}",
                  flush=True)

    if skipped:
        print(f"  skipped {skipped}", flush=True)
    return records


def cheap_baselines(records):
    n = len(records)
    from collections import Counter
    cnt = Counter(r["category"] for r in records)
    majority = max(cnt.values()) / n
    return {"n": n, "majority": majority, "class_dist": dict(cnt),
            "chance": 1.0 / 6}


def probe_cv(records, key, n_splits=5, seed=42):
    """6-class LogReg on records[i]['per_layer'][L][key]."""
    n_layers = records[0]["n_layers"]
    y = np.array([r["category_idx"] for r in records])

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    layer_acc = {}
    for layer_idx in range(n_layers):
        X = np.array([r["per_layer"][layer_idx][key] for r in records])
        accs = []
        for tr, te in skf.split(X, y):
            sc = StandardScaler().fit(X[tr])
            clf = LogisticRegression(max_iter=2000, C=0.1)
            clf.fit(sc.transform(X[tr]), y[tr])
            accs.append((clf.predict(sc.transform(X[te])) == y[te]).mean())
        layer_acc[layer_idx] = float(np.mean(accs))
    return layer_acc


def main():
    model, tokenizer, device = load_model()
    examples = load_task(TASK)
    print(f"Loaded {len(examples)} {TASK} examples", flush=True)

    print("\nCollecting hidden states (last/src/tr)…", flush=True)
    records = collect_records(examples, model, tokenizer, device)
    print(f"Collected {len(records)} records", flush=True)

    base = cheap_baselines(records)
    print("\n=== Cheap baselines ===")
    print(f"  N: {base['n']}")
    print(f"  Chance (1/6):       {base['chance']:.1%}")
    print(f"  Majority class:     {base['majority']:.1%}")
    print(f"  Class dist:         {base['class_dist']}")

    for key, label in [("last", "last token of prompt"),
                       ("tr",   "last token of translation"),
                       ("diff", "h_tr - h_src (differential)")]:
        print(f"\n=== Probe: {label} ===")
        accs = probe_cv(records, key)
        for layer_idx, acc in accs.items():
            marker = ""
            if acc > 0.30: marker = " ◀"
            if acc > 0.45: marker = " ◀◀"
            print(f"  L{layer_idx:>2}  {acc:>6.1%}{marker}", flush=True)
        best = max(accs.items(), key=lambda kv: kv[1])
        print(f"  Best: L{best[0]} = {best[1]:.1%}")

    # Concatenated probes: combine multiple positions at the best layer
    print("\n=== Concatenation probes (richer features) ===")
    n_layers = records[0]["n_layers"]
    y = np.array([r["category_idx"] for r in records])
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for combo_name, keys in [
        ("[last + tr]",        ["last", "tr"]),
        ("[last + tr + diff]", ["last", "tr", "diff"]),
        ("[src + tr + diff]",  ["src",  "tr", "diff"]),
    ]:
        layer_accs = {}
        for layer_idx in range(n_layers):
            X = np.array([
                np.concatenate([r["per_layer"][layer_idx][k] for k in keys])
                for r in records
            ])
            accs = []
            for tr, te in skf.split(X, y):
                sc = StandardScaler().fit(X[tr])
                clf = LogisticRegression(max_iter=2000, C=0.05)
                clf.fit(sc.transform(X[tr]), y[tr])
                accs.append((clf.predict(sc.transform(X[te])) == y[te]).mean())
            layer_accs[layer_idx] = float(np.mean(accs))
        best = max(layer_accs.items(), key=lambda kv: kv[1])
        print(f"  {combo_name:25s}  best L{best[0]:>2} = {best[1]:.1%}")

    # Cross-layer concatenation (last at L15 + tr at L13)
    X_cross = np.array([
        np.concatenate([r["per_layer"][15]["last"], r["per_layer"][13]["tr"]])
        for r in records
    ])
    accs = []
    for tr_idx, te_idx in skf.split(X_cross, y):
        sc = StandardScaler().fit(X_cross[tr_idx])
        clf = LogisticRegression(max_iter=2000, C=0.05)
        clf.fit(sc.transform(X_cross[tr_idx]), y[tr_idx])
        accs.append((clf.predict(sc.transform(X_cross[te_idx])) == y[te_idx]).mean())
    print(f"  cross-layer last@L15 + tr@L13 = {float(np.mean(accs)):.1%}")

    print(f"\nReference: chance={base['chance']:.1%}  "
          f"majority={base['majority']:.1%}  3-shot baseline ~28%")


if __name__ == "__main__":
    main()
