#!/usr/bin/env python3
"""Salient_translation per-option last-token probe diagnostic.

Each example has 6 option positions (A-F) with category-name texts
shuffled into them. Per-option last-token probing — the pattern that
landed snarks (74% CV) and ruin_names (85.5% CV) — is the natural next
attempt before declaring the model's representation insufficient.

Run:
    /Users/jdonaldson/Projects/swollm/.venv/bin/python \\
        experiments/salient_translation_option_probe.py
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

_OPTION_RE = re.compile(r"\(([A-F])\)\s+(.+?)(?=\n\(|\Z)", flags=re.S)


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
    encoded = tokenizer(text, return_offsets_mapping=True,
                        add_special_tokens=True)
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
    if set(positions) != set("ABCDEF"):
        return None
    return positions, encoded


def collect(examples, model, tokenizer, device):
    n_layers = model.config.num_hidden_layers + 1
    records = []
    skipped = 0
    for i, ex in enumerate(examples):
        text = ex["input"]
        target = ex["target"].strip()
        m_t = re.match(r"\(([A-F])\)", target)
        if not m_t:
            skipped += 1; continue
        correct = m_t.group(1)

        out = find_option_last_tokens(text, tokenizer)
        if out is None:
            skipped += 1; continue
        positions, encoded = out

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

        records.append({
            "correct": correct,
            "per_layer": per_layer,
            "n_layers": n_layers,
        })

        if (i + 1) % 25 == 0:
            print(f"  [{i+1}/{len(examples)}] records={len(records)}", flush=True)
    if skipped:
        print(f"  skipped {skipped}", flush=True)
    return records


def probe_per_option_cv(records, n_splits=5, seed=42):
    n_layers = records[0]["n_layers"]
    n_ex = len(records)
    y_class = np.array([ord(r["correct"]) - ord("A") for r in records])
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = list(skf.split(np.arange(n_ex), y_class))

    layer_acc = {}
    for layer_idx in range(n_layers):
        accs = []
        for tr_idx, te_idx in folds:
            X_tr, y_tr = [], []
            for i in tr_idx:
                for letter in "ABCDEF":
                    X_tr.append(records[i]["per_layer"][layer_idx][letter])
                    y_tr.append(1 if letter == records[i]["correct"] else 0)
            X_tr = np.array(X_tr); y_tr = np.array(y_tr)
            sc = StandardScaler().fit(X_tr)
            clf = LogisticRegression(
                max_iter=2000, C=0.1, class_weight="balanced"
            )
            clf.fit(sc.transform(X_tr), y_tr)

            correct = 0
            for i in te_idx:
                scores = {}
                for letter in "ABCDEF":
                    h = records[i]["per_layer"][layer_idx][letter]
                    scores[letter] = clf.predict_proba(sc.transform(h[None]))[0, 1]
                pred = max(scores, key=scores.get)
                if pred == records[i]["correct"]:
                    correct += 1
            accs.append(correct / len(te_idx))
        layer_acc[layer_idx] = float(np.mean(accs))
    return layer_acc


def main():
    model, tokenizer, device = load_model()
    examples = load_task(TASK)
    print(f"Loaded {len(examples)} {TASK} examples", flush=True)

    records = collect(examples, model, tokenizer, device)
    print(f"Collected {len(records)}", flush=True)

    print("\n=== Per-option last-token probe ===")
    accs = probe_per_option_cv(records)
    for layer_idx, acc in accs.items():
        marker = ""
        if acc > 0.30: marker = " ◀"
        if acc > 0.50: marker = " ◀◀"
        print(f"  L{layer_idx:>2}  {acc:>6.1%}{marker}", flush=True)
    best = max(accs.items(), key=lambda kv: kv[1])
    print(f"\nBest: L{best[0]} = {best[1]:.1%}")
    print(f"Reference: chance=16.7%  majority=22.4%  3-shot~28%  "
          f"last-of-translation best CV=44.8%")


if __name__ == "__main__":
    main()
