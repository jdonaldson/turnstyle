#!/usr/bin/env python3
"""Train and save the option-boundary structural probe artifact.

Output: turnstyle/src/turnstyle/data/structural_probes/option_boundary.npz

The probe is a per-token binary classifier — "is this token the start of an
option line" — trained on a mix of formats so it generalizes to unseen
formats at inference time.

Training set: 5 BBH multiple-choice tasks × 7 format rewrites each. Held-out
validation: a 6th task (logical_deduction_three_objects) plus a held-out
format (Roman numerals) the probe never sees during training.

Run:
    /Users/jdonaldson/Projects/swollm/.venv/bin/python \\
        experiments/train_option_boundary_probe.py
"""
from __future__ import annotations

import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, "/Users/jdonaldson/Projects/swollm/src")
from swollm.bench.bbh import load_task

MODEL_ID = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
TRAIN_TASKS = ["snarks", "ruin_names", "movie_recommendation",
               "salient_translation_error_detection", "disambiguation_qa"]
VAL_TASK = "logical_deduction_three_objects"
N_PER_TASK = 80
LAYERS_TO_TRY = [1, 2, 4, 8]   # prototype showed L4-L8 best F1 with mixed training

OUT_DIR = Path("/Users/jdonaldson/Projects/turnstyle/src/turnstyle/data/structural_probes")
OUT_PATH = OUT_DIR / "option_boundary.npz"

# Per-format anchor regexes (m.start() at the option marker's first char)
PAREN_RE       = re.compile(r"\(([A-Z])\)\s+(.+?)(?=\n\(|\Z)", flags=re.S)
LETTER_DOT_RE  = re.compile(r"^([A-Z])\.\s+", flags=re.M)
NUM_DOT_RE     = re.compile(r"^(\d+)\.\s+", flags=re.M)
CHOICE_RE      = re.compile(r"^Choice\s+([A-Z]):\s+", flags=re.M)
BRACKET_RE     = re.compile(r"\[([A-Z])\]\s+")
LOWER_PAREN_RE = re.compile(r"\(([a-z])\)\s+(.+?)(?=\n\(|\Z)", flags=re.S)
ROMAN_RE       = re.compile(r"\(([ivx]+)\)\s+(.+?)(?=\n\(|\Z)", flags=re.S | re.I)

# Training formats (the probe sees all of these)
TRAIN_FORMATS = ["(A)", "A.", "1.", "Choice A:", "[A]", "(a)"]
# Held-out format (probe never trains on this — tests true cross-format generalization)
HELDOUT_FORMATS = ["(i)"]   # roman numerals

FORMAT_RE = {
    "(A)": PAREN_RE,
    "A.": LETTER_DOT_RE,
    "1.": NUM_DOT_RE,
    "Choice A:": CHOICE_RE,
    "[A]": BRACKET_RE,
    "(a)": LOWER_PAREN_RE,
    "(i)": ROMAN_RE,
}


def rewrite(text, fmt):
    """Rewrite a (A)-format prompt into another format."""
    def replace(m):
        letter = m.group(1)
        content = m.group(2)
        if fmt == "(A)":
            return m.group(0)
        if fmt == "A.":
            return f"{letter}. {content}"
        if fmt == "1.":
            n = ord(letter) - ord("A") + 1
            return f"{n}. {content}"
        if fmt == "Choice A:":
            return f"Choice {letter}: {content}"
        if fmt == "[A]":
            return f"[{letter}] {content}"
        if fmt == "(a)":
            return f"({letter.lower()}) {content}"
        if fmt == "(i)":
            n = ord(letter) - ord("A") + 1
            roman = {1:"i",2:"ii",3:"iii",4:"iv",5:"v",6:"vi",7:"vii"}
            return f"({roman.get(n, 'x')}) {content}"
        return m.group(0)
    return PAREN_RE.sub(replace, text)


def label_starts(text, encoded, marker_re):
    """Return per-token labels: 1 if token contains start of option marker."""
    n = len(encoded["input_ids"])
    labels = np.zeros(n, dtype=int)
    for m in marker_re.finditer(text):
        start_char = m.start()
        for tok_idx, (s, e) in enumerate(encoded["offset_mapping"]):
            if s <= start_char < e and s != e:
                labels[tok_idx] = 1
                break
    return labels


def load_model():
    device = ("mps" if torch.backends.mps.is_available() else
              "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading {MODEL_ID} on {device}…", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    mdl = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.float16
    ).to(device).eval()
    return mdl, tok, device


def collect(prompts_with_fmt, model, tokenizer, device, layers):
    """For each (prompt, fmt) pair: forward pass, capture per-token hidden
    states at the chosen layers and per-token labels using the format's regex."""
    out = []
    for i, (text, fmt) in enumerate(prompts_with_fmt):
        encoded = tokenizer(text, return_offsets_mapping=True,
                            add_special_tokens=True)
        labels = label_starts(text, encoded, FORMAT_RE[fmt])
        ids = {
            "input_ids": torch.tensor([encoded["input_ids"]]).to(device),
            "attention_mask": torch.tensor([encoded["attention_mask"]]).to(device),
        }
        with torch.no_grad():
            o = model(**ids, output_hidden_states=True)
        per_layer = {
            l: o.hidden_states[l][0].float().cpu().numpy().astype(np.float16)
            for l in layers
        }
        out.append({"X": per_layer, "y": labels, "fmt": fmt,
                    "n_options": int(labels.sum())})
        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(prompts_with_fmt)}]", flush=True)
    return out


def evaluate(records, layer, sc, clf):
    """Per-token F1 + per-prompt boundary recovery (top-K)."""
    all_preds, all_labels = [], []
    recoveries = []
    for r in records:
        X = r["X"][layer].astype(np.float32)
        y = r["y"]
        scores = clf.predict_proba(sc.transform(X))[:, 1]
        preds = (scores > 0.5).astype(int)
        all_preds.extend(preds)
        all_labels.extend(y)
        K = int(y.sum())
        if K == 0:
            continue
        top_k = np.argsort(-scores)[:K]
        truth = np.where(y == 1)[0]
        recoveries.append(len(set(top_k) & set(truth)) / K)
    return {
        "f1": f1_score(all_labels, all_preds, zero_division=0),
        "recovery": float(np.mean(recoveries)),
        "perfect": sum(1 for r in recoveries if r == 1.0),
        "n": len(recoveries),
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model, tokenizer, device = load_model()

    # --- 1. Build training set: every train task × every train format ---
    train_pf = []
    for task in TRAIN_TASKS:
        examples = load_task(task)[:N_PER_TASK]
        for fmt in TRAIN_FORMATS:
            for ex in examples:
                train_pf.append((rewrite(ex["input"], fmt), fmt))
    print(f"\nTraining set: {len(train_pf)} (prompt, format) pairs "
          f"({len(TRAIN_TASKS)} tasks × {len(TRAIN_FORMATS)} formats × "
          f"{N_PER_TASK} prompts)")

    # --- 2. Held-out: held-out task in trained formats + held-out format ---
    val_examples = load_task(VAL_TASK)[:N_PER_TASK]
    val_in_train_fmts = []
    for fmt in TRAIN_FORMATS:
        for ex in val_examples:
            val_in_train_fmts.append((rewrite(ex["input"], fmt), fmt))
    val_in_heldout_fmts = []
    for fmt in HELDOUT_FORMATS:
        for ex in val_examples:
            val_in_heldout_fmts.append((rewrite(ex["input"], fmt), fmt))
    print(f"Validation: {len(val_in_train_fmts)} held-out-task in trained fmts, "
          f"{len(val_in_heldout_fmts)} held-out-task in held-out fmts")

    # --- 3. Collect ---
    print("\nCollecting training records…", flush=True)
    t0 = time.time()
    train_recs = collect(train_pf, model, tokenizer, device, LAYERS_TO_TRY)
    print(f"  done in {time.time()-t0:.1f}s")

    print("\nCollecting val records (trained formats)…", flush=True)
    t0 = time.time()
    val_train_recs = collect(val_in_train_fmts, model, tokenizer, device, LAYERS_TO_TRY)
    print(f"  done in {time.time()-t0:.1f}s")

    print("\nCollecting val records (held-out formats)…", flush=True)
    t0 = time.time()
    val_heldout_recs = collect(val_in_heldout_fmts, model, tokenizer, device, LAYERS_TO_TRY)
    print(f"  done in {time.time()-t0:.1f}s")

    # --- 4. Layer sweep ---
    print(f"\n=== Layer sweep ===")
    print(f"{'layer':>5}  {'F1(in)':>7}  {'rec(in)':>7}  "
          f"{'F1(holdfmt)':>11}  {'rec(holdfmt)':>11}")
    print("─" * 60)
    layer_results = {}
    for layer in LAYERS_TO_TRY:
        X_tr = np.concatenate([r["X"][layer].astype(np.float32) for r in train_recs])
        y_tr = np.concatenate([r["y"] for r in train_recs])
        sc = StandardScaler().fit(X_tr)
        clf = LogisticRegression(max_iter=2000, C=0.1, class_weight="balanced")
        clf.fit(sc.transform(X_tr), y_tr)

        in_fmt = evaluate(val_train_recs, layer, sc, clf)
        holdfmt = evaluate(val_heldout_recs, layer, sc, clf)
        layer_results[layer] = (sc, clf, in_fmt, holdfmt)
        print(f"  L{layer:<3}  {in_fmt['f1']:>6.3f}  {in_fmt['recovery']:>6.3f}  "
              f"{holdfmt['f1']:>10.3f}  {holdfmt['recovery']:>10.3f}")

    # --- 5. Per-format breakdown for the chosen layer ---
    chosen = max(layer_results, key=lambda L: (
        layer_results[L][2]["f1"] + layer_results[L][3]["f1"],
    ))
    sc, clf, _, _ = layer_results[chosen]
    print(f"\nChosen layer: L{chosen} (best F1 sum across in-fmt + held-out-fmt)")

    print(f"\n=== Per-format breakdown @ L{chosen} ===")
    print(f"{'format':<14}  {'F1':>6}  {'recov':>6}  perfect")
    print("─" * 40)
    all_recs = val_train_recs + val_heldout_recs
    for fmt in TRAIN_FORMATS + HELDOUT_FORMATS:
        recs_fmt = [r for r in all_recs if r["fmt"] == fmt]
        if not recs_fmt:
            continue
        r = evaluate(recs_fmt, chosen, sc, clf)
        flag = " (HELD-OUT)" if fmt in HELDOUT_FORMATS else ""
        print(f"  {fmt:<14}  {r['f1']:>5.3f}  {r['recovery']:>5.3f}  "
              f"{r['perfect']}/{r['n']}{flag}")

    # --- 6. Save artifact ---
    np.savez_compressed(
        OUT_PATH,
        layer=np.int32(chosen),
        mean=sc.mean_.astype(np.float16),
        std=sc.scale_.astype(np.float16),
        weights=clf.coef_[0].astype(np.float16),
        bias=np.float16(clf.intercept_[0]),
        model_id=np.array(MODEL_ID),
        train_formats=np.array(TRAIN_FORMATS),
    )
    print(f"\nSaved artifact: {OUT_PATH}")
    size_kb = OUT_PATH.stat().st_size / 1024
    print(f"  size: {size_kb:.1f} KB")


if __name__ == "__main__":
    main()
