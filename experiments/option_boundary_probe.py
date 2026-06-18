#!/usr/bin/env python3
"""Option-boundary probe prototype.

Per-token binary classifier: "is this token at the start of an option line?"
Trained on BBH prompts where regex finds options. Validates:
  1. Layer sweep — which layer encodes the boundary signal best
  2. Cross-task generalization — train on N tasks, test on held-out task
  3. Format generalization — rewrite prompts with alternative option markers
     (A./1./Choice A:/[A]) and check the probe still finds boundaries

If the probe at L1 (or wherever) generalizes across formats with high
boundary-recovery accuracy, we have a format-agnostic replacement for
the regex-based `.options()` selector.

Run:
    /Users/jdonaldson/Projects/swollm/.venv/bin/python \\
        experiments/option_boundary_probe.py
"""
from __future__ import annotations

import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import re
import sys
import time

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, precision_score, recall_score
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, "/Users/jdonaldson/Projects/swollm/src")
from swollm.bench.bbh import load_task

MODEL_ID = "HuggingFaceTB/SmolLM2-1.7B-Instruct"

TRAIN_TASKS = ["snarks", "ruin_names", "movie_recommendation"]
VAL_TASK = "disambiguation_qa"
LAYERS_TO_TRY = [1, 2, 4, 8]
N_TRAIN_PER_TASK = 60
N_VAL = 50

# Per-format anchor regex: match the BEGINNING of an option marker
# (m.start() should point at the first char of the marker — `(`, `A`, `1`, etc.)
PAREN_RE     = re.compile(r"\(([A-Z])\)\s+(.+?)(?=\n\(|\Z)", flags=re.S)
LETTER_DOT_RE = re.compile(r"^([A-Z])\.\s+", flags=re.M)
NUM_DOT_RE   = re.compile(r"^(\d+)\.\s+", flags=re.M)
CHOICE_RE    = re.compile(r"^Choice\s+([A-Z]):\s+", flags=re.M)
BRACKET_RE   = re.compile(r"\[([A-Z])\]\s+")

FORMAT_REGEX = {
    "(A)": PAREN_RE,
    "A.": LETTER_DOT_RE,
    "1.": NUM_DOT_RE,
    "Choice A:": CHOICE_RE,
    "[A]": BRACKET_RE,
}


def load_model():
    device = ("mps" if torch.backends.mps.is_available() else
              "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading {MODEL_ID} on {device}…", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    mdl = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.float16
    ).to(device).eval()
    return mdl, tok, device


# ── Labeling: which token is at the start of an option line ──────────────────

def label_starts(text, encoded, marker_re):
    """Return labels[i] = 1 if token i contains the start char of an option line."""
    n = len(encoded["input_ids"])
    labels = np.zeros(n, dtype=int)
    for m in marker_re.finditer(text):
        start_char = m.start()
        for tok_idx, (s, e) in enumerate(encoded["offset_mapping"]):
            if s <= start_char < e and s != e:
                labels[tok_idx] = 1
                break
    return labels


# ── Format rewriters for generalization test ────────────────────────────────

def rewrite_to(text, fmt):
    """Rewrite (A)/(B)/... markers to a different format.
    Returns (rewritten_text, marker_regex_for_new_format)."""

    def replace(m):
        letter = m.group(1)
        content = m.group(2)
        if fmt == "A.":
            return f"{letter}. {content}"
        if fmt == "1.":
            n = ord(letter) - ord("A") + 1
            return f"{n}. {content}"
        if fmt == "Choice A:":
            return f"Choice {letter}: {content}"
        if fmt == "[A]":
            return f"[{letter}] {content}"
        return m.group(0)

    rewritten = PAREN_RE.sub(replace, text)
    if fmt == "A.":
        new_re = re.compile(r"^([A-Z])\.\s+(.+?)(?=\n[A-Z]\.|\Z)", flags=re.S | re.M)
    elif fmt == "1.":
        new_re = re.compile(r"^(\d+)\.\s+(.+?)(?=\n\d+\.|\Z)", flags=re.S | re.M)
    elif fmt == "Choice A:":
        new_re = re.compile(r"^Choice\s+([A-Z]):\s+(.+?)(?=\nChoice\s+[A-Z]:|\Z)",
                            flags=re.S | re.M)
    elif fmt == "[A]":
        new_re = re.compile(r"^\[([A-Z])\]\s+(.+?)(?=\n\[[A-Z]\]|\Z)",
                            flags=re.S | re.M)
    else:
        new_re = PAREN_RE
    return rewritten, new_re


# ── Hidden-state collection (one forward pass, capture chosen layers) ────────

def collect(prompts, model, tokenizer, device, layers, marker_re=PAREN_RE,
            per_prompt_marker_res=None):
    """Return per-prompt list of (X_per_layer dict, labels). One forward
    pass per prompt; the chosen layers are gathered together.

    `per_prompt_marker_res` (optional): list of regex-per-prompt to override
    `marker_re` (used for mixed-format training)."""
    out = []
    for i, text in enumerate(prompts):
        encoded = tokenizer(text, return_offsets_mapping=True,
                            add_special_tokens=True)
        re_for_this = per_prompt_marker_res[i] if per_prompt_marker_res else marker_re
        labels = label_starts(text, encoded, re_for_this)

        ids = {
            "input_ids": torch.tensor([encoded["input_ids"]]).to(device),
            "attention_mask": torch.tensor([encoded["attention_mask"]]).to(device),
        }
        with torch.no_grad():
            o = model(**ids, output_hidden_states=True)

        per_layer = {
            layer: o.hidden_states[layer][0].float().cpu().numpy().astype(np.float16)
            for layer in layers
        }
        out.append({"X": per_layer, "y": labels, "n_options": int(labels.sum())})
        if (i + 1) % 25 == 0:
            print(f"  [{i+1}/{len(prompts)}]", flush=True)
    return out


# ── Probe training/evaluation ────────────────────────────────────────────────

def train_probe(records, layer):
    X = np.concatenate([r["X"][layer].astype(np.float32) for r in records])
    y = np.concatenate([r["y"] for r in records])
    sc = StandardScaler().fit(X)
    clf = LogisticRegression(max_iter=2000, C=0.1, class_weight="balanced")
    clf.fit(sc.transform(X), y)
    return sc, clf


def evaluate(records, layer, sc, clf):
    """Per-token F1 + per-prompt boundary recovery (top-K predictions vs
    K ground-truth boundaries)."""
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
        recovery = len(set(top_k) & set(truth)) / K
        recoveries.append(recovery)
    return {
        "f1": f1_score(all_labels, all_preds, zero_division=0),
        "precision": precision_score(all_labels, all_preds, zero_division=0),
        "recall": recall_score(all_labels, all_preds, zero_division=0),
        "boundary_recovery": float(np.mean(recoveries)),
        "n_prompts": len(recoveries),
        "perfect_prompts": sum(1 for r in recoveries if r == 1.0),
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    model, tokenizer, device = load_model()

    # Training prompts: BBH multiple-choice with (A)/(B)/... markers
    train_prompts = []
    for task in TRAIN_TASKS:
        train_prompts.extend(ex["input"] for ex in load_task(task)[:N_TRAIN_PER_TASK])
    print(f"\n{len(train_prompts)} training prompts from {TRAIN_TASKS}")

    val_prompts = [ex["input"] for ex in load_task(VAL_TASK)[:N_VAL]]
    print(f"{len(val_prompts)} held-out prompts from {VAL_TASK}\n")

    # --- 1. Collect train + held-out val (paren format) ---
    print("Collecting hidden states (training)…", flush=True)
    t0 = time.time()
    train_recs = collect(train_prompts, model, tokenizer, device, LAYERS_TO_TRY)
    print(f"  done in {time.time()-t0:.1f}s\n")

    print("Collecting hidden states (held-out val, paren format)…", flush=True)
    t0 = time.time()
    val_recs = collect(val_prompts, model, tokenizer, device, LAYERS_TO_TRY)
    print(f"  done in {time.time()-t0:.1f}s\n")

    # --- 2. Layer sweep: train per layer, evaluate on val ---
    print("=== Layer sweep (cross-task: train on TRAIN_TASKS, eval on VAL_TASK) ===")
    print(f"{'layer':>5}  {'F1':>6}  {'prec':>6}  {'recall':>6}  {'recov':>6}  perfect")
    print("─" * 60)
    layer_results = {}
    for layer in LAYERS_TO_TRY:
        sc, clf = train_probe(train_recs, layer)
        r = evaluate(val_recs, layer, sc, clf)
        layer_results[layer] = (sc, clf, r)
        print(f"  L{layer:<3}  {r['f1']:>5.3f}  {r['precision']:>5.3f}  "
              f"{r['recall']:>5.3f}  {r['boundary_recovery']:>5.3f}  "
              f"{r['perfect_prompts']}/{r['n_prompts']}")

    # Pick best layer by F1 (then recovery) — recovery alone is too lenient
    best_layer = max(layer_results, key=lambda L: (
        layer_results[L][2]["f1"],
        layer_results[L][2]["boundary_recovery"],
    ))
    print(f"\nBest layer (by F1): L{best_layer}")

    # --- 3. Format generalization at every layer ---
    print(f"\n=== Format generalization (probe trained on (A) format only, "
          f"tested on rewrites) ===")

    formats = ["(A)", "A.", "1.", "Choice A:", "[A]"]
    fmt_recs_per_format = {"(A)": val_recs}
    for fmt in formats:
        if fmt == "(A)":
            continue
        rewritten = []
        for p in val_prompts:
            new_text, _ = rewrite_to(p, fmt)
            rewritten.append(new_text)
        _, marker_re = rewrite_to(val_prompts[0], fmt)
        print(f"  collecting {fmt}…", flush=True)
        fmt_recs_per_format[fmt] = collect(rewritten, model, tokenizer, device,
                                            LAYERS_TO_TRY, marker_re=marker_re)

    print(f"\n{'fmt':<14}  " + "  ".join(f"L{l:<2}".rjust(8) for l in LAYERS_TO_TRY))
    print("─" * (16 + 10 * len(LAYERS_TO_TRY)))
    for fmt in formats:
        recs_fmt = fmt_recs_per_format[fmt]
        row = [f"{fmt:<14}"]
        for layer in LAYERS_TO_TRY:
            sc_l, clf_l, _ = layer_results[layer]
            r = evaluate(recs_fmt, layer, sc_l, clf_l)
            row.append(f"{r['boundary_recovery']:.2f}({r['perfect_prompts']:>2}/{r['n_prompts']})")
        print("  ".join(row))

    # --- 4. Mixed-format training: include a mix of (A)/A./1./Choice A:
    #     in the training data, test on held-out Choice A: + [A] ---
    print(f"\n=== Mixed-format training (rewrite 1/4 of training prompts each "
          f"to A./1./Choice A:, keep 1/4 as (A)) ===")
    mix_prompts = []
    quarters = [(0, 1, "(A)"), (1, 4, "A."), (2, 4, "1."), (3, 4, "Choice A:")]
    for ix, (start, total, fmt) in enumerate(quarters):
        for j, p in enumerate(train_prompts):
            if j % 4 == ix:
                if fmt == "(A)":
                    mix_prompts.append(p)
                else:
                    new_text, _ = rewrite_to(p, fmt)
                    mix_prompts.append(new_text)

    print(f"  collecting {len(mix_prompts)} mixed prompts…", flush=True)
    # Track each prompt's format → use the right marker regex when labeling
    mix_formats = []
    for ix, (start, total, fmt) in enumerate(quarters):
        for j, p in enumerate(train_prompts):
            if j % 4 == ix:
                mix_formats.append(fmt)
    mix_marker_res = [FORMAT_REGEX[fmt] for fmt in mix_formats]
    mix_recs = collect(mix_prompts, model, tokenizer, device, LAYERS_TO_TRY,
                       per_prompt_marker_res=mix_marker_res)

    print("  retraining probes on mixed-format data…", flush=True)
    mixed_results = {}
    for layer in LAYERS_TO_TRY:
        sc, clf = train_probe(mix_recs, layer)
        mixed_results[layer] = (sc, clf)

    print(f"\nMixed-trained probe, evaluated on each format (held-out task):")
    print(f"{'fmt':<14}  " + "  ".join(f"L{l:<2}".rjust(8) for l in LAYERS_TO_TRY))
    print("─" * (16 + 10 * len(LAYERS_TO_TRY)))
    for fmt in formats:
        recs_fmt = fmt_recs_per_format[fmt]
        row = [f"{fmt:<14}"]
        for layer in LAYERS_TO_TRY:
            sc, clf = mixed_results[layer]
            r = evaluate(recs_fmt, layer, sc, clf)
            row.append(f"{r['boundary_recovery']:.2f}({r['perfect_prompts']:>2}/{r['n_prompts']})")
        print("  ".join(row))


if __name__ == "__main__":
    main()
