#!/usr/bin/env python3
"""BIO span probe for object names in the init sentence of tracking_shuffled.

Trains a per-token B/I/O probe on hidden states from SmolLM2-1.7B-Instruct.
Labels are derived from character-offset prefix-tokenization (position-based,
not token-id-based) so "pink ball" and "green ball" get correct labels at
their respective positions.

Label scheme:
  B — first token of an object span (e.g. first token of "pink ball")
  I — continuation tokens within the same span
  O — everything else in the init sentence

Run:
    /Users/jdonaldson/Projects/swollm/.venv/bin/python experiments/bio_span_probe.py
"""

from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, "/Users/jdonaldson/Projects/swollm/src")
sys.path.insert(0, str(Path(__file__).parent))
from swollm.bench.bbh import load_task
from tracking_deterministic import detect_actors, parse_init

MODEL_ID   = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
TASK       = "tracking_shuffled_objects_three_objects"
N_EXAMPLES = 200
N_LAYERS   = 25   # SmolLM2: embedding (L0) + 24 transformer layers
CACHE_PATH = Path(__file__).parent / "bio_span_probe_data.npz"


# ── helpers ───────────────────────────────────────────────────────────────────

def char_to_tok(tokenizer, prompt: str, char_offset: int) -> int:
    return len(tokenizer.encode(prompt[:char_offset], add_special_tokens=False))


def label_init_tokens(
    tokenizer,
    prompt: str,
    init_sent: str,
    actors: list[str],
) -> tuple[list[int], list[str]]:
    """Return (absolute_token_positions, bio_labels) for init sentence tokens."""
    init_char_start = prompt.index(init_sent)
    init_char_end   = init_char_start + len(init_sent)

    tok_start = char_to_tok(tokenizer, prompt, init_char_start)
    tok_end   = char_to_tok(tokenizer, prompt, init_char_end)
    n         = tok_end - tok_start
    if n <= 0:
        return [], []

    positions = list(range(tok_start, tok_end))
    labels    = ["O"] * n

    state = parse_init(init_sent, actors)
    for obj_name in state.values():
        obj_name = obj_name.strip()
        try:
            rel = init_sent.index(obj_name)
        except ValueError:
            continue
        span_char_start = init_char_start + rel
        span_char_end   = span_char_start + len(obj_name)

        span_tok_start = char_to_tok(tokenizer, prompt, span_char_start)
        span_tok_end   = char_to_tok(tokenizer, prompt, span_char_end)

        for t in range(span_tok_start, span_tok_end):
            idx = t - tok_start
            if 0 <= idx < n:
                labels[idx] = "B" if t == span_tok_start else "I"

    return positions, labels


# ── feature extraction ────────────────────────────────────────────────────────

def extract_features(examples, tokenizer, model, device):
    all_vecs: list[list[np.ndarray]] = [[] for _ in range(N_LAYERS)]
    all_labels: list[str] = []

    for i, ex in enumerate(examples):
        text  = ex["input"].split("\nOptions:")[0].strip()
        lines = [l.strip() for l in text.split(".") if l.strip()]
        actors = detect_actors(lines[0] if lines else text)
        if not actors:
            continue

        init_sent = next(
            (l for l in lines if re.search(r"At the start", l, re.I)), None
        )
        if not init_sent or init_sent not in text:
            continue

        messages = [{"role": "user", "content": text}]
        prompt   = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)

        try:
            positions, labels = label_init_tokens(tokenizer, prompt, init_sent, actors)
        except ValueError:
            continue
        if not positions:
            continue

        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        seq_len = inputs["input_ids"].shape[1]
        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True)

        for pos, label in zip(positions, labels):
            if pos >= seq_len:
                continue
            for layer_idx in range(N_LAYERS):
                h = out.hidden_states[layer_idx][0, pos].cpu().float().numpy()
                all_vecs[layer_idx].append(h)
            all_labels.append(label)

        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{len(examples)}] tokens={len(all_labels)}", flush=True)

    return [np.array(v) for v in all_vecs], np.array(all_labels)


# ── probe per layer ───────────────────────────────────────────────────────────

def eval_layer(X: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Return (macro_F1, B_F1) via 5-fold stratified CV."""
    scaler  = StandardScaler()
    Xs      = scaler.fit_transform(X)
    macro   = float(cross_val_score(
        SGDClassifier(loss="log_loss", max_iter=2000, random_state=42),
        Xs, y, cv=5, scoring="f1_macro",
    ).mean())

    b_scores = []
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for tr, te in skf.split(Xs, y):
        clf = SGDClassifier(loss="log_loss", max_iter=2000, random_state=42)
        clf.fit(Xs[tr], y[tr])
        b_scores.append(
            f1_score(y[te], clf.predict(Xs[te]), labels=["B"],
                     average="macro", zero_division=0)
        )
    return macro, float(np.mean(b_scores))


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    if CACHE_PATH.exists():
        print(f"Loading cached features from {CACHE_PATH}…")
        npz        = np.load(CACHE_PATH, allow_pickle=True)
        X_layers   = [npz[f"layer_{i}"] for i in range(N_LAYERS)]
        y          = npz["y"].astype(str)
    else:
        device = (
            "mps"  if torch.backends.mps.is_available() else
            "cuda" if torch.cuda.is_available()          else
            "cpu"
        )
        print(f"Loading {MODEL_ID} on {device}…", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        model     = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, dtype=torch.float16,
        ).to(device).eval()

        examples = load_task(TASK)[:N_EXAMPLES]
        print(f"Extracting features from {len(examples)} examples…")
        X_layers, y = extract_features(examples, tokenizer, model, device)

        np.savez(CACHE_PATH, **{f"layer_{i}": X_layers[i] for i in range(N_LAYERS)}, y=y)
        print(f"Cached to {CACHE_PATH}")

    dist   = dict(Counter(y.tolist()))
    n      = len(y)
    chance = max(dist.values()) / n  # majority-class (O) baseline

    print(f"\nBIO span tagging — init sentence object tokens")
    print(f"n={n}  dist={dist}")
    print(f"Majority-class baseline: {chance:.1%}\n")
    print(f"{'Layer':<7}  {'macro-F1':>9}  {'B-F1':>8}")
    print("─" * 30)

    for layer_idx in range(N_LAYERS):
        X = X_layers[layer_idx]
        if len(X) == 0:
            print(f"  L{layer_idx:<3}  {'—':>9}  {'—':>8}")
            continue
        macro, b_f1 = eval_layer(X, y)
        marker = " ◀" if macro > chance + 0.10 else ""
        print(f"  L{layer_idx:<3}  {macro:>9.1%}  {b_f1:>8.1%}{marker}", flush=True)

    print(f"\nMajority-class baseline: {chance:.1%}  n={n}")


if __name__ == "__main__":
    main()
