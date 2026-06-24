#!/usr/bin/env python3
"""Probe whether model hidden states encode entity role at the token level.

For BBH tracking_shuffled_objects_three_objects, labels each occurrence of an
actor-name token as one of three roles:

  init  — actor appears in the 'At the start' assignment sentence
  swap  — actor appears in a swap/trade/switch sentence
  query — actor appears in the final question

Trains a 3-class linear probe (SGDClassifier, 5-fold CV) at each layer.
Chance = 33%. Accuracy >> 33% → model encodes role in the token representation.

Run:
    /Users/jdonaldson/Projects/swollm/.venv/bin/python \\
        experiments/entity_role_probe.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, "/Users/jdonaldson/Projects/swollm/src")
sys.path.insert(0, str(Path(__file__).parent))

from swollm.bench.bbh import load_task
from tracking_deterministic import detect_actors, parse_action, parse_init, parse_query

MODEL_ID   = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
TASK       = "tracking_shuffled_objects_three_objects"
N_EXAMPLES = 200


# ── model ─────────────────────────────────────────────────────────────────────

def load_model():
    device = (
        "mps"  if torch.backends.mps.is_available() else
        "cuda" if torch.cuda.is_available()          else
        "cpu"
    )
    print(f"Loading {MODEL_ID} on {device}…", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    mdl = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16,
    ).to(device).eval()
    return mdl, tok, device


# ── sentence role classification ──────────────────────────────────────────────

def label_sentences(body: str) -> list[tuple[str, str]]:
    """Return [(sentence_text, role), ...] for role-bearing sentences.

    Roles: init | swap | query
    Uses parse_query to find the queried actor and label the final sentence.
    """
    import re
    queried = parse_query(body)  # actor name or None

    sents = [s.strip() for s in re.split(r'(?<=[.?!])\s+', body) if s.strip()]
    result = []
    for idx, sent in enumerate(sents):
        is_last = idx == len(sents) - 1
        if "start" in sent.lower():
            result.append((sent, "init"))
        elif parse_action(sent):
            result.append((sent, "swap"))
        elif sent.endswith("?") or (is_last and queried and queried in sent):
            result.append((sent, "query"))
    return result


# ── token range mapping ───────────────────────────────────────────────────────

def sentence_token_ranges(
    labeled: list[tuple[str, str]],
    prompt: str,
    tokenizer,
) -> list[tuple[int, int, str]]:
    """Map (sentence_text, role) → (token_start, token_end, role).

    Uses prefix-tokenization to find where each sentence starts/ends in the
    full token sequence.
    """
    ranges = []
    for sent, role in labeled:
        try:
            char_start = prompt.index(sent)
        except ValueError:
            continue
        char_end = char_start + len(sent)
        tok_start = len(tokenizer.encode(prompt[:char_start], add_special_tokens=False))
        tok_end   = len(tokenizer.encode(prompt[:char_end],   add_special_tokens=False))
        ranges.append((tok_start, tok_end, role))
    return ranges


def classify_occurrence(pos: int, ranges: list[tuple[int, int, str]]) -> str | None:
    """Return the role of the sentence containing token position pos, or None."""
    for tok_start, tok_end, role in ranges:
        if tok_start <= pos < tok_end:
            return role
    return None


# ── hidden state extraction ───────────────────────────────────────────────────

def extract_hidden_states(prompt, model, tokenizer, device):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    token_ids = inputs["input_ids"][0].tolist()
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    layer_states = [h[0].cpu().float() for h in outputs.hidden_states]
    return token_ids, layer_states


# ── probe ─────────────────────────────────────────────────────────────────────

def probe_accuracy(X: np.ndarray, y: list[str], cv: int = 5) -> float:
    if len(set(y)) < 2 or len(y) < cv:
        return float("nan")
    scaler = StandardScaler()
    clf = SGDClassifier(loss="log_loss", max_iter=2000, random_state=42)
    return float(cross_val_score(clf, scaler.fit_transform(X), y, cv=cv).mean())


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    model, tokenizer, device = load_model()
    examples = load_task(TASK)[:N_EXAMPLES]

    n_layers = model.config.num_hidden_layers + 1  # +1 for embedding layer L0
    vecs:   list[list[np.ndarray]] = [[] for _ in range(n_layers)]
    labels: list[str] = []
    skipped = 0

    for i, ex in enumerate(examples):
        body = ex["input"].split("\nOptions:")[0].strip()
        lines = [l.strip() for l in body.split(".") if l.strip()]

        actors = detect_actors(lines[0] if lines else body)
        if not actors:
            skipped += 1
            continue

        labeled = label_sentences(body)
        if not labeled:
            skipped += 1
            continue

        messages = [{"role": "user", "content": body}]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)

        token_ids, layer_states = extract_hidden_states(
            prompt, model, tokenizer, device)

        ranges = sentence_token_ranges(labeled, prompt, tokenizer)

        found = 0
        for actor in actors:
            actor_tid = tokenizer.encode(" " + actor, add_special_tokens=False)
            if not actor_tid:
                continue
            tid = actor_tid[-1]

            for pos, t in enumerate(token_ids):
                if t != tid:
                    continue
                role = classify_occurrence(pos, ranges)
                if role is None:
                    continue
                for layer_idx, h in enumerate(layer_states):
                    vecs[layer_idx].append(h[pos].numpy())
                labels.append(role)
                found += 1

        if found == 0:
            skipped += 1

        if (i + 1) % 20 == 0:
            role_dist = {r: labels.count(r) for r in ("init", "swap", "query")}
            print(
                f"[{i+1}/{len(examples)}]  tokens={len(labels)}  "
                f"dist={role_dist}  skipped={skipped}",
                flush=True,
            )

    n = len(labels)
    role_counts = {r: labels.count(r) for r in set(labels)}
    chance = 1 / len(role_counts)
    print(f"\nCollected {n} labeled tokens  {role_counts}")
    print(f"Chance baseline: {chance:.1%}\n")
    print(f"{'Layer':>5}  {'accuracy':>10}")
    print("─" * 20)

    for layer_idx in range(n_layers):
        X   = np.array(vecs[layer_idx])
        acc = probe_accuracy(X, labels)
        marker = " ◀" if not np.isnan(acc) and acc > chance + 0.10 else ""
        print(f"  L{layer_idx:<3}  {acc:>9.1%}{marker}", flush=True)

    print(f"\nChance: {chance:.1%}  n={n}")


if __name__ == "__main__":
    main()
