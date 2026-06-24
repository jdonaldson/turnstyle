#!/usr/bin/env python3
"""Probe whether the model encodes actor identity in the query token.

For BBH tracking_shuffled_objects_three_objects, extracts the hidden state
of the queried actor's name token at two positions:

  init_pos   — actor token in the "At the start" sentence (encodes starting item)
  query_pos  — actor token in the final question         (encodes final item if
                                                          model tracks state)

Trains a linear probe at each layer × position to predict the correct answer
option (A/B/C), using 5-fold CV.

If query_pos accuracy exceeds init_pos at later layers, the model has
computed the answer in the entity token representation — not just inherited
the string identity from the init sentence.

Run:
    /Users/jdonaldson/Projects/swollm/.venv/bin/python \\
        experiments/entity_identity_probe.py
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
from tracking_deterministic import (
    detect_actors, parse_init, parse_action, parse_query, parse_options,
)

MODEL_ID   = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
TASK       = "tracking_shuffled_objects_three_objects"
N_EXAMPLES = 150   # three_objects: 250 total; use 150 for speed


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
        MODEL_ID, dtype=torch.float16
    ).to(device).eval()
    return mdl, tok, device


# ── ground-truth parsing ──────────────────────────────────────────────────────

def parse_example(text: str) -> dict | None:
    """Return {actors, init_state, final_state, queried_actor, correct_letter}."""
    lines = [l.strip() for l in text.split(".") if l.strip()]
    actors = detect_actors(lines[0] if lines else text)
    if not actors:
        return None

    init_sent = next((l for l in lines if "At the start" in l), None)
    if not init_sent:
        return None

    state = parse_init(init_sent, actors)
    if len(state) < len(actors):
        return None

    for line in lines:
        pair = parse_action(line)
        if pair:
            a1, a2 = pair
            if a1 in state and a2 in state:
                state[a1], state[a2] = state[a2], state[a1]

    queried = parse_query(text)
    if not queried or queried not in state:
        return None

    opts = parse_options(text)
    final_item = state[queried].lower()
    correct_letter = next(
        (letter for letter, val in opts.items() if val.lower() == final_item),
        None,
    )
    if correct_letter is None:
        return None

    return {
        "actors":        actors,
        "init_sent":     init_sent,
        "final_state":   state,
        "queried_actor": queried,
        "correct_letter": correct_letter,
    }


# ── token position search ─────────────────────────────────────────────────────

def find_actor_token_positions(
    token_ids: list[int],
    actor: str,
    tokenizer,
    init_sent: str,
    full_text: str,
) -> tuple[int | None, int | None]:
    """Return (init_pos, query_pos) — token indices of actor in each context.

    init_pos  — first occurrence of actor after 'start' in the token sequence
    query_pos — last occurrence of actor before the end (in the question)
    """
    # Actor token ID: use space-prefixed form since it appears mid-sentence
    actor_ids = tokenizer.encode(" " + actor, add_special_tokens=False)
    if not actor_ids:
        return None, None
    actor_id = actor_ids[-1]  # last subtoken of the name

    # Find all positions where actor_id appears
    positions = [i for i, t in enumerate(token_ids) if t == actor_id]
    if len(positions) < 2:
        return None, None

    # Locate the boundary between init sentence and the rest by tokenizing
    # the prefix up to the init sentence and counting tokens.
    prefix_up_to_init = full_text[: full_text.index(init_sent)]
    prefix_ids = tokenizer.encode(prefix_up_to_init, add_special_tokens=False)
    init_start_idx = len(prefix_ids)

    # Locate the boundary of the question: text after the last "."
    body_end = full_text.rfind("?")
    question_start_text = full_text[: body_end].rsplit(".", 1)[-1]
    prefix_ids_to_q = tokenizer.encode(
        full_text[: full_text.index(question_start_text.strip())],
        add_special_tokens=False,
    )
    query_start_idx = len(prefix_ids_to_q)

    # init_pos: first occurrence at or after init_start_idx
    init_pos = next((p for p in positions if p >= init_start_idx), None)

    # query_pos: last occurrence at or after query_start_idx
    query_candidates = [p for p in positions if p >= query_start_idx]
    query_pos = query_candidates[-1] if query_candidates else None

    return init_pos, query_pos


# ── hidden state extraction ───────────────────────────────────────────────────

def extract_hidden_states(
    prompt: str,
    model,
    tokenizer,
    device,
) -> tuple[list[int], list[torch.Tensor]]:
    """Forward pass with output_hidden_states=True.

    Returns (token_ids, layer_states) where layer_states[i] is shape (seq_len, hidden).
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    token_ids = inputs["input_ids"][0].tolist()

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    # outputs.hidden_states: tuple of (1, seq_len, hidden) per layer (L0=embedding)
    layer_states = [h[0].cpu().float() for h in outputs.hidden_states]
    return token_ids, layer_states


# ── probe training ────────────────────────────────────────────────────────────

def probe_accuracy(X: np.ndarray, y: list[str], cv: int = 5) -> float:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    clf = SGDClassifier(loss="log_loss", max_iter=2000, random_state=42)
    scores = cross_val_score(clf, X_scaled, y, cv=cv)
    return float(scores.mean())


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    model, tokenizer, device = load_model()
    examples = load_task(TASK)[:N_EXAMPLES]

    n_layers = model.config.num_hidden_layers  # 24 for SmolLM2-1.7B
    # +1 for embedding layer (index 0)
    init_vecs:  list[list[np.ndarray]] = [[] for _ in range(n_layers + 1)]
    query_vecs: list[list[np.ndarray]] = [[] for _ in range(n_layers + 1)]
    labels: list[str] = []

    skipped = 0
    for i, ex in enumerate(examples):
        parsed = parse_example(ex["input"])
        if parsed is None:
            skipped += 1
            continue

        actor  = parsed["queried_actor"]
        label  = parsed["correct_letter"]

        # Build prompt (no options — we want the question context, not the answer)
        body = ex["input"].split("\nOptions:")[0].strip()
        messages = [{"role": "user", "content": body}]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)

        token_ids, layer_states = extract_hidden_states(
            prompt, model, tokenizer, device)

        init_pos, query_pos = find_actor_token_positions(
            token_ids, actor, tokenizer, parsed["init_sent"], body)

        if init_pos is None or query_pos is None:
            skipped += 1
            continue

        for layer_idx, h in enumerate(layer_states):
            init_vecs[layer_idx].append(h[init_pos].numpy())
            query_vecs[layer_idx].append(h[query_pos].numpy())

        labels.append(label)

        if (i + 1) % 10 == 0:
            print(
                f"[{i+1}/{len(examples)}]  actor={actor}  label={label}  "
                f"init_pos={init_pos}  query_pos={query_pos}",
                flush=True,
            )

    n = len(labels)
    print(f"\nCollected {n} examples ({skipped} skipped)\n")
    print(f"{'Layer':>5}  {'init_pos→answer':>16}  {'query_pos→answer':>17}")
    print("─" * 44)

    for layer_idx in range(n_layers + 1):
        X_init  = np.array(init_vecs[layer_idx])
        X_query = np.array(query_vecs[layer_idx])

        acc_init  = probe_accuracy(X_init,  labels)
        acc_query = probe_accuracy(X_query, labels)

        marker = " ◀" if acc_query > acc_init + 0.05 else ""
        print(
            f"  L{layer_idx:<3}  {acc_init:>15.1%}  {acc_query:>16.1%}{marker}",
            flush=True,
        )

    chance = 1.0 / len(set(labels))
    print(f"\nChance baseline: {chance:.1%}  (n_classes={len(set(labels))})")


if __name__ == "__main__":
    main()
