#!/usr/bin/env python3
"""Probe whether model hidden states encode entity relationships at the token level.

Two probes, both on BBH tracking_shuffled_objects_three_objects:

  1. SWAP-PARTNER PROBE  (swap sentences)
     Label each actor token with the index (0/1/2) of the actor it's paired with.
     Chance = 50% (2 possible partners for 3 actors).
     Accuracy >> 50% → model encodes who you're swapping with.

  2. OBJECT-ASSIGNMENT PROBE  (init sentence)
     Label each actor token with the index (0/1/2) of the object they hold.
     Chance = 33%.
     Accuracy >> 33% → model encodes which object the actor holds in the init state.

Run:
    /Users/jdonaldson/Projects/swollm/.venv/bin/python \\
        experiments/relationship_probe.py
"""

from __future__ import annotations

import re
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
    detect_actors,
    parse_action,
    parse_init,
)

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
        MODEL_ID, dtype=torch.float16,
    ).to(device).eval()
    return mdl, tok, device


# ── hidden states ─────────────────────────────────────────────────────────────

def extract_hidden_states(prompt, model, tokenizer, device):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    token_ids = inputs["input_ids"][0].tolist()
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    layer_states = [h[0].cpu().float() for h in outputs.hidden_states]
    return token_ids, layer_states


# ── token position finder ─────────────────────────────────────────────────────

def find_actor_positions(token_ids: list[int], actor: str, tokenizer) -> list[int]:
    """All token positions where 'actor' appears (last subtoken of name)."""
    tids = tokenizer.encode(" " + actor, add_special_tokens=False)
    if not tids:
        return []
    tid = tids[-1]
    return [i for i, t in enumerate(token_ids) if t == tid]


def sentence_token_range(sent: str, prompt: str, tokenizer) -> tuple[int, int] | None:
    """(tok_start, tok_end) for sent inside prompt, or None."""
    try:
        char_start = prompt.index(sent)
    except ValueError:
        return None
    char_end = char_start + len(sent)
    ts = len(tokenizer.encode(prompt[:char_start], add_special_tokens=False))
    te = len(tokenizer.encode(prompt[:char_end],   add_special_tokens=False))
    return ts, te


def positions_in_range(positions: list[int], rng: tuple[int, int]) -> list[int]:
    ts, te = rng
    return [p for p in positions if ts <= p < te]


# ── probe ─────────────────────────────────────────────────────────────────────

def probe_accuracy(X: np.ndarray, y: list, cv: int = 5) -> float:
    if len(set(y)) < 2 or len(y) < cv * max(y.count(c) for c in set(y) if y.count(c) > 0):
        pass
    if len(set(y)) < 2 or len(y) < cv:
        return float("nan")
    scaler = StandardScaler()
    clf = SGDClassifier(loss="log_loss", max_iter=2000, random_state=42)
    return float(cross_val_score(clf, scaler.fit_transform(X), y, cv=cv).mean()
                 if len(y) >= cv else float("nan"))


def report(name: str, chance: float, vecs, labels, n_layers: int) -> None:
    n = len(labels)
    dist = {c: labels.count(c) for c in sorted(set(labels))}
    print(f"\n{'='*55}")
    print(f"  {name}")
    print(f"  n={n}  dist={dist}  chance={chance:.1%}")
    print(f"{'='*55}")
    print(f"{'Layer':>5}  {'accuracy':>10}")
    print("─" * 20)
    for layer_idx in range(n_layers):
        X = np.array(vecs[layer_idx])
        if len(X) == 0:
            print(f"  L{layer_idx:<3}  {'n/a':>9}")
            continue
        acc = probe_accuracy(X, labels)
        marker = " ◀" if not np.isnan(acc) and acc > chance + 0.10 else ""
        print(f"  L{layer_idx:<3}  {acc:>9.1%}{marker}", flush=True)
    print(f"\nChance: {chance:.1%}  n={n}")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    model, tokenizer, device = load_model()
    examples = load_task(TASK)[:N_EXAMPLES]
    n_layers = model.config.num_hidden_layers + 1

    # Probe 1: swap-partner (label = index of partner in actors list)
    swap_vecs:   list[list[np.ndarray]] = [[] for _ in range(n_layers)]
    swap_labels: list[int] = []

    # Probe 2: object-assignment from actor side (label = rank of held object)
    obj_vecs:   list[list[np.ndarray]] = [[] for _ in range(n_layers)]
    obj_labels: list[int] = []

    # Probe 3: object-owner from object side (label = actor index that holds it)
    own_vecs:   list[list[np.ndarray]] = [[] for _ in range(n_layers)]
    own_labels: list[int] = []

    skipped = 0

    for i, ex in enumerate(examples):
        body = ex["input"].split("\nOptions:")[0].strip()
        lines = [l.strip() for l in body.split(".") if l.strip()]

        actors = detect_actors(lines[0] if lines else body)
        if len(actors) < 2:
            skipped += 1
            continue

        # Init sentence
        init_sent = next((l for l in lines if re.search(r"At the start", l, re.I)), None)
        if not init_sent:
            skipped += 1
            continue

        init_state = parse_init(init_sent, actors)
        if len(init_state) < len(actors):
            skipped += 1
            continue

        # Sort objects alphabetically → consistent rank across examples
        # Label = rank of actor's object in sorted list, not actor's own index.
        # This means Alice gets a different label in each example depending on
        # what she holds, so the probe must use assignment info, not name identity.
        all_objects    = [init_state[a] for a in actors]
        sorted_objects = sorted(all_objects, key=str.lower)
        obj_rank       = {a: sorted_objects.index(init_state[a]) for a in actors}

        # Swap sentences
        swap_sents = [(l, parse_action(l)) for l in lines if parse_action(l)]

        messages = [{"role": "user", "content": body}]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)

        token_ids, layer_states = extract_hidden_states(
            prompt, model, tokenizer, device)

        # ── Probe 2: object assignment (init sentence) ────────────────────────
        init_rng = sentence_token_range(init_sent, prompt, tokenizer)
        if init_rng:
            for actor in actors:
                positions = find_actor_positions(token_ids, actor, tokenizer)
                in_init = positions_in_range(positions, init_rng)
                for pos in in_init:
                    for layer_idx, h in enumerate(layer_states):
                        obj_vecs[layer_idx].append(h[pos].numpy())
                    obj_labels.append(obj_rank[actor])

        # ── Probe 3: object-owner (init sentence, object-side) ────────────────
        # For each object token, does it encode which actor holds it?
        # Object tokens appear AFTER the actor in "Alice gets Ulysses" so they
        # have seen the actor in left context.
        if init_rng:
            for actor_idx, actor in enumerate(actors):
                obj_name = init_state[actor]
                # Use last subtoken of the object name (same strategy as actors)
                obj_tids = tokenizer.encode(" " + obj_name.split()[0], add_special_tokens=False)
                if not obj_tids:
                    continue
                obj_tid = obj_tids[-1]
                positions = [p for p, t in enumerate(token_ids) if t == obj_tid]
                in_init = positions_in_range(positions, init_rng)
                for pos in in_init:
                    for layer_idx, h in enumerate(layer_states):
                        own_vecs[layer_idx].append(h[pos].numpy())
                    own_labels.append(actor_idx)

        # ── Probe 1: swap partner (swap sentences) ─────────────────────────────
        for sent, pair in swap_sents:
            a1, a2 = pair
            if a1 not in actors or a2 not in actors:
                continue
            rng = sentence_token_range(sent, prompt, tokenizer)
            if not rng:
                continue
            partner = {a1: actors.index(a2), a2: actors.index(a1)}
            for actor, partner_idx in partner.items():
                positions = find_actor_positions(token_ids, actor, tokenizer)
                in_sent = positions_in_range(positions, rng)
                for pos in in_sent:
                    for layer_idx, h in enumerate(layer_states):
                        swap_vecs[layer_idx].append(h[pos].numpy())
                    swap_labels.append(partner_idx)

        if (i + 1) % 20 == 0:
            print(
                f"[{i+1}/{len(examples)}]  "
                f"swap={len(swap_labels)}  "
                f"obj={len(obj_labels)}  "
                f"own={len(own_labels)}  "
                f"skipped={skipped}",
                flush=True,
            )

    n_partners = len(set(swap_labels)) if swap_labels else 1
    n_objects  = len(set(obj_labels))  if obj_labels  else 1
    n_owners   = len(set(own_labels))  if own_labels  else 1

    report(
        "SWAP-PARTNER PROBE  (swap sentences — actor side)",
        chance=1 / n_partners,
        vecs=swap_vecs,
        labels=swap_labels,
        n_layers=n_layers,
    )

    report(
        "OBJECT-ASSIGNMENT PROBE  (init sentence — actor side)",
        chance=1 / n_objects,
        vecs=obj_vecs,
        labels=obj_labels,
        n_layers=n_layers,
    )

    report(
        "OBJECT-OWNER PROBE  (init sentence — object side)",
        chance=1 / n_owners,
        vecs=own_vecs,
        labels=own_labels,
        n_layers=n_layers,
    )


if __name__ == "__main__":
    main()
