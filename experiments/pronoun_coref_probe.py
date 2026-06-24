#!/usr/bin/env python3
"""Probe whether model hidden states resolve pronoun coreference.

Constructs synthetic swap sentences by replacing one actor name with a
gendered pronoun (Bob → "he", Alice/Claire → "she"), then measures whether
the pronoun token's hidden state is most similar to the correct antecedent
actor token (from the init sentence) at each layer.

Method:
  For each example:
    - Parse init sentence → actor→object mapping
    - For each swap sentence, replace one actor name with the matching pronoun
    - Run the modified prompt through the model
    - At each layer: compute cosine similarity between pronoun hidden state
      and each actor's hidden state from the init sentence
    - Correct = pronoun most similar to its antecedent

Chance = 33% (3 actors). Accuracy >> 33% → model resolves pronouns.

Run:
    /Users/jdonaldson/Projects/swollm/.venv/bin/python \\
        experiments/pronoun_coref_probe.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, "/Users/jdonaldson/Projects/swollm/src")
sys.path.insert(0, str(Path(__file__).parent))

from swollm.bench.bbh import load_task
from tracking_deterministic import detect_actors, parse_action, parse_init

MODEL_ID   = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
TASK       = "tracking_shuffled_objects_three_objects"
N_EXAMPLES = 200

# Pronoun assignment: name → pronoun (simplified)
PRONOUNS = {"Alice": "she", "Claire": "she", "Bob": "he",
            "Dave": "he", "Eve": "she", "Fred": "he", "Gertrude": "she"}


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


def hidden_states(prompt: str, model, tokenizer, device):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    tids   = inputs["input_ids"][0].tolist()
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)
    layers = [h[0].cpu().float() for h in out.hidden_states]
    return tids, layers


# ── token utilities ───────────────────────────────────────────────────────────

def last_subtoken_pos(tids: list[int], name: str, tokenizer,
                      start: int = 0, end: int | None = None) -> int | None:
    """First position of name's last subtoken within tids[start:end]."""
    subtids = tokenizer.encode(" " + name, add_special_tokens=False)
    if not subtids:
        return None
    target = subtids[-1]
    limit = end if end is not None else len(tids)
    for i in range(start, limit):
        if tids[i] == target:
            return i
    return None


def sentence_range(sent: str, prompt: str, tokenizer) -> tuple[int, int] | None:
    try:
        cs = prompt.index(sent)
    except ValueError:
        return None
    ce = cs + len(sent)
    ts = len(tokenizer.encode(prompt[:cs], add_special_tokens=False))
    te = len(tokenizer.encode(prompt[:ce], add_special_tokens=False))
    return ts, te


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    model, tokenizer, device = load_model()
    examples = load_task(TASK)[:N_EXAMPLES]
    n_layers = model.config.num_hidden_layers + 1

    # Per-layer correct count
    correct = [0] * n_layers
    total   = 0
    skipped = 0

    for i, ex in enumerate(examples):
        body  = ex["input"].split("\nOptions:")[0].strip()
        lines = [l.strip() for l in body.split(".") if l.strip()]

        actors = detect_actors(lines[0] if lines else body)
        if len(actors) < 2:
            skipped += 1
            continue

        init_sent = next((l for l in lines if re.search(r"At the start", l, re.I)), None)
        if not init_sent:
            skipped += 1
            continue
        init_state = parse_init(init_sent, actors)
        if len(init_state) < len(actors):
            skipped += 1
            continue

        swap_sents = [l for l in lines if parse_action(l)]
        if not swap_sents:
            skipped += 1
            continue

        # For each swap sentence, replace the FIRST actor with its pronoun
        for swap_sent in swap_sents:
            pair = parse_action(swap_sent)
            if not pair:
                continue
            replaced_actor = pair[0]          # the actor we'll replace
            pronoun        = PRONOUNS.get(replaced_actor)
            if not pronoun:
                continue

            # Build modified body: replace actor name in this swap sentence
            modified_sent = re.sub(
                rf"\b{replaced_actor}\b", pronoun, swap_sent, count=1
            )
            modified_body = body.replace(swap_sent, modified_sent, 1)

            messages = [{"role": "user", "content": modified_body}]
            prompt   = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)

            tids, layers = hidden_states(prompt, model, tokenizer, device)

            # Find init sentence range in the modified prompt
            init_rng = sentence_range(init_sent, prompt, tokenizer)
            if not init_rng:
                continue

            # Get hidden states of each actor in the init sentence
            actor_vecs: dict[str, dict[int, np.ndarray]] = {}
            for actor in actors:
                pos = last_subtoken_pos(tids, actor, tokenizer,
                                        start=init_rng[0], end=init_rng[1])
                if pos is not None:
                    actor_vecs[actor] = {
                        layer_idx: layers[layer_idx][pos].numpy()
                        for layer_idx in range(n_layers)
                    }

            if replaced_actor not in actor_vecs:
                continue

            # Find pronoun token position in the modified swap sentence
            mod_rng = sentence_range(modified_sent, prompt, tokenizer)
            if not mod_rng:
                continue
            pronoun_subtids = tokenizer.encode(" " + pronoun, add_special_tokens=False)
            if not pronoun_subtids:
                continue
            pronoun_tid = pronoun_subtids[-1]
            pron_pos = next(
                (p for p in range(mod_rng[0], mod_rng[1]) if tids[p] == pronoun_tid),
                None
            )
            if pron_pos is None:
                continue

            # At each layer: is pronoun most similar to replaced_actor?
            for layer_idx in range(n_layers):
                pron_vec = layers[layer_idx][pron_pos].numpy()
                sims = {
                    actor: cosine(pron_vec, vecs[layer_idx])
                    for actor, vecs in actor_vecs.items()
                }
                if not sims:
                    continue
                best = max(sims, key=sims.__getitem__)
                if best == replaced_actor:
                    correct[layer_idx] += 1

            total += 1

        if (i + 1) % 20 == 0:
            print(f"[{i+1}/{len(examples)}]  trials={total}  skipped={skipped}",
                  flush=True)

    chance = 1 / len(actors) if actors else 0.333
    print(f"\nPronoun coreference via cosine similarity")
    print(f"n_trials={total}  chance={chance:.1%}  skipped={skipped}\n")
    print(f"{'Layer':>5}  {'accuracy':>10}")
    print("─" * 20)
    for layer_idx in range(n_layers):
        acc    = correct[layer_idx] / total if total else 0.0
        marker = " ◀" if acc > chance + 0.10 else ""
        print(f"  L{layer_idx:<3}  {acc:>9.1%}{marker}", flush=True)
    print(f"\nChance: {chance:.1%}  n={total}")


if __name__ == "__main__":
    main()
