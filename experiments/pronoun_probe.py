#!/usr/bin/env python3
"""Trained probe for pronoun coreference resolution.

Extends pronoun_coref_probe.py (cosine similarity baseline) with a trained
linear probe to test whether contextual hidden states can resolve the
Alice/Claire ambiguity that caps cosine accuracy at ~67%.

Three actors: Alice (she), Bob (he), Claire (she).
  Cosine ceiling:  1/3 * 100% + 2/3 * 50% = 67%  (gender alone, "she" ambiguous)
  Probe ceiling:   100% (if context in hidden state distinguishes Alice vs Claire)

Experiment:
  For each swap sentence, replace one actor with its gendered pronoun.
  Collect hidden state of pronoun token at each layer.
  Train/test split (150/50).  Per-layer SGDClassifier.
  Report: probe accuracy vs cosine baseline, broken down by pronoun type.

Run:
    /Users/jdonaldson/Projects/swollm/.venv/bin/python \\
        experiments/pronoun_probe.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, "/Users/jdonaldson/Projects/swollm/src")
sys.path.insert(0, str(Path(__file__).parent))

from swollm.bench.bbh import load_task
from tracking_deterministic import detect_actors, parse_action, parse_init

MODEL_ID   = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
TASK       = "tracking_shuffled_objects_three_objects"
N_EXAMPLES = 200
N_TRAIN    = 150

PRONOUNS = {
    "Alice": "she", "Claire": "she", "Bob": "he",
    "Dave": "he", "Eve": "she", "Fred": "he", "Gertrude": "she",
}


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


def hidden_states(prompt, model, tokenizer, device):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    tids   = inputs["input_ids"][0].tolist()
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)
    layers = [h[0].cpu().float() for h in out.hidden_states]
    return tids, layers


# ── utilities ─────────────────────────────────────────────────────────────────

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(np.dot(a, b) / (na * nb)) if na and nb else 0.0


def sentence_range(sent: str, prompt: str, tokenizer) -> tuple[int, int] | None:
    try:
        cs = prompt.index(sent)
    except ValueError:
        return None
    ce = cs + len(sent)
    ts = len(tokenizer.encode(prompt[:cs], add_special_tokens=False))
    te = len(tokenizer.encode(prompt[:ce], add_special_tokens=False))
    return ts, te


def find_last_subtoken(tids, name, tokenizer, start=0, end=None):
    subtids = tokenizer.encode(" " + name, add_special_tokens=False)
    if not subtids:
        return None
    target = subtids[-1]
    limit = end if end is not None else len(tids)
    for i in range(start, limit):
        if tids[i] == target:
            return i
    return None


# ── data collection ───────────────────────────────────────────────────────────

def collect(examples, model, tokenizer, device):
    """Return list of dicts with hidden states + labels for each pronoun occurrence."""
    n_layers = model.config.num_hidden_layers + 1
    records  = []

    for i, ex in enumerate(examples):
        body  = ex["input"].split("\nOptions:")[0].strip()
        lines = [l.strip() for l in body.split(".") if l.strip()]

        actors = detect_actors(lines[0] if lines else body)
        if len(actors) < 2:
            continue

        init_sent = next((l for l in lines if re.search(r"At the start", l, re.I)), None)
        if not init_sent:
            continue
        init_state = parse_init(init_sent, actors)
        if len(init_state) < len(actors):
            continue

        swap_sents = [l for l in lines if parse_action(l)]
        if not swap_sents:
            continue

        for swap_sent in swap_sents:
            pair = parse_action(swap_sent)
            if not pair:
                continue
            replaced_actor = pair[0]
            pronoun        = PRONOUNS.get(replaced_actor)
            if not pronoun:
                continue

            # Build modified prompt with one actor replaced by pronoun
            modified_sent = re.sub(rf"\b{replaced_actor}\b", pronoun, swap_sent, count=1)
            modified_body = body.replace(swap_sent, modified_sent, 1)
            messages      = [{"role": "user", "content": modified_body}]
            prompt        = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)

            tids, layers = hidden_states(prompt, model, tokenizer, device)

            # Actor reference vectors from init sentence
            init_rng = sentence_range(init_sent, prompt, tokenizer)
            if not init_rng:
                continue
            actor_vecs: dict[str, np.ndarray] = {}
            for actor in actors:
                pos = find_last_subtoken(tids, actor, tokenizer,
                                         start=init_rng[0], end=init_rng[1])
                if pos is not None:
                    # Use L1 for reference (stable identity representation)
                    actor_vecs[actor] = layers[1][pos].numpy()

            if replaced_actor not in actor_vecs:
                continue

            # Pronoun token position in modified swap sentence
            mod_rng = sentence_range(modified_sent, prompt, tokenizer)
            if not mod_rng:
                continue
            pron_subtids = tokenizer.encode(" " + pronoun, add_special_tokens=False)
            if not pron_subtids:
                continue
            pron_tid = pron_subtids[-1]
            pron_pos = next(
                (p for p in range(mod_rng[0], mod_rng[1]) if tids[p] == pron_tid),
                None
            )
            if pron_pos is None:
                continue

            # Collect hidden states at pronoun position across all layers
            pron_vecs = {layer_idx: layers[layer_idx][pron_pos].numpy()
                         for layer_idx in range(n_layers)}

            records.append({
                "actor":       replaced_actor,
                "actor_idx":   actors.index(replaced_actor),
                "pronoun":     pronoun,
                "actors":      actors,
                "actor_vecs":  actor_vecs,
                "pron_vecs":   pron_vecs,
                "n_layers":    n_layers,
            })

        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{len(examples)}] records={len(records)}", flush=True)

    return records


# ── evaluation ────────────────────────────────────────────────────────────────

def cosine_accuracy(records):
    """Accuracy using cosine similarity to L1 actor reference vectors."""
    correct = 0
    for r in records:
        pron_vec = r["pron_vecs"][1]  # L1
        sims = {a: cosine(pron_vec, v) for a, v in r["actor_vecs"].items()}
        if max(sims, key=sims.__getitem__) == r["actor"]:
            correct += 1
    return correct / len(records) if records else 0.0


def probe_accuracy_by_layer(train_records, test_records):
    """Train per-layer probe, return (layer → accuracy, breakdown_by_pronoun)."""
    n_layers = train_records[0]["n_layers"]
    results  = {}

    for layer_idx in range(n_layers):
        X_tr = np.array([r["pron_vecs"][layer_idx] for r in train_records])
        y_tr = np.array([r["actor_idx"]            for r in train_records])
        X_te = np.array([r["pron_vecs"][layer_idx] for r in test_records])
        y_te = np.array([r["actor_idx"]            for r in test_records])

        sc  = StandardScaler().fit(X_tr)
        clf = SGDClassifier(loss="log_loss", max_iter=2000, random_state=42)
        clf.fit(sc.transform(X_tr), y_tr)
        preds = clf.predict(sc.transform(X_te))

        overall = float((preds == y_te).mean())

        # Breakdown: "he" (Bob) vs "she" (Alice/Claire)
        he_mask  = np.array([r["pronoun"] == "he"  for r in test_records])
        she_mask = np.array([r["pronoun"] == "she" for r in test_records])
        he_acc   = float((preds[he_mask]  == y_te[he_mask]).mean())  if he_mask.any()  else float("nan")
        she_acc  = float((preds[she_mask] == y_te[she_mask]).mean()) if she_mask.any() else float("nan")

        results[layer_idx] = (overall, he_acc, she_acc)

    return results


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    model, tokenizer, device = load_model()
    examples = load_task(TASK)[:N_EXAMPLES]

    print(f"\nCollecting pronoun hidden states…")
    records = collect(examples, model, tokenizer, device)

    train_records = [r for r in records
                     if examples.index(next(e for e in examples
                         if r["actor"] in e["input"][:200])) < N_TRAIN]

    # Simpler split: first 75% train, last 25% test
    n = len(records)
    train_records = records[:int(n * 0.75)]
    test_records  = records[int(n * 0.75):]

    print(f"\nTotal records: {n}  train={len(train_records)}  test={len(test_records)}")

    # Pronoun distribution
    he_n  = sum(1 for r in records if r["pronoun"] == "he")
    she_n = sum(1 for r in records if r["pronoun"] == "she")
    print(f"Pronoun dist: he={he_n}  she={she_n}")
    print(f"Theoretical ceiling (gender only): "
          f"{he_n/n:.0%}×100% + {she_n/n:.0%}×50% = "
          f"{he_n/n + she_n/n*0.5:.1%}\n")

    # Cosine baseline on test set
    cos_acc = cosine_accuracy(test_records)
    print(f"Cosine baseline (L1 similarity): {cos_acc:.1%}")

    # Trained probe sweep
    print(f"\nTraining per-layer probes…")
    layer_results = probe_accuracy_by_layer(train_records, test_records)

    chance = 1 / len(records[0]["actors"]) if records else 0.333
    print(f"\n{'Layer':>5}  {'overall':>8}  {'he':>7}  {'she':>7}")
    print("─" * 38)
    best_layer, best_acc = max(layer_results.items(), key=lambda kv: kv[1][0])
    for layer_idx, (overall, he_acc, she_acc) in layer_results.items():
        marker = " ◀" if overall > chance + 0.10 else ""
        best_m = " ★" if layer_idx == best_layer else ""
        print(f"  L{layer_idx:<3}  {overall:>7.1%}  {he_acc:>6.1%}  {she_acc:>6.1%}{marker}{best_m}",
              flush=True)

    print(f"\nChance: {chance:.1%}  Cosine baseline: {cos_acc:.1%}  "
          f"Best probe: {best_acc[0]:.1%} @ L{best_layer}")


if __name__ == "__main__":
    main()
