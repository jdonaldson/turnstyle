#!/usr/bin/env python3
"""Compare relational probe accuracy across small (~1B) models.

Measures partner-probe and owner-probe accuracy at each layer for multiple
models on the same 200 BBH tracking_shuffled_objects_three_objects examples.
Identifies the best layer per model and reports a summary table.

Models compared:
  SmolLM2-1.7B         (baseline)
  Qwen2.5-1.5B         (structured-data trained)
  TinyLlama-1.1B
  SmolLM2-360M         (smaller reference)
  BitNet-b1.58-2B-4T   (1-bit ternary weights, bf16 activations)

Run:
    /Users/jdonaldson/Projects/swollm/.venv/bin/python \\
        experiments/model_comparison.py
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
    ALL_ACTORS, detect_actors, parse_action, parse_init,
)

TASK       = "tracking_shuffled_objects_three_objects"
N_EXAMPLES = 200

MODELS = [
    "HuggingFaceTB/SmolLM2-360M-Instruct",
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    "microsoft/bitnet-b1.58-2B-4T-bf16",
]


# ── model ─────────────────────────────────────────────────────────────────────

def load_model(model_id: str):
    device = (
        "mps"  if torch.backends.mps.is_available() else
        "cuda" if torch.cuda.is_available()          else
        "cpu"
    )
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # BitNet models use bf16 activations — float16 overflows to NaN
    dtype = torch.bfloat16 if "bitnet" in model_id.lower() else torch.float16
    mdl = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=dtype,
    ).to(device).eval()
    return mdl, tok, device


def hidden_states(prompt, model, tokenizer, device):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    tids   = inputs["input_ids"][0].tolist()
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)
    layers = [h[0].cpu().float() for h in out.hidden_states]
    return tids, layers


# ── token utilities ───────────────────────────────────────────────────────────

def sentence_range(sent: str, prompt: str, tokenizer):
    try:
        cs = prompt.index(sent)
    except ValueError:
        return None
    ce = cs + len(sent)
    ts = len(tokenizer.encode(prompt[:cs], add_special_tokens=False))
    te = len(tokenizer.encode(prompt[:ce], add_special_tokens=False))
    return ts, te


def actor_positions(tids, actor, tokenizer, start=0, end=None):
    subtids = tokenizer.encode(" " + actor, add_special_tokens=False)
    if not subtids:
        return []
    target = subtids[-1]
    limit  = end if end is not None else len(tids)
    return [i for i in range(start, limit) if tids[i] == target]


# ── data collection ───────────────────────────────────────────────────────────

def collect(examples, model, tokenizer, device):
    """Return vecs/labels for partner probe and owner probe at every layer."""
    n_layers = model.config.num_hidden_layers + 1

    partner_vecs:  list[list[np.ndarray]] = [[] for _ in range(n_layers)]
    partner_labels: list[int] = []
    owner_vecs:    list[list[np.ndarray]] = [[] for _ in range(n_layers)]
    owner_labels:  list[int] = []

    for ex in examples:
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

        swap_sents = [(l, parse_action(l)) for l in lines if parse_action(l)]

        try:
            messages = [{"role": "user", "content": body}]
            prompt   = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            prompt = body

        tids, layers = hidden_states(prompt, model, tokenizer, device)

        init_rng = sentence_range(init_sent, prompt, tokenizer)

        # ── owner probe: object-side, last token of full object name ──────
        if init_rng:
            try:
                sent_char_start = prompt.index(init_sent)
            except ValueError:
                sent_char_start = None

            if sent_char_start is not None:
                obj_pos_labels: dict[int, int] = {}
                for actor_idx, actor in enumerate(actors):
                    obj_name = init_state[actor]
                    try:
                        obj_in_sent = init_sent.index(obj_name)
                    except ValueError:
                        continue
                    pc_end = sent_char_start + obj_in_sent + len(obj_name)
                    te     = len(tokenizer.encode(prompt[:pc_end], add_special_tokens=False))
                    obj_pos_labels[te - 1] = actor_idx

                for pos in range(init_rng[0], init_rng[1]):
                    label = obj_pos_labels.get(pos, len(actors))  # len(actors) = bg
                    for li, h in enumerate(layers):
                        owner_vecs[li].append(h[pos].numpy())
                    owner_labels.append(label)

        # ── partner probe: swap sentence actor tokens ──────────────────────
        for sent, pair in swap_sents:
            a1, a2 = pair
            if a1 not in actors or a2 not in actors:
                continue
            srng = sentence_range(sent, prompt, tokenizer)
            if not srng:
                continue
            for actor, partner in ((a1, a2), (a2, a1)):
                for pos in actor_positions(tids, actor, tokenizer,
                                           start=srng[0], end=srng[1]):
                    for li, h in enumerate(layers):
                        partner_vecs[li].append(h[pos].numpy())
                    partner_labels.append(actors.index(partner))

    return (
        [np.array(v) for v in partner_vecs], partner_labels,
        [np.array(v) for v in owner_vecs],   owner_labels,
    )


# ── probe sweep ───────────────────────────────────────────────────────────────

def best_accuracy(vecs_per_layer, labels, bg_class=None, cv=5):
    """Return (best_acc, best_layer) across all layers."""
    y = np.array(labels)
    if bg_class is not None:
        mask = y != bg_class
        y    = y[mask]
    else:
        mask = np.ones(len(y), dtype=bool)

    if len(set(y.tolist())) < 2:
        return float("nan"), -1

    best_acc, best_layer = 0.0, 0
    for li, X_full in enumerate(vecs_per_layer):
        X = X_full[mask]
        if len(X) < cv * 2:
            continue
        sc  = StandardScaler()
        clf = SGDClassifier(loss="log_loss", max_iter=2000, random_state=42)
        acc = float(cross_val_score(clf, sc.fit_transform(X), y, cv=cv).mean())
        if acc > best_acc:
            best_acc, best_layer = acc, li
    return best_acc, best_layer


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    examples = load_task(TASK)[:N_EXAMPLES]
    device   = (
        "mps"  if torch.backends.mps.is_available() else
        "cuda" if torch.cuda.is_available()          else
        "cpu"
    )

    results = {}

    for model_id in MODELS:
        short = model_id.split("/")[-1]
        print(f"\n{'='*60}")
        print(f"  {short}", flush=True)
        print(f"{'='*60}")

        try:
            model, tokenizer, device = load_model(model_id)
        except Exception as e:
            print(f"  FAILED to load: {e}")
            continue

        n_params = sum(p.numel() for p in model.parameters()) / 1e9
        n_layers = model.config.num_hidden_layers + 1
        print(f"  {n_params:.1f}B params  {n_layers} layers", flush=True)

        print("  Collecting hidden states…", flush=True)
        pv, pl, ov, ol = collect(examples, model, tokenizer, device)

        n_actors = 3  # three_objects
        bg_class = n_actors

        print("  Sweeping partner probe…", flush=True)
        p_acc, p_layer = best_accuracy(pv, pl, bg_class=None)

        print("  Sweeping owner probe…", flush=True)
        o_acc, o_layer = best_accuracy(ov, ol, bg_class=bg_class)

        results[short] = {
            "params":  n_params,
            "n_layers": n_layers - 1,
            "partner_acc":   p_acc,
            "partner_layer": p_layer,
            "owner_acc":     o_acc,
            "owner_layer":   o_layer,
        }

        print(f"  partner: {p_acc:.1%} @ L{p_layer}   owner: {o_acc:.1%} @ L{o_layer}")

        # Free memory before next model
        del model
        if device == "mps":
            torch.mps.empty_cache()
        elif device == "cuda":
            torch.cuda.empty_cache()

    # ── summary table ─────────────────────────────────────────────────────
    print(f"\n\n{'='*70}")
    print(f"  SUMMARY")
    print(f"{'='*70}")
    print(f"{'Model':<32}  {'Params':>6}  {'Layers':>6}  "
          f"{'Partner':>9}  {'Layer':>5}  {'Owner':>7}  {'Layer':>5}")
    print("─" * 70)
    for name, r in results.items():
        print(f"  {name:<30}  {r['params']:>5.1f}B  {r['n_layers']:>6}  "
              f"{r['partner_acc']:>8.1%}  L{r['partner_layer']:<4}  "
              f"{r['owner_acc']:>6.1%}  L{r['owner_layer']:<4}")


if __name__ == "__main__":
    main()
