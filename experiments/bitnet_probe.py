#!/usr/bin/env python3
"""Run partner+owner probe sweep for BitNet-b1.58-2B-4T only.

Finding: partner=87.8%@L15, owner=100%@L17 — comparable to SmolLM2-1.7B.
Requires bfloat16 (float16 overflows to NaN).
"""
from __future__ import annotations
import re
import numpy as np
import torch
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

from common import load_model
from swollm.bench.bbh import load_task
from tracking_deterministic import detect_actors, parse_action, parse_init

TASK = "tracking_shuffled_objects_three_objects"
N_EXAMPLES = 200
MODEL_ID = "microsoft/bitnet-b1.58-2B-4T-bf16"


def hidden_states(prompt, model, tokenizer, device):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    tids = inputs["input_ids"][0].tolist()
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)
    layers = [h[0].cpu().float() for h in out.hidden_states]
    return tids, layers


def sentence_range(sent, prompt, tokenizer):
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
    limit = end if end is not None else len(tids)
    return [i for i in range(start, limit) if tids[i] == target]


def collect(examples, model, tokenizer, device):
    n_layers = model.config.num_hidden_layers + 1
    partner_vecs = [[] for _ in range(n_layers)]
    partner_labels = []
    owner_vecs = [[] for _ in range(n_layers)]
    owner_labels = []

    for ex in examples:
        body = ex["input"].split("\nOptions:")[0].strip()
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
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            prompt = body

        tids, layers = hidden_states(prompt, model, tokenizer, device)
        init_rng = sentence_range(init_sent, prompt, tokenizer)

        if init_rng:
            try:
                sent_char_start = prompt.index(init_sent)
            except ValueError:
                sent_char_start = None
            if sent_char_start is not None:
                obj_pos_labels = {}
                for actor_idx, actor in enumerate(actors):
                    obj_name = init_state[actor]
                    try:
                        obj_in_sent = init_sent.index(obj_name)
                    except ValueError:
                        continue
                    pc_end = sent_char_start + obj_in_sent + len(obj_name)
                    te = len(tokenizer.encode(prompt[:pc_end], add_special_tokens=False))
                    obj_pos_labels[te - 1] = actor_idx
                for pos in range(init_rng[0], init_rng[1]):
                    label = obj_pos_labels.get(pos, len(actors))
                    for li, h in enumerate(layers):
                        owner_vecs[li].append(h[pos].numpy())
                    owner_labels.append(label)

        for sent, pair in swap_sents:
            a1, a2 = pair
            if a1 not in actors or a2 not in actors:
                continue
            srng = sentence_range(sent, prompt, tokenizer)
            if not srng:
                continue
            for actor, partner in ((a1, a2), (a2, a1)):
                for pos in actor_positions(tids, actor, tokenizer, start=srng[0], end=srng[1]):
                    for li, h in enumerate(layers):
                        partner_vecs[li].append(h[pos].numpy())
                    partner_labels.append(actors.index(partner))

    return (
        [np.array(v) for v in partner_vecs], partner_labels,
        [np.array(v) for v in owner_vecs], owner_labels,
    )


def best_accuracy(vecs_per_layer, labels, bg_class=None, cv=5):
    y = np.array(labels)
    if bg_class is not None:
        mask = y != bg_class
        y = y[mask]
    else:
        mask = np.ones(len(y), dtype=bool)
    if len(set(y.tolist())) < 2:
        return float("nan"), -1
    best_acc, best_layer = 0.0, 0
    for li, X_full in enumerate(vecs_per_layer):
        X = X_full[mask]
        if len(X) < cv * 2:
            continue
        sc = StandardScaler()
        clf = SGDClassifier(loss="log_loss", max_iter=2000, random_state=42)
        acc = float(cross_val_score(clf, sc.fit_transform(X), y, cv=cv).mean())
        if acc > best_acc:
            best_acc, best_layer = acc, li
    return best_acc, best_layer


print(f"Loading {MODEL_ID}...", flush=True)
tokenizer, model, device = load_model(MODEL_ID)
n_params = sum(p.numel() for p in model.parameters()) / 1e9
n_layers = model.config.num_hidden_layers + 1
print(f"  {n_params:.1f}B params  {n_layers} layers  device={device}", flush=True)

examples = load_task(TASK)[:N_EXAMPLES]
print("Collecting hidden states...", flush=True)
pv, pl, ov, ol = collect(examples, model, tokenizer, device)

n_actors = 3
print("Sweeping partner probe...", flush=True)
p_acc, p_layer = best_accuracy(pv, pl, bg_class=None)
print("Sweeping owner probe...", flush=True)
o_acc, o_layer = best_accuracy(ov, ol, bg_class=n_actors)

print(f"\nBitNet-b1.58-2B-4T: partner={p_acc:.1%} @ L{p_layer}   owner={o_acc:.1%} @ L{o_layer}")
