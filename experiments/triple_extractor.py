#!/usr/bin/env python3
"""Probe-based triple extractor for tracking_shuffled_objects.

Reconstructs (subject, predicate, object) triples from hidden states alone,
with no regex parsing of the swap/init structure and no LLM extraction.

Three probes (trained once, applied at inference):
  Role probe    (L1)  — actor token → init | swap | query
  Partner probe (L14) — actor token in swap sentence → partner actor index
  Owner probe   (L14) — any token → actor index (0/1/2) | background (3)

Extraction pipeline for a new prompt:
  1. detect_actors() — fast name lookup, the only external call
  2. Forward pass → hidden states at L1 and L14
  3. Actor token positions via last-subtoken matching
  4. Role probe  → classify each actor occurrence
  5. Partner probe → swap-role actors emit (A, swap, B) triples
  6. Owner probe  → scan all tokens, non-background hits emit (A, has, token)

Evaluation: train/test split, report triple-level precision/recall.

Run:
    /Users/jdonaldson/Projects/swollm/.venv/bin/python \\
        experiments/triple_extractor.py
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, "/Users/jdonaldson/Projects/swollm/src")
sys.path.insert(0, str(Path(__file__).parent))

from swollm.bench.bbh import load_task
from tracking_deterministic import (
    ALL_ACTORS,
    detect_actors,
    parse_action,
    parse_init,
)

MODEL_ID   = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
TASK       = "tracking_shuffled_objects_three_objects"
N_EXAMPLES = 200
N_TRAIN    = 150

ROLE_LAYER    = 1
PARTNER_LAYER = 14
OWNER_LAYER   = 14
OWNER_BG      = 3   # background class for owner probe
OWN_CONF_THR  = 0.6 # min softmax confidence to emit an object triple


# ── data types ────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Triple:
    subj: str
    pred: str
    obj:  str

    def __str__(self) -> str:
        return f"({self.subj}, {self.pred}, {self.obj})"


class ProbeSet(NamedTuple):
    role_scaler:    StandardScaler
    role_clf:       SGDClassifier
    partner_scaler: StandardScaler
    partner_clf:    SGDClassifier
    owner_scaler:   StandardScaler
    owner_clf:      SGDClassifier


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

def actor_positions(tids: list[int], actor: str, tokenizer) -> list[int]:
    """Positions of actor's last subtoken in tids."""
    subtids = tokenizer.encode(" " + actor, add_special_tokens=False)
    if not subtids:
        return []
    target = subtids[-1]
    return [i for i, t in enumerate(tids) if t == target]


def sentence_range(sent: str, prompt: str, tokenizer) -> tuple[int, int] | None:
    try:
        cs = prompt.index(sent)
    except ValueError:
        return None
    ce = cs + len(sent)
    ts = len(tokenizer.encode(prompt[:cs], add_special_tokens=False))
    te = len(tokenizer.encode(prompt[:ce], add_special_tokens=False))
    return ts, te


def in_range(positions: list[int], rng: tuple[int, int]) -> list[int]:
    return [p for p in positions if rng[0] <= p < rng[1]]


# ── ground truth (for training labels and evaluation) ─────────────────────────

def ground_truth_triples(body: str, actors: list[str]) -> list[Triple]:
    lines = [l.strip() for l in body.split(".") if l.strip()]
    triples: list[Triple] = []

    init_sent = next((l for l in lines if re.search(r"At the start", l, re.I)), None)
    if init_sent:
        state = parse_init(init_sent, actors)
        for actor, obj in state.items():
            triples.append(Triple(actor, "has", obj))

    for line in lines:
        pair = parse_action(line)
        if pair:
            a1, a2 = pair
            triples.append(Triple(a1, "swap", a2))
            triples.append(Triple(a2, "swap", a1))

    return triples


# ── probe training ────────────────────────────────────────────────────────────

def _new_clf() -> SGDClassifier:
    return SGDClassifier(loss="log_loss", max_iter=2000, random_state=42)


def train_probes(
    examples: list[dict],
    model,
    tokenizer,
    device,
) -> ProbeSet:
    """Collect hidden states + labels for all three probes, fit, return ProbeSet."""

    role_X:    list[np.ndarray] = []
    role_y:    list[str]        = []
    part_X:    list[np.ndarray] = []
    part_y:    list[int]        = []
    owner_X:   list[np.ndarray] = []
    owner_y:   list[int]        = []

    for ex in examples:
        body   = ex["input"].split("\nOptions:")[0].strip()
        lines  = [l.strip() for l in body.split(".") if l.strip()]
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
        query_sent = next((l for l in lines if re.search(r"At the end", l, re.I)), None)

        messages = [{"role": "user", "content": body}]
        prompt   = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)

        tids, layers = hidden_states(prompt, model, tokenizer, device)
        L1  = layers[ROLE_LAYER]
        L14 = layers[OWNER_LAYER]  # same as PARTNER_LAYER

        init_rng  = sentence_range(init_sent,  prompt, tokenizer)
        query_rng = sentence_range(query_sent, prompt, tokenizer) if query_sent else None

        # ── Role probe labels ────────────────────────────────────────────────
        for actor in actors:
            for pos in actor_positions(tids, actor, tokenizer):
                if init_rng and init_rng[0] <= pos < init_rng[1]:
                    role_X.append(L1[pos].numpy()); role_y.append("init")
                elif query_rng and query_rng[0] <= pos < query_rng[1]:
                    role_X.append(L1[pos].numpy()); role_y.append("query")
                else:
                    # Check swap sentences
                    for sent, _ in swap_sents:
                        srng = sentence_range(sent, prompt, tokenizer)
                        if srng and srng[0] <= pos < srng[1]:
                            role_X.append(L1[pos].numpy()); role_y.append("swap")
                            break

        # ── Partner probe labels (swap sentences) ────────────────────────────
        for sent, pair in swap_sents:
            a1, a2 = pair
            if a1 not in actors or a2 not in actors:
                continue
            srng = sentence_range(sent, prompt, tokenizer)
            if not srng:
                continue
            for actor, partner in ((a1, a2), (a2, a1)):
                for pos in in_range(actor_positions(tids, actor, tokenizer), srng):
                    part_X.append(L14[pos].numpy())
                    part_y.append(actors.index(partner))

        # ── Owner probe labels (init sentence) ───────────────────────────────
        if not init_rng:
            continue
        # Use character-offset prefix-tokenization to pin the exact last token
        # of each object span.  This handles repeated tokens (e.g. three actors
        # all holding "ball") because position is unique, token_id is not.
        try:
            sent_char_start = prompt.index(init_sent)
        except ValueError:
            continue
        obj_pos_labels: dict[int, int] = {}  # token position → actor_idx
        for actor_idx, actor in enumerate(actors):
            obj_name = init_state[actor]
            try:
                obj_in_sent = init_sent.index(obj_name)
            except ValueError:
                continue
            pc_end   = sent_char_start + obj_in_sent + len(obj_name)
            te       = len(tokenizer.encode(prompt[:pc_end], add_special_tokens=False))
            anchor   = te - 1  # last token position of this object
            obj_pos_labels[anchor] = actor_idx
        # Scan init sentence: anchored positions → actor_idx, rest → background
        for pos in range(init_rng[0], init_rng[1]):
            owner_X.append(L14[pos].numpy())
            owner_y.append(obj_pos_labels.get(pos, OWNER_BG))

    print(f"Training probes: role={len(role_y)}  partner={len(part_y)}  owner={len(owner_y)}")

    def fit(X_list, y_list):
        X = np.array(X_list)
        sc = StandardScaler().fit(X)
        clf = _new_clf()
        clf.fit(sc.transform(X), y_list)
        return sc, clf

    r_sc, r_clf = fit(role_X,  role_y)
    p_sc, p_clf = fit(part_X,  part_y)
    o_sc, o_clf = fit(owner_X, owner_y)
    return ProbeSet(r_sc, r_clf, p_sc, p_clf, o_sc, o_clf)


# ── triple extraction ─────────────────────────────────────────────────────────

def extract_triples(
    prompt: str,
    actors: list[str],
    probes: ProbeSet,
    model,
    tokenizer,
    device,
) -> list[Triple]:
    tids, layers = hidden_states(prompt, model, tokenizer, device)
    L1  = layers[ROLE_LAYER]
    L14 = layers[OWNER_LAYER]
    triples: list[Triple] = []

    # ── Role classification of all actor token occurrences ───────────────────
    init_positions:  list[int] = []
    swap_positions:  dict[int, str] = {}   # pos → actor name

    for actor in actors:
        for pos in actor_positions(tids, actor, tokenizer):
            vec   = L1[pos].numpy().reshape(1, -1)
            role  = probes.role_clf.predict(probes.role_scaler.transform(vec))[0]
            if role == "init":
                init_positions.append(pos)
            elif role == "swap":
                swap_positions[pos] = actor

    # ── Partner probe → swap triples ─────────────────────────────────────────
    emitted_swaps: set[tuple[str, str]] = set()
    for pos, actor in swap_positions.items():
        vec    = L14[pos].numpy().reshape(1, -1)
        pred   = probes.partner_clf.predict(probes.partner_scaler.transform(vec))[0]
        if 0 <= pred < len(actors):
            partner = actors[pred]
            key = tuple(sorted((actor, partner)))
            if key not in emitted_swaps:
                triples.append(Triple(actor,   "swap", partner))
                triples.append(Triple(partner, "swap", actor))
                emitted_swaps.add(key)

    # ── Owner probe → has triples ─────────────────────────────────────────────
    if not init_positions:
        return triples

    # Approximate init sentence range: from first actor_pos − 60 to last + 60
    lo = max(0, min(init_positions) - 60)
    hi = min(len(tids), max(init_positions) + 60)

    window_vecs = np.array([L14[p].numpy() for p in range(lo, hi)])
    probs = np.exp(probes.owner_clf.predict_log_proba(
        probes.owner_scaler.transform(window_vecs)
    ))

    # For each actor class, find the position with peak confidence (the anchor
    # = last subtoken of the object name).  Expand leftward to recover the
    # full multi-word object name by including preceding tokens up to the
    # nearest comma, colon, or actor name boundary.
    for actor_idx in range(len(actors)):
        col = probs[:, actor_idx]
        peak_offset = int(np.argmax(col))
        peak_conf   = float(col[peak_offset])
        if peak_conf < OWN_CONF_THR:
            continue

        anchor = lo + peak_offset   # absolute token position of anchor

        # Expand left: include tokens until we hit a comma/colon or actor name
        left = anchor
        while left > lo:
            prev_tid = tids[left - 1]
            prev_tok = tokenizer.decode([prev_tid]).strip()
            if prev_tok in (",", ":", ";") or prev_tok in ALL_ACTORS:
                break
            left -= 1

        obj_text = tokenizer.decode(tids[left:anchor + 1]).strip().strip(",.:;")
        if obj_text and obj_text not in ALL_ACTORS:
            triples.append(Triple(actors[actor_idx], "has", obj_text))

    return triples


# ── evaluation ────────────────────────────────────────────────────────────────

def evaluate(examples: list[dict], probes: ProbeSet, model, tokenizer, device) -> None:
    swap_tp = swap_fp = swap_fn = 0
    has_tp  = has_fp  = has_fn  = 0
    n_examples = 0

    for ex in examples:
        body   = ex["input"].split("\nOptions:")[0].strip()
        lines  = [l.strip() for l in body.split(".") if l.strip()]
        actors = detect_actors(lines[0] if lines else body)
        if not actors:
            continue

        messages = [{"role": "user", "content": body}]
        prompt   = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)

        gt      = ground_truth_triples(body, actors)
        gt_swap = {t for t in gt if t.pred == "swap"}
        gt_has  = {t for t in gt if t.pred == "has"}

        pred      = extract_triples(prompt, actors, probes, model, tokenizer, device)
        pred_swap = {t for t in pred if t.pred == "swap"}
        pred_has  = {t for t in pred if t.pred == "has"}

        # For has-triples, match on (subj, pred) pair only — object name may be partial
        gt_has_pairs   = {(t.subj, t.pred) for t in gt_has}
        pred_has_pairs = {(t.subj, t.pred) for t in pred_has}

        swap_tp += len(pred_swap & gt_swap)
        swap_fp += len(pred_swap - gt_swap)
        swap_fn += len(gt_swap - pred_swap)

        has_tp  += len(pred_has_pairs & gt_has_pairs)
        has_fp  += len(pred_has_pairs - gt_has_pairs)
        missed   = gt_has_pairs - pred_has_pairs
        has_fn  += len(missed)
        for subj, _ in missed:
            obj_gt = next(t.obj for t in gt_has if t.subj == subj)
            obj_pred = next((t.obj for t in pred_has if t.subj == subj), "<missing>")
            print(f"  MISS  {subj:8s} gt={obj_gt!r:30s}  pred={obj_pred!r}")
        n_examples += 1

    def prf(tp, fp, fn):
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f = 2*p*r / (p+r)  if p + r  else 0.0
        return p, r, f

    sp, sr, sf = prf(swap_tp, swap_fp, swap_fn)
    hp, hr, hf = prf(has_tp,  has_fp,  has_fn)

    print(f"\nEvaluation on {n_examples} examples")
    print(f"{'':20s}  {'prec':>6}  {'rec':>6}  {'F1':>6}")
    print("─" * 44)
    print(f"  swap triples       {sp:>6.1%}  {sr:>6.1%}  {sf:>6.1%}  "
          f"(tp={swap_tp} fp={swap_fp} fn={swap_fn})")
    print(f"  has  triples       {hp:>6.1%}  {hr:>6.1%}  {hf:>6.1%}  "
          f"(tp={has_tp}  fp={has_fp}  fn={has_fn})")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    model, tokenizer, device = load_model()
    examples = load_task(TASK)[:N_EXAMPLES]

    train_ex = examples[:N_TRAIN]
    test_ex  = examples[N_TRAIN:]

    print(f"Train: {len(train_ex)}  Test: {len(test_ex)}", flush=True)
    probes = train_probes(train_ex, model, tokenizer, device)

    print("\nRunning extraction on test set…", flush=True)
    evaluate(test_ex, probes, model, tokenizer, device)


if __name__ == "__main__":
    main()
