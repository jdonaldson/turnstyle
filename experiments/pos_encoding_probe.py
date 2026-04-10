#!/usr/bin/env python3
"""POS (syntactic role) encoding probe.

Does SmolLM2 encode syntactic role at token positions for ambiguous words?

Target words where our SVO extraction breaks:
  "left"   — VERB (departure): "he left it"
           — NOUN  (direction): "take a left"
  "right"  — NOUN  (direction): "turn right"
           — ADJ   (correct):   "that is right"
  "take"   — VERB  (movement):  "take 3 steps north"
           — VERB  (transfer):  "Alice takes the ball"
  "second" — ORDINAL:           "the second book"
           — NOUN  (time):      "wait a second"

No training needed — pure geometry test.
Intra-role cosine vs inter-role cosine → separation score.

Run:
    /Users/jdonaldson/Projects/swollm/.venv/bin/python \\
        experiments/pos_encoding_probe.py [--model smollm2|phi]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# ── model config ───────────────────────────────────────────────────────

_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument("--model", default="smollm2", choices=["smollm2", "phi"])
_args, _ = _parser.parse_known_args()

MODEL_CONFIGS = {
    "smollm2": {
        "model_id": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
        "layers": [2, 4, 6, 8, 10, 12],
        "layer_path": "model.layers",
    },
    "phi": {
        "model_id": "microsoft/Phi-4-mini-instruct",
        "layers": [2, 4, 6, 8, 10, 12],
        "layer_path": "model.layers",
    },
}

_cfg = MODEL_CONFIGS[_args.model]
MODEL_ID = _cfg["model_id"]
LAYERS = _cfg["layers"]

# ── minimal pairs ──────────────────────────────────────────────────────
# (sentence, target_word, role_label)
# role_label is the syntactic/semantic role we want to distinguish

MINIMAL_PAIRS: list[tuple[str, str, str]] = [

    # ── "left" ────────────────────────────────────────────────────────
    # VERB = past-tense "to leave" (departure)
    ("He left the room.",                          "left", "left:VERB"),
    ("Alice left the ball on the table.",          "left", "left:VERB"),
    ("She left without saying goodbye.",           "left", "left:VERB"),
    ("Bob left his keys behind.",                  "left", "left:VERB"),
    ("They left early yesterday.",                 "left", "left:VERB"),
    ("Carol left the book at home.",               "left", "left:VERB"),
    ("The cat left a mark on the door.",           "left", "left:VERB"),
    ("He left it there for you.",                  "left", "left:VERB"),
    # NOUN/ADV = spatial direction
    ("Turn left at the corner.",                   "left", "left:DIR"),
    ("Take a left at the next intersection.",      "left", "left:DIR"),
    ("The book is to the left of the pen.",        "left", "left:DIR"),
    ("Move 2 steps to the left.",                  "left", "left:DIR"),
    ("The door on the left opens first.",          "left", "left:DIR"),
    ("Look to the left and then the right.",       "left", "left:DIR"),
    ("Alice is to the left of Bob.",               "left", "left:DIR"),
    ("The red box is on the left side.",           "left", "left:DIR"),

    # ── "right" ───────────────────────────────────────────────────────
    # DIR = spatial direction
    ("Turn right at the stop sign.",               "right", "right:DIR"),
    ("Take a right at the corner.",                "right", "right:DIR"),
    ("The book is to the right of the pen.",       "right", "right:DIR"),
    ("Move 3 steps to the right.",                 "right", "right:DIR"),
    ("Alice is directly to the right of Bob.",     "right", "right:DIR"),
    ("The exit is on the right side.",             "right", "right:DIR"),
    ("Look to the right.",                         "right", "right:DIR"),
    ("The chair on the right belongs to Carol.",   "right", "right:DIR"),
    # ADJ = correct / true
    ("That answer is right.",                      "right", "right:ADJ"),
    ("You are right about that.",                  "right", "right:ADJ"),
    ("She got every question right.",              "right", "right:ADJ"),
    ("Bob was right all along.",                   "right", "right:ADJ"),
    ("Is that the right answer?",                  "right", "right:ADJ"),
    ("The right choice here is to wait.",          "right", "right:ADJ"),
    ("He knew it was right.",                      "right", "right:ADJ"),
    ("Only one of them is right.",                 "right", "right:ADJ"),

    # ── "take" ────────────────────────────────────────────────────────
    # MOVE = navigation (take N steps direction)
    ("Take 3 steps north.",                        "take", "take:MOVE"),
    ("Take 2 steps forward.",                      "take", "take:MOVE"),
    ("Take 1 step back.",                          "take", "take:MOVE"),
    ("Take 4 steps to the east.",                  "take", "take:MOVE"),
    ("Take 5 steps south then turn.",              "take", "take:MOVE"),
    ("Take one step left.",                        "take", "take:MOVE"),
    ("Take two steps west.",                       "take", "take:MOVE"),
    ("Take 3 steps and turn right.",               "take", "take:MOVE"),
    # GET = possession / object transfer (imperative, same form as MOVE)
    ("Take the ball from Bob.",                    "take", "take:GET"),
    ("Take the book off the table.",               "take", "take:GET"),
    ("Take the pen from Carol.",                   "take", "take:GET"),
    ("Take the coin and give it to Alice.",        "take", "take:GET"),
    ("Take the gift from the shelf.",              "take", "take:GET"),
    ("Take the trophy home.",                      "take", "take:GET"),
    ("Take the last remaining item.",              "take", "take:GET"),
    ("Take the red box from Bob.",                 "take", "take:GET"),

    # ── "second" ─────────────────────────────────────────────────────
    # ORD = ordinal position
    ("The second book from the left.",             "second", "second:ORD"),
    ("She finished in second place.",              "second", "second:ORD"),
    ("The pen is second from the right.",          "second", "second:ORD"),
    ("Bob is the second tallest.",                 "second", "second:ORD"),
    ("Take the second door on the left.",          "second", "second:ORD"),
    ("The second item in the list.",               "second", "second:ORD"),
    ("Alice arrived second.",                      "second", "second:ORD"),
    ("The second box contains the answer.",        "second", "second:ORD"),
    # TIME = unit of time
    ("Wait a second.",                             "second", "second:TIME"),
    ("Just a second please.",                      "second", "second:TIME"),
    ("He paused for a second.",                    "second", "second:TIME"),
    ("Give me one second.",                        "second", "second:TIME"),
    ("It happened in a second.",                   "second", "second:TIME"),
    ("She thought for a second.",                  "second", "second:TIME"),
    ("Not for a second did he doubt it.",          "second", "second:TIME"),
    ("Within a second the answer appeared.",       "second", "second:TIME"),
]


# ── model loading ──────────────────────────────────────────────────────

def load_model():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = (
        "mps"  if torch.backends.mps.is_available()  else
        "cuda" if torch.cuda.is_available()           else
        "cpu"
    )
    print(f"Loading {MODEL_ID} on {device}…", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    mdl = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=getattr(__import__("torch"), "float32")
    ).to(device).eval()
    return mdl, tok, device


# ── hidden state extraction ────────────────────────────────────────────

def get_layer(model, idx: int):
    obj = model
    for part in _cfg["layer_path"].split("."):
        obj = getattr(obj, part)
    return obj[idx]


def extract_at_token(
    sentence: str, target_word: str,
    model, tokenizer, device,
) -> dict[int, np.ndarray] | None:
    """Extract hidden state at first occurrence of target_word in sentence."""
    import torch

    inputs = tokenizer(sentence, return_tensors="pt").to(device)
    ids = inputs["input_ids"][0].tolist()
    token_strings = [tokenizer.decode([t]).strip().lower() for t in ids]

    # find first token that matches target (exact or stripped)
    target = target_word.lower()
    pos = None
    for i, s in enumerate(token_strings):
        if s == target or s.strip(",.;:") == target:
            pos = i
            break
    if pos is None:
        return None

    captured: dict[int, "torch.Tensor"] = {}
    handles = []

    def make_hook(layer_idx: int):
        def hook_fn(_m, _i, output):
            h = output[0] if isinstance(output, tuple) else output
            captured[layer_idx] = h.detach().cpu()
        return hook_fn

    for L in LAYERS:
        handles.append(get_layer(model, L).register_forward_hook(make_hook(L)))

    try:
        with torch.no_grad():
            model(**inputs)
    finally:
        for h in handles:
            h.remove()

    return {L: captured[L][0, pos, :].numpy() for L in LAYERS if L in captured}


# ── geometry analysis ──────────────────────────────────────────────────

def mean_cosine(A: np.ndarray, B: np.ndarray, seed: int = 0) -> float:
    A = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-8)
    B = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-8)
    rng = np.random.RandomState(seed)
    n = min(500, len(A) * len(B))
    ia = rng.randint(0, len(A), n)
    ib = rng.randint(0, len(B), n)
    return float(np.mean(np.sum(A[ia] * B[ib], axis=1)))


def analyse_geometry(
    vecs_by_role: dict[str, dict[int, list[np.ndarray]]]
) -> None:
    roles = sorted(vecs_by_role.keys())
    # group by target word
    words = sorted({r.split(":")[0] for r in roles})

    for word in words:
        word_roles = [r for r in roles if r.startswith(word + ":")]
        if len(word_roles) < 2:
            continue
        print(f"\n  ── '{word}' ──")
        print(f"  {'layer':<6}  {'role A':<14} {'role B':<14}  "
              f"{'intra-A':>8} {'intra-B':>8} {'inter':>8} {'sep':>6}")
        print("  " + "─" * 62)

        for L in LAYERS:
            mats = {}
            for r in word_roles:
                arrs = vecs_by_role[r][L]
                if arrs:
                    mats[r] = np.vstack(arrs)

            if len(mats) < 2:
                continue

            rA, rB = word_roles[0], word_roles[1]
            if rA not in mats or rB not in mats:
                continue

            intra_A = mean_cosine(mats[rA], mats[rA])
            intra_B = mean_cosine(mats[rB], mats[rB])
            inter   = mean_cosine(mats[rA], mats[rB])
            sep     = min(intra_A, intra_B) - inter

            print(f"  L{L:<5}  {rA:<14} {rB:<14}  "
                  f"{intra_A:8.3f} {intra_B:8.3f} {inter:8.3f} {sep:6.3f}")


# ── main ───────────────────────────────────────────────────────────────

def main():
    model, tokenizer, device = load_model()

    # {role_label: {layer: [vec, ...]}}
    vecs_by_role: dict[str, dict[int, list[np.ndarray]]] = {}

    n_miss = 0
    for sentence, target, role in MINIMAL_PAIRS:
        hiddens = extract_at_token(sentence, target, model, tokenizer, device)
        if hiddens is None:
            print(f"  [MISS] '{target}' not found in: {sentence}", flush=True)
            n_miss += 1
            continue
        if role not in vecs_by_role:
            vecs_by_role[role] = {L: [] for L in LAYERS}
        for L, vec in hiddens.items():
            vecs_by_role[role][L].append(vec)

    print(f"\n{len(MINIMAL_PAIRS) - n_miss}/{len(MINIMAL_PAIRS)} sentences "
          f"found target token  ({n_miss} misses)\n")

    # per-role counts
    for role in sorted(vecs_by_role):
        n = len(vecs_by_role[role][LAYERS[0]])
        print(f"  {role:<18}  n={n}")

    print(f"\n=== Geometry: intra-role vs inter-role cosine ({_args.model}) ===")
    print("  sep = min(intra_A, intra_B) − inter  (higher = more separable)\n")
    analyse_geometry(vecs_by_role)


if __name__ == "__main__":
    main()
