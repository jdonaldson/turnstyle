#!/usr/bin/env python3
"""SVO extraction with centroid-based disambiguation gate.

Builds role centroids from the minimal-pair data in pos_encoding_probe.py,
then re-runs SVO extraction on BBH tasks.  For ambiguous words (left, right,
second) it replaces the dictionary classification with the nearest-centroid
prediction before deciding whether a segment encodes a real relationship.

'take' is handled by bigram rule (sentence-initial; no centroid separation).

Run:
    /Users/jdonaldson/Projects/swollm/.venv/bin/python \\
        experiments/svo_gate.py
"""

from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, "/Users/jdonaldson/Projects/swollm/src")
from swollm.bench.bbh import load_task

sys.path.insert(0, str(Path(__file__).parent))
from segment_homogeneity import SEGMENTERS, sentence_segments
from svo_extraction import VERB_CANON, extract_canonical
from pos_encoding_probe import MINIMAL_PAIRS, MODEL_CONFIGS

# ── config ─────────────────────────────────────────────────────────────

GATE_LAYER = 8   # best separation for "left" (sep=0.194)

# Roles that indicate the word is NOT functioning as a relationship verb.
# When the gate predicts one of these, suppress the SVO hit.
NON_RELATIONSHIP_ROLES = {"left:VERB", "right:ADJ", "second:TIME"}

# Bigram rules for sentence-initial verbs where centroid can't help.
# Pattern → replacement verb_canon (or None to suppress)
BIGRAM_RULES: list[tuple[re.Pattern, str | None]] = [
    # "take N steps direction" → NAVIGATION
    (re.compile(r"^take\s+\d+\s+steps?\b", re.I), "NAVIGATION"),
    # "take N steps forward/backward/left/right" → NAVIGATION
    (re.compile(r"^take\s+\d+\s+steps?\s+(forward|backward|left|right)\b", re.I), "NAVIGATION"),
]

TASKS = {
    "web_of_lies":                             "dependency",
    "tracking_shuffled_objects_three_objects": "ordered_ops",
    "logical_deduction_five_objects":          "ordered_ops",
    "navigate":                                "ordered_ops",
}

# ── model loading ──────────────────────────────────────────────────────

def load_model(model_key: str = "smollm2"):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    cfg = MODEL_CONFIGS[model_key]
    device = (
        "mps"  if torch.backends.mps.is_available()  else
        "cuda" if torch.cuda.is_available()           else
        "cpu"
    )
    print(f"Loading {cfg['model_id']} on {device}…", flush=True)
    tok = AutoTokenizer.from_pretrained(cfg["model_id"])
    mdl = AutoModelForCausalLM.from_pretrained(
        cfg["model_id"], dtype=getattr(__import__("torch"), "float32")
    ).to(device).eval()
    return mdl, tok, device, cfg


# ── hidden state extraction ────────────────────────────────────────────

def _get_layer(model, layer_path: str, idx: int):
    obj = model
    for part in layer_path.split("."):
        obj = getattr(obj, part)
    return obj[idx]


def extract_at_token(
    sentence: str, target_word: str,
    model, tokenizer, device, layer_path: str,
    layer: int = GATE_LAYER,
) -> np.ndarray | None:
    import torch
    inputs = tokenizer(sentence, return_tensors="pt").to(device)
    ids = inputs["input_ids"][0].tolist()
    token_strings = [tokenizer.decode([t]).strip().lower() for t in ids]

    target = target_word.lower()
    pos = next(
        (i for i, s in enumerate(token_strings)
         if s == target or s.strip(",.;:") == target),
        None,
    )
    if pos is None:
        return None

    captured: dict[int, "torch.Tensor"] = {}
    handle = _get_layer(model, layer_path, layer).register_forward_hook(
        lambda _m, _i, out: captured.__setitem__(
            layer, (out[0] if isinstance(out, tuple) else out).detach().cpu()
        )
    )
    try:
        with torch.no_grad():
            model(**inputs)
    finally:
        handle.remove()

    if layer not in captured:
        return None
    return captured[layer][0, pos, :].numpy()


# ── centroid building ──────────────────────────────────────────────────

def build_centroids(model, tokenizer, device, layer_path: str) -> dict:
    """
    Compute per-role mean centroids at GATE_LAYER from MINIMAL_PAIRS.
    Returns {word: {role: unit_centroid_vec}}.
    """
    role_vecs: dict[str, dict[str, list[np.ndarray]]] = {}

    for sentence, target, role in MINIMAL_PAIRS:
        vec = extract_at_token(
            sentence, target, model, tokenizer, device, layer_path
        )
        if vec is None:
            continue
        word = role.split(":")[0]
        role_vecs.setdefault(word, {}).setdefault(role, []).append(vec)

    centroids: dict[str, dict[str, np.ndarray]] = {}
    for word, roles in role_vecs.items():
        centroids[word] = {}
        for role, vecs in roles.items():
            c = np.vstack(vecs).mean(axis=0)
            centroids[word][role] = c / (np.linalg.norm(c) + 1e-8)
    return centroids


# ── disambiguation gate ────────────────────────────────────────────────

def gate_classify(
    sentence: str, word: str,
    centroids: dict,
    model, tokenizer, device, layer_path: str,
) -> str | None:
    """
    Return nearest-centroid role label, or None if word not in centroids.
    """
    if word not in centroids:
        return None
    vec = extract_at_token(
        sentence, word, model, tokenizer, device, layer_path
    )
    if vec is None:
        return None
    vec_n = vec / (np.linalg.norm(vec) + 1e-8)
    best_role, best_sim = None, -1.0
    for role, centroid in centroids[word].items():
        sim = float(np.dot(vec_n, centroid))
        if sim > best_sim:
            best_sim, best_role = sim, role
    return best_role


def apply_bigram_rules(seg: str, result: dict) -> dict:
    """Apply bigram rules for sentence-initial verbs (e.g. 'take N steps')."""
    for pattern, replacement in BIGRAM_RULES:
        if pattern.match(seg.strip()):
            result = dict(result)
            result["verb_canon"] = replacement
            result["gated_by"] = "bigram"
            return result
    return result


# ── per-task analysis ──────────────────────────────────────────────────

def analyse_task(
    task: str, list_type: str,
    centroids: dict, model, tokenizer, device, layer_path: str,
) -> None:
    examples  = load_task(task)[:150]
    segmenter = SEGMENTERS.get(task, sentence_segments)

    n_segs = 0
    before_ct: Counter = Counter()
    after_ct:  Counter = Counter()
    changes:   list[dict] = []

    for ex in examples:
        for seg in segmenter(ex["input"]):
            n_segs += 1
            result = extract_canonical(seg)
            if result is None:
                continue

            before_canon = result["verb_canon"]
            before_ct[before_canon] += 1

            # 1. bigram rules (no model needed)
            result = apply_bigram_rules(seg, result)

            # 2. centroid gate for known ambiguous words
            word = result["verb"]
            if word in centroids:
                predicted_role = gate_classify(
                    seg, word, centroids, model, tokenizer, device, layer_path
                )
                if predicted_role in NON_RELATIONSHIP_ROLES:
                    result = dict(result)
                    result["verb_canon"] = "NON_RELATIONSHIP"
                    result["gated_by"]   = f"centroid({predicted_role})"

            after_canon = result["verb_canon"]
            after_ct[after_canon] += 1

            if before_canon != after_canon:
                changes.append({
                    "seg":    seg[:80],
                    "before": before_canon,
                    "after":  after_canon,
                    "how":    result.get("gated_by", "?"),
                })

    print(f"\n{'─'*64}")
    print(f"{task}  [{list_type}]  ({n_segs} segments)")

    all_types = sorted(set(list(before_ct) + list(after_ct)))
    print(f"\n  {'verb_canon':<20} {'before':>8} {'after':>8} {'delta':>6}")
    print(f"  {'─'*44}")
    for vt in all_types:
        b, a = before_ct.get(vt, 0), after_ct.get(vt, 0)
        print(f"  {vt:<20} {b:>8} {a:>8} {a-b:>+6}")

    if changes:
        print(f"\n  {len(changes)} classifications changed:")
        for c in changes[:10]:
            print(f"    [{c['before']} → {c['after']}] via {c['how']}")
            print(f"      \"{c['seg']}\"")
        if len(changes) > 10:
            print(f"    … and {len(changes)-10} more")


# ── main ───────────────────────────────────────────────────────────────

def main():
    model, tokenizer, device, cfg = load_model("smollm2")

    print("\nBuilding centroids from minimal pairs…", flush=True)
    centroids = build_centroids(model, tokenizer, device, cfg["layer_path"])
    for word, roles in sorted(centroids.items()):
        print(f"  {word}: {list(roles.keys())}")

    print("\n=== SVO extraction: before vs after gate ===")
    for task, list_type in TASKS.items():
        analyse_task(
            task, list_type,
            centroids, model, tokenizer, device, cfg["layer_path"],
        )


if __name__ == "__main__":
    main()
