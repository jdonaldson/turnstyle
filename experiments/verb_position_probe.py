#!/usr/bin/env python3
"""Verb-position hidden state probe.

Does the model encode relationship type (claim vs state_transition) at the
hidden state of verb tokens in segment bodies? And does it transfer OOD
(train on one task's verbs, predict the other)?

Two tasks with clear inter-entity verb relationships:
  web_of_lies       → claim verbs  (says, tells, lies)
  tracking_shuffled → state_transition verbs  (swaps, gives, gets, trades)

Probe: LogReg on hidden state at verb token position, L4 and L8.
Test:  Leave-one-task-out — train on tracking verbs, predict web_of_lies
       and vice versa.

Run:
    /Users/jdonaldson/Projects/swollm/.venv/bin/python \\
        experiments/verb_position_probe.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/Users/jdonaldson/Projects/swollm/src")
from swollm.bench.bbh import load_task

import argparse

_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument("--model", default="smollm2",
                     choices=["smollm2", "phi"],
                     help="Which model to probe")
_args, _ = _parser.parse_known_args()

MODEL_CONFIGS = {
    "smollm2": {
        "model_id": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
        "layers": [4, 8, 12, 16],
        "layer_path": "model.layers",   # LlamaForCausalLM → .model → LlamaModel → .layers
    },
    "phi": {
        "model_id": "microsoft/Phi-4-mini-instruct",
        "layers": [8, 12, 16, 20, 24],
        "layer_path": "model.layers",   # Phi3ForCausalLM → .model → Phi3Model → .layers
    },
}

_cfg       = MODEL_CONFIGS[_args.model]
MODEL_ID   = _cfg["model_id"]
LAYERS     = _cfg["layers"]
MAX_EXAMPLES = 150
CACHE_PATH = Path(__file__).parent / f"verb_probe_data_{_args.model}.npz"

# ── verb vocabulary ────────────────────────────────────────────────────
# Lowercase surface forms. We search for these in the tokenized prompt.

VERB_CLASSES = {
    "claim": [
        "says", "say", "tells", "tell", "told",
        "lies", "lie", "lied",
    ],
    "state_transition": [
        "swaps", "swap", "swapped",
        "gives", "give", "gave",
        "gets", "get", "got",
        "trades", "trade", "traded",
        "receives", "receive", "received",
        "moves", "move", "moved",
    ],
    # Only single-token surface forms (verified against SmolLM2 tokenizer)
    "comparison": [
        "older", "newer", "oldest", "newest",
        "left", "right",
        "expensive", "cheapest",
        "second", "third",
    ],
}

TASK_EXPECTED_CLASS = {
    "web_of_lies":                             "claim",
    "tracking_shuffled_objects_three_objects": "state_transition",
    "logical_deduction_five_objects":          "comparison",
}

HELD_OUT_VERB = {
    "web_of_lies":                             "lies",
    "tracking_shuffled_objects_three_objects": "gets",
    "logical_deduction_five_objects":          "older",
}

# ── text helpers ───────────────────────────────────────────────────────

def strip_options(text: str) -> str:
    for marker in ("Options:", "\n- Yes", "\nOptions"):
        idx = text.find(marker)
        if idx != -1:
            text = text[:idx]
    return text.strip()


def find_verb_tokens(
    text: str, tokenizer, verb_class: str
) -> list[tuple[int, str]]:
    """Return (position, surface_verb) for all verb-class tokens in text."""
    tokens = tokenizer.encode(text, add_special_tokens=True)
    token_strings = [
        tokenizer.decode([t]).strip().lower() for t in tokens
    ]
    verbs = VERB_CLASSES[verb_class]
    hits = []
    for i, s in enumerate(token_strings):
        for v in verbs:
            if s == v or s.rstrip("s") == v.rstrip("s"):
                hits.append((i, v))
                break
    return hits


# ── model loading ──────────────────────────────────────────────────────

def load_model():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = (
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    print(f"Loading {MODEL_ID} on {device}…", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float32
    ).to(device).eval()
    return model, tokenizer, device


# ── hidden state extraction ────────────────────────────────────────────

def _get_layer_module(model, layer_path: str, idx: int):
    """Navigate 'a.b.c' attribute path on model, then index with idx."""
    obj = model
    for part in layer_path.split("."):
        obj = getattr(obj, part)
    return obj[idx]


def extract_verb_hiddens(
    text: str,
    verb_hits: list[tuple[int, str]],
    model,
    tokenizer,
    device,
    layers: list[int],
) -> tuple[dict[int, np.ndarray], list[str]] | None:
    """Return ({layer: (n_verbs, hidden_dim)}, surface_labels) for all verb positions."""
    import torch

    if not verb_hits:
        return None

    captured: dict[int, "torch.Tensor"] = {}
    handles = []
    layer_path = _cfg["layer_path"]

    def make_hook(layer_idx: int):
        def hook_fn(_module, _input, output):
            h = output[0] if isinstance(output, tuple) else output
            captured[layer_idx] = h.detach().cpu()
        return hook_fn

    for layer_idx in layers:
        handles.append(
            _get_layer_module(model, layer_path, layer_idx).register_forward_hook(make_hook(layer_idx))
        )

    try:
        inputs = tokenizer(text, return_tensors="pt").to(device)
        seq_len = inputs["input_ids"].shape[1]
        valid_hits = [(p, v) for p, v in verb_hits if p < seq_len]
        if not valid_hits:
            return None
        valid_pos = [p for p, _ in valid_hits]
        with torch.no_grad():
            model(**inputs)
    finally:
        for h in handles:
            h.remove()

    return {
        layer: captured[layer][0, valid_pos, :].numpy()
        for layer in layers
        if layer in captured
    }, [v for _, v in valid_hits]


# ── data collection ────────────────────────────────────────────────────

def collect(model, tokenizer, device) -> dict:
    """Collect verb-position hidden states for both tasks."""
    data: dict[str, dict] = {}   # task → {layer → [arrays], labels → [...]}

    for task, verb_class in TASK_EXPECTED_CLASS.items():
        print(f"\n{task} ({verb_class})", flush=True)
        examples = load_task(task)[:MAX_EXAMPLES]
        layer_vecs: dict[int, list[np.ndarray]] = {L: [] for L in LAYERS}
        n_found = 0

        verb_surfaces: list[str] = []

        for i, ex in enumerate(examples):
            text = strip_options(ex["input"])
            hits = find_verb_tokens(text, tokenizer, verb_class)
            if not hits:
                continue
            result = extract_verb_hiddens(
                text, hits, model, tokenizer, device, LAYERS
            )
            if result is None:
                continue
            hiddens, surfaces = result
            for L in LAYERS:
                if L in hiddens:
                    for vec in hiddens[L]:
                        layer_vecs[L].append(vec)
            verb_surfaces.extend(surfaces * (len(LAYERS)))  # one label per (layer, verb)
            n_found += len(hits)
            if (i + 1) % 25 == 0:
                print(f"  [{i+1}/{len(examples)}] verb tokens found: {n_found}", flush=True)

        # verb_surfaces has labels per (layer × verb occurrence) — need per-layer split
        # Store surfaces once (same order for all layers)
        n_per_layer = [len(layer_vecs[L]) for L in LAYERS if layer_vecs[L]]
        surfaces_per_layer = verb_surfaces[:n_per_layer[0]] if n_per_layer else []

        data[task] = {
            "verb_class": verb_class,
            "layers": {L: np.array(layer_vecs[L]) for L in LAYERS if layer_vecs[L]},
            "verb_surfaces": np.array(surfaces_per_layer),
            "n_verbs": n_found,
        }
        print(f"  total verb tokens: {n_found}", flush=True)

    return data


# ── probe training + eval ──────────────────────────────────────────────

def probe_eval(data: dict):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler

    tasks = list(data.keys())
    class_to_int = {c: i for i, c in enumerate(
        set(data[t]["verb_class"] for t in tasks)
    )}

    # ── Test 1: pooled 5-fold CV ───────────────────────────────────────
    # Both tasks pooled; folds are stratified by class.
    # Tells us: are claim and state_transition separable at verb positions?
    print("\n=== Test 1: pooled 5-fold CV (both tasks, stratified) ===\n")
    for L in LAYERS:
        X_all, y_all = [], []
        for t in tasks:
            if L not in data[t]["layers"]:
                continue
            H = data[t]["layers"][L]
            y = class_to_int[data[t]["verb_class"]]
            X_all.append(H)
            y_all.extend([y] * len(H))
        if not X_all or len(set(y_all)) < 2:
            continue
        X = np.vstack(X_all)
        y = np.array(y_all)
        scaler = StandardScaler()
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
        accs, aucs = [], []
        for tr, te in skf.split(X, y):
            Xs = scaler.fit_transform(X[tr])
            Xt = scaler.transform(X[te])
            clf = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=0)
            clf.fit(Xs, y[tr])
            accs.append(clf.score(Xt, y[te]))
            p = clf.predict_proba(Xt)[:, 1]
            if len(set(y[te])) > 1:
                n_cls = len(set(y))
                if n_cls > 2:
                    aucs.append(roc_auc_score(y[te], clf.predict_proba(Xt), multi_class="ovr", average="macro"))
                else:
                    aucs.append(roc_auc_score(y[te], p))
        print(f"  L{L}  acc={np.mean(accs):.3f}±{np.std(accs):.3f}  auc={np.mean(aucs):.3f}")

    # ── Test 2: leave-one-VERB-out within each task ────────────────────
    # Within web_of_lies: train on "says"+"tells", test on "lies".
    # Within tracking: train on "swaps"+"gives", test on "gets".
    # Tells us: does the probe generalise across verb surface forms within
    # the same relationship class?
    print("\n=== Test 2: leave-one-verb-out within task ===\n")
    print("(train on most verbs of each class, test on held-out verb surface form)\n")

    print(f"  {'layer':<6}  {'task':<14}  {'held-out verb':<14}  {'n_test':>6}  {'acc':>6}")
    print("  " + "-" * 58)

    for L in LAYERS:
        for task in tasks:
            if L not in data[task]["layers"]:
                continue
            surfaces = data[task].get("verb_surfaces", np.array([]))
            if len(surfaces) == 0:
                print(f"  L{L:<5}  {task.split('_')[0]:<14}  (no surface labels)")
                continue

            H       = data[task]["layers"][L]
            held    = HELD_OUT_VERB[task]
            y_focal = class_to_int[data[task]["verb_class"]]

            train_mask = surfaces != held
            test_mask  = surfaces == held
            if test_mask.sum() == 0:
                print(f"  L{L:<5}  {task.split('_')[0]:<14}  '{held}' not found")
                continue

            # train on non-held verbs from ALL tasks (n-class)
            X_train, y_train = [], []
            for t in tasks:
                if L not in data[t]["layers"]:
                    continue
                H_t = data[t]["layers"][L]
                y_t = class_to_int[data[t]["verb_class"]]
                mask = (surfaces != held) if t == task else np.ones(len(H_t), dtype=bool)
                X_train.append(H_t[mask])
                y_train.extend([y_t] * mask.sum())

            X_test = H[test_mask]
            y_test = np.array([y_focal] * test_mask.sum())

            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(np.vstack(X_train))
            X_te_s = scaler.transform(X_test)

            clf = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=0)
            clf.fit(X_tr_s, np.array(y_train))
            acc = clf.score(X_te_s, y_test)
            print(f"  L{L:<5}  {task.split('_')[0]:<14}  '{held}'  {test_mask.sum():>6}  {acc:>6.3f}")
    print()

    # ── Test 2b: leave-one-TASK-out (cross-template transfer) ──────────
    # Train on 2 tasks, predict the 3rd entirely.
    # Critical test: does comparison transfer from logical_deduction
    # when the probe has never seen that template?
    print("=== Test 2b: leave-one-task-out (cross-template transfer) ===\n")
    print(f"  {'layer':<6}  {'train tasks':<32}  {'test task':<14}  {'acc':>6}  {'note'}")
    print("  " + "-" * 72)

    for L in LAYERS:
        for held_task in tasks:
            if L not in data[held_task]["layers"]:
                continue
            train_tasks = [t for t in tasks if t != held_task]

            X_train, y_train = [], []
            for t in train_tasks:
                if L not in data[t]["layers"]:
                    continue
                X_train.append(data[t]["layers"][L])
                y_train.extend([class_to_int[data[t]["verb_class"]]] * len(data[t]["layers"][L]))

            if len(set(y_train)) < 2:
                continue

            X_test  = data[held_task]["layers"][L]
            y_test  = np.array([class_to_int[data[held_task]["verb_class"]]] * len(X_test))

            scaler  = StandardScaler()
            X_tr_s  = scaler.fit_transform(np.vstack(X_train))
            X_te_s  = scaler.transform(X_test)

            clf = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=0)
            clf.fit(X_tr_s, np.array(y_train))
            acc = clf.score(X_te_s, y_test)

            train_str = " + ".join(t.split("_")[0] for t in train_tasks)
            test_str  = held_task.split("_")[0]
            note = "← cross-template" if held_task == "logical_deduction_five_objects" else ""
            print(f"  L{L:<5}  {train_str:<32}  {test_str:<14}  {acc:>6.3f}  {note}")
        print()

    # ── Test 3: geometry — cosine between verb classes ─────────────────
    # Are claim verb positions closer to each other than to
    # state_transition verb positions, purely in terms of cosine distance?
    # No classifier needed — just pairwise cosine.
    print("=== Test 3: inter-class vs intra-class cosine similarity ===\n")
    for L in LAYERS:
        vecs_by_class: dict[str, np.ndarray] = {}
        for t in tasks:
            if L not in data[t]["layers"]:
                continue
            vc = data[t]["verb_class"]
            H  = data[t]["layers"][L].astype(np.float32)
            H  = H / (np.linalg.norm(H, axis=1, keepdims=True) + 1e-8)
            vecs_by_class[vc] = H

        if len(vecs_by_class) < 2:
            continue

        MAX_SAMP = 200
        rng = np.random.RandomState(0)
        sampled = {
            c: v[rng.choice(len(v), min(MAX_SAMP, len(v)), replace=False)]
            for c, v in vecs_by_class.items()
        }

        def mean_cosine(A: np.ndarray, B: np.ndarray) -> float:
            n   = min(500, len(A) * len(B))
            rng2 = np.random.RandomState(1)
            ia  = rng2.randint(0, len(A), n)
            ib  = rng2.randint(0, len(B), n)
            return float(np.mean(np.sum(A[ia] * B[ib], axis=1)))

        classes_list = sorted(sampled.keys())
        print(f"  L{L}:")
        for i, ca in enumerate(classes_list):
            for cb in classes_list[i:]:
                label = "intra" if ca == cb else "inter"
                sim   = mean_cosine(sampled[ca], sampled[cb])
                print(f"    {label}  {ca:<18} × {cb:<18}  {sim:.4f}")


# ── main ───────────────────────────────────────────────────────────────

def main():
    if CACHE_PATH.exists():
        print(f"Loading cached data from {CACHE_PATH}")
        npz = np.load(CACHE_PATH, allow_pickle=True)
        data = npz["data"].item()
    else:
        model, tokenizer, device = load_model()
        data = collect(model, tokenizer, device)
        np.savez(CACHE_PATH, data=np.array(data, dtype=object))
        print(f"\nCached to {CACHE_PATH}")

    probe_eval(data)


if __name__ == "__main__":
    main()
