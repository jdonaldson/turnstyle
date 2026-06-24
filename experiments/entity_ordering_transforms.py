#!/usr/bin/env python3
"""Exp/log-aware entity ordering probes for logical_deduction.

Compares 5 probe variants to test whether transforming hidden states before
probing improves entity-ordering accuracy:

  baseline   — raw hidden state h (current approach, ~88-91%)
  log-mag    — sign(h) * log(1 + |h|)  (compress if representation is exp-scale)
  exp        — exp(h / T), T = norm-based scaling  (expand if log-scale)
  softmax-proj — softmax(h @ W_rand) for k=128  (mimic attention's native exp)
  mlp-2layer — ReLU(hW1+b1)W2+b2  (learnable nonlinear, upper bound)

Tests at L13, L14, L15 (known-good layers from entity-token probe experiments).
Reports per-item CV accuracy and answer-level accuracy via answer_from_ordering().

Run:
    /Users/jdonaldson/Projects/swollm/.venv/bin/python \
        experiments/entity_ordering_transforms.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, "/Users/jdonaldson/Projects/swollm/src")
sys.path.insert(0, str(Path(__file__).parent))

from swollm.bench.bbh import load_task
from logical_deduction_deterministic import (
    extract_items,
    gt_ordering,
    parse_options,
    answer_from_ordering,
)

MODEL_ID = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
TASKS = [
    "logical_deduction_three_objects",
    "logical_deduction_five_objects",
    "logical_deduction_seven_objects",
]
PROBE_LAYERS = [13, 14, 15]
N_FOLDS = 5
N_EXAMPLES = 250  # per task


# ── model ─────────────────────────────────────────────────────────────────────

def load_model():
    device = (
        "mps"  if torch.backends.mps.is_available() else
        "cuda" if torch.cuda.is_available()          else
        "cpu"
    )
    print(f"Loading {MODEL_ID} on {device}...", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    mdl = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16,
    ).to(device).eval()
    return mdl, tok, device


# ── hidden state extraction ──────────────────────────────────────────────────

def extract_hidden_states(
    prompt: str, model, tokenizer, device,
) -> tuple[list[int], list[torch.Tensor]]:
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    token_ids = inputs["input_ids"][0].tolist()
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    layer_states = [h[0].cpu().float() for h in outputs.hidden_states]
    return token_ids, layer_states


# ── BPE-safe item token search ───────────────────────────────────────────────

def find_item_last_occurrence(
    token_ids: list[int], item: str, tokenizer,
) -> int | None:
    """Find last occurrence of item's last subtoken in token_ids.

    BPE-safe: encodes ' {item}', takes last subtoken, scans from end.
    """
    item_ids = tokenizer.encode(" " + item, add_special_tokens=False)
    if not item_ids:
        return None
    target = item_ids[-1]

    for i in range(len(token_ids) - 1, -1, -1):
        if token_ids[i] == target:
            return i
    return None


# ── data collection ──────────────────────────────────────────────────────────

def collect_data(
    model, tokenizer, device,
) -> tuple[dict[int, list[np.ndarray]], list[int], list[dict]]:
    """Extract entity-token hidden states and ground-truth position labels.

    Returns:
        layer_vecs: {layer_idx: [hidden_state_vectors]}
        labels: position indices (0..N-1) for each vector
        example_meta: per-example metadata for answer-level accuracy
    """
    layer_vecs: dict[int, list[np.ndarray]] = {l: [] for l in PROBE_LAYERS}
    labels: list[int] = []
    example_meta: list[dict] = []  # {items, ordering, options, item_indices}

    total_items = 0
    skipped_examples = 0

    for task_name in TASKS:
        examples = load_task(task_name)[:N_EXAMPLES]
        for ex_idx, ex in enumerate(examples):
            text = ex["input"]
            items = extract_items(text)
            if not items:
                skipped_examples += 1
                continue

            ordering = gt_ordering(text, items)
            if ordering is None:
                skipped_examples += 1
                continue

            options = parse_options(text)

            # Build prompt (full text including options for position context)
            body = text.split("\nOptions:")[0].strip()
            messages = [{"role": "user", "content": body}]
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)

            token_ids, layer_states = extract_hidden_states(
                prompt, model, tokenizer, device)

            # Map each item to its ground-truth position (0-indexed)
            item_to_pos = {item: ordering.index(item) for item in items}

            item_indices_in_labels = []  # indices into labels[] for this example
            found = 0
            for item in items:
                pos = find_item_last_occurrence(token_ids, item, tokenizer)
                if pos is None:
                    continue

                label_idx = item_to_pos[item]
                for layer_idx in PROBE_LAYERS:
                    layer_vecs[layer_idx].append(
                        layer_states[layer_idx][pos].numpy()
                    )
                item_indices_in_labels.append(len(labels))
                labels.append(label_idx)
                found += 1

            if found > 0:
                example_meta.append({
                    "items": items,
                    "ordering": ordering,
                    "options": options,
                    "item_indices": item_indices_in_labels,
                    "target": ex["target"].strip(),
                })
                total_items += found
            else:
                skipped_examples += 1

            if (ex_idx + 1) % 50 == 0:
                print(
                    f"  [{task_name}] {ex_idx + 1}/{len(examples)}  "
                    f"items={total_items}  skipped={skipped_examples}",
                    flush=True,
                )

    print(f"\nTotal: {total_items} item vectors, {len(example_meta)} examples, "
          f"{skipped_examples} skipped\n")
    return layer_vecs, labels, example_meta


# ── transforms ───────────────────────────────────────────────────────────────

def transform_baseline(X: np.ndarray) -> np.ndarray:
    """Identity — raw hidden states."""
    return X


def transform_log_mag(X: np.ndarray) -> np.ndarray:
    """sign(h) * log(1 + |h|) — compress if representation is exp-scale."""
    return np.sign(X) * np.log1p(np.abs(X))


def transform_exp(X: np.ndarray) -> np.ndarray:
    """exp(h / T) — expand if representation is log-scale.

    T is set per-dimension to the std of that dimension, clamped to avoid
    overflow. This normalizes scale before exponentiation.
    """
    std = X.std(axis=0, keepdims=True)
    std = np.clip(std, 1e-6, None)
    scaled = X / std
    # Clamp to avoid overflow (exp(88) ~ float64 max)
    scaled = np.clip(scaled, -20, 20)
    return np.exp(scaled)


def transform_softmax_proj(X: np.ndarray, k: int = 128, seed: int = 42) -> np.ndarray:
    """softmax(h @ W_rand) — random projection then softmax.

    Mimics attention's native exp operation with fixed random weights.
    """
    rng = np.random.RandomState(seed)
    d = X.shape[1]
    W = rng.randn(d, k).astype(np.float32) / np.sqrt(d)
    logits = X @ W
    # Row-wise softmax
    logits_max = logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits - logits_max)
    return exp_logits / exp_logits.sum(axis=1, keepdims=True)


# ── MLP probe (PyTorch) ─────────────────────────────────────────────────────

class MLPProbe(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, n_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, x):
        return self.net(x)


def train_mlp_fold(
    X_train: np.ndarray, y_train: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
    n_classes: int,
    hidden_dim: int = 256,
    epochs: int = 200,
    lr: float = 1e-3,
) -> float:
    """Train a 2-layer MLP and return test accuracy."""
    device = "cpu"  # small data, CPU is fine
    X_tr = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_tr = torch.tensor(y_train, dtype=torch.long, device=device)
    X_te = torch.tensor(X_test, dtype=torch.float32, device=device)
    y_te = torch.tensor(y_test, dtype=torch.long, device=device)

    model = MLPProbe(X_train.shape[1], hidden_dim, n_classes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        loss = loss_fn(model(X_tr), y_tr)
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        preds = model(X_te).argmax(dim=1)
        acc = (preds == y_te).float().mean().item()
    return acc


# ── probe evaluation ─────────────────────────────────────────────────────────

def cv_accuracy_linear(
    X: np.ndarray, y: np.ndarray, n_folds: int = 5,
) -> tuple[float, np.ndarray]:
    """K-fold CV with SGDClassifier. Returns (mean_acc, per-sample predictions)."""
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    preds = np.full(len(y), -1)
    fold_accs = []

    for train_idx, test_idx in skf.split(X, y):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train_idx])
        X_test = scaler.transform(X[test_idx])
        y_train, y_test = y[train_idx], y[test_idx]

        clf = SGDClassifier(loss="log_loss", max_iter=2000, random_state=42)
        clf.fit(X_train, y_train)
        fold_preds = clf.predict(X_test)
        preds[test_idx] = fold_preds
        fold_accs.append((fold_preds == y_test).mean())

    return float(np.mean(fold_accs)), preds


def cv_accuracy_mlp(
    X: np.ndarray, y: np.ndarray, n_classes: int, n_folds: int = 5,
) -> tuple[float, np.ndarray]:
    """K-fold CV with 2-layer MLP. Returns (mean_acc, per-sample predictions)."""
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    preds = np.full(len(y), -1)
    fold_accs = []

    for train_idx, test_idx in skf.split(X, y):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train_idx])
        X_test = scaler.transform(X[test_idx])
        y_train, y_test = y[train_idx], y[test_idx]

        acc = train_mlp_fold(X_train, y_train, X_test, y_test, n_classes)
        fold_accs.append(acc)

        # Get predictions for answer-level accuracy
        device = "cpu"
        X_te = torch.tensor(X_test, dtype=torch.float32, device=device)
        model = MLPProbe(X_train.shape[1], 256, n_classes).to(device)
        # Re-train to get predictions (simpler than saving per-fold models)
        X_tr = torch.tensor(X_train, dtype=torch.float32, device=device)
        y_tr = torch.tensor(y_train, dtype=torch.long, device=device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = nn.CrossEntropyLoss()
        model.train()
        for _ in range(200):
            opt.zero_grad()
            loss_fn(model(X_tr), y_tr).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            preds[test_idx] = model(X_te).argmax(dim=1).numpy()

    return float(np.mean(fold_accs)), preds


# ── answer-level accuracy ────────────────────────────────────────────────────

def answer_accuracy(
    preds: np.ndarray, example_meta: list[dict],
) -> tuple[float, int, int]:
    """Compute answer-level accuracy from per-item position predictions.

    For each example, reconstruct ordering from predicted positions and
    match against ground-truth answer via answer_from_ordering().
    """
    correct = 0
    total = 0

    for meta in example_meta:
        indices = meta["item_indices"]
        items = meta["items"]
        options = meta["options"]
        target = meta["target"]

        # Get predicted positions for this example's items
        predicted_positions = []
        for idx in indices:
            predicted_positions.append(int(preds[idx]))

        if len(predicted_positions) != len(items):
            continue

        # Reconstruct ordering: sort items by predicted position
        item_pos_pairs = list(zip(items, predicted_positions))
        item_pos_pairs.sort(key=lambda x: x[1])
        predicted_ordering = [item for item, _ in item_pos_pairs]

        pred_answer = answer_from_ordering(predicted_ordering, options)
        if pred_answer is not None and pred_answer == target:
            correct += 1
        total += 1

    return correct / total if total > 0 else 0.0, correct, total


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    model, tokenizer, device = load_model()

    print("Collecting entity-token hidden states...\n")
    layer_vecs, labels_list, example_meta = collect_data(model, tokenizer, device)

    y = np.array(labels_list)
    n_classes = len(set(labels_list))
    print(f"Classes: {n_classes} (positions 0..{n_classes - 1})")
    print(f"Class distribution: {np.bincount(y).tolist()}\n")

    # Define transforms
    transforms = {
        "baseline": transform_baseline,
        "log-mag": transform_log_mag,
        "exp": transform_exp,
        "softmax-proj": transform_softmax_proj,
    }

    # Results table
    print(f"{'Layer':>5}  {'Variant':>14}  {'Item CV':>8}  {'Answer':>8}  {'n_correct':>10}")
    print("-" * 55)

    for layer_idx in PROBE_LAYERS:
        X_raw = np.array(layer_vecs[layer_idx])

        for name, tfn in transforms.items():
            X = tfn(X_raw)
            item_acc, preds = cv_accuracy_linear(X, y, N_FOLDS)
            ans_acc, n_correct, n_total = answer_accuracy(
                preds, example_meta)
            print(
                f"  L{layer_idx:<3}  {name:>14}  {item_acc:>7.1%}  "
                f"{ans_acc:>7.1%}  {n_correct:>5}/{n_total}",
                flush=True,
            )

        # MLP (separate because it uses PyTorch, not sklearn)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_raw)
        mlp_acc, mlp_preds = cv_accuracy_mlp(X_scaled, y, n_classes, N_FOLDS)
        ans_acc, n_correct, n_total = answer_accuracy(
            mlp_preds, example_meta)
        print(
            f"  L{layer_idx:<3}  {'mlp-2layer':>14}  {mlp_acc:>7.1%}  "
            f"{ans_acc:>7.1%}  {n_correct:>5}/{n_total}",
            flush=True,
        )
        print()

    # Summary interpretation
    print("=" * 55)
    print("Interpretation guide:")
    print("  - If log-mag or exp beats baseline by >=2pp: representation mismatch")
    print("  - If mlp-2layer beats all transforms: optimal transform is nonlinear")
    print("  - If baseline matches mlp-2layer: StandardScaler absorbs the encoding")
    print("  - The 12.7pp gap (87.3% -> 100%) may be capacity, not representation")


if __name__ == "__main__":
    main()
