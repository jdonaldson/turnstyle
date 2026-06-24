#!/usr/bin/env python3
"""Weber's Law verification for SmolLM2: does it encode magnitude on a log scale?

Generates 100 prompts ("The number is {n}.") for n in 1..100, extracts hidden
states at the number-token position for all 24 layers, then computes RSA
(Representational Similarity Analysis) correlations:

  - Neural pairwise cosine distance vs log(|a - b| + 1) predictor
  - Neural pairwise cosine distance vs |a - b| predictor

If log > linear at most layers, Weber's Law holds for SmolLM2.

Also produces an MDS embedding at the best layer, colored by magnitude.

Run:
    /Users/jdonaldson/Projects/swollm/.venv/bin/python \
        experiments/weber_probe.py
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.spatial.distance import cosine as cosine_dist
from scipy.stats import pearsonr, spearmanr
from sklearn.manifold import MDS
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
N_NUMBERS = 100  # 1..100


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


# ── number token position ────────────────────────────────────────────────────

def find_number_token(token_ids: list[int], n: int, tokenizer) -> int | None:
    """Find the last subtoken position of the number n in the token sequence.

    BPE-safe: encodes ' {n}' and finds the last subtoken ID in the sequence.
    """
    num_ids = tokenizer.encode(f" {n}", add_special_tokens=False)
    if not num_ids:
        return None
    target = num_ids[-1]

    # Search from end to find the number in context (not in chat template)
    for i in range(len(token_ids) - 1, -1, -1):
        if token_ids[i] == target:
            # Verify this is actually our number by checking subtokens match
            if len(num_ids) == 1:
                return i
            # Multi-subtoken: check preceding tokens match
            start = i - len(num_ids) + 1
            if start >= 0 and token_ids[start:i + 1] == num_ids:
                return i
    return None


# ── hidden state extraction ──────────────────────────────────────────────────

def extract_hidden_states(
    prompt: str, model, tokenizer, device,
) -> tuple[list[int], list[torch.Tensor]]:
    """Forward pass with output_hidden_states=True.

    Returns (token_ids, layer_states) where layer_states[i] is (seq_len, hidden).
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    token_ids = inputs["input_ids"][0].tolist()

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    layer_states = [h[0].cpu().float() for h in outputs.hidden_states]
    return token_ids, layer_states


# ── RSA computation ──────────────────────────────────────────────────────────

def pairwise_cosine_distances(vecs: np.ndarray) -> np.ndarray:
    """Compute upper-triangle pairwise cosine distances.

    vecs: (N, D) array.
    Returns: flat array of N*(N-1)/2 distances.
    """
    n = len(vecs)
    dists = []
    for i in range(n):
        for j in range(i + 1, n):
            dists.append(cosine_dist(vecs[i], vecs[j]))
    return np.array(dists)


def predictor_matrix(numbers: list[int], transform: str) -> np.ndarray:
    """Compute upper-triangle pairwise predictor values.

    transform: 'log' for log(|a-b|+1), 'linear' for |a-b|.
    """
    n = len(numbers)
    dists = []
    for i in range(n):
        for j in range(i + 1, n):
            diff = abs(numbers[i] - numbers[j])
            if transform == "log":
                dists.append(np.log(diff + 1))
            else:
                dists.append(float(diff))
    return np.array(dists)


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    model, tokenizer, device = load_model()

    numbers = list(range(1, N_NUMBERS + 1))
    n_layers = model.config.num_hidden_layers + 1  # +1 for embedding (L0)

    # Collect hidden states at number token position for each number
    # Shape: (n_layers, N_NUMBERS, hidden_dim)
    all_vecs: list[list[np.ndarray]] = [[] for _ in range(n_layers)]
    valid_numbers: list[int] = []
    skipped = 0

    for idx, n in enumerate(numbers):
        prompt_text = f"The number is {n}."
        messages = [{"role": "user", "content": prompt_text}]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)

        token_ids, layer_states = extract_hidden_states(
            prompt, model, tokenizer, device)

        pos = find_number_token(token_ids, n, tokenizer)
        if pos is None:
            skipped += 1
            continue

        for layer_idx, h in enumerate(layer_states):
            all_vecs[layer_idx].append(h[pos].numpy())
        valid_numbers.append(n)

        if (idx + 1) % 20 == 0:
            print(
                f"[{idx + 1}/{N_NUMBERS}]  n={n}  pos={pos}  "
                f"token='{tokenizer.decode([token_ids[pos]])}'",
                flush=True,
            )

    n_valid = len(valid_numbers)
    print(f"\nCollected {n_valid} number representations ({skipped} skipped)\n")

    if n_valid < 10:
        print("Too few valid numbers for RSA analysis.")
        return

    # Compute predictor matrices (same for all layers)
    log_pred = predictor_matrix(valid_numbers, "log")
    lin_pred = predictor_matrix(valid_numbers, "linear")

    # RSA at each layer — Pearson (sensitive to scaling) and Spearman (rank-only)
    # Spearman can't distinguish log vs linear (monotonic transform preserves ranks),
    # so Pearson is the primary comparison. Spearman included for reference.
    print(f"{'Layer':>5}  {'Log(P)':>8}  {'Lin(P)':>8}  {'Delta':>7}  {'Spearman':>9}  {'Winner':>8}")
    print("-" * 56)

    best_log_layer = -1
    best_log_rsa = -1.0
    pearson_log_all = []
    pearson_lin_all = []
    spearman_all = []

    for layer_idx in range(n_layers):
        vecs = np.array(all_vecs[layer_idx])  # (n_valid, hidden_dim)
        neural_dists = pairwise_cosine_distances(vecs)

        # Pearson: sensitive to whether neural distance scales with log or linear
        r_log, _ = pearsonr(neural_dists, log_pred)
        r_lin, _ = pearsonr(neural_dists, lin_pred)
        # Spearman: identical for log/linear (monotonic), shown for reference
        rho, _ = spearmanr(neural_dists, log_pred)

        pearson_log_all.append(r_log)
        pearson_lin_all.append(r_lin)
        spearman_all.append(rho)

        delta = r_log - r_lin
        winner = "LOG" if delta > 0.005 else ("linear" if delta < -0.005 else "tie")
        print(
            f"  L{layer_idx:<3}  {r_log:>7.4f}  {r_lin:>7.4f}  {delta:>+6.4f}  {rho:>8.4f}  {winner:>8}",
            flush=True,
        )

        if r_log > best_log_rsa:
            best_log_rsa = r_log
            best_log_layer = layer_idx

    # Summary
    log_wins = sum(1 for l, n in zip(pearson_log_all, pearson_lin_all) if l > n + 0.005)
    lin_wins = sum(1 for l, n in zip(pearson_log_all, pearson_lin_all) if n > l + 0.005)
    total = len(pearson_log_all)
    print(f"\nLog wins: {log_wins}/{total}  Linear wins: {lin_wins}/{total}  "
          f"Ties: {total - log_wins - lin_wins}/{total}")
    print(f"Best log Pearson: L{best_log_layer} ({best_log_rsa:.4f})")

    avg_log = np.mean(pearson_log_all)
    avg_lin = np.mean(pearson_lin_all)
    print(f"Mean Pearson — log: {avg_log:.4f}, linear: {avg_lin:.4f}, "
          f"delta: {avg_log - avg_lin:+.4f}")

    # MDS at best layer
    print(f"\nMDS embedding at L{best_log_layer}...")
    vecs_best = np.array(all_vecs[best_log_layer])
    mds = MDS(n_components=2, dissimilarity="euclidean", random_state=42,
              normalized_stress="auto")
    coords = mds.fit_transform(vecs_best)

    print("\nMDS coordinates (number, x, y):")
    print(f"{'n':>4}  {'x':>8}  {'y':>8}")
    for n, (x, y) in zip(valid_numbers, coords):
        print(f"{n:>4}  {x:>8.3f}  {y:>8.3f}")

    # Save results for plotting
    results_path = "/Users/jdonaldson/Projects/turnstyle/experiments/weber_probe_results.npz"
    np.savez(
        results_path,
        pearson_log=np.array(pearson_log_all),
        pearson_lin=np.array(pearson_lin_all),
        spearman=np.array(spearman_all),
        mds_coords=coords,
        numbers=np.array(valid_numbers),
        best_layer=best_log_layer,
    )
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
