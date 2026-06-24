"""Does the NanoGPT log-encoder + dispatcher pattern survive scaling?

Replicates two key NanoGPT tests on SmolLM2-1.7B (24 layers):

  (1) Per-layer log-vs-raw R² for mul and add. NanoGPT result:
        - mul: log(a*b+1) wins at every layer (1-18pp), peak at L2.
        - add: raw a+b wins at every layer (4-9pp).
      Tests whether the operator-specific log preference is a tiny-model
      artifact or a transformer-arithmetic motif.

  (2) Cross-op =-token cosine, paired by (a, b). NanoGPT result:
        - cos(mul, add) drops 0.80 → 0.60 between L0 and L1, biggest
          single-block step; saturates ~0.20 by L5.
      Tests whether a clean "dispatcher block" exists in SmolLM2.

Setup: `\\n5*3=` format (100% baseline accuracy for mul and add on both
base and Instruct). Probe site = the last token (`=`) at every layer.

100 prompts per operator. Skips sub (49-67% baseline; would need
restricted subset).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "HuggingFaceTB/SmolLM2-1.7B"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
CACHE = Path(__file__).parent / "data" / "smol_arithmetic"
CACHE.mkdir(parents=True, exist_ok=True)


@torch.no_grad()
def collect_states(model, tok, op: str, device: str = "cpu"):
    n_layers = model.config.num_hidden_layers
    H_per_layer = [[] for _ in range(n_layers + 1)]
    a_arr, b_arr, target_arr = [], [], []
    for a in range(10):
        for b in range(10):
            prompt = f"\n{a}{op}{b}="
            ids = tok.encode(prompt, add_special_tokens=False,
                             return_tensors="pt").to(device)
            out = model(ids, use_cache=False, output_hidden_states=True)
            # out.hidden_states: tuple of (n_layers + 1) tensors, each
            # (batch=1, seq, hidden). Index 0 = embeddings, 1..n = block outputs.
            for l, h in enumerate(out.hidden_states):
                H_per_layer[l].append(
                    h[0, -1, :].detach().to("cpu").float().numpy()
                )
            a_arr.append(a); b_arr.append(b)
            if op == "*":
                target_arr.append(a * b)
            elif op == "+":
                target_arr.append(a + b)
            else:
                target_arr.append(a - b)
    H = np.stack([np.stack(per) for per in H_per_layer], axis=0)
    # shape: (n_layers+1, 100, hidden)
    return {"a": np.array(a_arr, dtype=np.int32),
            "b": np.array(b_arr, dtype=np.int32),
            "target": np.array(target_arr, dtype=np.int32),
            "H": H}


def cv_r2(X, y, alpha=1.0, n_splits=5, seeds=3):
    r2s = []
    for seed in range(seeds):
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for tr, te in kf.split(X):
            s = StandardScaler().fit(X[tr])
            m = Ridge(alpha=alpha, fit_intercept=True).fit(s.transform(X[tr]),
                                                           y[tr])
            r2s.append(m.score(s.transform(X[te]), y[te]))
    return float(np.mean(r2s))


def main():
    print(f"Device: {DEVICE}")
    print(f"Loading {MODEL_NAME} ...")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float32, local_files_only=True
    ).to(DEVICE).eval()
    n_layers = model.config.num_hidden_layers
    print(f"  layers={n_layers}, hidden={model.config.hidden_size}")

    # Collect (or load cached) states
    for op in ("*", "+"):
        cache_file = CACHE / f"hidden_states_{ {'*': 'mul', '+': 'add'}[op] }.npz"
        if not cache_file.exists():
            print(f"\nCollecting states for op={op!r} ...")
            d = collect_states(model, tok, op, device=DEVICE)
            np.savez(cache_file, **d)
            print(f"  saved {cache_file.name}, H shape = {d['H'].shape}")
        else:
            print(f"\nUsing cached {cache_file.name}")

    del model
    if DEVICE == "mps":
        torch.mps.empty_cache()

    # Load
    mul = dict(np.load(CACHE / "hidden_states_mul.npz"))
    add = dict(np.load(CACHE / "hidden_states_add.npz"))
    H_mul = mul["H"]; H_add = add["H"]
    a = mul["a"]; b = mul["b"]
    n_states = H_mul.shape[0]  # n_layers + 1

    # ── (1) log-vs-raw R² per layer ──
    print()
    print("=" * 100)
    print("MULTIPLICATION (a*b) — log vs raw decodability per layer")
    print("=" * 100)
    y_raw = mul["target"].astype(float)
    y_log = np.log(y_raw + 1)
    y_la = np.log(a.astype(float) + 1)
    y_lb = np.log(b.astype(float) + 1)
    print(f"  {'state':>20}  {'raw':>8}  {'log':>8}  {'log−raw':>8}  "
          f"{'log(a)':>8}  {'log(b)':>8}")
    for l in range(n_states):
        label = "embedding" if l == 0 else f"L{l-1}"
        r2_raw = cv_r2(H_mul[l], y_raw)
        r2_log = cv_r2(H_mul[l], y_log)
        r2_la = cv_r2(H_mul[l], y_la)
        r2_lb = cv_r2(H_mul[l], y_lb)
        print(f"  {label:>20}  {r2_raw:+8.3f}  {r2_log:+8.3f}  "
              f"{r2_log - r2_raw:+8.3f}  {r2_la:+8.3f}  {r2_lb:+8.3f}")

    print()
    print("=" * 100)
    print("ADDITION (a+b) — log vs raw decodability per layer")
    print("=" * 100)
    y_raw = add["target"].astype(float)
    y_log = np.log(y_raw + 1)
    y_la = np.log(add["a"].astype(float) + 1)
    y_lb = np.log(add["b"].astype(float) + 1)
    print(f"  {'state':>20}  {'raw':>8}  {'log':>8}  {'log−raw':>8}  "
          f"{'log(a)':>8}  {'log(b)':>8}")
    for l in range(n_states):
        label = "embedding" if l == 0 else f"L{l-1}"
        r2_raw = cv_r2(H_add[l], y_raw)
        r2_log = cv_r2(H_add[l], y_log)
        r2_la = cv_r2(H_add[l], y_la)
        r2_lb = cv_r2(H_add[l], y_lb)
        print(f"  {label:>20}  {r2_raw:+8.3f}  {r2_log:+8.3f}  "
              f"{r2_log - r2_raw:+8.3f}  {r2_la:+8.3f}  {r2_lb:+8.3f}")

    # ── (2) Cross-op =-token cosine, paired by (a, b) ──
    print()
    print("=" * 100)
    print("Cross-op =-token cosine, paired by (a, b)")
    print("=" * 100)

    def avg_cos(X, Y):
        xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
        yn = Y / (np.linalg.norm(Y, axis=1, keepdims=True) + 1e-9)
        return float(np.mean(np.sum(xn * yn, axis=1)))

    print(f"  {'state':>20}  {'cos(mul, add)':>14}")
    cos_series = []
    for l in range(n_states):
        c = avg_cos(H_mul[l], H_add[l])
        cos_series.append(c)
        label = "embedding" if l == 0 else f"L{l-1}"
        print(f"  {label:>20}  {c:+14.4f}")

    # ── (3) Biggest single-layer drop in cross-op cosine ──
    print()
    print("Biggest per-layer drops in cos(mul, add):")
    drops = [(l - 1, cos_series[l - 1] - cos_series[l])
             for l in range(1, n_states)]
    drops.sort(key=lambda x: -x[1])
    for l, d in drops[:5]:
        s_label = "embedding→L0" if l == 0 else f"L{l-1}→L{l}"
        print(f"  {s_label}  Δcos = {-d:+.4f}")


if __name__ == "__main__":
    main()
