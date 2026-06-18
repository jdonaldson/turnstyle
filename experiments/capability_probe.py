#!/usr/bin/env python3
"""L18 Capability Probe Experiment — Phases 0/1/2/3.

Does a hidden-state probe at L18 (keyed on (task, solver) pairs) beat a
sequential try-in-order fallback chain? All phases run in **no-regex mode**
— regex fast paths are bypassed; SQL/IR/poll solvers are called directly.

Phases
------
0. Feasibility gate
     For each task × example:
       - One forward pass through SmolLM2 caches hidden states at L8, L12,
         L16, L18, L20, L22 (last-token pooling).
       - Each applicable tier runs in isolation; we record answered/correct.
     Computes sequential (first-answered wins) vs oracle (any tier correct).
     Gate: if max gap < 2pp, the sequential chain is already near-optimal;
     stop with a negative result.

1. Probe training (conditional on Phase 0 gate)
     One binary LogReg probe per (task, tier) at L18, 5-fold CV, class-weighted.
     Reports accuracy, AUC, Brier, and gain over majority baseline.

2. CapabilityRouter eval (conditional on Phase 1 gate)
     For each example, sort tiers by probe-predicted success and walk that
     order. Compares against sequential and oracle. Uses CV out-of-fold
     probabilities to avoid train-on-test leakage.

3. Layer ablation
     Re-trains probes at L8/L12/L16/L18/L20/L22 using the same cached
     hidden states. Identifies the best layer per (task, tier).

Usage
-----
    python experiments/capability_probe.py --phase 0
    python experiments/capability_probe.py --phase 1
    python experiments/capability_probe.py --phase 2
    python experiments/capability_probe.py --phase 3
"""

from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from pathlib import Path

import numpy as np


# ──────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────

BBH_CACHE = "/Users/jdonaldson/Projects/swollm/data/bbh_cache"
MODEL_ID = "HuggingFaceTB/SmolLM2-1.7B-Instruct"

EXPERIMENT_DIR = Path(__file__).parent
CACHE_PATH = EXPERIMENT_DIR / "capability_probe_data.npz"
PROBE_PATH = EXPERIMENT_DIR / "capability_probe_models.npz"

# Layers captured in a single forward pass. Phase 3 reuses these.
DEFAULT_LAYERS = [8, 12, 16, 18, 20, 22]
# Primary layer for the capability probe.
# SmolLM2 has 24 layers; L18 ≈ 75% depth.
# For other architectures, parameterize as layer = int(round(0.75 * n_layers)).
L_PRIMARY = 18

# Per-task tier list. Order is the production sequential fallback order.
TASK_TIERS: dict[str, list[str]] = {
    "penguins_in_a_table": ["sql", "knowledge_poll", "logit_poll", "baseline"],
    "tracking_shuffled_objects_three_objects": ["sql", "logit_poll", "baseline"],
    "object_counting": ["sql", "baseline"],
    "navigate": ["ir", "baseline"],
    "web_of_lies": ["ir", "baseline"],
}

# Gate thresholds
PHASE0_GAP_THRESHOLD = 0.02   # at least 2pp gap somewhere to proceed
PHASE1_GAIN_THRESHOLD = 0.05  # at least 5pp probe gain over majority


# ──────────────────────────────────────────────────────────────────────────
# Setup helpers
# ──────────────────────────────────────────────────────────────────────────

def load_task_examples(name: str) -> list[dict]:
    with open(os.path.join(BBH_CACHE, f"{name}.json")) as f:
        return json.load(f)


def detect_device() -> str:
    import torch
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def build_solver(task: str, model, tokenizer, device):
    """Construct a turnstyle solver in no-regex mode for this task.

    We call the SQLTurnstyle / IRSolver / SentenceIRSolver internals directly
    and never invoke the regex fast paths in swollm/solvers/*.
    """
    if task == "penguins_in_a_table":
        from turnstyle.sql import SQLTurnstyle
        from swollm.solvers.penguins import parse_tables
        return SQLTurnstyle(
            model, tokenizer, device,
            parse_tables_fn=parse_tables,
            logit_poll_fallback=True,
        )
    if task == "tracking_shuffled_objects_three_objects":
        from turnstyle.sql import SQLTurnstyle
        from swollm.solvers.tracking_shuffled import parse_tables
        return SQLTurnstyle(
            model, tokenizer, device,
            parse_tables_fn=parse_tables,
            logit_poll_fallback=True,
        )
    if task == "object_counting":
        from turnstyle.sql import SQLTurnstyle
        from swollm.solvers.object_counting import parse_tables
        return SQLTurnstyle(
            model, tokenizer, device,
            parse_tables_fn=parse_tables,
        )
    if task == "navigate":
        from turnstyle.ir import IRSolver
        from swollm.solvers.navigate import get_ir_spec
        return IRSolver(model, tokenizer, device, spec=get_ir_spec())
    if task == "web_of_lies":
        from turnstyle.ir import SentenceIRSolver
        from swollm.solvers.web_of_lies import get_sentence_ir_spec
        return SentenceIRSolver(
            model, tokenizer, device, spec=get_sentence_ir_spec())
    raise ValueError(f"unknown task: {task}")


# ──────────────────────────────────────────────────────────────────────────
# Hidden-state extraction
# ──────────────────────────────────────────────────────────────────────────

def extract_last_token_hiddens(
    text: str, model, tokenizer, device, layers: list[int],
) -> dict[int, np.ndarray]:
    """Forward pass with hooks on multiple layers simultaneously.

    One pass, hooks fire as the model walks through each layer. Returns
    {layer_idx: (hidden_dim,) float32 numpy array} — last-token pooling.
    """
    import torch

    captured: dict[int, "torch.Tensor"] = {}
    handles = []

    def make_hook(layer_idx: int):
        def hook_fn(_module, _input, output):
            h = output[0] if isinstance(output, tuple) else output
            captured[layer_idx] = h.detach()
        return hook_fn

    for layer_idx in layers:
        handle = model.model.layers[layer_idx].register_forward_hook(
            make_hook(layer_idx))
        handles.append(handle)

    try:
        messages = [{"role": "user", "content": text}]
        chat_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(chat_text, return_tensors="pt").to(device)
        with torch.no_grad():
            model(**inputs)
    finally:
        for h in handles:
            h.remove()

    return {
        layer_idx: captured[layer_idx][0, -1].cpu().float().numpy()
        for layer_idx in layers
    }


# ──────────────────────────────────────────────────────────────────────────
# Per-tier isolated execution
# ──────────────────────────────────────────────────────────────────────────

def run_tier(
    tier: str, text: str, solver, baseline_ctx: dict,
) -> str | None:
    """Run one solver tier in isolation. Returns answer string or None."""
    from turnstyle.sql import extract_options, extract_question

    try:
        if tier == "sql":
            result = solver._sql_solve(text)
            if result is None:
                return None
            _, answer = result
            return answer

        if tier == "knowledge_poll":
            question = extract_question(text)
            options = extract_options(text)
            if not question or not options:
                return None
            result = solver._knowledge_poll(question, options)
            if result is None:
                return None
            _, answer = result
            return answer

        if tier == "logit_poll":
            options = extract_options(text)
            if not options:
                return None
            result = solver._logit_poll(text, options)
            if result is None:
                return None
            _, answer = result
            return answer

        if tier == "ir":
            return solver.solve(text)

        if tier == "baseline":
            # Always-answering fallback via class-token logits over the 3-shot
            # prompt. Matches swollm.bench.bbh.run_bypass_diagnose's fallback.
            return _baseline_classify(text, baseline_ctx)

    except Exception:
        return None

    return None


def _baseline_classify(text: str, ctx: dict) -> str:
    import torch
    from swollm.bench.bbh import format_3shot

    prompt = format_3shot(ctx["examples"], ctx["exemplar_indices"], text)
    inputs = ctx["tokenizer"](prompt, return_tensors="pt").to(ctx["device"])
    with torch.no_grad():
        logits = ctx["model"](**inputs).logits[0, -1]
    probs = torch.softmax(logits.float(), dim=-1)
    class_probs = {
        cls: sum(probs[t].item() for t in toks)
        for cls, toks in ctx["class_tokens"].items()
    }
    return max(class_probs, key=class_probs.get)


def build_baseline_ctx(
    examples: list[dict], exemplar_indices, model, tokenizer, device,
) -> dict:
    targets = [ex["target"].strip() for ex in examples]
    classes = sorted(set(targets))
    class_tokens = {}
    for cls in classes:
        toks = tokenizer.encode(f" {cls}", add_special_tokens=False)
        class_tokens[cls] = toks[:1]
    return {
        "examples": examples,
        "exemplar_indices": exemplar_indices,
        "classes": classes,
        "class_tokens": class_tokens,
        "model": model,
        "tokenizer": tokenizer,
        "device": device,
    }


# ──────────────────────────────────────────────────────────────────────────
# Phase 0 — Feasibility gate
# ──────────────────────────────────────────────────────────────────────────

def phase0(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from swollm.bench.bbh import get_exemplars

    device = detect_device()
    print(f"Device: {device}")
    print(f"Model: {MODEL_ID}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16).to(device)
    model.eval()
    hidden_dim = model.config.hidden_size

    cache: dict[str, np.ndarray] = {}
    if CACHE_PATH.exists() and not args.overwrite:
        existing = np.load(CACHE_PATH, allow_pickle=True)
        cache.update({k: existing[k] for k in existing.files})
        print(f"Loaded existing cache with {len(cache)} keys", flush=True)

    task_filter = set(args.tasks.split(",")) if args.tasks else None

    summary_rows: list[dict] = []
    for task, tiers in TASK_TIERS.items():
        if task_filter and task not in task_filter:
            continue

        # Skip if already cached
        already_done = all(
            f"{task}__{t}__correct" in cache for t in tiers
        ) and f"{task}__hidden_l{L_PRIMARY}" in cache
        if already_done and not args.overwrite:
            print(f"\n=== {task} (cached) ===", flush=True)
        else:
            print(f"\n=== {task} ===", flush=True)
            _run_task_phase0(
                task, tiers, model, tokenizer, device, hidden_dim, cache)
            # Persist after each task (cheap, safeguards progress)
            np.savez(CACHE_PATH, **cache)
            print(f"  saved cache → {CACHE_PATH.name}", flush=True)

        summary_rows.append(_summarize_task_phase0(task, tiers, cache))

    _write_phase0_summary(summary_rows)

    if not summary_rows:
        print("\nNo tasks processed.")
        return

    max_gap = max(r["gap"] for r in summary_rows)
    print(f"\nMax gap across tasks: {max_gap:+.3f}")
    if max_gap >= PHASE0_GAP_THRESHOLD:
        print(f"GATE PASSED (gap ≥ {PHASE0_GAP_THRESHOLD:.2f}) → proceed to Phase 1")
    else:
        print(f"GATE FAILED (gap < {PHASE0_GAP_THRESHOLD:.2f}) → "
              "sequential chain is near-optimal in no-regex mode")


def _run_task_phase0(
    task: str, tiers: list[str], model, tokenizer, device, hidden_dim, cache,
):
    from swollm.bench.bbh import get_exemplars

    examples = load_task_examples(task)
    exemplar_indices = get_exemplars(examples)
    test_indices = [i for i in range(len(examples)) if i not in exemplar_indices]
    targets = [examples[i]["target"].strip() for i in test_indices]
    inputs = [examples[i]["input"] for i in test_indices]
    n = len(test_indices)

    solver = build_solver(task, model, tokenizer, device)
    baseline_ctx = build_baseline_ctx(
        examples, exemplar_indices, model, tokenizer, device)

    hiddens = {L: np.zeros((n, hidden_dim), dtype=np.float32)
               for L in DEFAULT_LAYERS}
    per_tier_answered = {t: np.zeros(n, dtype=np.int8) for t in tiers}
    per_tier_correct = {t: np.zeros(n, dtype=np.int8) for t in tiers}
    per_tier_answer: dict[str, list[str]] = {t: ["" for _ in range(n)] for t in tiers}

    t_start = time.time()
    for i, (text, target) in enumerate(zip(inputs, targets)):
        # 1. Hidden states (one forward pass, hooks on all layers)
        h = extract_last_token_hiddens(
            text, model, tokenizer, device, DEFAULT_LAYERS)
        for L in DEFAULT_LAYERS:
            hiddens[L][i] = h[L]

        # 2. Run each tier independently
        for tier in tiers:
            ans = run_tier(tier, text, solver, baseline_ctx)
            if ans is None:
                per_tier_answered[tier][i] = 0
                per_tier_correct[tier][i] = 0
                per_tier_answer[tier][i] = ""
            else:
                per_tier_answered[tier][i] = 1
                per_tier_answer[tier][i] = ans
                per_tier_correct[tier][i] = int(ans.strip() == target)

        if (i + 1) % 25 == 0:
            elapsed = time.time() - t_start
            rate = (i + 1) / max(elapsed, 1e-6)
            eta = (n - i - 1) / max(rate, 1e-6)
            print(
                f"  {i+1}/{n} ({elapsed:.0f}s, eta={eta:.0f}s)",
                flush=True,
            )

    # Save into cache dict
    for L in DEFAULT_LAYERS:
        cache[f"{task}__hidden_l{L}"] = hiddens[L]
    cache[f"{task}__targets"] = np.array(targets, dtype=object)
    cache[f"{task}__test_indices"] = np.array(test_indices, dtype=np.int32)
    for tier in tiers:
        cache[f"{task}__{tier}__answered"] = per_tier_answered[tier]
        cache[f"{task}__{tier}__correct"] = per_tier_correct[tier]
        cache[f"{task}__{tier}__answer"] = np.array(
            per_tier_answer[tier], dtype=object)

    # Free solver resources (model is shared across tasks, don't touch it)
    del solver


def _summarize_task_phase0(task: str, tiers: list[str], cache) -> dict:
    n = len(cache[f"{task}__targets"])
    answered = {t: cache[f"{task}__{t}__answered"] for t in tiers}
    correct = {t: cache[f"{task}__{t}__correct"] for t in tiers}

    seq_correct = 0
    oracle_correct = 0
    for i in range(n):
        # Sequential: first answered wins
        chosen = None
        for tier in TASK_TIERS[task]:
            if answered[tier][i]:
                chosen = tier
                break
        if chosen is not None and correct[chosen][i]:
            seq_correct += 1
        # Oracle: any tier correct
        if any(correct[t][i] for t in tiers):
            oracle_correct += 1

    seq_acc = seq_correct / n
    oracle_acc = oracle_correct / n
    gap = oracle_acc - seq_acc

    per_tier_overall = {t: float(correct[t].mean()) for t in tiers}
    per_tier_answered = {t: float(answered[t].mean()) for t in tiers}
    per_tier_when_answered = {}
    for t in tiers:
        k = int(answered[t].sum())
        per_tier_when_answered[t] = (
            float(correct[t].sum() / k) if k > 0 else 0.0
        )

    print(f"  sequential={seq_acc:.3f}  oracle={oracle_acc:.3f}  gap={gap:+.3f}")
    for t in tiers:
        print(
            f"    {t:16s}  answered={per_tier_answered[t]:.1%}  "
            f"when_ans={per_tier_when_answered[t]:.1%}  "
            f"overall={per_tier_overall[t]:.1%}"
        )

    return {
        "task": task,
        "n": n,
        "sequential": seq_acc,
        "oracle": oracle_acc,
        "gap": gap,
        "tiers": tiers,
        "per_tier_overall": per_tier_overall,
        "per_tier_answered": per_tier_answered,
        "per_tier_when_answered": per_tier_when_answered,
    }


def _write_phase0_summary(rows: list[dict]):
    path = EXPERIMENT_DIR / "capability_probe_phase0.md"
    lines = [
        "# Phase 0 — Feasibility Gate",
        "",
        "Model: `HuggingFaceTB/SmolLM2-1.7B-Instruct`, no-regex mode.",
        "",
        "## Sequential vs Oracle",
        "",
        "| task | n | sequential | oracle | gap |",
        "|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| `{r['task']}` | {r['n']} | {r['sequential']:.3f} | "
            f"{r['oracle']:.3f} | {r['gap']:+.3f} |"
        )
    lines.extend(["", "## Per-tier breakdown", ""])
    for r in rows:
        lines.append(f"### `{r['task']}`")
        lines.append("")
        lines.append("| tier | answered | when answered | overall |")
        lines.append("|---|---|---|---|")
        for t in r["tiers"]:
            lines.append(
                f"| {t} | {r['per_tier_answered'][t]:.1%} | "
                f"{r['per_tier_when_answered'][t]:.1%} | "
                f"{r['per_tier_overall'][t]:.1%} |"
            )
        lines.append("")

    max_gap = max((r["gap"] for r in rows), default=0)
    gate = "PASSED" if max_gap >= PHASE0_GAP_THRESHOLD else "FAILED"
    lines.extend([
        "## Gate",
        "",
        f"Max gap: **{max_gap:+.3f}**. "
        f"Threshold {PHASE0_GAP_THRESHOLD:+.2f}. **{gate}.**",
        "",
    ])
    path.write_text("\n".join(lines))
    print(f"Wrote {path}")


# ──────────────────────────────────────────────────────────────────────────
# Phase 1 — probe training at L18
# ──────────────────────────────────────────────────────────────────────────

def phase1(args):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import brier_score_loss, roc_auc_score
    from sklearn.model_selection import cross_val_predict
    from sklearn.preprocessing import StandardScaler

    cache = np.load(CACHE_PATH, allow_pickle=True)

    rows: list[dict] = []
    probe_data: dict[str, np.ndarray] = {}

    for task, tiers in TASK_TIERS.items():
        if f"{task}__hidden_l{L_PRIMARY}" not in cache.files:
            print(f"  skip {task}: no cached hiddens")
            continue

        print(f"\n=== {task} ===")
        X = cache[f"{task}__hidden_l{L_PRIMARY}"]

        for tier in tiers:
            y_key = f"{task}__{tier}__correct"
            if y_key not in cache.files:
                continue
            y = cache[y_key].astype(int)
            p_pos = float(y.mean())
            baseline_acc = max(p_pos, 1.0 - p_pos)

            if y.sum() == 0 or y.sum() == len(y):
                print(
                    f"  {tier:16s}  p_succ={p_pos:.2f}  trivial (all "
                    f"{'1' if y[0] == 1 else '0'})"
                )
                rows.append({
                    "task": task, "tier": tier, "p_succ": p_pos,
                    "acc": baseline_acc, "baseline": baseline_acc,
                    "gain": 0.0, "auc": float("nan"), "brier": 0.0,
                    "trivial": True,
                })
                # Store constant probability for router
                probe_data[f"{task}__{tier}__cv_proba"] = np.full(
                    len(y), p_pos, dtype=np.float32)
                continue

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            clf = LogisticRegression(
                max_iter=1000, solver="lbfgs", C=1.0,
                class_weight="balanced", random_state=0,
            )
            try:
                cv_proba = cross_val_predict(
                    clf, X_scaled, y, cv=5, method="predict_proba")[:, 1]
            except Exception as e:
                print(f"  {tier}: CV failed ({e})")
                continue
            cv_pred = (cv_proba >= 0.5).astype(int)
            acc = float((cv_pred == y).mean())
            auc = float(roc_auc_score(y, cv_proba)) if len(set(y)) > 1 else float("nan")
            brier = float(brier_score_loss(y, cv_proba))
            gain = acc - baseline_acc
            print(
                f"  {tier:16s}  p_succ={p_pos:.2f}  acc={acc:.3f} "
                f"(base={baseline_acc:.3f}, gain={gain:+.3f})  "
                f"auc={auc:.3f}  brier={brier:.3f}"
            )
            rows.append({
                "task": task, "tier": tier, "p_succ": p_pos,
                "acc": acc, "baseline": baseline_acc, "gain": gain,
                "auc": auc, "brier": brier, "trivial": False,
            })

            # Fit final (full-data) probe too, for reusable weights
            clf_full = LogisticRegression(
                max_iter=1000, solver="lbfgs", C=1.0,
                class_weight="balanced", random_state=0,
            )
            clf_full.fit(X_scaled, y)
            probe_data[f"{task}__{tier}__cv_proba"] = cv_proba.astype(np.float32)
            probe_data[f"{task}__{tier}__coef"] = clf_full.coef_[0].astype(np.float32)
            probe_data[f"{task}__{tier}__intercept"] = np.asarray(
                clf_full.intercept_, dtype=np.float32)
            probe_data[f"{task}__{tier}__scaler_mean"] = scaler.mean_.astype(np.float32)
            probe_data[f"{task}__{tier}__scaler_scale"] = scaler.scale_.astype(np.float32)

    np.savez(PROBE_PATH, **probe_data)
    print(f"\nSaved probe data → {PROBE_PATH}")

    _write_phase1_summary(rows)

    informative = [r for r in rows if not r.get("trivial")]
    max_gain = max((r["gain"] for r in informative), default=0.0)
    print(f"\nMax probe gain over majority baseline: {max_gain:+.3f}")
    if max_gain >= PHASE1_GAIN_THRESHOLD:
        print(f"GATE PASSED → probe routing is informative")
    else:
        print(
            f"GATE FAILED (max gain < {PHASE1_GAIN_THRESHOLD:+.2f}) → "
            "L18 does not carry routing signal"
        )


def _write_phase1_summary(rows: list[dict]):
    path = EXPERIMENT_DIR / "capability_probe_phase1.md"
    lines = [
        "# Phase 1 — Probe Training at L18",
        "",
        "5-fold CV, LogReg with `class_weight='balanced'`.",
        "",
        "| task | tier | p(succ) | acc | baseline | gain | AUC | Brier |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| `{r['task']}` | {r['tier']} | {r['p_succ']:.2f} | "
            f"{r['acc']:.3f} | {r['baseline']:.3f} | {r['gain']:+.3f} | "
            f"{r['auc']:.3f} | {r['brier']:.3f} |"
        )
    path.write_text("\n".join(lines))
    print(f"Wrote {path}")


# ──────────────────────────────────────────────────────────────────────────
# Phase 2 — router evaluation
# ──────────────────────────────────────────────────────────────────────────

def phase2(args):
    cache = np.load(CACHE_PATH, allow_pickle=True)
    if not PROBE_PATH.exists():
        print(f"Missing {PROBE_PATH}. Run --phase 1 first.")
        return
    probes = np.load(PROBE_PATH, allow_pickle=True)

    rows: list[dict] = []
    for task, tiers in TASK_TIERS.items():
        if f"{task}__targets" not in cache.files:
            continue
        print(f"\n=== {task} ===")

        n = len(cache[f"{task}__targets"])
        answered = np.stack([cache[f"{task}__{t}__answered"] for t in tiers])
        correct = np.stack([cache[f"{task}__{t}__correct"] for t in tiers])

        # Out-of-fold probabilities per tier
        proba = []
        for tier in tiers:
            key = f"{task}__{tier}__cv_proba"
            if key in probes.files:
                proba.append(probes[key])
            else:
                # Fallback: use empirical success rate
                p = float(correct[tiers.index(tier)].mean())
                proba.append(np.full(n, p, dtype=np.float32))
        proba_mat = np.stack(proba)  # (T, N)

        seq_correct = 0
        routed_correct = 0
        oracle_correct = 0
        n_tries_routed = 0

        for i in range(n):
            # Sequential
            for tier in TASK_TIERS[task]:
                tidx = tiers.index(tier)
                if answered[tidx, i]:
                    if correct[tidx, i]:
                        seq_correct += 1
                    break

            # Probe-routed: sort tiers by predicted success descending
            order = np.argsort(-proba_mat[:, i])
            for k, tidx in enumerate(order):
                n_tries_routed += 1
                if answered[tidx, i]:
                    if correct[tidx, i]:
                        routed_correct += 1
                    break
            else:
                # No tier answered — shouldn't happen if "baseline" is present
                pass

            # Oracle
            if correct[:, i].max() > 0:
                oracle_correct += 1

        seq_acc = seq_correct / n
        routed_acc = routed_correct / n
        oracle_acc = oracle_correct / n
        gap = oracle_acc - seq_acc
        closure = (routed_acc - seq_acc) / gap if gap > 1e-9 else 0.0
        avg_tries = n_tries_routed / n

        print(
            f"  sequential={seq_acc:.3f}  routed={routed_acc:.3f}  "
            f"oracle={oracle_acc:.3f}  "
            f"gap_closure={closure:+.1%}  avg_tries={avg_tries:.2f}"
        )
        rows.append({
            "task": task, "n": n,
            "sequential": seq_acc, "routed": routed_acc, "oracle": oracle_acc,
            "gap": gap, "closure": closure, "avg_tries": avg_tries,
        })

    _write_phase2_summary(rows)


def _write_phase2_summary(rows: list[dict]):
    path = EXPERIMENT_DIR / "capability_probe_phase2.md"
    lines = [
        "# Phase 2 — Router Evaluation",
        "",
        "Sequential (production order) vs probe-routed vs oracle. "
        "Router uses out-of-fold CV probabilities.",
        "",
        "| task | n | sequential | routed | oracle | gap closed | avg tries |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| `{r['task']}` | {r['n']} | {r['sequential']:.3f} | "
            f"{r['routed']:.3f} | {r['oracle']:.3f} | "
            f"{r['closure']:+.1%} | {r['avg_tries']:.2f} |"
        )
    path.write_text("\n".join(lines))
    print(f"Wrote {path}")


# ──────────────────────────────────────────────────────────────────────────
# Phase 3 — layer ablation
# ──────────────────────────────────────────────────────────────────────────

def phase3(args):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_predict
    from sklearn.preprocessing import StandardScaler

    cache = np.load(CACHE_PATH, allow_pickle=True)

    rows: list[dict] = []
    for task, tiers in TASK_TIERS.items():
        if f"{task}__hidden_l{L_PRIMARY}" not in cache.files:
            continue
        print(f"\n=== {task} ===")
        for tier in tiers:
            y_key = f"{task}__{tier}__correct"
            if y_key not in cache.files:
                continue
            y = cache[y_key].astype(int)
            if y.sum() == 0 or y.sum() == len(y):
                continue

            row = {"task": task, "tier": tier}
            for L in DEFAULT_LAYERS:
                X = cache[f"{task}__hidden_l{L}"]
                scaler = StandardScaler()
                Xs = scaler.fit_transform(X)
                clf = LogisticRegression(
                    max_iter=1000, solver="lbfgs", C=1.0,
                    class_weight="balanced", random_state=0,
                )
                try:
                    proba = cross_val_predict(
                        clf, Xs, y, cv=5, method="predict_proba")[:, 1]
                except Exception:
                    row[f"L{L}"] = float("nan")
                    continue
                pred = (proba >= 0.5).astype(int)
                row[f"L{L}"] = float((pred == y).mean())
            layer_scores = [(L, row[f"L{L}"]) for L in DEFAULT_LAYERS]
            best_layer = max(
                layer_scores,
                key=lambda x: (x[1] if not np.isnan(x[1]) else -1),
            )[0]
            row["best"] = best_layer
            scores_str = "  ".join(
                f"L{L}={row[f'L{L}']:.3f}" for L in DEFAULT_LAYERS
            )
            print(f"  {tier:16s}  {scores_str}  best=L{best_layer}")
            rows.append(row)

    _write_phase3_summary(rows)


def _write_phase3_summary(rows: list[dict]):
    path = EXPERIMENT_DIR / "capability_probe_phase3.md"
    header = "| task | tier | " + " | ".join(f"L{L}" for L in DEFAULT_LAYERS) + " | best |"
    sep = "|---|---|" + "|".join("---" for _ in DEFAULT_LAYERS) + "|---|"
    lines = [
        "# Phase 3 — Layer Ablation",
        "",
        "5-fold CV LogReg probes trained on last-token hidden states at each layer.",
        "",
        header,
        sep,
    ]
    for r in rows:
        cells = " | ".join(f"{r[f'L{L}']:.3f}" for L in DEFAULT_LAYERS)
        lines.append(
            f"| `{r['task']}` | {r['tier']} | {cells} | L{r['best']} |"
        )
    path.write_text("\n".join(lines))
    print(f"Wrote {path}")


# ──────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", type=int, choices=[0, 1, 2, 3], required=True)
    parser.add_argument(
        "--tasks", type=str, default=None,
        help="Comma-separated task subset (Phase 0 only).",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Recompute cached tasks in Phase 0.",
    )
    args = parser.parse_args()

    try:
        if args.phase == 0:
            phase0(args)
        elif args.phase == 1:
            phase1(args)
        elif args.phase == 2:
            phase2(args)
        elif args.phase == 3:
            phase3(args)
    except KeyboardInterrupt:
        print("\nInterrupted.")
    except Exception:
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
