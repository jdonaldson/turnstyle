"""Separate the two confounds the pooled parallelism result couldn't:

Q1 "does it just detect similar sentences?" -> per-task breakdown across a
   similarity gradient:
     snarks / ruin_names         = near-duplicate options (1 word / typo)
     date_understanding          = rigid template, different digits
     logical_deduction_three     = different orderings
     movie_recommendation        = dissimilar titles, varied length
   If the beyond-lexical residual (aligned_diff - misaligned) survives on
   movie_recommendation, it's structural; if it collapses, it's similarity.

Q2 "do options have to be at the end?" -> causal-mask prediction: an option
   token's hidden state depends only on PRECEDING tokens, so appending a suffix
   after the options must leave their cosines byte-identical. Verify it.
"""
from __future__ import annotations

import json
import re
import sys
from collections import defaultdict

import numpy as np
import torch

sys.path.insert(0, "/Users/jdonaldson/Projects/turnstyle/experiments")
from common import load_model  # noqa: E402

BBH_CACHE = "/Users/jdonaldson/Projects/swollm/data/bbh_cache"
LAYERS = [1, 4, 8]
TASKS = ["snarks", "ruin_names", "date_understanding",
         "logical_deduction_three_objects", "movie_recommendation"]
N = 15
OPTION_RE = re.compile(r"^\(([A-Z])\)", re.MULTILINE)


def load_task(name):
    with open(f"{BBH_CACHE}/{name}.json") as f:
        return json.load(f)


def option_spans(text):
    marks = list(OPTION_RE.finditer(text))
    out = []
    for k, m in enumerate(marks):
        end = marks[k + 1].start() if k + 1 < len(marks) else len(text)
        out.append((m.start(), end, k))
    return out


def token_layout(text, tokenizer):
    enc = tokenizer(text, return_offsets_mapping=True, return_tensors="pt")
    offsets = enc["offset_mapping"][0].tolist()
    spans = option_spans(text)
    opt_id = np.full(len(offsets), -1, dtype=int)
    for ti, (a, b) in enumerate(offsets):
        if a == b:
            continue
        mid = (a + b) / 2
        for s, e, oid in spans:
            if s <= mid < e:
                opt_id[ti] = oid
                break
    within = np.full(len(offsets), -1, dtype=int)
    for oid in set(opt_id[opt_id >= 0]):
        for k, ti in enumerate(np.where(opt_id == oid)[0]):
            within[ti] = k
    return enc, opt_id, within


def cosine_matrix(H):
    Hn = H / (np.linalg.norm(H, axis=1, keepdims=True) + 1e-8)
    return Hn @ Hn.T


def hs(enc, mdl, device):
    with torch.no_grad():
        out = mdl(input_ids=enc["input_ids"].to(device),
                  attention_mask=enc["attention_mask"].to(device),
                  output_hidden_states=True)
    return out.hidden_states


def main():
    tok, mdl, device = load_model()
    print(f"model loaded on {device}\n", flush=True)

    # per task: layer -> lists for the lexical-controlled residual
    res = {t: {L: defaultdict(list) for L in LAYERS} for t in TASKS}

    for task in TASKS:
        for ex in load_task(task)[:N]:
            text = ex["input"]
            enc, opt_id, within = token_layout(text, tok)
            ids = enc["input_ids"][0].tolist()
            opt_tok = np.where(opt_id >= 0)[0]
            if len(set(opt_id[opt_id >= 0])) < 2:
                continue
            H = hs(enc, mdl, device)
            for L in LAYERS:
                S = cosine_matrix(H[L][0].float().cpu().numpy())
                for x in range(len(opt_tok)):
                    for y in range(x + 1, len(opt_tok)):
                        ti, tj = int(opt_tok[x]), int(opt_tok[y])
                        if opt_id[ti] == opt_id[tj]:
                            continue
                        c = float(S[ti, tj])
                        if within[ti] == within[tj]:
                            key = "aligned_same" if ids[ti] == ids[tj] else "aligned_diff"
                            res[task][L][key].append(c)
                        else:
                            res[task][L]["misaligned"].append(c)
        # report this task
        print(f"\n=== {task} ===")
        for L in LAYERS:
            r = res[task][L]
            a_d = np.mean(r["aligned_diff"]) if r["aligned_diff"] else float("nan")
            mis = np.mean(r["misaligned"]) if r["misaligned"] else float("nan")
            frac_same = len(r["aligned_same"]) / max(
                1, len(r["aligned_same"]) + len(r["aligned_diff"]))
            print(f"  L{L}:  aligned_diff={a_d:+.3f}  misaligned={mis:+.3f}  "
                  f"beyond-lexical Δ={a_d - mis:+.3f}   "
                  f"(template repeat frac={frac_same:.2f}, "
                  f"n_diff={len(r['aligned_diff'])})")

    # --- Q2: suffix invariance (causal mask => option cosines unchanged) ---
    print("\n=== Q2: suffix-invariance (option-token states vs trailing text) ===")
    suffix = "\n\nLet's think step by step before answering."
    maxdiff = 0.0
    for ex in load_task("snarks")[:5]:
        text = ex["input"]
        enc, opt_id, _ = token_layout(text, tok)
        enc2, opt_id2, _ = token_layout(text + suffix, tok)
        opt_tok = np.where(opt_id >= 0)[0]
        H1 = hs(enc, mdl, device)[4][0].float().cpu().numpy()[opt_tok]
        H2 = hs(enc2, mdl, device)[4][0].float().cpu().numpy()[opt_tok]
        maxdiff = max(maxdiff, float(np.max(np.abs(H1 - H2))))
    print(f"  max |Δ| in L4 option-token hidden states after appending suffix: "
          f"{maxdiff:.2e}")
    print("  (≈0 confirms: option representations ignore everything after them;"
          " 'at the end' is irrelevant.)")


if __name__ == "__main__":
    main()
