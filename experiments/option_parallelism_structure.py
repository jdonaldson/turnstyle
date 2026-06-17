"""Are MC options *parallel templates* — corresponding positions across options
aligned in activation space?

Follow-up to option_region_coherence.py, which falsified within-option
coherence (within <= across at matched distance) and pointed here: options
look like parallel repeats of a shared template, so the structure is
CROSS-option positional alignment, not within-option cohesion.

Test: among cross-option token pairs, are pairs at the SAME within-option
position ("aligned") more similar than pairs at DIFFERENT positions
("misaligned"), holding token distance |i-j| fixed?

Collinearity trap: for two equal-length options, aligned pairs all sit at one
gap G == distance, so within a single prompt aligned-ness and distance are
confounded. Fix: pool across many prompts with varying option lengths, so each
distance bin contains aligned pairs (prompts where G~=d) AND misaligned pairs
(prompts with other gaps). Then compare within bin.

Lexical control: split aligned pairs into same-token-id vs different-token-id.
If aligned-DIFFERENT-token pairs still beat the misaligned baseline, the model
encodes parallel position beyond mere repeated wording.
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
TASKS = {"snarks": 15, "ruin_names": 15, "logical_deduction_three_objects": 10}
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
    """Per token: option id (-1 if none) and within-option index (-1 if none)."""
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
        idxs = np.where(opt_id == oid)[0]
        for k, ti in enumerate(idxs):
            within[ti] = k
    return enc, opt_id, within


def cosine_matrix(H):
    Hn = H / (np.linalg.norm(H, axis=1, keepdims=True) + 1e-8)
    return Hn @ Hn.T


def main():
    tok, mdl, device = load_model()
    print(f"model loaded on {device}\n", flush=True)

    # layer -> distance -> category -> list[cos]
    # categories: 'within', 'cross_aligned', 'cross_misaligned'
    acc = {L: defaultdict(lambda: defaultdict(list)) for L in LAYERS}
    # lexical split for aligned cross pairs (distance-pooled): same vs diff token id
    lex = {L: {"aligned_same": [], "aligned_diff": [], "misaligned": []} for L in LAYERS}

    for task, n in TASKS.items():
        for i, ex in enumerate(load_task(task)[:n]):
            text = ex["input"]
            enc, opt_id, within = token_layout(text, tok)
            ids = enc["input_ids"][0].tolist()
            opt_tok = np.where(opt_id >= 0)[0]
            if len(set(opt_id[opt_id >= 0])) < 2:
                continue
            with torch.no_grad():
                out = mdl(input_ids=enc["input_ids"].to(device),
                          attention_mask=enc["attention_mask"].to(device),
                          output_hidden_states=True)
            for L in LAYERS:
                S = cosine_matrix(out.hidden_states[L][0].float().cpu().numpy())
                for x in range(len(opt_tok)):
                    for y in range(x + 1, len(opt_tok)):
                        ti, tj = int(opt_tok[x]), int(opt_tok[y])
                        d = tj - ti
                        c = float(S[ti, tj])
                        same_opt = opt_id[ti] == opt_id[tj]
                        if same_opt:
                            acc[L][d]["within"].append(c)
                        else:
                            aligned = within[ti] == within[tj]
                            cat = "cross_aligned" if aligned else "cross_misaligned"
                            acc[L][d][cat].append(c)
                            if aligned:
                                key = "aligned_same" if ids[ti] == ids[tj] else "aligned_diff"
                                lex[L][key].append(c)
                            else:
                                lex[L]["misaligned"].append(c)
            print(f"  [{task} {i}] opt_tokens={len(opt_tok)}", flush=True)

    def controlled_delta(L, cat_a, cat_b):
        num, den = 0.0, 0
        bins = 0
        for d, dd in acc[L].items():
            if dd[cat_a] and dd[cat_b]:
                w = min(len(dd[cat_a]), len(dd[cat_b]))
                num += (np.mean(dd[cat_a]) - np.mean(dd[cat_b])) * w
                den += w
                bins += 1
        return (num / den if den else float("nan")), bins

    print("\n=== Parallelism test (controlled for token distance |i-j|) ===")
    print("Δ1 = cross_aligned - cross_misaligned   (positional alignment effect)")
    print("Δ2 = cross_aligned - within             (alignment vs within-option cohesion)\n")
    for L in LAYERS:
        d1, b1 = controlled_delta(L, "cross_aligned", "cross_misaligned")
        d2, b2 = controlled_delta(L, "cross_aligned", "within")
        print(f"  L{L}:  Δ1 = {d1:+.4f} ({b1} bins)   Δ2 = {d2:+.4f} ({b2} bins)")

    print("\n=== Lexical control (distance-pooled aligned cross pairs) ===")
    print("If aligned_diff > misaligned, parallelism is beyond repeated wording.\n")
    for L in LAYERS:
        a_s = np.mean(lex[L]["aligned_same"]) if lex[L]["aligned_same"] else float("nan")
        a_d = np.mean(lex[L]["aligned_diff"]) if lex[L]["aligned_diff"] else float("nan")
        mis = np.mean(lex[L]["misaligned"]) if lex[L]["misaligned"] else float("nan")
        print(f"  L{L}:  aligned_same={a_s:+.3f} (n={len(lex[L]['aligned_same'])})  "
              f"aligned_diff={a_d:+.3f} (n={len(lex[L]['aligned_diff'])})  "
              f"misaligned={mis:+.3f} (n={len(lex[L]['misaligned'])})  "
              f"|  diff-vs-mis = {a_d - mis:+.3f}")


if __name__ == "__main__":
    main()
