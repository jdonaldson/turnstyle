"""How many DISTINCT option-gathering patterns does the model have?

The selection heads aren't one mechanism. Characterize every head that attends
from the answer position onto the option spans by *what* it gathers, then
cluster the signatures. Number of clusters = number of distinct gathering
patterns.

Per-head signature (averaged over prompts, answer-position attention):
  winner_share  : frac of option-attn on the eventually-chosen option (selectivity)
  first_share   : frac on the first option        (positional, early)
  last_share    : frac on the last option         (positional, late / recency)
  uniformity    : normalized perplexity over options (1 = attends all equally)
  marker_share  : frac on the option MARKER token  ("(A)") vs content
  lasttok_share : frac on the option's LAST token  (where per-option scores live)
"""
from __future__ import annotations

import json
import re
from collections import defaultdict

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

MODEL_ID = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
BBH_CACHE = "/Users/jdonaldson/Projects/swollm/data/bbh_cache"
TASKS = ["snarks", "date_understanding", "logical_deduction_three_objects",
         "movie_recommendation", "ruin_names"]
N = 15
OPTION_RE = re.compile(r"^\(([A-Z])\)", re.MULTILINE)
MASS_THRESH = 0.45          # min mean option-span attention to count as a gatherer
FEATS = ["winner_share", "first_share", "last_share", "uniformity",
         "marker_share", "lasttok_share"]


def load_task(name):
    with open(f"{BBH_CACHE}/{name}.json") as f:
        return json.load(f)


def option_spans(text):
    marks = list(OPTION_RE.finditer(text))
    return [(m.start(),
             marks[k + 1].start() if k + 1 < len(marks) else len(text),
             m.group(1)) for k, m in enumerate(marks)]


def span_tokens(offsets, s, e):
    return [ti for ti, (a, b) in enumerate(offsets) if a != b and s <= (a + b) / 2 < e]


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    mdl = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.float16, attn_implementation="eager").to(device).eval()
    print(f"model loaded on {device}\n", flush=True)
    n_layers, n_heads = mdl.config.num_hidden_layers, mdl.config.num_attention_heads
    norm, head_w = mdl.model.norm, mdl.lm_head
    letter_ids = {c: tok.encode(c, add_special_tokens=False)[0] for c in "ABCDEFGH"}

    # (layer, head) -> list of per-prompt feature dicts + mass
    acc = defaultdict(lambda: defaultdict(list))

    for task in TASKS:
        for ex in load_task(task)[:N]:
            text = ex["input"]
            spans = option_spans(text)
            if len(spans) < 2:
                continue
            present = [s[2] for s in spans]
            enc = tok(text + "\nAnswer: (", return_offsets_mapping=True, return_tensors="pt")
            offsets = enc["offset_mapping"][0].tolist()
            opt_tokens = [span_tokens(offsets, s, e) for (s, e, _) in spans]
            if any(len(t) == 0 for t in opt_tokens):
                continue
            ids = enc["input_ids"].to(device)
            ans = ids.shape[1] - 1
            with torch.no_grad():
                out = mdl(input_ids=ids, output_hidden_states=True, output_attentions=True)
                logits = head_w(norm(out.hidden_states[-1][0, ans]))
            cand = torch.tensor([letter_ids[c] for c in present], device=device)
            winner = int(torch.argmax(logits[cand]).item())

            for L in range(1, n_layers):       # skip L0 (BOS-sink artifact)
                A = out.attentions[L][0, :, ans, :].float().cpu().numpy()  # [heads, seq]
                for h in range(n_heads):
                    row = A[h]
                    per_opt = np.array([row[t].sum() for t in opt_tokens])
                    total = per_opt.sum()
                    if total < 1e-4:
                        continue
                    p = per_opt / total
                    marker = sum(row[t[0]] for t in opt_tokens) / total
                    lasttok = sum(row[t[-1]] for t in opt_tokens) / total
                    perpl = np.exp(-(p * np.log(p + 1e-12)).sum())
                    f = acc[(L, h)]
                    f["mass"].append(float(total))
                    f["winner_share"].append(float(p[winner]))
                    f["first_share"].append(float(p[0]))
                    f["last_share"].append(float(p[-1]))
                    f["uniformity"].append(float(perpl / len(p)))
                    f["marker_share"].append(float(marker))
                    f["lasttok_share"].append(float(lasttok))
        print(f"  scanned {task}", flush=True)

    # select gatherer heads
    heads, sigs, meta = [], [], []
    for (L, h), f in acc.items():
        if np.mean(f["mass"]) >= MASS_THRESH and len(f["mass"]) >= 10:
            heads.append((L, h))
            sigs.append([np.mean(f[k]) for k in FEATS])
            meta.append((L, np.mean(f["mass"])))
    sigs = np.array(sigs)
    allmass = [np.mean(f["mass"]) for f in acc.values() if len(f["mass"]) >= 10]
    for thr in [0.3, 0.45, 0.6, 0.75]:
        print(f"  heads with mean option-mass >= {thr}: {sum(m >= thr for m in allmass)}")
    print(f"\n{len(heads)} gatherer heads (mean option-mass >= {MASS_THRESH})\n")

    if len(heads) < 4:
        print("too few to cluster"); return

    X = StandardScaler().fit_transform(sigs)
    best = None
    for k in range(2, 7):
        km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(X)
        s = silhouette_score(X, km.labels_)
        print(f"  k={k}: silhouette={s:.3f}")
        if best is None or s > best[0]:
            best = (s, k, km)
    _, k, km = best
    print(f"\n=> {k} distinct gathering patterns (best silhouette)\n")

    for c in range(k):
        idx = np.where(km.labels_ == c)[0]
        cent = sigs[idx].mean(axis=0)
        layers = [meta[i][0] for i in idx]
        desc = ", ".join(f"{FEATS[j].split('_')[0]}={cent[j]:.2f}" for j in range(len(FEATS)))
        # name the dominant trait
        name = name_pattern(cent)
        print(f"  Pattern {c} [{name}]  n={len(idx)}  layers {min(layers)}-{max(layers)} "
              f"(median {int(np.median(layers))})")
        print(f"     {desc}")
        ex_heads = sorted([(meta[i][0], heads[i][1]) for i in idx])[:6]
        print(f"     heads: {', '.join(f'L{L}H{h}' for L, h in ex_heads)}")


def name_pattern(c):
    winner, first, last, unif, marker, lasttok = c
    if unif > 0.85:
        return "uniform broad gather"
    if winner > 0.55:
        return "winner-selective"
    if last > 0.55:
        return "last-option (recency)"
    if first > 0.55:
        return "first-option (primacy)"
    if marker > 0.5:
        return "marker-locked"
    if lasttok > 0.5:
        return "score-token (last-of-option)"
    return "mixed"


if __name__ == "__main__":
    main()
