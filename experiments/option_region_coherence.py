"""Do options live as *coherent activation regions*, beyond mere locality?

Hypothesis: the tokens spanning one multiple-choice option share an activation
signature that distinguishes them as a unit from the next option — so options
could be detected from hidden states alone (no regex marker, format-free).

The confound is LOCALITY: adjacent tokens are similar just from position, so a
raw token×token similarity matrix looks blocky for any text. We control for it
by binning every token pair by separation |i-j| and, within each bin, comparing
within-option pairs vs across-option pairs. Δ = sim(within) - sim(across) at
matched distance. Δ > 0 across bins = genuine option-coherence beyond locality.

Option ids are parsed from "(X)" markers and used ONLY as ground truth to score
the structure — never as input to it.

Outputs:
  - per-layer locality-controlled Δ (the headline number)
  - data/option_region_coherence/simmatrix_<task>_<i>.png  (one heatmap to eyeball)
"""
from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/Users/jdonaldson/Projects/turnstyle/experiments")
from common import load_model  # noqa: E402

BBH_CACHE = "/Users/jdonaldson/Projects/swollm/data/bbh_cache"
LAYERS = [1, 4, 8]
TASKS = {"snarks": 10, "ruin_names": 10}     # long 2-option + short multi-option
OPTION_RE = re.compile(r"^\(([A-Z])\)", re.MULTILINE)
OUT_DIR = Path("/Users/jdonaldson/Projects/turnstyle/experiments/data/option_region_coherence")


def load_task(name: str) -> list[dict]:
    with open(f"{BBH_CACHE}/{name}.json") as f:
        return json.load(f)


def option_spans(text: str) -> list[tuple[int, int, int]]:
    """Return [(char_start, char_end, option_id), ...] for each (X) option line."""
    marks = list(OPTION_RE.finditer(text))
    spans = []
    for k, m in enumerate(marks):
        start = m.start()
        end = marks[k + 1].start() if k + 1 < len(marks) else len(text)
        spans.append((start, end, k))
    return spans


def token_option_ids(text, tokenizer):
    """Map each token to its option id (or -1 for body/non-option tokens)."""
    enc = tokenizer(text, return_offsets_mapping=True, return_tensors="pt")
    offsets = enc["offset_mapping"][0].tolist()
    spans = option_spans(text)
    ids = np.full(len(offsets), -1, dtype=int)
    for ti, (a, b) in enumerate(offsets):
        if a == b:                       # special / empty token
            continue
        mid = (a + b) / 2
        for s, e, oid in spans:
            if s <= mid < e:
                ids[ti] = oid
                break
    return enc, ids


def hidden_states(enc, model, device):
    with torch.no_grad():
        out = model(input_ids=enc["input_ids"].to(device),
                    attention_mask=enc["attention_mask"].to(device),
                    output_hidden_states=True)
    return out.hidden_states            # tuple len n_layers+1, each [1, T, D]


def cosine_matrix(H):
    """H: [T, D] -> [T, T] cosine similarity."""
    Hn = H / (np.linalg.norm(H, axis=1, keepdims=True) + 1e-8)
    return Hn @ Hn.T


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tok, mdl, device = load_model()
    print(f"model loaded on {device}\n", flush=True)

    # accumulator: layer -> distance_bin -> {'within': [cos...], 'across': [cos...]}
    acc = {L: defaultdict(lambda: {"within": [], "across": []}) for L in LAYERS}
    saved_heatmap = False

    for task, n in TASKS.items():
        examples = load_task(task)[:n]
        for i, ex in enumerate(examples):
            text = ex["input"]
            enc, opt_ids = token_option_ids(text, tok)
            n_opts = len(set(opt_ids[opt_ids >= 0]))
            if n_opts < 2:
                print(f"  [{task} {i}] <2 options parsed, skip", flush=True)
                continue
            hs = hidden_states(enc, mdl, device)

            opt_tok = np.where(opt_ids >= 0)[0]      # only option-region tokens
            for L in LAYERS:
                H = hs[L][0].float().cpu().numpy()
                S = cosine_matrix(H)
                for a_idx in range(len(opt_tok)):
                    for b_idx in range(a_idx + 1, len(opt_tok)):
                        ti, tj = opt_tok[a_idx], opt_tok[b_idx]
                        d = int(tj - ti)
                        same = opt_ids[ti] == opt_ids[tj]
                        acc[L][d]["within" if same else "across"].append(S[ti, tj])

            # save one heatmap (first prompt with >=3 options for visible blocks)
            if not saved_heatmap and n_opts >= 3:
                try:
                    save_heatmap(hs, opt_tok, opt_ids, L=4, task=task, idx=i)
                except ImportError:
                    print("  (matplotlib missing — skipping heatmap)", flush=True)
                saved_heatmap = True
            print(f"  [{task} {i}] T={enc['input_ids'].shape[1]} "
                  f"opts={n_opts} opt_tokens={len(opt_tok)}", flush=True)

    print("\n=== Locality-controlled coherence (Δ = within - across, matched |i-j|) ===")
    for L in LAYERS:
        # only distance bins that contain BOTH within and across pairs are comparable
        wdiffs, weights = [], []
        for d, dd in sorted(acc[L].items()):
            if dd["within"] and dd["across"]:
                w, a = np.mean(dd["within"]), np.mean(dd["across"])
                n = min(len(dd["within"]), len(dd["across"]))
                wdiffs.append((w - a) * n)
                weights.append(n)
        if weights:
            delta = sum(wdiffs) / sum(weights)
            # also raw (uncontrolled) for contrast
            allw = [c for d in acc[L] for c in acc[L][d]["within"]]
            alla = [c for d in acc[L] for c in acc[L][d]["across"]]
            raw = np.mean(allw) - np.mean(alla)
            print(f"  L{L}:  controlled Δ = {delta:+.4f}   "
                  f"(raw uncontrolled Δ = {raw:+.4f}, "
                  f"comparable bins = {len(weights)})")
        else:
            print(f"  L{L}:  no comparable distance bins")


def save_heatmap(hs, opt_tok, opt_ids, L, task, idx):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    H = hs[L][0].float().cpu().numpy()[opt_tok]
    S = cosine_matrix(H)
    ids = opt_ids[opt_tok]
    bounds = [k for k in range(1, len(ids)) if ids[k] != ids[k - 1]]

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(S, cmap="viridis", aspect="equal")
    for b in bounds:
        ax.axhline(b - 0.5, color="red", lw=0.8)
        ax.axvline(b - 0.5, color="red", lw=0.8)
    ax.set_title(f"{task} ex{idx} — L{L} token×token cosine\n(red = option boundaries)")
    fig.colorbar(im, ax=ax, fraction=0.046)
    out = OUT_DIR / f"simmatrix_{task}_{idx}.png"
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"  saved heatmap -> {out}", flush=True)


if __name__ == "__main__":
    main()
