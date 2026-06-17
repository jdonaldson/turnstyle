"""When does the argmax selection FIRE? Per-example decision layer.

At an "Answer: (" position, logit-lens the candidate option letters at every
layer. The DECISION LAYER for an example is the earliest layer L such that the
candidate argmax at every layer >= L equals the final-layer pick — i.e. the
choice locks in and never changes again. Distribution of decision layers tells
us whether selection is a sharp late event or gradual, and whether wrong picks
commit as late as right ones.

Companion to option_list_then_select.py (which showed selection is a late,
final-position event; this localizes WHEN per example).
"""
from __future__ import annotations

import json
import re
import sys

import numpy as np
import torch

sys.path.insert(0, "/Users/jdonaldson/Projects/turnstyle/experiments")
from common import load_model  # noqa: E402

BBH_CACHE = "/Users/jdonaldson/Projects/swollm/data/bbh_cache"
TASKS = ["snarks", "date_understanding", "logical_deduction_three_objects",
         "movie_recommendation"]
N = 15
OPTION_RE = re.compile(r"^\(([A-Z])\)", re.MULTILINE)


def load_task(name):
    with open(f"{BBH_CACHE}/{name}.json") as f:
        return json.load(f)


def main():
    tok, mdl, device = load_model()
    print(f"model loaded on {device}\n", flush=True)
    n_layers = mdl.config.num_hidden_layers
    norm, head = mdl.model.norm, mdl.lm_head
    letter_ids = {c: tok.encode(c, add_special_tokens=False)[0] for c in "ABCDEFGH"}

    rows = []  # (task, decision_layer, depth, correct, stable)
    for task in TASKS:
        for ex in load_task(task)[:N]:
            text = ex["input"]
            present = OPTION_RE.findall(text)
            if len(present) < 2:
                continue
            g = re.search(r"\(([A-Z])\)", ex["target"])
            if not g:
                continue
            gold = g.group(1)
            cand = torch.tensor([letter_ids[c] for c in present], device=device)
            enc = tok(text + "\nAnswer: (", return_tensors="pt")
            with torch.no_grad():
                hs = mdl(input_ids=enc["input_ids"].to(device),
                         attention_mask=enc["attention_mask"].to(device),
                         output_hidden_states=True).hidden_states
            picks = []
            for l in range(n_layers + 1):
                logits = head(norm(hs[l][0, -1]))
                picks.append(present[int(torch.argmax(logits[cand]).item())])
            final = picks[-1]
            # earliest layer from which picks are constant == final
            dl = n_layers
            for l in range(n_layers, -1, -1):
                if picks[l] == final:
                    dl = l
                else:
                    break
            rows.append((task, dl, dl / n_layers, final == gold))

    arr_dl = np.array([r[1] for r in rows])
    correct = np.array([r[3] for r in rows])
    print(f"n={len(rows)} prompts, {n_layers} layers\n")
    print(f"Decision layer (model locks its pick):  "
          f"median={np.median(arr_dl):.0f}  mean={arr_dl.mean():.1f}  "
          f"depth={np.median(arr_dl)/n_layers:.0%}")
    print(f"  correct picks (n={correct.sum()}):   "
          f"median decision layer = {np.median(arr_dl[correct]):.0f}")
    print(f"  wrong   picks (n={(~correct).sum()}): "
          f"median decision layer = {np.median(arr_dl[~correct]):.0f}")

    print("\nDecision-layer histogram (each ■ = one prompt):")
    edges = list(range(0, n_layers + 2, 2))
    for lo in edges[:-1]:
        hi = lo + 2
        c = int(((arr_dl >= lo) & (arr_dl < hi)).sum())
        print(f"  L{lo:2d}-{hi-1:2d}: {'■' * c} {c}")

    # how late: fraction committing only in the last quarter of the stack
    late = (arr_dl >= 0.75 * n_layers).mean()
    print(f"\nFraction committing in the final quarter (>=L{int(0.75*n_layers)}): {late:.0%}")
    print("(high => selection is a late event; low => committed earlier)")


if __name__ == "__main__":
    main()
