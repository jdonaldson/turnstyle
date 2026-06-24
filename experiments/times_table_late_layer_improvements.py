"""What specifically improves at late layers?

For each (prompt, layer), compute:
  - P(correct first-character)        ← confidence in right answer
  - Top wrong character and its prob   ← strongest competitor
  - Margin = P(correct) − P(top wrong)
  - Rank of the correct character      ← where it sits in the ordering

Track these across layers to distinguish:
  - "Sharpening" (margin grows, but rank already 1) — pure refinement
  - "Computing"  (rank goes from >1 to 1)           — real work
  - "Aligning"   (P grows from tiny to large but rank already low) — unembed alignment

A model that's coasting at late layers shows only sharpening.  A model
doing real work at late layers shows rank improvements.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from times_table_trace import STOI, load  # noqa: E402

STATES = (Path(__file__).parent / "data" / "nanogpt_times_table"
          / "hidden_states.npz")


def main():
    model = load("cpu")
    data = np.load(STATES)
    a = data["a"]; b = data["b"]; prod = data["product"]
    l_arr = data["layer"]
    H_np = data["H"]
    n_layers = int(l_arr.max()) + 1
    vocab = model.cfg.vocab_size

    H = torch.from_numpy(H_np).float()
    with torch.no_grad():
        logits = model.head(model.ln_f(H))
        probs = torch.softmax(logits, dim=-1).numpy()

    true_ids = np.array([STOI[str(int(p))[0]] for p in prod])
    p_correct = probs[np.arange(len(probs)), true_ids]

    # Margin and rank of the correct answer per row
    p_copy = probs.copy()
    rows = np.arange(len(probs))
    p_copy[rows, true_ids] = -1.0  # exclude correct
    top_wrong_p = p_copy.max(axis=1)
    margins = p_correct - top_wrong_p
    # Rank: 1 = best, vocab = worst
    sorted_desc = np.argsort(-probs, axis=1)
    ranks = (sorted_desc == true_ids[:, None]).argmax(axis=1) + 1

    print(f"{'layer':>6}  {'P(correct)':>10}  {'margin':>10}  "
          f"{'P(top wrong)':>12}  {'rank=1%':>8}  {'mean rank':>10}")
    print("-" * 70)
    for l in range(n_layers):
        m = l_arr == l
        print(f"  L{l}    "
              f"{p_correct[m].mean():10.3f}  "
              f"{margins[m].mean():+10.3f}  "
              f"{top_wrong_p[m].mean():12.3f}  "
              f"{(ranks[m] == 1).mean() * 100:7.1f}%  "
              f"{ranks[m].mean():10.2f}")

    # Decompose what L5 specifically adds.
    # For the 100 pairs, compare L4 prediction to L5 prediction.
    print()
    print("L4 → L5 transition breakdown:")
    by_pair_l4 = {}
    by_pair_l5 = {}
    for i in range(len(l_arr)):
        key = (int(a[i]), int(b[i]))
        if l_arr[i] == 4:
            by_pair_l4[key] = (int(np.argmax(probs[i])), float(p_correct[i]),
                               int(ranks[i]))
        elif l_arr[i] == 5:
            by_pair_l5[key] = (int(np.argmax(probs[i])), float(p_correct[i]),
                               int(ranks[i]))

    transitions = {"already correct → still correct (refinement)": 0,
                   "wrong → correct (rank improvement)": 0,
                   "correct → wrong (regression — shouldn't happen)": 0,
                   "still wrong (failed pair)": 0}

    avg_pcorrect_change = []
    rank_change_wins = []

    for key in by_pair_l4:
        pred_l4, p_l4, rank_l4 = by_pair_l4[key]
        pred_l5, p_l5, rank_l5 = by_pair_l5[key]
        l4_correct = (rank_l4 == 1)
        l5_correct = (rank_l5 == 1)
        if l4_correct and l5_correct:
            transitions["already correct → still correct (refinement)"] += 1
        elif not l4_correct and l5_correct:
            transitions["wrong → correct (rank improvement)"] += 1
            rank_change_wins.append((key, rank_l4, p_l4, p_l5))
        elif l4_correct and not l5_correct:
            transitions["correct → wrong (regression — shouldn't happen)"] += 1
        else:
            transitions["still wrong (failed pair)"] += 1
        avg_pcorrect_change.append(p_l5 - p_l4)

    for label, count in transitions.items():
        print(f"  {label}: {count}")

    print(f"  Average P(correct) gain from L4 → L5: "
          f"{float(np.mean(avg_pcorrect_change)):+.3f}")

    # For the rank-improvement pairs, what was L4's rank of the correct char?
    print()
    print("For pairs that flipped wrong → correct at L5:")
    print(f"{'pair':>6}  {'L4 rank of correct':>18}  {'L4 P(correct)':>14}  "
          f"{'L5 P(correct)':>14}")
    rank_change_wins.sort(key=lambda x: x[1])  # sort by L4 rank
    for (a_v, b_v), r4, p4, p5 in rank_change_wins:
        print(f"  {a_v}*{b_v}    rank={r4:<3d}            "
              f"{p4:14.3f}  {p5:14.3f}")


if __name__ == "__main__":
    main()
