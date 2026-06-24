"""Are there shared output circuits across the three operators?

If the L5 state at the = position encodes "next character to emit",
then prompts producing the same first character of the result should
cluster together at L5 regardless of which operator they used.

Example collisions (same first char):
    1*5=5     5+0=5     7-2=5      → all emit '5' next
    2*3=6     1+5=6     8-2=6
    3+8=11    2*6=12    -          → both emit '1' next

Test:
1. Run 100 prompts per operator × 3 operators = 300 prompts.
2. Capture L5 state at the '=' position.
3. Compute cosine similarity within-(first-char-class) vs across-classes.
4. Per pair, ask: of the 5 nearest L5 neighbors (excluding same prompt),
   how many share the next character?  How many share the operator?
   Higher next-char-match means the circuit is sorting by output.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from times_table_trace import GPT, encode, STOI, load  # noqa: E402


def build_prompts():
    rows = []
    for op in ("+", "-", "*"):
        for a in range(10):
            for b in range(10):
                r = (a + b) if op == "+" else (a - b) if op == "-" else (a * b)
                prompt = f"{a}{op}{b}="
                first_char = str(r)[0]
                rows.append(dict(prompt=prompt, op=op, a=a, b=b,
                                 result=r, first_char=first_char))
    return rows


@torch.no_grad()
def capture_l5_states(model: GPT, rows: list[dict]) -> np.ndarray:
    states = []
    for r in rows:
        idx = torch.tensor([encode(r["prompt"])], dtype=torch.long)
        _, _, all_states = model(idx, return_states=True)
        states.append(all_states[-1][0, -1].cpu().numpy())  # L5, last token
    return np.stack(states)


def main():
    model = load("cpu")
    rows = build_prompts()
    print(f"Built {len(rows)} prompts across 3 operators")

    print("\nFirst-character distribution:")
    chars = [r["first_char"] for r in rows]
    for c in sorted(set(chars)):
        n = chars.count(c)
        ops = [r["op"] for r in rows if r["first_char"] == c]
        per_op = {o: ops.count(o) for o in ("+", "-", "*")}
        print(f"  '{c}': {n} total  (+{per_op['+']}  -{per_op['-']}  *{per_op['*']})")

    H = capture_l5_states(model, rows)
    print(f"\nCaptured L5 states: shape {H.shape}")

    # Normalize for cosine
    Hn = H / (np.linalg.norm(H, axis=1, keepdims=True) + 1e-8)
    sim = Hn @ Hn.T  # (N, N)
    np.fill_diagonal(sim, -np.inf)  # exclude self

    K = 5
    same_char, same_op, same_both = 0, 0, 0
    total = 0
    for i in range(len(rows)):
        top = np.argsort(-sim[i])[:K]
        c_i = rows[i]["first_char"]
        o_i = rows[i]["op"]
        for j in top:
            if rows[int(j)]["first_char"] == c_i:
                same_char += 1
            if rows[int(j)]["op"] == o_i:
                same_op += 1
            if (rows[int(j)]["first_char"] == c_i
                    and rows[int(j)]["op"] == o_i):
                same_both += 1
            total += 1

    p_char_random = sum((chars.count(c) / len(rows)) ** 2
                        for c in set(chars))
    p_op_random = 1 / 3
    print(f"\nFor each pair, top-{K} L5 neighbors:")
    print(f"  share next character:  {same_char/total:.3f}  "
          f"(chance {p_char_random:.3f})")
    print(f"  share operator:        {same_op/total:.3f}  "
          f"(chance {p_op_random:.3f})")
    print(f"  share both:            {same_both/total:.3f}  "
          f"(chance {p_char_random * p_op_random:.3f})")

    # Per-first-character: mean cosine within vs across
    print(f"\nWithin-class vs across-class cosine similarity per first char:")
    for c in sorted(set(chars)):
        mask = np.array([r["first_char"] == c for r in rows])
        if mask.sum() < 2:
            continue
        within = sim[np.ix_(mask, mask)]
        across = sim[np.ix_(mask, ~mask)]
        within_mean = float(within[within > -np.inf].mean())
        across_mean = float(across.mean())
        ops_in_class = sorted(set(rows[i]["op"] for i in range(len(rows))
                                  if mask[i]))
        print(f"  '{c}'  n={int(mask.sum()):3d}  ops={'/'.join(ops_in_class)}  "
              f"within={within_mean:+.3f}  across={across_mean:+.3f}  "
              f"gap={within_mean - across_mean:+.3f}")


if __name__ == "__main__":
    main()
