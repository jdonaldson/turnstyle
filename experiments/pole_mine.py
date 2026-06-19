"""Recon: measure within-paragraph antonym mixing in logical_deduction.

Determines whether absolute pole knowledge is load-bearing on BBH, or whether
root-consistency (morphology) suffices. Pole knowledge is only needed when a
single ordering axis appears in BOTH poles (e.g. "newer than" in the body but
"the oldest" in the query) — root-matching cannot bridge antonyms.
"""
import json
import re
from collections import Counter

BBH = "/Users/jdonaldson/Projects/swollm/data/bbh_cache"
TASKS = ["logical_deduction_three_objects",
         "logical_deduction_five_objects",
         "logical_deduction_seven_objects"]

# axis -> {word: pole}  (+1 HIGH end, -1 LOW end)
AXES = {
    "age":   {"newer": +1, "newest": +1, "older": -1, "oldest": -1},
    "price": {"most expensive": +1, "expensive": +1, "pricier": +1,
              "cheapest": -1, "cheaper": -1, "cheap": -1},
    "space": {"rightmost": +1, "right": +1, "leftmost": -1, "left": -1},
}


def words_in(s):
    s = s.lower()
    out = []
    for axis, wmap in AXES.items():
        for w, pole in wmap.items():
            if re.search(r"\b" + re.escape(w) + r"\b", s):
                out.append((w, axis, pole))
    return out


def main():
    total = 0
    mixed = 0
    cat = Counter()
    for t in TASKS:
        for ex in json.load(open(f"{BBH}/{t}.json")):
            total += 1
            full = ex["input"]
            allw = words_in(full)
            axes = sorted(set(a for _, a, _ in allw))
            cat[tuple(axes)] += 1
            for ax in axes:
                poles = set(p for _, a, p in allw if a == ax)
                if len(poles) > 1:
                    mixed += 1
                    break
    print(f"total={total}")
    print(f"antonym-mixing examples (pole knowledge load-bearing): "
          f"{mixed} ({mixed / total:.1%})")
    print("axis composition:")
    for k, v in cat.most_common():
        print(f"  {','.join(k) or '(none)':20s} {v}")


if __name__ == "__main__":
    main()
