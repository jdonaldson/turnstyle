"""Honest held-out accuracy for the order-augmented movie selection probe.

Train (order-augmented) on movie_recommendation[40:250], evaluate on the UNTOUCHED
[0:40] — so the test examples never appear in training in any ordering. Reports
held-out accuracy in natural order AND under a fixed option reorder (out-of-sample
order-robustness), vs the generation floor.

  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python experiments/movie_holdout.py
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import re
import random
from turnstyle.bbh import load_task, answer_matches
from turnstyle.dispatch import _score_selection_marginalized, _split_canonical_options, normalize_option_markers

LET = re.compile(r"\(([A-Z])\)")


def reorder(ex, seed):
    """Return a copy of ex with options shuffled and the target letter remapped."""
    parsed = _split_canonical_options(normalize_option_markers(ex["input"]))
    m = LET.search(ex["target"])
    if parsed is None or m is None:
        return ex
    head, contents = parsed
    n = len(contents)
    correct = ord(m.group(1)) - ord("A")
    perm = list(range(n)); random.Random(seed).shuffle(perm)
    new_contents = [contents[perm[s]] for s in range(n)]
    slot = perm.index(correct)
    new_input = head + "\n".join(f"({chr(ord('A')+s)}) {c}" for s, c in enumerate(new_contents))
    return {"input": new_input, "target": f"({chr(ord('A')+slot)})"}


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from turnstyle.dispatch_turnstyle import DispatchTurnstyle
    mid = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(mid)
    mdl = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.float16).to(dev)
    dt = DispatchTurnstyle(mdl, tok, dev)

    import sys
    K = int(sys.argv[1]) if len(sys.argv) > 1 else 6        # augment orderings
    SEEDS = int(sys.argv[2]) if len(sys.argv) > 2 else 5    # reorder seeds to average

    exs = load_task("movie_recommendation")
    test, train = exs[:40], exs[40:]
    print(f"train {len(train)} (augmented k={K}), held-out test {len(test)}, "
          f"reorder over {SEEDS} seeds", flush=True)

    # Fit the order-augmented selection probe on the DISJOINT train split (task=None
    # → don't touch the profile; just set ctx.choice_artifact to this held-out probe).
    res = dt.fit_selection(train, task=None, verbose=True, augment_orderings=K)
    if not res.ship:
        print("did not ship on held-out train:", res.reason); return

    def acc(items):
        ok = n = 0
        for ex in items:
            sel = _score_selection_marginalized(ex["input"], dt.ctx)
            if sel is None:
                continue
            n += 1
            ok += int(answer_matches(sel[0], ex["target"].strip()))
        return ok / n if n else 0.0

    print(f"\n=== HELD-OUT movie (probe trained on [40:250] k={K}, eval on [0:40]) ===")
    a_nat = acc(test)
    print(f"  natural order: {a_nat:.3f}", flush=True)
    reord_accs = []
    for s in range(SEEDS):                       # distinct permutation per seed
        a = acc([reorder(ex, s * 1000 + i) for i, ex in enumerate(test)])
        reord_accs.append(a)
        print(f"  reordered[seed {s}]: {a:.3f}", flush=True)
    mean_reord = sum(reord_accs) / len(reord_accs)
    print(f"\n  reordered mean = {mean_reord:.3f}  (over {SEEDS} seeds)")
    print(f"  reorder Δ = {(mean_reord - a_nat)*100:+.1f}pp   (generation floor ~0.22)")


if __name__ == "__main__":
    main()
