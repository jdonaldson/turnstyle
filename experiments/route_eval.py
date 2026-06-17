"""Mixed-stream router eval: run dispatch.run() over a SHUFFLED mix of BBH tasks.

The per-variant smokes tested each task in isolation. This tests the part that
actually matters — does parse()'s priority ordering correctly route a mixed
stream without collisions, and where does FreeForm/abstain land? It's also the
first real consumer of the ADT, which is what tells us whether Abstain needs a
type.

No model/artifact: deterministic variants solve; MultipleChoice + knowledge
tasks abstain (that's the signal — those are where a model/probe would plug in).
"""
from __future__ import annotations

import json
import random
import sys
from collections import Counter, defaultdict

sys.path.insert(0, "/Users/jdonaldson/Projects/turnstyle/src")
from turnstyle.dispatch import parse, run, Ctx  # noqa: E402

BBH = "/Users/jdonaldson/Projects/swollm/data/bbh_cache"
N_PER_TASK = 30

# task -> expected routed variant (hypothesis; eval flags mismatches)
EXPECT = {
    "multistep_arithmetic_two": "Arithmetic",
    "boolean_expressions": "Boolean",
    "dyck_languages": "Dyck",
    "word_sorting": "Sorting",
    "navigate": "Spatial",
    "web_of_lies": "TruthChain",
    "date_understanding": "DateCalc",
    "logical_deduction_three_objects": "Ordering",
    "snarks": "MultipleChoice",            # routes to MC, abstains w/o artifact
    "movie_recommendation": "MultipleChoice",
    "causal_judgement": "FreeForm",        # knowledge, no structure
}


def main():
    ctx = Ctx()  # no model
    stream = []
    for task in EXPECT:
        try:
            data = json.load(open(f"{BBH}/{task}.json"))[:N_PER_TASK]
        except FileNotFoundError:
            print(f"  (skip {task}: not in cache)")
            continue
        stream += [(task, ex) for ex in data]

    random.seed(0)
    random.shuffle(stream)
    print(f"mixed stream: {len(stream)} examples, {len({t for t, _ in stream})} tasks\n")

    route: dict[str, Counter] = defaultdict(Counter)
    correct, total, abstain = Counter(), Counter(), Counter()
    for task, ex in stream:
        variant = type(parse(ex["input"], ctx)).__name__
        ans = run(ex["input"], ctx)
        route[task][variant] += 1
        total[task] += 1
        if ans.source == "abstain":
            abstain[task] += 1
        elif ans.text == ex["target"]:
            correct[task] += 1

    print(f"{'task':32s} {'routed → variant(s)':27s} {'acc':>5s} {'abstain':>8s}")
    print("-" * 80)
    misroutes = 0
    for task in EXPECT:
        if total[task] == 0:
            continue
        variants = route[task]
        clean = len(variants) == 1 and next(iter(variants)) == EXPECT[task]
        misroutes += 0 if clean else 1
        vstr = ",".join(f"{v}:{c}" for v, c in variants.most_common())
        flag = "" if clean else f"  ⚠ expected {EXPECT[task]}"
        print(f"{task:32s} {vstr:27s} {correct[task]/total[task]:4.0%} "
              f"{abstain[task]/total[task]:7.0%}{flag}")

    n_tasks = sum(1 for t in EXPECT if total[t])
    print(f"\nclean routing: {n_tasks - misroutes}/{n_tasks} tasks → single expected variant")
    print(f"solved (deterministic): {sum(correct.values())}/{sum(total.values())}")
    print(f"abstained (MC/knowledge, no model): {sum(abstain.values())}/{sum(total.values())}")


if __name__ == "__main__":
    main()
