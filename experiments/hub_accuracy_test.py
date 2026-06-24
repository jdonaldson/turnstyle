#!/usr/bin/env python3
"""Test RoutingTurnstyle end-to-end answer accuracy on held-out BBH examples.

Training window: indices 0-29.
Test window: indices 30-59 (held out).

Checks both routing and answer correctness through the full hub pipeline:
  probe route → extraction_spec / sentence_ir_spec → deterministic compute / logit bias
"""
from common import load_hub
from swollm.bench.bbh import load_task

TASKS = [
    ("boolean_expressions",                       "boolean",             "boolean"),
    ("dyck_languages",                            "dyck",                "dyck"),
    ("word_sorting",                              "sorting",             "sorting"),
    ("date_understanding",                        "date",                "date"),
    ("navigate",                                  "spatial_navigation",  "spatial_navigation"),
    ("web_of_lies",                               "truth_chain",         "truth_chain"),
    ("logical_deduction_three_objects",           "comparison_ordering", "comparison_ordering"),
    ("tracking_shuffled_objects_three_objects",   "object_tracking",     "object_tracking"),
]

N_TEST    = 10   # examples per task (indices 30-39)
START_IDX = 30


def answer_correct(text: str, target: str) -> bool:
    """Check if target appears in generated text (case-insensitive)."""
    t = text.strip().lower()
    g = target.strip().lower()
    return g in t or t.startswith(g)


print("Loading model + hub...", flush=True)
tok, mdl, solvers, hub = load_hub()

print(f"\n{'Task':<44}  {'Routed':<22}  {'Acc':>5}  {'Details'}")
print("─" * 100)

total_correct = total_count = 0

for task_name, expected_label, display_label in TASKS:
    examples = load_task(task_name)
    test_examples = examples[START_IDX:START_IDX + N_TEST]

    correct = wrong_route = 0
    failures = []

    for ex in test_examples:
        prompt = ex["input"]
        target = ex["target"]
        results = hub.solve(prompt, max_new_tokens=80)
        got_label = results[0].solver if results else "none"
        got_text  = results[0].text   if results else ""

        ok = answer_correct(got_text, target)
        if got_label != expected_label:
            wrong_route += 1
        if ok:
            correct += 1
        else:
            failures.append((got_label, got_text[:40], target))

    acc = correct / len(test_examples)
    route_note = f" ({wrong_route} wrong route)" if wrong_route else ""
    print(f"  {display_label:<42}  {expected_label:<22}  {acc:>4.0%}{route_note}", flush=True)
    if failures:
        for route, text, tgt in failures[:3]:
            print(f"    ✗ routed={route!r}  got={text!r}  expected={tgt!r}")

    total_correct += correct
    total_count   += len(test_examples)

print(f"\nAggregate: {total_correct}/{total_count} ({total_correct/total_count:.1%})")
