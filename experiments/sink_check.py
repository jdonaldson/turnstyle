"""
sink_check.py — verify the "goals = DAG sinks" formalization on the toy library.

Claims checked (no model):
  1. GENERATOR:  regress(sinks(P) ∪ checks(P)) == P  for every gold plan.
  2. MINIMALITY: dropping any single sink from the seed loses nodes.
  3. CORPUS CONSISTENCY: goal_probe's label-by-construction goal sets are exactly
     the sink sets of their regressed plans (i.e. the probe was already trained
     on sink labels, just without the definition).

Sink def: a ∈ P is a sink iff no OTHER consumer in P reads any of a's writes,
where consumers = actions' reads ∪ fork-guard keys (a branch condition consuming
a check action's boolean counts as consumption).
"""
from __future__ import annotations
import os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dag_extract import ACTIONS, REQUIREMENTS, regress
from goal_probe import build_corpus


def sinks(nodes: frozenset, guard_keys: frozenset = frozenset()) -> frozenset:
    consumed = set(guard_keys)
    for a in nodes:
        consumed |= set(ACTIONS[a][0])          # reads
    return frozenset(a for a in nodes if not (set(ACTIONS[a][1]) & consumed))


def check_gold_plans():
    ok_gen = ok_min = 0
    for rid, req, steps in REQUIREMENTS:
        nodes = frozenset(a for a, _, _ in steps)
        checks = frozenset(a for a, bo, _ in steps if bo is not None)
        guards = frozenset(bo.removeprefix("not ") for _, bo, _ in steps if bo)
        s = sinks(nodes, guards)
        seed = s | checks
        gen = regress(seed) == nodes
        # minimality: dropping any sink loses nodes
        minimal = all(regress((s - {x}) | checks) != nodes for x in s)
        ok_gen += gen; ok_min += minimal
        status = "OK " if gen and minimal else "FAIL"
        print(f"  [{rid}] {status} sinks={sorted(s)}"
              + (f" checks={sorted(checks)}" if checks else ""))
        if not gen:
            print(f"        regress(seed)={sorted(regress(seed))} != nodes={sorted(nodes)}")
    print(f"  generator {ok_gen}/{len(REQUIREMENTS)}, minimality {ok_min}/{len(REQUIREMENTS)}")


def check_corpus_labels():
    agree = 0
    reqs = build_corpus()
    disagreements = []
    for r in reqs:
        # rebuild the plan the way reconstruct() would: goals (+ check for cond)
        if r["template"] == "cond":
            check = r["guard"]["check"]
            guards = frozenset(ACTIONS[check][1])   # the check's boolean = the guard
            plan = regress(r["goals"] | {check})
        else:
            guards = frozenset()
            plan = regress(r["goals"])
        if sinks(plan, guards) == r["goals"]:
            agree += 1
        else:
            disagreements.append((r["req"], sorted(r["goals"]),
                                  sorted(sinks(plan, guards))))
    print(f"  corpus label ↔ sink-rule agreement: {agree}/{len(reqs)}")
    for req, g, s in disagreements[:5]:
        print(f"    DISAGREE {req!r}: labeled={g} sinks={s}")


if __name__ == "__main__":
    print("== gold plans (dag_extract.REQUIREMENTS) ==")
    check_gold_plans()
    print("\n== corpus labels (goal_probe.build_corpus) ==")
    check_corpus_labels()
