"""Calibrate the route-classification probe, persist it, then validate END-TO-END:
a fresh DispatchTurnstyle (router auto-loaded) must auto-route HELD-OUT probe prompts
to the right recognition probe WITHOUT use_probe, while symbolic + NONE prompts route
correctly and don't misfire.

  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python experiments/calibrate_router.py
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from turnstyle.bbh import load_task, answer_matches
from turnstyle.dispatch_turnstyle import DispatchTurnstyle

MID = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
PROBE_SOURCES = {"choice_probe", "selection_probe", "pmi_floor"}


def main():
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MID)
    mdl = AutoModelForCausalLM.from_pretrained(MID, dtype=torch.float16).to(dev)
    dt = DispatchTurnstyle(mdl, tok, dev)
    print(f"probe tasks: {dt.profile_tasks}", flush=True)

    acc = dt.calibrate_router(threshold=0.9, verbose=True)
    path = dt.persist()
    print(f"router CV acc = {acc:.3f}; persisted → {path}", flush=True)

    # fresh instance auto-loads the router from the profile
    dt2 = DispatchTurnstyle(mdl, tok, dev)
    print(f"router loaded on fresh instance: {dt2.ctx.router is not None}", flush=True)

    print("\n=== AUTO-ROUTE held-out probe prompts (NO use_probe) ===")
    routed_ok = ans_ok = n = 0
    for t in dt2.profile_tasks:
        for ex in load_task(t)[60:66]:        # held out from router training ([:60])
            dt2.ctx.choice_artifact = None    # simulate "auto" — nothing pre-selected
            a = dt2.parse(ex["input"])
            n += 1
            is_probe = a is not None and a.source in PROBE_SOURCES
            routed_ok += int(is_probe)
            ans_ok += int(a is not None and answer_matches(a.text, ex["target"].strip()))
        print(f"  {t:42s} routed-to-probe so far…", flush=True)
    print(f"  routed-to-a-probe: {routed_ok}/{n} ; correct answer: {ans_ok}/{n}")

    print("\n=== symbolic + NONE prompts must NOT misroute ===")
    checks = [
        ("multistep_arithmetic_two", 0, "arithmetic"),
        ("date_understanding", 8, "date_calc"),       # symbolic date path (probe-task too, but DateCalc first)
        ("logical_deduction_three_objects", 0, "ordering"),
        ("penguins_in_a_table", 0, "table_query"),
        ("boolean_expressions", 9, "boolean"),
    ]
    for t, i, want in checks:
        dt2.ctx.choice_artifact = None
        a = dt2.parse(load_task(t)[i]["input"])
        src = a.source if a else None
        print(f"  {t:38s} → {src:14s} (want {want}) {'OK' if src == want else 'MISROUTE?'}")


if __name__ == "__main__":
    main()
