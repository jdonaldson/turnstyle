"""
negation_bbh_conflict.py — does the negation-aware FrameOrdering path change
ANY behavior on BBH?

The wiring adds a hedge-probe verdict inside solve_frame_ordering. Conflict
surface: (1) a plain BBH prompt scoring above the clear line would turn an
old commit into an abstain (mid band) or a complement flip (negated band);
(2) any prompt where old and new answers differ. The gate (>=2 short
entity-like options + a superlative that routes to a frame) is checked with
pure string ops + membership routing over EVERY example of EVERY task;
model forwards run only where the gate fires.

Report per task: examples, gate-fired count, detector verdicts on fired
examples, and every old!=new diff (prompt head + both answers + hedge z).

Run:  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python \
        experiments/negation_bbh_conflict.py
Writes experiments/results/negation_bbh_conflict.json.
"""
from __future__ import annotations
import json, os, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

BBH = "/Users/jdonaldson/Projects/swollm/data/bbh_cache"
MODEL = "HuggingFaceTB/SmolLM2-1.7B-Instruct"


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from turnstyle.frame_library import load_library
    from turnstyle.operators import load_detector
    from turnstyle.frame_ordering import solve_frame_ordering, _candidates
    from turnstyle.sql import extract_options, extract_question

    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL)
    mdl = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.float16).to(dev)
    lib = load_library(mdl)
    det = load_detector(mdl)
    assert det is not None, "no cached detector — run calibrate_negation first"
    print(f"detector: L{det.layer} clear<{det.z_clear:.2f} "
          f"negated>={det.z_negated:.2f}\n", flush=True)

    def gate(prompt: str):
        """Mirror solve_frame_ordering's cheap gates; return (question, attr)
        for the first routable superlative, else None. No model forwards."""
        options = extract_options(prompt)
        if len(options) < 2:
            return None
        if any(len(t.split()) > 4 for t in options.values()):
            return None
        question = extract_question(prompt) or prompt
        for attr, _least in _candidates(question):
            if lib.route(attr) is not None:
                return question, attr
        return None

    tasks = sorted(f[:-5] for f in os.listdir(BBH) if f.endswith(".json"))
    results = {"model": MODEL, "tasks": {}, "diffs": []}
    tot_ex = tot_fired = tot_diff = 0
    for task in tasks:
        data = json.load(open(f"{BBH}/{task}.json"))
        fired, verdicts, diffs = 0, {"True": 0, "False": 0, "None": 0}, []
        for i, ex in enumerate(data):
            prompt = ex["input"]
            g = gate(prompt)
            if g is None:
                continue
            fired += 1
            question, attr = g
            z = det.score_word(mdl, tok, dev, question, attr)
            v = det.detect_word(mdl, tok, dev, question, attr)
            verdicts[str(v)] += 1
            old = solve_frame_ordering(prompt, lib, mdl, tok, dev)
            new = solve_frame_ordering(prompt, lib, mdl, tok, dev,
                                       detector=det)
            if old != new:
                d = {"task": task, "i": i, "attr": attr,
                     "z": round(z, 2), "verdict": str(v),
                     "old": old, "new": new,
                     "target": ex.get("target"),
                     "q": question[:90]}
                diffs.append(d)
                results["diffs"].append(d)
        results["tasks"][task] = {"n": len(data), "gate_fired": fired,
                                  "verdicts": verdicts if fired else {}}
        tot_ex += len(data); tot_fired += fired; tot_diff += len(diffs)
        flag = f"  DIFFS {len(diffs)}" if diffs else ""
        print(f"  {task:38s} n={len(data):3d} fired={fired:3d} "
              f"{verdicts if fired else ''}{flag}", flush=True)

    print(f"\ntotal: {tot_ex} examples, gate fired {tot_fired}, "
          f"old!=new {tot_diff}")
    for d in results["diffs"]:
        print(f"  DIFF {d['task']}[{d['i']}] attr={d['attr']} z={d['z']} "
              f"v={d['verdict']} {d['old']} -> {d['new']} "
              f"(target {d['target']})  \"{d['q']}\"")

    # ── production-path check: FULL dispatch.parse on every fired example ────
    # (direct solve_frame_ordering bypasses the earlier deterministic solvers —
    # penguins' SQL path etc. may claim these prompts first in real dispatch)
    from turnstyle.dispatch import Ctx, parse
    fired_examples = []
    for task in tasks:
        data = json.load(open(f"{BBH}/{task}.json"))
        for i, ex in enumerate(data):
            if gate(ex["input"]) is not None:
                fired_examples.append((task, i, ex["input"]))
    print(f"\n=== production parse() on {len(fired_examples)} fired examples "
          "(old ctx vs detector ctx) ===")
    prod_diffs = 0
    for task, i, prompt in fired_examples:
        old_t = parse(prompt, Ctx(model=mdl, tokenizer=tok, device=dev,
                                  frame_library=lib, pole_cache={}))
        new_t = parse(prompt, Ctx(model=mdl, tokenizer=tok, device=dev,
                                  frame_library=lib, negation_detector=det,
                                  pole_cache={}))
        same = type(old_t) is type(new_t) and \
            getattr(old_t, "answer", None) == getattr(new_t, "answer", None)
        if not same:
            prod_diffs += 1
            print(f"  PROD-DIFF {task}[{i}]: "
                  f"{type(old_t).__name__}({getattr(old_t, 'answer', '')}) -> "
                  f"{type(new_t).__name__}({getattr(new_t, 'answer', '')})")
    print(f"production diffs: {prod_diffs}/{len(fired_examples)}")
    results["production_diffs"] = prod_diffs
    results["production_fired"] = len(fired_examples)

    outp = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "results", "negation_bbh_conflict.json")
    with open(outp, "w") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {outp}")


if __name__ == "__main__":
    main()
