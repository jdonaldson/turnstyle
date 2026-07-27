"""Why does dispatch abstain on 11 BBH tasks? Routing failure or solve failure? (task #14)

`abstention_headroom.py` found 11 tasks at coverage 0.00 — dispatch never commits,
everything falls through to raw generation. That is the real capability gap.

But several of them have KNOWN solutions that may simply never have been wired into
native dispatch: object_counting and colored_objects have SQL-first paths,
tracking_shuffled has intercept-and-correct at 98/94/84%, penguins reaches 99.3% via
SQL in swollm, movie and salient have shipped recognition probes. And the Task ADT
already declares ObjectCount / TableQuery / SceneQuery / StateTracking / SvgPath
variants.

So before building anything, separate the failure modes — the diagnose-first rule:

  ROUTING   parse() does not produce the right variant (or produces FreeForm), so the
            solver is never reached. Cheap to fix; the solver already exists.
  SOLVE     parse() routes correctly but solve() returns Abstain. The solver is
            reached and declines — a real gap in the solver.
  WALL      neither; genuinely open (causal_judgement and sports_understanding are
            the documented knowledge walls).

Reports the variant each prompt routes to and the abstain reason, per task.

    uv run python experiments/coverage_zero_triage.py --limit 10
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

RESULTS = Path(__file__).parent / "results"
BBH = Path.home() / "Projects" / "swollm" / "data" / "bbh_cache"
MODEL = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
TASKS = [
    "object_counting", "reasoning_about_colored_objects", "penguins_in_a_table",
    "tracking_shuffled_objects_three_objects", "tracking_shuffled_objects_five_objects",
    "tracking_shuffled_objects_seven_objects", "geometric_shapes",
    "movie_recommendation", "salient_translation_error_detection",
    "causal_judgement", "sports_understanding",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=10)
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from turnstyle.bbh import load_task
    from turnstyle.dispatch import Answer, normalize_option_markers, parse, enrich, solve
    from turnstyle.dispatch_turnstyle import DispatchTurnstyle

    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to(dev).eval()
    dt = DispatchTurnstyle(model, tok, dev)

    out = {}
    print(f"{'task':40s} {'variant(s) routed to':34s} {'outcome'}")
    print("-" * 100)
    for t in TASKS:
        try:
            exs = load_task(t, cache_dir=BBH)[: args.limit]
        except Exception as e:
            print(f"{t:40s} SKIP {type(e).__name__}")
            continue
        dt.ctx.choice_artifact = None
        dt.use_probe(t)
        variants, outcomes, reasons = Counter(), Counter(), Counter()
        for ex in exs:
            p = normalize_option_markers(ex["input"])
            try:
                task = parse(p, dt.ctx)
                variants[type(task).__name__] += 1
                res = solve(enrich(task, p, dt.ctx), p, dt.ctx)
                if isinstance(res, Answer):
                    outcomes["answer"] += 1
                else:
                    outcomes["abstain"] += 1
                    reasons[getattr(res, "reason", "") or "(none)"] += 1
            except Exception as e:
                outcomes[f"error:{type(e).__name__}"] += 1
        vs = ", ".join(f"{k}×{v}" for k, v in variants.most_common(3))
        os_ = ", ".join(f"{k}×{v}" for k, v in outcomes.most_common(3))
        # classify
        n = sum(outcomes.values()) or 1
        routed_freeform = variants.get("FreeForm", 0) / n
        if outcomes.get("answer", 0) / n > 0.5:
            cls = "ANSWERS (so the e2e abstain is elsewhere)"
        elif routed_freeform > 0.5:
            cls = "ROUTING — lands in FreeForm, solver never reached"
        else:
            cls = "SOLVE — routed correctly, solver declines"
        out[t] = {"variants": dict(variants), "outcomes": dict(outcomes),
                  "reasons": dict(reasons), "class": cls}
        print(f"{t:40s} {vs:34s} {os_}")
        print(f"{'':40s} {'-> ' + cls}")
        if reasons:
            print(f"{'':40s}    reasons: {dict(reasons.most_common(3))}")

    print(f"\n{'=' * 100}\nTRIAGE\n{'=' * 100}")
    for cls in ("ROUTING — lands in FreeForm, solver never reached",
                "SOLVE — routed correctly, solver declines",
                "ANSWERS (so the e2e abstain is elsewhere)"):
        hits = [t for t, v in out.items() if v["class"] == cls]
        print(f"  {cls}\n    {hits or '(none)'}")

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "coverage_zero_triage.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {RESULTS / 'coverage_zero_triage.json'}")


if __name__ == "__main__":
    main()
