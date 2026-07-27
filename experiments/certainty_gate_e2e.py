"""End-to-end: does the certainty gate actually help on BBH? (task #1)

The gate shipped with an offline number — AUC 0.653, +0.120 over margin at 50%
coverage — measured in a standalone harness, NOT through `DispatchTurnstyle.parse()`.
That number cannot settle whether the gate helps, for one reason:

  A veto is not a free win. When the gate abstains, DispatchTurnstyle falls back to
  plain generation. If the fallback answers as well as the vetoed answer would have,
  the gate buys nothing no matter how good its AUC. The offline harness never
  modelled the fallback at all.

So the metric is OVERALL accuracy after fallback, not accuracy-on-what-remains.
`committed_acc` should rise almost by construction (we are removing the least
confident commits); the question is whether `accuracy` follows.

HELD OUT: the gate was calibrated on examples [0:40], so this runs offset=40. Same
model, same profile, gate the only difference.

Includes a symbolic control task. The gate may only veto GATED_SOURCES
(choice_probe, pmi_floor); boolean_expressions is solved symbolically, so it must
come out byte-identical. If it moves, the carve-out leaks.

    uv run python experiments/certainty_gate_e2e.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

RESULTS = Path(__file__).parent / "results"
BBH = Path.home() / "Projects" / "swollm" / "data" / "bbh_cache"
MODEL = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
TASKS = ["snarks", "disambiguation_qa", "movie_recommendation",
         "date_understanding", "hyperbaton", "boolean_expressions"]
SYMBOLIC_CONTROL = "boolean_expressions"


def summarize(rep, tasks):
    tot_c = tot_n = com = com_c = 0
    for t in tasks:
        r = rep.get(t)
        if not r:
            continue
        tot_c += r["correct"]
        tot_n += r["n"]
        com += r["committed"]
        com_c += int(round((r["committed_acc"] or 0) * r["committed"]))
    return {"acc": tot_c / max(1, tot_n), "n": tot_n,
            "coverage": com / max(1, tot_n),
            "committed_acc": (com_c / com) if com else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=40)
    ap.add_argument("--offset", type=int, default=40, help="MUST exceed the "
                    "calibration slice or the result is leaked")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from turnstyle.bbh import evaluate
    from turnstyle.certainty_gate import load_gate
    from turnstyle.dispatch_turnstyle import DispatchTurnstyle
    from turnstyle.profile import model_fingerprint

    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to(dev).eval()
    fp = model_fingerprint(model)
    gate = load_gate(fp)
    if gate is None:
        raise SystemExit(f"no gate for {fp} — run calibrate_certainty_gate.py first")
    print(f"gate: threshold {gate.threshold:+.3f} L{gate.layer} "
          f"(calibrated AUC {gate.auc:.3f}, n={gate.n_calibration})")
    print(f"eval slice: examples [{args.offset}:{args.offset + args.limit}] "
          f"— disjoint from calibration [0:40]\n")

    out = {}
    for label, use_gate in (("baseline", False), ("gated", True)):
        dt = DispatchTurnstyle(model, tok, dev)
        if use_gate:
            dt.install_certainty_gate(gate)
        print(f"--- {label} ---", flush=True)
        rep = evaluate(dt, tasks=TASKS, cache_dir=BBH, limit=args.limit,
                       offset=args.offset, verbose_first=0, progress=False)
        out[label] = rep

    gated_tasks = [t for t in TASKS if t != SYMBOLIC_CONTROL]
    print(f"\n{'=' * 78}\nEND-TO-END (overall accuracy is what matters; the fallback "
          f"gets the vetoed ones)\n{'=' * 78}")
    print(f"  {'':26s} {'baseline':>10s} {'gated':>10s} {'delta':>8s}")
    b, g = summarize(out["baseline"], gated_tasks), summarize(out["gated"], gated_tasks)
    print(f"  {'OVERALL accuracy':26s} {b['acc']:>10.3f} {g['acc']:>10.3f} "
          f"{g['acc'] - b['acc']:>+8.3f}   <- the real question")
    print(f"  {'coverage (committed)':26s} {b['coverage']:>10.3f} {g['coverage']:>10.3f} "
          f"{g['coverage'] - b['coverage']:>+8.3f}")
    ba = b["committed_acc"] or 0.0
    ga = g["committed_acc"] or 0.0
    print(f"  {'accuracy on committed':26s} {ba:>10.3f} {ga:>10.3f} {ga - ba:>+8.3f}"
          f"   <- rises almost by construction")

    print(f"\n  per task:")
    print(f"  {'task':26s} {'base acc':>9s} {'gated acc':>10s} {'delta':>8s} "
          f"{'cov base':>9s} {'cov gated':>10s}")
    for t in TASKS:
        rb, rg = out["baseline"].get(t), out["gated"].get(t)
        if not rb or not rg:
            continue
        mark = "  (symbolic control)" if t == SYMBOLIC_CONTROL else ""
        print(f"  {t:26s} {rb['accuracy']:>9.3f} {rg['accuracy']:>10.3f} "
              f"{rg['accuracy'] - rb['accuracy']:>+8.3f} "
              f"{rb['committed'] / rb['n']:>9.2f} {rg['committed'] / rg['n']:>10.2f}{mark}")

    cb, cg = out["baseline"].get(SYMBOLIC_CONTROL), out["gated"].get(SYMBOLIC_CONTROL)
    if cb and cg:
        leaked = (cb["accuracy"] != cg["accuracy"]) or (cb["committed"] != cg["committed"])
        print(f"\n  symbolic control {SYMBOLIC_CONTROL}: "
              f"{'LEAKED — the gate touched a proved answer' if leaked else 'untouched (correct)'}")

    d = g["acc"] - b["acc"]
    print(f"\n{'=' * 78}\nVERDICT\n{'=' * 78}")
    if d > 0.01:
        print(f"  HELPS: +{d:.3f} overall accuracy. The vetoed answers were worse than")
        print(f"  what the fallback produced for them.")
    elif d < -0.01:
        print(f"  HURTS: {d:+.3f}. The fallback is WORSE than the answers being vetoed —")
        print(f"  a good confidence signal is not enough if escalation has nowhere better")
        print(f"  to go. The gate should not ship enabled.")
    else:
        print(f"  NEUTRAL ({d:+.3f}). The gate identifies weak answers (committed_acc "
              f"{ga - ba:+.3f}) but the fallback is no better at them, so overall accuracy")
        print(f"  is unchanged. AUC 0.653 does not convert into end-to-end gain here.")

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "certainty_gate_e2e.json").write_text(json.dumps(
        {"model": MODEL, "offset": args.offset, "limit": args.limit,
         "summary": {"baseline": b, "gated": g}, "reports": out}, indent=2, default=str))
    print(f"\nwrote {RESULTS / 'certainty_gate_e2e.json'}")


if __name__ == "__main__":
    main()
