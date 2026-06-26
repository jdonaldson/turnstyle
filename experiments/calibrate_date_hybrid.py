"""Build + verify the hybrid DateCalc. The architecture already routes DateCalc-abstain ->
MultipleChoice -> _solve_choice; this calibrates a date per-option probe (fit_choice ->
autoprobe), so abstains hit the probe instead of the pmi_floor/generation fallback. Commits
still go to exact symbolic (never reach the probe).

Honest verification: train the probe on date_understanding[125:250], evaluate end-to-end on
the DISJOINT [0:125] via parse->enrich->solve (no leakage), with the probe ON vs OFF, and a
per-source breakdown proving the routing (abstains -> choice_probe).

  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python experiments/calibrate_date_hybrid.py
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
from collections import Counter
from turnstyle.bbh import load_task, answer_matches, _load_model
from turnstyle.dispatch_turnstyle import DispatchTurnstyle
from turnstyle import dispatch as D


def evaluate(dt, exs):
    correct = 0
    by_src = Counter(); ok_src = Counter()
    for ex in exs:
        tgt = ex["target"].strip()
        res = D.run(ex["input"], dt.ctx)
        src = getattr(res, "source", "abstain")
        ok = answer_matches(getattr(res, "text", ""), tgt)
        correct += ok; by_src[src] += 1; ok_src[src] += int(ok)
    n = len(exs)
    return correct / n, by_src, ok_src


def main():
    mdl, tok, dev = _load_model("HuggingFaceTB/SmolLM2-1.7B-Instruct", "auto")
    dt = DispatchTurnstyle(mdl, tok, dev)

    train = load_task("date_understanding")[125:250]
    evalset = load_task("date_understanding")[0:125]
    print(f"train {len(train)}  eval {len(evalset)} (disjoint)", flush=True)

    res = dt.fit_choice(train, task="date_understanding", verbose=True)
    print(f"\nfit_choice: ship={res.ship}  chosen={res.chosen}", flush=True)
    if not res.ship:
        print("probe did not ship; aborting"); return
    dt.persist()
    print("persisted to user cache", flush=True)

    # OFF: no choice probe -> abstains fall to pmi_floor/generation
    saved = dt.ctx.choice_artifact
    dt.ctx.choice_artifact = None
    acc_off, src_off, ok_off = evaluate(dt, evalset)
    # ON: date probe active -> abstains route to it
    dt.ctx.choice_artifact = saved
    acc_on, src_on, ok_on = evaluate(dt, evalset)

    print(f"\n=== end-to-end on disjoint eval (n={len(evalset)}) ===")
    print(f"  probe OFF (symbolic + pmi_floor/gen): {acc_off:.3f}")
    print(f"  probe ON  (symbolic + date probe):    {acc_on:.3f}   ({acc_on-acc_off:+.3f})")
    print(f"\n  source breakdown (ON):  acc by source")
    for s in sorted(src_on):
        print(f"    {s:14s} n={src_on[s]:3d}  acc={ok_on[s]/src_on[s]:.3f}")
    print(f"\n  source breakdown (OFF):")
    for s in sorted(src_off):
        print(f"    {s:14s} n={src_off[s]:3d}  acc={ok_off[s]/src_off[s]:.3f}")


if __name__ == "__main__":
    main()
