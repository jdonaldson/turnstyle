"""Is there ANY abstention headroom in turnstyle's dispatch? (task #13)

The certainty gate failed end-to-end at -0.220 ([[abstention_needs_a_better_fallback]])
because the fallback (plain generation) is worse than the answers it vetoed. Before
recalibrating the gate it is worth asking the prior question: does a gate have
headroom ANYWHERE here?

The criterion needs no new compute. For each task the native BBH report already
records accuracy on COMMITTED answers and enough to derive accuracy on ABSTAINED
ones. A confidence gate can only help where

    accuracy(abstained, i.e. fallback) > accuracy(committed, i.e. primary)

because a gate moves examples from the first column to the second.

    uv run python experiments/abstention_headroom.py
"""

from __future__ import annotations

import json
from pathlib import Path

RESULTS = Path(__file__).parent / "results"
REPORT = Path(__file__).parent / "data" / "bbh_native_report.json"


def main():
    d = json.loads(REPORT.read_text())
    rows = []
    for t, r in sorted(d.items()):
        if t.startswith("_"):
            continue
        n, c, com, ab = r["n"], r["correct"], r["committed"], r["abstained"]
        ca = r["committed_acc"]
        aa = None
        if ab and com:
            aa = (c - round((ca or 0) * com)) / ab
        rows.append({"task": t, "n": n, "coverage": com / n, "committed_acc": ca,
                     "abstained_acc": aa,
                     "headroom": None if aa is None else aa - (ca or 0)})

    print("HEADROOM = accuracy(fallback) - accuracy(primary); positive means a gate helps")
    print(f"  {'task':38s} {'cov':>5s} {'primary':>8s} {'fallback':>9s} {'headroom':>9s}")
    for r in rows:
        ca = "    n/a" if r["committed_acc"] is None else f"{r['committed_acc']:7.3f}"
        aa = "     n/a" if r["abstained_acc"] is None else f"{r['abstained_acc']:8.3f}"
        hd = "      n/a" if r["headroom"] is None else f"{r['headroom']:+9.3f}"
        flag = "  <-- HEADROOM" if (r["headroom"] or -1) > 0.05 else ""
        print(f"  {r['task']:38s} {r['coverage']:>5.2f} {ca} {aa} {hd}{flag}")

    good = [r for r in rows if (r["headroom"] or -1) > 0.05]
    solved = [r for r in rows if r["coverage"] >= 0.99 and (r["committed_acc"] or 0) >= 0.95]
    none_ = [r for r in rows if r["coverage"] <= 0.01]

    print(f"\n{'=' * 78}\nSTRUCTURE\n{'=' * 78}")
    print(f"  tasks with headroom > 5pp:                    {len(good)}/{len(rows)}")
    print(f"  SOLVED (coverage ~1.0, primary >= .95):        {len(solved)}  "
          f"-> a gate can only DELETE correct answers")
    print(f"  ALREADY FULLY ABSTAINING (coverage ~0.0):      {len(none_)}  "
          f"-> nothing for a gate to veto")
    print(f"  in between:                                    "
          f"{len(rows) - len(solved) - len(none_)}")

    print(f"\n  Dispatch is ALREADY a calibrated abstention system: `commitment = coverage`")
    print(f"  commits only when the parse explains the whole prompt, which yields ~1.0")
    print(f"  accuracy on everything it commits to. A confidence gate layers a second,")
    print(f"  worse abstention decision on top of a correct one.")
    print(f"\n  The capability gap is the {len(none_)} coverage-0.00 tasks — no solver at all,")
    print(f"  falling through to raw generation. Build there; do not second-guess the")
    print(f"  paths that already work.")
    print(f"    {[r['task'] for r in none_]}")

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "abstention_headroom.json").write_text(json.dumps(
        {"rows": rows, "n_headroom": len(good), "n_solved": len(solved),
         "n_no_coverage": len(none_),
         "no_solver_tasks": [r["task"] for r in none_]}, indent=2))
    print(f"\nwrote {RESULTS / 'abstention_headroom.json'}")


if __name__ == "__main__":
    main()
