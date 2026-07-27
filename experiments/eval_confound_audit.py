"""Is the bundled frame family confounded with EVALUATION?

Building a tokenization-clean potency lexicon turned up an incidental finding: the
SHIPPED curated `potency` lexicon scores |corr(label, ACT Evaluation)| = 0.911 over
the 8 words it shares with ACT. If that holds, the bundled potency frame is
substantially measuring evaluation -- the same failure class as `space`, which had
shipped while actually measuring word rarity.

The frequency audit became a standing rule after `space`. There has never been an
equivalent EVALUATION gate, and no reason to think potency is the only frame that
would fail one. So audit the whole family, not the symptom.

TWO LEVELS, because they can disagree and only the second one is decisive:

  1. LEXICON level -- corr(frame's own labels, ACT E) over shared words. This is
     what raised the alarm. It is necessary but not sufficient: a lexicon can be
     E-correlated while the fitted direction is not, if the model encodes the two
     properties separately and the ridge picks up the intended one.

  2. DIRECTION level -- |cos| between the fitted frame direction and the fitted
     EVALUATION direction, in the bundled library. `opinion` is the family's
     evaluation rung, so opinion.direction is the reference. THIS is what decides
     whether the shipped artifact is actually confounded, and the library already
     advertises mean off-diagonal |cos| = 0.037.

A frame is flagged only if BOTH levels agree. Low overlap with ACT is reported as
UNKNOWN, never as clean.

    uv run python experiments/eval_confound_audit.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from turnstyle.frame_library import CANONICAL_FRAMES, load_library  # noqa: E402

LEX = Path(__file__).parent / "data" / "lexicons"
RESULTS = Path(__file__).parent / "results"
BUNDLED = Path(__file__).resolve().parents[1] / "src" / "turnstyle" / "data" / "frames"
MIN_OVERLAP = 6
LABEL_GATE = 0.50      # |corr(label, E)| above this is a lexicon-level flag
COS_GATE = 0.30        # |cos| with the evaluation direction above this is real


def load_act_E():
    """{term: E} from both ACT files (modifiers and identities)."""
    out = {}
    for kind in ("act_mods.csv", "act_identities.csv"):
        with open(LEX / kind, newline="") as f:
            for row in csv.DictReader(f):
                t = row["term"].strip().lower()
                if not t or "_" in t or " " in t:
                    continue
                try:
                    e = float(row["E"])
                except (ValueError, KeyError):
                    continue
                out[t] = (out[t] + e) / 2 if t in out else e
    return out


def main():
    E = load_act_E()
    print(f"ACT terms with an Evaluation rating: {len(E)}\n")

    print("=" * 84)
    print("LEVEL 1 — LEXICON: does the frame's own label track ACT Evaluation?")
    print("=" * 84)
    print(f"  {'frame':14s} {'overlap':>8s} {'|corr(label,E)|':>16s}   status")
    lex = {}
    for name, spec in CANONICAL_FRAMES.items():
        data = spec["data"]
        common = [w for w in data if w in E]
        if len(common) < MIN_OVERLAP:
            lex[name] = None
            print(f"  {name:14s} {len(common):>3d}/{len(data):<4d} "
                  f"{'—':>16s}   UNKNOWN (too little ACT overlap)")
            continue
        lab = np.array([data[w] for w in common], float)
        ev = np.array([E[w] for w in common], float)
        # zero variance on either side => the correlation is nan, which must read as
        # UNKNOWN and not slip through as "ok" (the same unknown-is-not-clean rule the
        # tokenization control enforces)
        if lab.std() < 1e-9 or ev.std() < 1e-9:
            lex[name] = None
            print(f"  {name:14s} {len(common):>3d}/{len(data):<4d} {'—':>16s}   "
                  f"UNKNOWN (no label variance among shared words)")
            continue
        c = abs(float(np.corrcoef(lab, ev)[0, 1]))
        lex[name] = c
        flag = "FLAG" if c >= LABEL_GATE else "ok"
        print(f"  {name:14s} {len(common):>3d}/{len(data):<4d} {c:>16.3f}   {flag}")

    print("\n" + "=" * 84)
    print("LEVEL 2 — DIRECTION: is the FITTED frame aligned with the fitted")
    print("          evaluation direction? (`opinion` is the family's eval rung)")
    print("=" * 84)
    npz = sorted(BUNDLED.glob("*.npz"))
    if not npz:
        print("  no bundled library found — level 2 skipped")
        return
    from turnstyle.frame_library import FrameLibrary
    lib = FrameLibrary.load_npz(npz[0])
    if "opinion" not in lib:
        print("  bundled library has no `opinion` frame — no evaluation reference")
        return
    ref = lib.frames["opinion"].axis.direction
    ref = ref / (np.linalg.norm(ref) + 1e-12)

    print(f"  reference = opinion @L{lib.frames['opinion'].layer}  "
          f"({len(lib)} frames in {npz[0].name})")
    print(f"\n  {'frame':14s} {'layer':>6s} {'|cos| vs opinion':>17s}   status")
    rows = []
    for name, f in lib.frames.items():
        if name == "opinion":
            continue
        d = f.axis.direction
        d = d / (np.linalg.norm(d) + 1e-12)
        cs = abs(float(d @ ref))
        rows.append((name, f.layer, cs))
    for name, layer, cs in sorted(rows, key=lambda r: -r[2]):
        flag = "FLAG" if cs >= COS_GATE else "ok"
        print(f"  {name:14s} {'L' + str(layer):>6s} {cs:>17.3f}   {flag}")

    print("\n" + "=" * 84)
    print("VERDICT — flagged only where BOTH levels agree")
    print("=" * 84)
    cos = {n: c for n, _, c in rows}
    both, lex_only, unknown = [], [], []
    for name in CANONICAL_FRAMES:
        if name == "opinion":
            continue
        l, c = lex.get(name), cos.get(name)
        if l is None:
            unknown.append(name)
        elif l >= LABEL_GATE and c is not None and c >= COS_GATE:
            both.append((name, l, c))
        elif l >= LABEL_GATE:
            lex_only.append((name, l, c))
    for n, l, c in both:
        print(f"  CONFOUNDED  {n:14s} label {l:.3f}, direction {c:.3f}")
    if not both:
        print("  no frame is confounded at BOTH levels")
    for n, l, c in lex_only:
        cs = "n/a" if c is None else f"{c:.3f}"
        print(f"  lexicon-only {n:13s} label {l:.3f} but direction {cs} — the fit "
              f"did NOT inherit it")
    if unknown:
        print(f"  UNKNOWN (insufficient ACT overlap, not clean): {unknown}")

    p = RESULTS / "eval_confound_audit.json"
    p.write_text(json.dumps({"lexicon": lex, "direction": cos,
                             "gates": {"label": LABEL_GATE, "cos": COS_GATE},
                             "confounded_both_levels": [n for n, _, _ in both],
                             "lexicon_only": [n for n, _, _ in lex_only],
                             "unknown": unknown}, indent=2))
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
