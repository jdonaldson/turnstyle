"""Build a tokenization-clean `potency` lexicon from real ACT norms.

The controlled refit (`depth_stok_compare.py`) killed per-model localization but
left the surviving frame-level claim resting on ONE clean exemplar, `warmth`,
because `weight`/`potency`/`competence`/`shape` lose too many words under the
shared-single-token filter to fit at all (weight 12->3, potency 12->5).  Those
lexicons were built for semantic precision, which selected for exactly the long
rare words tokenizers split: weightless, featherlight, ponderous, virtuosic.

`potency` is the one of the four that can be rebuilt from REAL norms rather than
values assigned by eye: ACT `act_mods.csv` gives 621 single-word adjectives with an
explicit P (potency) rating, of which 211 survive the 6-tokenizer filter.  Adjectives
are also the right part of speech for the frame's template.

Three gates, all required -- a fresh lexicon can trade one confound for another:

  1. shared single-token across the whole panel        (the new control)
  2. LOW |corr(P, E)| in the selected subset           (ACT's P correlates with E;
     a naive stratified sample would build a frame that is partly Evaluation, and
     the curated lexicon was deliberately potency-pure)
  3. LOW |corr(P, zipf)| -- the standing frequency rule that killed the `space`
     frame, which had shipped while actually measuring word rarity

Selection is greedy: from a stratified spine across the P range, keep adding the
candidate that most improves P spread while holding both confounds down.

    uv run python experiments/build_stok_lexicon.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from wordfreq import zipf_frequency

RESULTS = Path(__file__).parent / "results"
LEX = Path(__file__).parent / "data" / "lexicons"
TEMPLATE = "They are a {w} person."
PANEL = [
    "mistralai/Mistral-7B-Instruct-v0.3",
    "microsoft/Phi-4-mini-instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    "HuggingFaceTB/SmolLM2-360M-Instruct",
    "google/gemma-3n-E2B-it",
]
TARGET_N = 20
MAX_ABS_CORR = 0.35          # ceiling for |corr(P,E)| and |corr(P,zipf)|


def load_act_mods():
    """{term: (E, P, A)} for single-word ACT modifiers."""
    out = {}
    with open(LEX / "act_mods.csv", newline="") as f:
        for row in csv.DictReader(f):
            t = row["term"].strip().lower()
            if not t or "_" in t or " " in t:
                continue
            try:
                out[t] = (float(row["E"]), float(row["P"]), float(row["A"]))
            except (ValueError, KeyError):
                continue
    return out


def residualize(target: dict, on: dict) -> dict:
    """target with the linear part explained by `on` removed, over the candidate pool.

    Necessary because ACT's P correlates with E at +0.857 across its modifiers: the
    P-extremes ARE the E-extremes, so ANY subset selected to span raw P inherits the
    confound (a first attempt seeded on the P-extremes and reached |corr(P,E)| = .98).
    Residualizing first makes the target a potency axis with evaluation removed --
    high = powerful-but-bad (abusive, domineering, ruthless), low = weak (powerless,
    cowardly, timid).

    Fitted over the candidate pool rather than the chosen subset: computing it inside
    the selection would make the orthogonality true by construction and untestable.
    """
    ws = sorted(target)
    y = np.array([target[w] for w in ws])
    x = np.array([on[w] for w in ws])
    resid = y - np.polyval(np.polyfit(x, y, 1), x)
    return dict(zip(ws, resid))


def greedy_select(cands, T, E, Z, n, max_corr):
    """Grow a set spanning target T while holding |corr(T,E)| and |corr(T,zipf)| down.

    Seeded on the extreme-T pair. With T = residualized potency this is safe; with
    raw P it was not, because the raw extremes carry the E confound.
    """
    chosen = [min(cands, key=lambda w: T[w]), max(cands, key=lambda w: T[w])]
    pool = [w for w in cands if w not in chosen]

    def confounds(ws):
        t = np.array([T[w] for w in ws])
        if t.std() < 1e-9:
            return 1.0, 1.0
        e = np.array([E[w] for w in ws])
        z = np.array([Z[w] for w in ws])
        ce = 0.0 if e.std() < 1e-9 else abs(np.corrcoef(t, e)[0, 1])
        cz = 0.0 if z.std() < 1e-9 else abs(np.corrcoef(t, z)[0, 1])
        return ce, cz

    while len(chosen) < n and pool:
        best = None
        for w in pool:
            trial = chosen + [w]
            ce, cz = confounds(trial)
            if len(trial) >= 6 and (ce > max_corr or cz > max_corr):
                continue
            score = np.std([T[x] for x in trial]) - 0.5 * (ce + cz)
            if best is None or score > best[0]:
                best = (score, w)
        if best is None:
            break
        chosen.append(best[1])
        pool.remove(best[1])
    return chosen


def main():
    from transformers import AutoTokenizer
    from turnstyle.frame_library import shared_single_token_words

    act = load_act_mods()
    print(f"ACT single-word modifiers with EPA: {len(act)}")

    toks = [AutoTokenizer.from_pretrained(m) for m in PANEL]
    keep = shared_single_token_words(toks, sorted(act), TEMPLATE)
    print(f"gate 1 — shared single-token across {len(PANEL)} tokenizers: {len(keep)}")

    E = {w: act[w][0] for w in keep}
    P = {w: act[w][1] for w in keep}
    Z = {w: zipf_frequency(w, "en") for w in keep}
    # a word absent from wordfreq would read as maximally rare and skew the audit
    missing = [w for w in keep if Z[w] == 0.0]
    if missing:
        print(f"  dropping {len(missing)} words with no frequency data: {missing[:6]}")
        keep = [w for w in keep if Z[w] > 0.0]

    pool_pe = abs(float(np.corrcoef([P[w] for w in keep], [E[w] for w in keep])[0, 1]))
    print(f"  |corr(P, E)| across the surviving pool = {pool_pe:.3f}"
          f"  -> targeting RESIDUALIZED potency")
    T = residualize(P, E)

    chosen = greedy_select(keep, T, E, Z, TARGET_N, MAX_ABS_CORR)
    t = np.array([T[w] for w in chosen])
    e = np.array([E[w] for w in chosen])
    z = np.array([Z[w] for w in chosen])
    ce = abs(float(np.corrcoef(t, e)[0, 1]))
    cz = abs(float(np.corrcoef(t, z)[0, 1]))

    print(f"\nselected n={len(chosen)}  target range {t.min():+.2f}..{t.max():+.2f} "
          f"(sd {t.std():.2f})")
    print(f"gate 2 — |corr(target, Evaluation)| = {ce:.3f}   "
          f"{'PASS' if ce <= MAX_ABS_CORR else 'FAIL'}")
    print(f"gate 3 — |corr(target, zipf)|       = {cz:.3f}   "
          f"{'PASS' if cz <= MAX_ABS_CORR else 'FAIL'}")

    # baseline: what the ORIGINAL curated lexicon scores on the same gates
    from turnstyle.frame_library import CANONICAL_FRAMES
    orig = CANONICAL_FRAMES["potency"]["data"]
    oz = {w: zipf_frequency(w, "en") for w in orig}
    ov = np.array([orig[w] for w in orig])
    ozv = np.array([oz[w] for w in orig])
    o_keep = shared_single_token_words(toks, list(orig), TEMPLATE)
    # Does the ORIGINAL curated lexicon pass the same gates? It was never checked
    # against Evaluation, and if it fails, residualizing is an upgrade rather than a
    # workaround forced by tokenization.
    o_e = {w: act[w][0] for w in orig if w in act}
    print(f"\noriginal curated potency: n={len(orig)}, "
          f"shared-single-token {len(o_keep)}/{len(orig)}, "
          f"|corr(label, zipf)| = {abs(float(np.corrcoef(ov, ozv)[0, 1])):.3f}")
    if len(o_e) >= 6:
        common = [w for w in orig if w in o_e]
        oc = abs(float(np.corrcoef([orig[w] for w in common],
                                   [o_e[w] for w in common])[0, 1]))
        print(f"  |corr(curated label, ACT Evaluation)| = {oc:.3f} over {len(common)} "
              f"words it shares with ACT   {'PASS' if oc <= MAX_ABS_CORR else 'FAIL'}")

    print(f"\n{'word':14s} {'target':>7s} {'rawP':>6s} {'E':>6s} {'zipf':>6s}")
    for w in sorted(chosen, key=lambda w: T[w]):
        print(f"  {w:12s} {T[w]:>+7.2f} {P[w]:>+6.2f} {E[w]:>+6.2f} {Z[w]:>6.2f}")

    out = {"frame": "potency_stok", "template": TEMPLATE,
           "source": "ACT act_mods.csv, P residualized on E over the candidate pool",
           "pool_corr_P_E": pool_pe,
           "gates": {"shared_single_token": True, "corr_target_E": ce,
                     "corr_target_zipf": cz, "max_abs_corr": MAX_ABS_CORR},
           "data": {w: round(float(T[w]), 3) for w in chosen}}
    p_out = RESULTS / "stok_lexicon_potency.json"
    p_out.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {p_out}")
    if ce > MAX_ABS_CORR or cz > MAX_ABS_CORR:
        raise SystemExit("a gate FAILED — do not fit this lexicon as-is")


if __name__ == "__main__":
    main()
