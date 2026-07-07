"""
frame_operators_negation.py — WHAT negation does, given frame_operators.py's
surprise: "not" is never a reflection; it is a layer-gated COLLAPSE + SHIFT.

frame_operators.py found: (1) affine fits c(not w) = b*c(w) + a are excellent,
but b is always POSITIVE; late-layer frames get b << 1 with a < 0, and
certainty is the limit case b=0.03, r2=0.00 — c(not w) is a CONSTANT.
(2) The effect is layer-gated (profile bottoms ~L14); early-read frames
(age L2) are negation-BLIND. (3) un- words genuinely reflect. This script
adjudicates the mechanism.

Pre-registered questions:
  Q1 SHARED VECTOR: is the negation delta state(not w) - state(w) one
     direction in hidden space shared across frames (cos of per-frame mean
     deltas at a common layer), i.e. "not" adds a generic hedge vector?
     And per frame: how much delta variance is (i) the mean shift,
     (ii) the frame-axis component prop. to -c(w) (reflection share),
     (iii) residual?
  Q2 DOUBLE NEGATION: if negation is collapse-to-hedge, "not unkind" lands
     at the same hedged constant as "not kind" (WRONG semantically —
     litotes says mildly-kind). If the model composes, c(not un-X) > 0.
  Q3 END ANCHOR: at the sentence-final token, is the bare frame readable,
     and does negation show different structure there (composition may
     live at the aggregation site, not the adjective token)?
  Q4 corrected litotes: consistent centering (bare-lexicon mean is the
     origin for ALL conditions); per-pole LANDINGS, not own-centered slopes.

Run:  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python \
        experiments/frame_operators_negation.py
Writes experiments/results/frame_operators_negation.json.
"""
from __future__ import annotations
import json, os, sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from frame_operators import TEMPLATES, UN_PAIRS, ridge_axis, project, ols, MODEL

COMMON_L = 14          # negation-effect trough from the layer profile


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from turnstyle.frame_library import load_library

    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL)
    mdl = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.float16).to(dev)
    lib = load_library(mdl)
    frames = {n: lib.frames[n] for n in TEMPLATES if n in lib.frames}

    def collect2(sent: str, w: str) -> np.ndarray:
        """(L+1, 2, H): anchor 0 = last subword of w, anchor 1 = final token."""
        cs = sent.rfind(w); ce = cs + len(w)
        enc = tok(sent, return_offsets_mapping=True, return_tensors="pt")
        offs = enc.pop("offset_mapping")[0].tolist()
        enc = {k: v.to(dev) for k, v in enc.items()}
        with torch.no_grad():
            hs = mdl(**enc, output_hidden_states=True).hidden_states
        stk = torch.stack(hs, 0)[:, 0, :, :]
        idxs = [k for k, (s, e) in enumerate(offs) if e > cs and s < ce] or [-1]
        both = torch.stack([stk[:, idxs[-1], :], stk[:, -1, :]], 1)
        return both.float().cpu().numpy().astype(np.float16)

    acts: dict = {}
    for fname, fr in frames.items():
        tmpl = TEMPLATES[fname]
        for w in fr.data:
            for op in ("", "not "):
                acts[(fname, w, op)] = collect2(tmpl.format(m=op, w=w), w)
        print(f"  collected {fname}", flush=True)
    n_layers = next(iter(acts.values())).shape[0]

    results: dict = {"model": MODEL}

    # ── Q4: corrected litotes — one origin (bare-lexicon mean), landings ────
    print("\n=== Q4 landings (origin = bare lexicon mean; adj token, frame "
          "layer) ===")
    landings = {}
    for fname, fr in frames.items():
        words = list(fr.data)
        y = np.array([fr.data[w] for w in words], float)
        L = fr.layer
        Xb = np.array([acts[(fname, w, "")][L, 0] for w in words], float)
        Xn = np.array([acts[(fname, w, "not ")][L, 0] for w in words], float)
        mu, sd, d = ridge_axis(Xb, y)
        cb = project(Xb, mu, sd, d); c0 = cb.mean()
        cb -= c0
        cn = project(Xn, mu, sd, d) - c0
        s = float(cb.std()) or 1.0
        hi, lo = y > 0, y < 0
        # sd ratio separates true collapse (low output variance) from mere
        # unpredictability (slope->0 with high residual variance)
        sd_ratio = float(cn.std() / s)
        landings[fname] = {
            "hi_bare": round(float(cb[hi].mean() / s), 2),
            "hi_not": round(float(cn[hi].mean() / s), 2),
            "lo_bare": round(float(cb[lo].mean() / s), 2),
            "lo_not": round(float(cn[lo].mean() / s), 2),
            "sd_ratio": round(sd_ratio, 2),
            "layer": L,
        }
        r = landings[fname]
        print(f"  {fname:12s} L{L:2d}  hi {r['hi_bare']:+5.2f} -> "
              f"{r['hi_not']:+5.2f}   lo {r['lo_bare']:+5.2f} -> "
              f"{r['lo_not']:+5.2f}   sd(not)/sd(bare) {sd_ratio:.2f}")
    results["landings"] = landings

    # ── Q1: negation-delta geometry ─────────────────────────────────────────
    print("\n=== Q1 negation-delta decomposition (adj token) ===")
    q1 = {}
    mean_deltas = {}
    for fname, fr in frames.items():
        words = list(fr.data)
        y = np.array([fr.data[w] for w in words], float)
        L = fr.layer
        Xb = np.array([acts[(fname, w, "")][L, 0] for w in words], float)
        Xn = np.array([acts[(fname, w, "not ")][L, 0] for w in words], float)
        D = Xn - Xb
        mu, sd, d = ridge_axis(Xb, y)
        cb = project(Xb, mu, sd, d); cb -= cb.mean()
        # shares of delta energy: mean vector vs frame-axis vs residual
        m = D.mean(0)
        tot = float((D ** 2).sum())
        sh_mean = float((np.outer(np.ones(len(D)), m) ** 2).sum()) / tot
        Dr = D - m
        # frame-axis component of the residual, and its relation to -c(w)
        d_raw = d / (sd + 1e-9)                       # axis back in raw space
        d_raw /= np.linalg.norm(d_raw) + 1e-12
        comp = Dr @ d_raw
        sh_axis = float((comp ** 2).sum()) / tot
        # NOTE: corr(axis-comp, -c) is inflated by regression-to-mean (any
        # attenuation, incl. noise, produces it); the load-bearing evidence
        # for proportional erasure is the affine fit's HIGH r2 with slope<1
        # in frame_operators.py, not this column.
        r_reflect = float(np.corrcoef(comp, -cb)[0, 1]) if cb.std() > 0 else 0.0
        q1[fname] = {"share_mean": round(sh_mean, 3),
                     "share_axis_resid": round(sh_axis, 3),
                     "r_axiscomp_vs_negc": round(r_reflect, 3)}
        print(f"  {fname:12s} mean-shift {sh_mean:.0%}  axis-resid "
              f"{sh_axis:.1%}  corr(axis comp, -c) {r_reflect:+.2f}")
        # common-layer mean delta for cross-frame cos
        XbC = np.array([acts[(fname, w, "")][COMMON_L, 0] for w in words], float)
        XnC = np.array([acts[(fname, w, "not ")][COMMON_L, 0] for w in words], float)
        mean_deltas[fname] = (XnC - XbC).mean(0)
    names = list(mean_deltas)
    M = np.array([[float(np.dot(mean_deltas[a], mean_deltas[b]) /
                         (np.linalg.norm(mean_deltas[a]) *
                          np.linalg.norm(mean_deltas[b]) + 1e-12))
                   for b in names] for a in names])
    off = M[~np.eye(len(names), dtype=bool)]
    print(f"  cross-frame cos of mean deltas @L{COMMON_L}: "
          f"mean {off.mean():.2f}  min {off.min():.2f}  max {off.max():.2f}")
    results["q1"] = {"per_frame": q1, "crossframe_cos_mean": round(float(off.mean()), 3),
                     "crossframe_cos_min": round(float(off.min()), 3)}

    # ── Q2: double negation ──────────────────────────────────────────────────
    print("\n=== Q2 double negation (not un-X) ===")
    q2 = []
    for fname, pairs in UN_PAIRS.items():
        fr = frames[fname]; tmpl = TEMPLATES[fname]
        words = list(fr.data)
        y = np.array([fr.data[w] for w in words], float)
        L = fr.layer
        Xb = np.array([acts[(fname, w, "")][L, 0] for w in words], float)
        mu, sd, d = ridge_axis(Xb, y)
        c0 = project(Xb, mu, sd, d).mean()
        s = float(project(Xb, mu, sd, d).std()) or 1.0
        for base, un in pairs:
            row = {"frame": fname, "base": base}
            for label, m, w in (("base", "", base), ("un", "", un),
                                ("not_base", "not ", base),
                                ("not_un", "not ", un)):
                st = collect2(tmpl.format(m=m, w=w), w)[L, 0].astype(float)
                row[label] = round(float((project(st[None], mu, sd, d)[0]
                                          - c0) / s), 2)
            q2.append(row)
            print(f"  {fname:10s} {base:>9s} {row['base']:+5.2f} | un "
                  f"{row['un']:+5.2f} | not-base {row['not_base']:+5.2f} | "
                  f"NOT-UN {row['not_un']:+5.2f}")
    nu = np.array([r["not_un"] for r in q2])
    nb = np.array([r["not_base"] for r in q2])
    un = np.array([r["un"] for r in q2])
    print(f"  mean: un {un.mean():+.2f} -> not-un {nu.mean():+.2f} "
          f"(composition predicts > 0; collapse predicts ~ not-base "
          f"{nb.mean():+.2f})")
    results["q2"] = q2

    # ── Q3: end-token anchor ─────────────────────────────────────────────────
    # layer selection must be HELD-OUT: n~15 words in 2048 dims fits any layer
    # in-sample at r=1.0, so in-sample selection is arbitrary (v1 bug).
    from turnstyle.frame_library import _cv_r
    print("\n=== Q3 end-token anchor (best CV layer per frame; bare cv_r, "
          "then not-slope there) ===")
    q3 = {}
    for fname, fr in frames.items():
        words = list(fr.data)
        y = np.array([fr.data[w] for w in words], float)
        best = (-9, -1)
        for L in range(n_layers):
            Xb = np.array([acts[(fname, w, "")][L, 1] for w in words], float)
            r = _cv_r(Xb, y)
            if r == r and r > best[0]:
                best = (r, L)
        r, L = best
        Xb = np.array([acts[(fname, w, "")][L, 1] for w in words], float)
        Xn = np.array([acts[(fname, w, "not ")][L, 1] for w in words], float)
        mu, sd, d = ridge_axis(Xb, y)
        cb = project(Xb, mu, sd, d); c0 = cb.mean(); cb -= c0
        cn = project(Xn, mu, sd, d) - c0
        b, a, r2 = ols(cb, cn)
        s = float(cb.std()) or 1.0
        q3[fname] = {"bare_cv_r": round(r, 3), "layer": L,
                     "not_slope": round(b, 3),
                     "not_shift_sd": round(a / s, 2), "r2": round(r2, 3),
                     "sd_ratio": round(float(cn.std() / s), 2)}
        print(f"  {fname:12s} bare cv_r {r:+.2f} @L{L:2d}   not: slope "
              f"{b:+5.2f}  shift {a / s:+5.2f} sd  r2 {r2:.2f}  "
              f"sdr {cn.std() / s:.2f}")
    results["q3"] = q3

    outp = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "results", "frame_operators_negation.json")
    with open(outp, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {outp}")


if __name__ == "__main__":
    main()
