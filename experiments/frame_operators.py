"""
frame_operators.py — do modifier words act as GEOMETRIC OPERATORS on frame
coordinates?

The frame family gives every adjective a scalar coordinate per frame. The
compositional question: is c("not heavy") a lawful function of c("heavy") —
and is that law the SAME across frames (an operator), or per-frame idiom?
Negation has bitten simpler probes repeatedly (certainty needed prefix
balancing; the DAG polarity heads beat neg-regex precisely because surface
negation is unreliable), so a measured account of what negation does to
coordinates is the framework we want.

Design — paired within-word: each adjective from each frame's own lexicon is
read bare and under each modifier in the SAME predicative template
("It is {m}{w}." etc.), last-subword readout. Word identity is constant
across conditions, so zipf frequency cannot produce an operator effect
(the standing frequency kill-law is satisfied by construction; the un-
morphology section, where the surface word changes, is flagged separately).
Axis = ridge refit on the BARE predicative states at the library frame's
layer (removes template domain-shift); the bundled library axis is kept as
a sanity column.

Pre-registered predictions:
  P1 control: "also" (position-shifting, degree-neutral) slope ~ 1 on every
     frame — outside [0.7, 1.3] the instrument is broken, stop.
  P2 intensifiers are gains: extremely/very slope > 1, with a monotone
     ladder extremely >= very >= quite > somewhat >= slightly >= barely,
     the lower rungs < 1.
  P3 negation is an ATTENUATED REFLECTION: slope negative, |slope| < 1
     ("not huge" != "tiny").
  P4 litotes asymmetry: negating high-pole words moves further than
     negating low-pole words ("not good" is worse than "not bad" is good).
  P5 operator consistency: per-operator slope sd across frames is small
     relative to its distance from 1 (an operator, not 8 idioms).
  P6 negation's layer profile: slope near +1 in the earliest layers (the
     adjective token not yet contextualized), turning negative by the
     frame's decodable layer — operator application has a depth.
  P7 specificity: "very"/"not" deltas land on the word's own frame, ~0 on
     the other frames (in coord_scale units).
  P8 morphological negation lands where syntactic negation lands:
     c(un-X) ~ c(not X).

Run:  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python \
        experiments/frame_operators.py
Writes experiments/results/frame_operators.json.
"""
from __future__ import annotations
import json, os, sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

MODEL = "HuggingFaceTB/SmolLM2-1.7B-Instruct"

# predicative templates with a modifier slot directly before the adjective
TEMPLATES = {
    "opinion": "It is {m}{w}.",
    "size": "It is {m}{w}.",
    "age": "It is {m}{w}.",
    "weight": "It is {m}{w}.",
    "warmth": "They are {m}{w}.",
    "competence": "They are {m}{w}.",
    "potency": "They are {m}{w}.",
    "certainty": "It is {m}{w} that it happened.",
}
OPS = ["", "also ", "extremely ", "very ", "quite ", "somewhat ",
       "slightly ", "barely ", "not ", "more ", "less "]
CONTROL_OP = "also "
PROFILE_OPS = ["also ", "very ", "not "]          # layer-profile subjects

UN_PAIRS = {  # frame -> [(base, morphological negation)]
    "warmth": [("kind", "unkind"), ("friendly", "unfriendly"),
               ("pleasant", "unpleasant")],
    "certainty": [("likely", "unlikely"), ("certain", "uncertain"),
                  ("clear", "unclear")],
    "opinion": [("fair", "unfair"), ("happy", "unhappy")],
}


def ridge_axis(X: np.ndarray, y: np.ndarray, alpha: float = 10.0):
    """Standardization + unit ridge direction oriented toward high y.
    Returns (mu, sd, dir) so any state can be projected consistently."""
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Z = (X - mu) / sd
    yc = y - y.mean()
    a = np.linalg.solve(Z @ Z.T + alpha * np.eye(len(y)), yc)
    d = Z.T @ a
    d /= np.linalg.norm(d) + 1e-12
    if (Z @ d) @ yc < 0:
        d = -d
    return mu, sd, d


def project(states: np.ndarray, mu, sd, d) -> np.ndarray:
    return ((states - mu) / sd) @ d


def ols(x: np.ndarray, y: np.ndarray):
    """slope, intercept, r2 of y on x."""
    vx = x.var()
    if vx < 1e-12:
        return 0.0, float(y.mean()), 0.0
    b = float(np.cov(x, y, bias=True)[0, 1] / vx)
    a = float(y.mean() - b * x.mean())
    r = float(np.corrcoef(x, y)[0, 1])
    return b, a, r * r


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from turnstyle.frame_library import load_library

    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL)
    mdl = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.float16).to(dev)
    lib = load_library(mdl)
    assert lib is not None, "no frame library for this fingerprint"
    frames = {n: lib.frames[n] for n in TEMPLATES if n in lib.frames}
    print(f"frames: {list(frames)}", flush=True)

    def collect(sent: str, w: str) -> np.ndarray:
        """(L+1, H) hidden states at the last subword of w."""
        cs = sent.rfind(w); ce = cs + len(w)
        enc = tok(sent, return_offsets_mapping=True, return_tensors="pt")
        offs = enc.pop("offset_mapping")[0].tolist()
        enc = {k: v.to(dev) for k, v in enc.items()}
        with torch.no_grad():
            hs = mdl(**enc, output_hidden_states=True).hidden_states
        stk = torch.stack(hs, 0)[:, 0, :, :]                # (L+1, T, H)
        idxs = [k for k, (s, e) in enumerate(offs) if e > cs and s < ce] or [-1]
        return stk[:, idxs[-1], :].float().cpu().numpy().astype(np.float16)

    # ── collection: every (frame, word, op) ─────────────────────────────────
    acts: dict = {}
    for fname, fr in frames.items():
        tmpl = TEMPLATES[fname]
        words = list(fr.data)
        for w in words:
            for op in OPS:
                acts[(fname, w, op)] = collect(tmpl.format(m=op, w=w), w)
        print(f"  collected {fname}: {len(words)} words x {len(OPS)} ops",
              flush=True)
    n_layers = next(iter(acts.values())).shape[0]

    results: dict = {"model": MODEL, "ops": OPS, "frames": {}, }

    # ── per-frame refit axis at the library layer; slopes per op ────────────
    print("\n=== operator slopes (refit axis @ frame layer; y = c(op w), "
          "x = c(w)) ===")
    slopes: dict = {op: {} for op in OPS if op}
    axes: dict = {}
    hdr = "frame        lib_r " + " ".join(f"{op.strip():>9s}" for op in OPS if op)
    print(hdr)
    for fname, fr in frames.items():
        words = list(fr.data)
        y = np.array([fr.data[w] for w in words], float)
        L = fr.layer
        Xb = np.array([acts[(fname, w, "")][L] for w in words], float)
        mu, sd, d = ridge_axis(Xb, y)
        axes[fname] = (mu, sd, d, L)
        cb = project(Xb, mu, sd, d)
        cb -= cb.mean()
        # sanity: does the refit axis read the lexicon? does the LIB axis?
        r_fit = float(np.corrcoef(cb, y)[0, 1])
        lib_proj = np.array([fr.project(acts[(fname, w, "")][L].astype(float))
                             for w in words])
        r_lib = float(np.corrcoef(lib_proj, y)[0, 1])
        row = {"r_refit": r_fit, "r_lib": r_lib, "layer": L, "ops": {}}
        cells = []
        for op in OPS:
            if not op:
                continue
            Xm = np.array([acts[(fname, w, op)][L] for w in words], float)
            cm = project(Xm, mu, sd, d) - project(Xb, mu, sd, d).mean()
            b, a, r2 = ols(cb, cm)
            slopes[op][fname] = b
            row["ops"][op.strip()] = {"slope": round(b, 3),
                                      "intercept": round(a, 3),
                                      "r2": round(r2, 3)}
            cells.append(f"{b:+6.2f}/{r2:.2f}"[:9].rjust(9))
        results["frames"][fname] = row
        print(f"{fname:12s} {r_lib:+.2f} " + " ".join(cells)
              + f"   (refit r {r_fit:+.2f}, L{L})")

    # ── operator consistency across frames ──────────────────────────────────
    print("\n=== operator signatures (mean slope +/- sd across frames) ===")
    sig = {}
    for op in slopes:
        v = np.array(list(slopes[op].values()))
        sig[op.strip()] = {"mean": round(float(v.mean()), 3),
                           "sd": round(float(v.std()), 3)}
        print(f"  {op.strip():>10s}: {v.mean():+5.2f} +/- {v.std():.2f}")
    results["signatures"] = sig

    # ── negation asymmetry (litotes) ─────────────────────────────────────────
    print("\n=== negation asymmetry (per-pole slope of c(not w) vs c(w)) ===")
    asym = {}
    hi_all, lo_all = [], []
    for fname, fr in frames.items():
        words = list(fr.data)
        y = np.array([fr.data[w] for w in words], float)
        mu, sd, d, L = axes[fname]
        cb = project(np.array([acts[(fname, w, "")][L] for w in words], float),
                     mu, sd, d)
        cn = project(np.array([acts[(fname, w, "not ")][L] for w in words],
                              float), mu, sd, d)
        cb -= cb.mean(); cn -= cn.mean()
        hi, lo = y > 0, y < 0
        for m, box in ((hi, hi_all), (lo, lo_all)):
            box.extend(zip(cb[m], cn[m]))
        bh = ols(cb[hi], cn[hi])[0] if hi.sum() > 2 else float("nan")
        bl = ols(cb[lo], cn[lo])[0] if lo.sum() > 2 else float("nan")
        asym[fname] = {"hi_slope": round(bh, 3), "lo_slope": round(bl, 3)}
        print(f"  {fname:12s} hi-pole {bh:+5.2f}   lo-pole {bl:+5.2f}")
    for tag, box in (("HI", hi_all), ("LO", lo_all)):
        xs, ys = map(np.array, zip(*box))
        b, _, r2 = ols(xs, ys)
        asym[f"pooled_{tag.lower()}"] = round(b, 3)
        print(f"  pooled {tag}: slope {b:+5.2f} (r2 {r2:.2f}, n={len(xs)})")
    results["negation_asymmetry"] = asym

    # ── layer profile: where do operators apply? (refit per layer) ──────────
    print("\n=== layer profile (pooled slope across frames per layer) ===")
    prof = {op.strip(): [] for op in PROFILE_OPS}
    for L in range(n_layers):
        pooled = {op: [] for op in PROFILE_OPS}
        for fname, fr in frames.items():
            words = list(fr.data)
            y = np.array([fr.data[w] for w in words], float)
            Xb = np.array([acts[(fname, w, "")][L] for w in words], float)
            mu, sd, d = ridge_axis(Xb, y)
            cb = project(Xb, mu, sd, d); cb -= cb.mean()
            if np.corrcoef(cb, y)[0, 1] < 0.5:
                continue                    # axis unreadable at this layer
            for op in PROFILE_OPS:
                Xm = np.array([acts[(fname, w, op)][L] for w in words], float)
                cm = project(Xm, mu, sd, d); cm -= cm.mean()
                pooled[op].append(ols(cb, cm)[0])
        for op in PROFILE_OPS:
            prof[op.strip()].append(
                round(float(np.mean(pooled[op])), 3) if pooled[op] else None)
        if L % 2 == 0:
            cells = " ".join(
                f"{op.strip()}={prof[op.strip()][-1]:+.2f}"
                if prof[op.strip()][-1] is not None else f"{op.strip()}=  --"
                for op in PROFILE_OPS)
            print(f"  L{L:2d}: {cells}")
    results["layer_profile"] = prof

    # ── specificity: do "very"/"not" move only the word's own frame? ────────
    print("\n=== specificity (|delta| in coord_scale units: own frame vs "
          "others) ===")
    spec = {}
    for op in ("very ", "not "):
        own, other = [], []
        for fname, fr in frames.items():
            for w in fr.data:
                for gname, gr in frames.items():
                    Lg = gr.layer
                    db = gr.project(acts[(fname, w, "")][Lg].astype(float))
                    dm = gr.project(acts[(fname, w, op)][Lg].astype(float))
                    delta = abs(dm - db) / (gr.coord_scale or 1.0)
                    (own if gname == fname else other).append(delta)
        spec[op.strip()] = {"own": round(float(np.mean(own)), 3),
                            "other": round(float(np.mean(other)), 3)}
        print(f"  {op.strip():>5s}: own {np.mean(own):.2f}  "
              f"other {np.mean(other):.2f}  "
              f"ratio {np.mean(own)/max(np.mean(other),1e-9):.1f}x")
    results["specificity"] = spec

    # ── morphological vs syntactic negation ──────────────────────────────────
    print("\n=== un- vs not (coordinates on the pair's frame) ===")
    unrows = []
    for fname, pairs in UN_PAIRS.items():
        fr = frames[fname]; tmpl = TEMPLATES[fname]
        mu, sd, d, L = axes[fname]
        # center from the frame's bare lexicon
        cb_lex = project(np.array([acts[(fname, w, "")][L] for w in fr.data],
                                  float), mu, sd, d)
        c0 = cb_lex.mean()
        for base, un in pairs:
            sb = collect(tmpl.format(m="", w=base), base)[L].astype(float)
            su = collect(tmpl.format(m="", w=un), un)[L].astype(float)
            sn = collect(tmpl.format(m="not ", w=base), base)[L].astype(float)
            cbase = float(project(sb[None], mu, sd, d)[0] - c0)
            cun = float(project(su[None], mu, sd, d)[0] - c0)
            cnot = float(project(sn[None], mu, sd, d)[0] - c0)
            unrows.append({"frame": fname, "base": base, "un": un,
                           "c_base": round(cbase, 2), "c_un": round(cun, 2),
                           "c_not": round(cnot, 2)})
            print(f"  {fname:10s} {base:>9s} {cbase:+5.2f} | "
                  f"un- {cun:+5.2f} | not {cnot:+5.2f}")
    cu = np.array([r["c_un"] for r in unrows])
    cn = np.array([r["c_not"] for r in unrows])
    cb_ = np.array([r["c_base"] for r in unrows])
    print(f"  corr(un, not) = {np.corrcoef(cu, cn)[0,1]:+.2f}   "
          f"corr(un, -base) = {np.corrcoef(cu, -cb_)[0,1]:+.2f}")
    results["un_pairs"] = unrows

    outp = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "results", "frame_operators.json")
    with open(outp, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {outp}")


if __name__ == "__main__":
    main()
