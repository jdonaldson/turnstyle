"""
frame_operators_replicate.py — PRE-REGISTERED replication of the end-token
negation reflection (frame_operators_negation.py Q3, which was exploratory).

Claim under test: at the sentence-FINAL token (CV-selected layer per frame),
"not" acts as an attenuated REFLECTION (negative slope of c(not w) on c(w))
for warmth / size / competence / potency, while at the adjective token it
never does. Fresh predicative templates (past tense, different subjects);
same lexicons (they define the frames).

Pre-registered predictions (from run A, template set A):
  R1 not-slope at end token NEGATIVE for warmth, size, competence, potency
     (warmth the most negative).
  R2 not-slope ~0 (|b| < 0.15) for age and certainty (collapse, not
     reflection — certainty negation is non-compositional).
  R3 at the ADJECTIVE token, no frame goes negative (all slopes > 0).

Run:  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python \
        experiments/frame_operators_replicate.py
Writes experiments/results/frame_operators_replicate.json.
"""
from __future__ import annotations
import json, os, sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from frame_operators import ridge_axis, project, ols, MODEL

TEMPLATES_B = {
    "opinion": "That was {m}{w}.",
    "size": "That was {m}{w}.",
    "age": "That was {m}{w}.",
    "weight": "That was {m}{w}.",
    "warmth": "He was {m}{w}.",
    "competence": "He was {m}{w}.",
    "potency": "He was {m}{w}.",
    "certainty": "It was {m}{w} that they left.",
}
PRED_NEG = ("warmth", "size", "competence", "potency")   # R1
PRED_ZERO = ("age", "certainty")                          # R2


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from turnstyle.frame_library import load_library, _cv_r

    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL)
    mdl = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.float16).to(dev)
    lib = load_library(mdl)
    frames = {n: lib.frames[n] for n in TEMPLATES_B if n in lib.frames}

    def collect2(sent: str, w: str) -> np.ndarray:
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
        for w in fr.data:
            for op in ("", "not "):
                acts[(fname, w, op)] = collect2(
                    TEMPLATES_B[fname].format(m=op, w=w), w)
        print(f"  collected {fname}", flush=True)
    n_layers = next(iter(acts.values())).shape[0]

    results: dict = {"model": MODEL, "templates": TEMPLATES_B, "rows": {}}
    print("\nframe          END: cv_r @L   slope  r2   |  ADJ: slope (frame "
          "layer)")
    verdicts = []
    for fname, fr in frames.items():
        words = list(fr.data)
        y = np.array([fr.data[w] for w in words], float)
        # end anchor, CV-selected layer
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
        b_end, _, r2_end = ols(cb, cn)
        # adjective anchor at the library frame layer
        La = fr.layer
        Xb = np.array([acts[(fname, w, "")][La, 0] for w in words], float)
        Xn = np.array([acts[(fname, w, "not ")][La, 0] for w in words], float)
        mu, sd, d = ridge_axis(Xb, y)
        cb = project(Xb, mu, sd, d); cb -= cb.mean()
        cn = project(Xn, mu, sd, d) - project(Xb, mu, sd, d).mean()
        b_adj, _, _ = ols(cb, cn)
        if fname in PRED_NEG:
            ok = b_end < 0
            pred = "R1 neg"
        elif fname in PRED_ZERO:
            ok = abs(b_end) < 0.15
            pred = "R2 ~0"
        else:
            ok = None
            pred = "  --"
        verdicts.append((fname, pred, ok))
        results["rows"][fname] = {
            "end_cv_r": round(r, 3), "end_layer": L,
            "end_not_slope": round(b_end, 3), "end_r2": round(r2_end, 3),
            "adj_not_slope": round(b_adj, 3), "prediction": pred,
            "ok": ok}
        flag = "PASS" if ok else ("FAIL" if ok is False else "    ")
        print(f"  {fname:12s} {r:+.2f} @L{L:2d}  {b_end:+5.2f}  "
              f"{r2_end:.2f}  |  {b_adj:+5.2f}   {pred} {flag}")

    r3 = all(results["rows"][f]["adj_not_slope"] > 0 for f in frames)
    n_pass = sum(1 for _, _, ok in verdicts if ok)
    n_test = sum(1 for _, _, ok in verdicts if ok is not None)
    print(f"\nR1+R2: {n_pass}/{n_test} pass   R3 (all adj slopes > 0): "
          f"{'PASS' if r3 else 'FAIL'}")
    results["r3_all_adj_positive"] = r3

    outp = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "results", "frame_operators_replicate.json")
    with open(outp, "w") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {outp}")


if __name__ == "__main__":
    main()
