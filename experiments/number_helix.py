"""
number_helix.py — adjudicate the L10 number loop (homology_sweep, z=22.7).

Candidate mechanisms for the two..twelve H1 loop:
  A. DECADE CONGRUENCE (helix): units-digit circle (period 10) + weak ordinal
     axis — eleven sits near one, twelve near two (cf. Kantamneni & Tegmark,
     numbers-as-helix).
  B. SURFACE MORPHOLOGY: twelve~two share "tw"; eleven~seven share "even";
     thir/teen transparently contain the unit morpheme. A lexical fake.
  C. Plain line + cluster-boundary artifact (no residual mod-10 signal).

The adversarial pair: for "eleven", congruence predicts close-to-ONE while
morphology predicts close-to-SEVEN. They cannot both win.

Per layer, per (template x pooling):
  1. cong_ratio: mean d(i, i+10) over the 10 congruent pairs vs mean d over
     non-congruent pairs with ordinal gap in {8,9,11,12}; z vs B random
     equal-size draws from the control pairs.
  2. star distances: d(eleven,one) vs d(eleven,seven)/d(eleven,six),
     d(twelve,two) vs d(twelve,seven).
  3. RSA (Spearman on upper triangles): line |i-j|, circle period 10 (and
     control periods 7, 12), helix(rho) over a pitch grid.
  4. SEMI-PARTIAL circ10: correlate D with the circle model AFTER removing
     the line model (OLS residual) — the null that preserves the line and
     tests only the loop (the CLAUDE.md rule). Permutation p over word->value
     reassignment (rebuilds both models per draw).
  5. LOO dual-ridge readout of t, cos(2*pi*t/10), sin(2*pi*t/10).

Run:  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python \
        experiments/number_helix.py [model_id]
Writes experiments/results/number_helix.json.
"""
from __future__ import annotations
import json, os, sys

import numpy as np

WORDS = ["one", "two", "three", "four", "five", "six", "seven", "eight",
         "nine", "ten", "eleven", "twelve", "thirteen", "fourteen",
         "fifteen", "sixteen", "seventeen", "eighteen", "nineteen", "twenty"]
VAL = {w: i + 1 for i, w in enumerate(WORDS)}
TEMPLATES = {"neutral": "They talked about {w}.",
             "number": "The number {w} was written on the board."}
RHO_GRID = [0.25, 0.5, 1.0, 2.0, 4.0]
B_CONG = 1000
B_PERM = 500


def collect(model, tok, device, words, template):
    """Last-subword AND mean-over-word hidden states, all layers."""
    import torch
    out = {}
    for w in words:
        sent = template.format(w=w)
        cs = sent.rfind(w); ce = cs + len(w)
        enc = tok(sent, return_offsets_mapping=True, return_tensors="pt")
        offs = enc.pop("offset_mapping")[0].tolist()
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            hs = model(**enc, output_hidden_states=True).hidden_states
        stk = torch.stack(hs, 0)[:, 0, :, :]
        idxs = [k for k, (s, e) in enumerate(offs) if e > cs and s < ce] or [-1]
        out[w] = (stk[:, idxs[-1], :].float().cpu().numpy(),
                  stk[:, idxs, :].mean(1).float().cpu().numpy())
    return out


def pdist(X):
    sq = (X ** 2).sum(1)
    return np.sqrt(np.maximum(sq[:, None] + sq[None, :] - 2 * X @ X.T, 0))


def upper(M):
    return M[np.triu_indices_from(M, k=1)]


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    return float((ra @ rb) / (np.linalg.norm(ra) * np.linalg.norm(rb) + 1e-12))


def model_dists(t):
    """Model distance matrices for a value assignment t (len n)."""
    dt = np.abs(t[:, None] - t[None, :]).astype(float)
    out = {"line": dt}
    for P in (7, 10, 12):
        th = 2 * np.pi * t / P
        c, s = np.cos(th), np.sin(th)
        out[f"circ{P}"] = np.sqrt((c[:, None] - c[None, :]) ** 2
                                  + (s[:, None] - s[None, :]) ** 2)
    return out


def helix_dist(t, rho):
    th = 2 * np.pi * t / 10.0
    pts = np.stack([rho * th, np.cos(th), np.sin(th)], 1)
    return pdist(pts)


def semi_partial(Du, t):
    """Spearman(circ10, D residual after OLS removal of line)."""
    md = model_dists(t)
    lin, c10 = upper(md["line"]), upper(md["circ10"])
    A = np.stack([lin, np.ones_like(lin)], 1)
    beta, *_ = np.linalg.lstsq(A, Du, rcond=None)
    return spearman(Du - A @ beta, c10)


def loo_ridge_r(X, y, lam=10.0):
    """LOO dual-ridge correlation(pred, y)."""
    n = X.shape[0]
    Xc = X - X.mean(0)
    K = Xc @ Xc.T
    preds = np.empty(n)
    for i in range(n):
        tr = np.array([j for j in range(n) if j != i])
        Kt = K[np.ix_(tr, tr)] + lam * np.eye(n - 1)
        a = np.linalg.solve(Kt, y[tr] - y[tr].mean())
        preds[i] = K[i, tr] @ a + y[tr].mean()
    return float(np.corrcoef(preds, y)[0, 1])


def analyze_layer(X, rng):
    t = np.array([VAL[w] for w in WORDS], dtype=float)
    X = (X - X.mean(0)) / (X.std(0) + 1e-6)
    D = pdist(X)
    Du = upper(D)
    wi = {w: i for i, w in enumerate(WORDS)}

    # 1. congruence ratio: gap-10 pairs vs non-congruent gap 8/9/11/12 pairs
    cong = [(i, i + 10) for i in range(1, 11)]
    ctrl = [(a, b) for a in range(1, 21) for b in range(a + 1, 21)
            if b - a in (8, 9, 11, 12)]
    d_cong = np.array([D[wi[WORDS[a - 1]], wi[WORDS[b - 1]]] for a, b in cong])
    d_ctrl = np.array([D[wi[WORDS[a - 1]], wi[WORDS[b - 1]]] for a, b in ctrl])
    ratio = float(d_cong.mean() / (d_ctrl.mean() + 1e-12))
    null = np.array([d_ctrl[rng.choice(len(d_ctrl), len(cong), replace=False)].mean()
                     for _ in range(B_CONG)])
    z = float((d_cong.mean() - null.mean()) / (null.std() + 1e-12))

    # 2. star distances (standardized-space euclidean)
    star = {k: round(float(D[wi[a], wi[b]]), 2) for k, (a, b) in {
        "d(eleven,one)": ("eleven", "one"),
        "d(eleven,seven)": ("eleven", "seven"),
        "d(eleven,six)": ("eleven", "six"),
        "d(twelve,two)": ("twelve", "two"),
        "d(twelve,seven)": ("twelve", "seven")}.items()}

    # 3. RSA
    md = model_dists(t)
    rsa = {k: round(spearman(Du, upper(v)), 3) for k, v in md.items()}
    hx = [(rho, spearman(Du, upper(helix_dist(t, rho)))) for rho in RHO_GRID]
    best_rho, best_hx = max(hx, key=lambda p: p[1])
    rsa["helix"] = round(best_hx, 3)

    # 4. semi-partial circ10 beyond line, permutation p
    sp_obs = semi_partial(Du, t)
    perm = np.array([semi_partial(Du, rng.permutation(t)) for _ in range(B_PERM)])
    sp_p = float((np.sum(perm >= sp_obs) + 1) / (B_PERM + 1))

    # 5. LOO readout of t, cos, sin
    th = 2 * np.pi * t / 10.0
    reads = {"t": loo_ridge_r(X, t),
             "cos10": loo_ridge_r(X, np.cos(th)),
             "sin10": loo_ridge_r(X, np.sin(th))}

    return {"cong_ratio": round(ratio, 3), "cong_z": round(z, 1),
            "star": star, "rsa": rsa, "best_rho": best_rho,
            "semi_partial_circ10": round(sp_obs, 3), "sp_p": round(sp_p, 4),
            "readout": {k: round(v, 3) for k, v in reads.items()}}


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    mid = sys.argv[1] if len(sys.argv) > 1 else "mistralai/Mistral-7B-Instruct-v0.3"
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(mid)
    mdl = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.float16).to(dev)

    for w in WORDS:
        print(f"  tok[{w}] = {tok.tokenize(TEMPLATES['neutral'].format(w=w))}",
              flush=True)

    results = {"model": mid, "conditions": {}}
    for tname, tmpl in TEMPLATES.items():
        acts = collect(mdl, tok, dev, WORDS, tmpl)
        n_layers = acts[WORDS[0]][0].shape[0]
        for pi, pname in enumerate(["last", "mean"]):
            key = f"{tname}/{pname}"
            print(f"\n=== {key} ===", flush=True)
            print("  L | ratio    z | line c10 c7 c12 helix(rho) | "
                  "sp_c10 (p) | read t/cos/sin | d(11,1) d(11,7) d(12,2)",
                  flush=True)
            rng = np.random.default_rng(0)
            layers = {}
            for L in range(n_layers):
                X = np.array([acts[w][pi][L] for w in WORDS], dtype=np.float32)
                r = analyze_layer(X, rng)
                layers[L] = r
                s, rd = r["star"], r["readout"]
                print(f" {L:2d} | {r['cong_ratio']:.3f} {r['cong_z']:5.1f} | "
                      f"{r['rsa']['line']:.2f} {r['rsa']['circ10']:.2f} "
                      f"{r['rsa']['circ7']:.2f} {r['rsa']['circ12']:.2f} "
                      f"{r['rsa']['helix']:.2f}({r['best_rho']}) | "
                      f"{r['semi_partial_circ10']:+.2f} ({r['sp_p']:.3f}) | "
                      f"{rd['t']:.2f}/{rd['cos10']:.2f}/{rd['sin10']:.2f} | "
                      f"{s['d(eleven,one)']:.1f} {s['d(eleven,seven)']:.1f} "
                      f"{s['d(twelve,two)']:.1f}", flush=True)
            results["conditions"][key] = layers

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "results", "number_helix.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
