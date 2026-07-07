"""
torus_hunt.py — higher homology in the joint calendar representation.

ring_denoise established: factor rings (months, weekdays) are HOLLOW in their
top-2-PC readout plane MID-stack (H1r ~ 2 = synthetic-clean-circle magnitude),
and late-stack circularity is order-only. The next rung on the complexity
ladder is a PRODUCT: is the joint representation of "a {weekday} in {month}"
(7x12 = 84 cells) a TORUS (H1 rank 2 + H2), or do the factor rings not
compose?

Conditions:
  wd_x_mo   "a Monday in January" grid — both factors are known readout
            rings mid-stack -> torus candidate.
  wd_x_hr   "on Monday at three o'clock" grid — clock hours inherit the
            number LINE (late-geometry sweep), so the prediction is a
            CYLINDER (H1 rank 1, no H2). Contrast condition: if the
            instrument calls this a torus too, distrust the instrument.
  hue_sup   the ring_denoise loose end: hue words projected onto their
            SUPERVISED cos/sin readout plane (top-2 PCs missed it because
            hue's plane is not variance-dominant).

Per layer, per torus condition (cells averaged over K templates):
  - full-space spectro (expect contractible — full space is always filled)
  - top-4-PC projection spectro (unsupervised — strongest claim if it fires)
  - supervised 4D projection: ridge readout to [cos,sin]x[wd,mo], project
    onto the orthonormalized readout span, spectro maxdim=2. CAVEAT printed:
    the plane is selected by supervision; magnitude calibration + shuffle-z
    guard, but the unsupervised column is the trustworthy one.
  - factor MARGINALS: weekday states averaged over months x templates
    (96 samples/point) and vice versa (56/point) -> top-2-PC spectro.
    Extreme denoising: do the factor rings survive inside the composite?
  - RSA of the projected cloud vs the product (torus) metric.

Run:  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python \
        experiments/torus_hunt.py [model_id]
Writes experiments/results/torus_hunt.json.
"""
from __future__ import annotations
import json, os, sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from homology_resweep import spectro, pdist

WD = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday",
      "Sunday"]
MO = ["January", "February", "March", "April", "May", "June", "July",
      "August", "September", "October", "November", "December"]
HR = [f"{h} o'clock" for h in
      ["one", "two", "three", "four", "five", "six", "seven", "eight",
       "nine", "ten", "eleven", "twelve"]]
HUE = ["red", "orange", "yellow", "green", "cyan", "blue", "violet",
       "purple", "magenta", "pink"]
HUE_DEG = [0, 30, 60, 120, 180, 240, 270, 285, 300, 330]

T_WDMO = [
    "They met on a {a} in {b}.",
    "The deadline fell on a {a} in {b}.",
    "It happened on a {a} in {b}.",
    "She was born on a {a} in {b}.",
    "The store opened on a {a} in {b}.",
    "We arrived on a {a} in {b}.",
    "The letter was dated a {a} in {b}.",
    "His interview was on a {a} in {b}.",
]
T_WDHR = [
    "They met on {a} at {b}.",
    "The call is on {a} at {b}.",
    "It happened on {a} at {b}.",
    "Her shift starts on {a} at {b}.",
    "The train leaves on {a} at {b}.",
    "We arrived on {a} at {b}.",
    "The alarm rang on {a} at {b}.",
    "His interview is on {a} at {b}.",
]
T_HUE = [
    "They talked about {b}.", "The report mentioned {b}.",
    "She wrote {b} on the board.", "Everyone kept discussing {b}.",
    "He circled {b} on the list.", "A note about {b} was left behind.",
    "The word {b} appeared in the article.", "The label said {b}.",
    "I remember {b} clearly.", "The answer was {b}.",
    "The card had {b} printed on it.", "The teacher underlined {b}.",
    "Their favorite was {b}.", "The sign read {b}.",
    "We settled on {b}.", "That reminded her of {b}.",
    "The quiz asked about {b}.", "Nobody expected {b}.",
    "The sample was {b}.", "Its cover was {b}.",
]


def collect_grid(model, tok, device, pairs, templates):
    """States at (1) the last subword of the {b} span and (2) the final
    sentence token, same forward pass. -> {anchor: [n_pairs, T, L, H]}.
    The dual anchor guards against anchoring bias: reading at the {b} token
    structurally favors factor B's geometry."""
    import torch
    out_b, out_e = [], []
    for a, b in pairs:
        per_b, per_e = [], []
        for tmpl in templates:
            sent = tmpl.format(a=a, b=b)
            cs = sent.rfind(b); ce = cs + len(b)
            enc = tok(sent, return_offsets_mapping=True, return_tensors="pt")
            offs = enc.pop("offset_mapping")[0].tolist()
            enc = {k: v.to(device) for k, v in enc.items()}
            with torch.no_grad():
                hs = model(**enc, output_hidden_states=True).hidden_states
            stk = torch.stack(hs, 0)[:, 0, :, :]
            idxs = [k for k, (s, e) in enumerate(offs)
                    if e > cs and s < ce] or [-1]
            per_b.append(stk[:, idxs[-1], :].float().cpu().numpy())
            per_e.append(stk[:, -1, :].float().cpu().numpy())
        out_b.append(np.stack(per_b, 0))
        out_e.append(np.stack(per_e, 0))
    return {"b_tok": np.stack(out_b, 0), "end": np.stack(out_e, 0)}


def std(X):
    return (X - X.mean(0)) / (X.std(0) + 1e-6)


def upper(M):
    return M[np.triu_indices_from(M, k=1)]


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    return float((ra @ rb) / (np.linalg.norm(ra) * np.linalg.norm(rb) + 1e-12))


def top_pcs(X, k):
    Xc = X - X.mean(0)
    U, S, _ = np.linalg.svd(Xc, full_matrices=False)
    return (U[:, :k] * S[:k]).astype(np.float32)


def ridge_readout(X, Y, folds=5, lam_scale=1.0):
    """Dual ridge X->Y. Returns (proj [n,k] onto orthonormalized readout
    span, mean 5-fold CV r across targets)."""
    n = X.shape[0]
    Xc = X - X.mean(0)
    K = Xc @ Xc.T
    lam = lam_scale * np.trace(K) / n + 1e-6     # epsilon: K can be all-zero
    A = np.linalg.solve(K + lam * np.eye(n), Y - Y.mean(0))
    W = Xc.T @ A                                   # [H, k]
    Q, _ = np.linalg.qr(W)
    proj = (Xc @ Q).astype(np.float32)
    idx = np.arange(n); rs = []
    rng = np.random.default_rng(0)
    rng.shuffle(idx)
    for j in range(Y.shape[1]):
        preds = np.empty(n)
        for f in range(folds):
            te = idx[f::folds]; tr = np.setdiff1d(idx, te)
            Kt = K[np.ix_(tr, tr)]
            lam_f = lam_scale * np.trace(Kt) / len(tr) + 1e-6
            a = np.linalg.solve(Kt + lam_f * np.eye(len(tr)),
                                Y[tr, j] - Y[tr, j].mean())
            preds[te] = K[np.ix_(te, tr)] @ a + Y[tr, j].mean()
        rs.append(0.0 if preds.std() < 1e-12
                  else float(np.corrcoef(preds, Y[:, j])[0, 1]))
    return proj, float(np.mean(rs))


def torus_metric(na, nb):
    """Product circle metric over the (a,b) grid, both factors unit rings."""
    ia, ib = np.divmod(np.arange(na * nb), nb)
    def circd(i, P):
        th = 2 * np.pi * i / P
        c, s = np.cos(th), np.sin(th)
        return ((c[:, None] - c[None, :]) ** 2
                + (s[:, None] - s[None, :]) ** 2)
    return np.sqrt(circd(ia, na) + circd(ib, nb))


def run_torus(name, acts, na, nb, results, rng):
    """acts: {anchor: [na*nb, T, L, H]}.

    Torus verdict = STRUCTURAL, not projection-based: base ring hollow
    (margB) AND mean fiber hollow (margA) AND fibers consistent across the
    base (fibcons). The pc4/sup4 columns are descriptive: the 84 cells form
    tight clumps, so medNN-relative ratios there are scale-unstable, and
    per-dim shuffle nulls in low-D projections are harsh."""
    Ytor = np.stack([f(2 * np.pi * v / P) for v, P in
                     ((np.divmod(np.arange(na * nb), nb)[0], na),
                      (np.divmod(np.arange(na * nb), nb)[1], nb))
                     for f in (np.cos, np.sin)], 1)
    tm = upper(torus_metric(na, nb))
    for anchor, ac in acts.items():
        n_layers = ac.shape[2]
        print(f"\n=== {name} ({na}x{nb}, K={ac.shape[1]}, anchor={anchor}) "
              f"===", flush=True)
        print("  L | full | pc4 cls (H1r n1 n2) | sup4 cls (H1r n1 n2) cv | "
              "margA H1r/z | margB H1r/z | fibcons | torusRSA(pc4)",
              flush=True)
        rows = {}
        for L in range(n_layers):
            cells = ac[:, :, L, :].mean(1)             # [na*nb, H]
            Xa = std(cells)
            g = Xa.reshape(na, nb, -1)
            r_full = spectro(Xa, rng)
            P4 = top_pcs(Xa, 4)
            r_pc4 = spectro(P4, rng)
            S4, cv = ridge_readout(Xa, Ytor)
            r_sup = spectro(S4, rng)
            r_a = spectro(top_pcs(std(g.mean(1)), 2), rng)
            r_b = spectro(top_pcs(std(g.mean(0)), 2), rng)
            # fiber consistency: A-fiber shape (residual vs B-centroid),
            # rank-correlated across the base B
            resid = g - g.mean(0, keepdims=True)
            fd = [upper(pdist(resid[:, b, :])) for b in range(nb)]
            fc = [spearman(fd[i], fd[j])
                  for i in range(nb) for j in range(i + 1, nb)]
            fibcons = float(np.mean(fc))
            rsa = spearman(upper(pdist(P4)), tm)
            rows[L] = {"full": r_full["cls"], "pc4": r_pc4, "sup4": r_sup,
                       "cv": round(cv, 2), "margA": r_a, "margB": r_b,
                       "fibcons": round(fibcons, 2),
                       "torus_rsa_pc4": round(rsa, 2)}
            print(f" {L:2d} | {r_full['H1_ratio']:4.2f} | "
                  f"{r_pc4['cls']:12s} ({r_pc4['H1_ratio']:5.2f} "
                  f"{r_pc4['n1']} {r_pc4['n2']}) | "
                  f"{r_sup['cls']:12s} ({r_sup['H1_ratio']:5.2f} "
                  f"{r_sup['n1']} {r_sup['n2']}) {cv:+.2f} | "
                  f"{r_a['H1_ratio']:4.2f}/{str(r_a['z1']):>5s} | "
                  f"{r_b['H1_ratio']:4.2f}/{str(r_b['z1']):>5s} | "
                  f"{fibcons:+.2f} | {rsa:+.2f}", flush=True)
        results[f"{name}/{anchor}"] = rows


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    mid = sys.argv[1] if len(sys.argv) > 1 else "mistralai/Mistral-7B-Instruct-v0.3"
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(mid)
    mdl = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.float16).to(dev)
    rng = np.random.default_rng(0)
    results: dict = {"model": mid}

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "results", "torus_hunt.json")

    def dump():
        with open(out, "w") as f:
            json.dump(results, f, indent=2)

    pairs_wdmo = [(a, b) for a in WD for b in MO]
    acts = collect_grid(mdl, tok, dev, pairs_wdmo, T_WDMO)
    run_torus("wd_x_mo", acts, len(WD), len(MO), results, rng)
    dump()

    pairs_wdhr = [(a, b) for a in WD for b in HR]
    acts = collect_grid(mdl, tok, dev, pairs_wdhr, T_WDHR)
    run_torus("wd_x_hr", acts, len(WD), len(HR), results, rng)
    dump()

    # hue on its SUPERVISED plane
    acts = collect_grid(mdl, tok, dev, [("", b) for b in HUE], T_HUE)["b_tok"]
    th = np.deg2rad(np.array(HUE_DEG, dtype=float))
    Yh = np.stack([np.cos(th), np.sin(th)], 1)
    circ = np.sqrt((Yh[:, 0][:, None] - Yh[:, 0][None, :]) ** 2
                   + (Yh[:, 1][:, None] - Yh[:, 1][None, :]) ** 2)
    print(f"\n=== hue_sup (n={len(HUE)}, K={len(T_HUE)}) ===", flush=True)
    print("  L | sup2 cls (H1r z1) cv | circRSA(sup2)", flush=True)
    rows = {}
    for L in range(acts.shape[2]):
        Xa = std(acts[:, :, L, :].mean(1))
        S2, cv = ridge_readout(Xa, Yh)
        r = spectro(S2, rng)
        rsa = spearman(upper(pdist(S2)), upper(circ))
        rows[L] = {"sup2": r, "cv": round(cv, 2), "circ_rsa": round(rsa, 2)}
        print(f" {L:2d} | {r['cls']:12s} ({r['H1_ratio']:4.2f} "
              f"{str(r['z1']):>5s}) {cv:+.2f} | {rsa:+.2f}", flush=True)
    results["hue_sup"] = rows
    dump()
    print("\nCAVEAT: sup-plane columns are supervision-selected; the "
          "structural verdict is margA+margB+fibcons.", flush=True)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
