"""Make-or-break test: does the linear month axis (fit on 12 literal English months)
project IMPLICIT time terms — seasons, holidays, activities, cross-lingual words —
onto the right month region? If yes, it's a generalizing time-normalizer + the
FrameOrdering-over-time capability. If they scatter, it's a fancy month lookup.

Fit: ridge(StandardScaler(acts_of_12_EN_months) -> month index 1..12) per layer.
Project each held-out probe term, compare predicted month to its assigned true month.
Report Pearson (linear) AND circular correlation (fair to cyclic ground truth), plus a
per-term table at the principled layer. Ground truth is Northern-Hemisphere / Western-
calendar center-of-mass; WRAP terms (winter/skiing near the Dec-Jan boundary) are
flagged — a LINEAR axis structurally cannot place Dec next to Jan, so those are the
expected failure mode, not a surprise.

  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python experiments/time_projection.py
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import numpy as np

EN_MONTHS = ["January", "February", "March", "April", "May", "June",
             "July", "August", "September", "October", "November", "December"]

# (term, true month center 1..12, wrap-ambiguous near Dec/Jan boundary?)
PROBES = [
    # seasons (NH)
    ("spring", 4, False), ("summer", 7, False), ("autumn", 10, False),
    ("fall", 10, False), ("winter", 1, True),
    # Western holidays / observances
    ("New Year", 1, True), ("Valentine's Day", 2, False), ("Easter", 4, False),
    ("Halloween", 10, False), ("Thanksgiving", 11, False), ("Christmas", 12, True),
    # seasonal activities / weather
    ("harvest", 10, False), ("back-to-school", 9, False), ("graduation", 6, False),
    ("planting", 4, False), ("blizzard", 1, True), ("heatwave", 7, False),
    ("skiing", 1, True),
    # cross-lingual seasons (es/fr)
    ("verano", 7, False), ("invierno", 1, True), ("primavera", 4, False),
    ("été", 7, False), ("hiver", 1, True),
]

TMPL = "It happens in {w}."
M = 12
MIDX = np.arange(1, M + 1).astype(float)


def collect_last(model, tok, device, words, template):
    import torch
    out = []
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
        out.append(stk[:, idxs[-1], :].float().cpu().numpy())
    return np.stack(out, 1)                        # [L, n, H]


def circ_corr(a, b):
    am = np.arctan2(np.sin(a).sum(), np.cos(a).sum())
    bm = np.arctan2(np.sin(b).sum(), np.cos(b).sum())
    sa, sb = np.sin(a - am), np.sin(b - bm)
    return float((sa * sb).sum() / (np.sqrt((sa**2).sum() * (sb**2).sum()) + 1e-9))


def fit_predict(Xm, Xp, alpha=10.0):
    """Fit ridge months->index, predict probe months. Standardize on month stats."""
    mu, sd = Xm.mean(0), Xm.std(0) + 1e-6
    Ms = (Xm - mu) / sd
    w = np.linalg.solve(Ms.T @ Ms + alpha * np.eye(Ms.shape[1]), Ms.T @ (MIDX - MIDX.mean()))
    b = MIDX.mean()
    pred_m = ((Xm - mu) / sd) @ w + b
    pred_p = ((Xp - mu) / sd) @ w + b
    return pred_m, pred_p


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    mid = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(mid)
    mdl = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.float16).to(dev)
    print(f"model on {dev}", flush=True)

    pterms = [p[0] for p in PROBES]
    ptrue = np.array([p[1] for p in PROBES], float)
    pwrap = np.array([p[2] for p in PROBES])

    Xm = collect_last(mdl, tok, dev, EN_MONTHS, TMPL)
    Xp = collect_last(mdl, tok, dev, pterms, TMPL)
    nL = Xm.shape[0]

    theta_true = 2 * np.pi * (ptrue - 1) / M
    nowrap = ~pwrap

    print("\n=== probe-set alignment per layer (predicted month vs true) ===")
    print(f"  {'L':>3s} | {'pearson':>7s} {'circ':>6s} | {'pearson(no-wrap)':>16s} {'circ(no-wrap)':>13s}")
    best = (-9, -1)
    for L in range(nL):
        _, pp = fit_predict(Xm[L], Xp[L])
        pear = float(np.corrcoef(pp, ptrue)[0, 1])
        circ = circ_corr(2 * np.pi * (pp - 1) / M, theta_true)
        pear_nw = float(np.corrcoef(pp[nowrap], ptrue[nowrap])[0, 1])
        circ_nw = circ_corr(2 * np.pi * (pp[nowrap] - 1) / M, theta_true[nowrap])
        if pear_nw > best[0]:
            best = (pear_nw, L)
        if L % 2 == 0 or L == nL - 1:
            print(f"  {L:>3d} | {pear:>+7.3f} {circ:>+6.3f} | {pear_nw:>+16.3f} {circ_nw:>+13.3f}", flush=True)
    bestL = best[1]
    print(f"  >>> best no-wrap pearson r={best[0]:+.3f} @L{bestL}")

    pm, pp = fit_predict(Xm[bestL], Xp[bestL])
    print(f"\n=== per-term predictions @L{bestL}  (month sanity: r="
          f"{np.corrcoef(pm, MIDX)[0,1]:+.3f}) ===")
    order = np.argsort(ptrue)
    for i in order:
        err = abs(pp[i] - ptrue[i])
        cerr = min(err, M - err)
        flag = " WRAP" if pwrap[i] else ""
        hit = "ok" if cerr <= 1.5 else ("~ " if cerr <= 2.5 else "XX")
        print(f"  {pterms[i]:18s} true={ptrue[i]:4.1f}  pred={pp[i]:5.1f}  "
              f"cerr={cerr:4.1f}  {hit}{flag}", flush=True)


if __name__ == "__main__":
    main()
