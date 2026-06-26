"""Follow-up to time_ring: is the month-ring SEMANTIC (transfers cross-lingually,
survives mid-stack) or LEXICAL (English-only, dies where the model computes)?

Two frame-relevant tests, more sensitive than pairwise distance:

A. RECOVERABILITY (English, LOO-CV) per layer:
   - linear: ridge(acts -> month index 1..12), corr(pred, index)
   - cyclic: ridge(acts -> cos theta), ridge(acts -> sin theta); reconstruct angle;
     CIRCULAR correlation of predicted vs true phase.
   Robustness: two EN templates. If cyclic >> linear at the computational mid-stack,
   the ring is the working representation; if linear wins mid-stack, it's ordinal+mod.

B. CROSS-LINGUAL TRANSFER (semantic test): fit cyclic + linear directions on EN,
   project ES & FR months (EN standardization, per color_crosslingual), correlate
   recovered phase/index with the language-independent calendar target, per layer.
   High & language-uniform => the ring is semantic (shared concept). ~0 => lexical.

  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python experiments/time_ring_followup.py
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import numpy as np

EN = ["January", "February", "March", "April", "May", "June",
      "July", "August", "September", "October", "November", "December"]
ES = ["enero", "febrero", "marzo", "abril", "mayo", "junio",
      "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre"]
FR = ["janvier", "février", "mars", "avril", "mai", "juin",
      "juillet", "août", "septembre", "octobre", "novembre", "décembre"]

TMPL = {
    "en": "The event happened in {w}.",
    "en2": "My birthday is in {w}.",
    "es": "El evento ocurrió en {w}.",
    "fr": "L'événement s'est produit en {w}.",
}

M = 12
IDX = np.arange(1, M + 1).astype(float)          # linear target
THETA = 2 * np.pi * np.arange(M) / M             # true phase
COS, SIN = np.cos(THETA), np.sin(THETA)


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
    return np.stack(out, 1)                        # [L, M, H]


def circ_corr(a, b):
    """Jammalamadaka-Sarma circular correlation between two angle arrays."""
    am = np.arctan2(np.sin(a).sum(), np.cos(a).sum())
    bm = np.arctan2(np.sin(b).sum(), np.cos(b).sum())
    sa, sb = np.sin(a - am), np.sin(b - bm)
    return float((sa * sb).sum() / (np.sqrt((sa**2).sum() * (sb**2).sum()) + 1e-9))


def ridge_dir(Xs, y, alpha=10.0):
    return np.linalg.solve(Xs.T @ Xs + alpha * np.eye(Xs.shape[1]), Xs.T @ (y - y.mean()))


def loo_recover(X):
    """LOO-CV at one layer. X=[M,H]. Returns (linear_r, cyclic_circ_r)."""
    pred_idx, pred_c, pred_s = np.zeros(M), np.zeros(M), np.zeros(M)
    for h in range(M):
        tr = [k for k in range(M) if k != h]
        Xtr, Xte = X[tr], X[h]
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
        Xtrs = (Xtr - mu) / sd
        xte = (Xte - mu) / sd
        wl = ridge_dir(Xtrs, IDX[tr]); pred_idx[h] = xte @ wl
        wc = ridge_dir(Xtrs, COS[tr]); pred_c[h] = xte @ wc
        ws = ridge_dir(Xtrs, SIN[tr]); pred_s[h] = xte @ ws
    lin_r = float(np.corrcoef(pred_idx, IDX)[0, 1])
    cyc_r = circ_corr(np.arctan2(pred_s, pred_c), THETA)
    return lin_r, cyc_r


def transfer(Xen, Xfor):
    """Fit on all EN at one layer, project foreign. Returns (linear_r, cyclic_circ_r)."""
    mu, sd = Xen.mean(0), Xen.std(0) + 1e-6
    Es = (Xen - mu) / sd
    Fs = (Xfor - mu) / sd
    wl = ridge_dir(Es, IDX); wc = ridge_dir(Es, COS); ws = ridge_dir(Es, SIN)
    lin_r = float(np.corrcoef(Fs @ wl, IDX)[0, 1])
    cyc_r = circ_corr(np.arctan2(Fs @ ws, Fs @ wc), THETA)
    return lin_r, cyc_r


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    mid = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(mid)
    mdl = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.float16).to(dev)
    print(f"model on {dev}", flush=True)

    acts = {
        "en":  collect_last(mdl, tok, dev, EN, TMPL["en"]),
        "en2": collect_last(mdl, tok, dev, EN, TMPL["en2"]),
        "es":  collect_last(mdl, tok, dev, ES, TMPL["es"]),
        "fr":  collect_last(mdl, tok, dev, FR, TMPL["fr"]),
    }
    nL = acts["en"].shape[0]

    print("\n=== A. EN recoverability (LOO-CV): linear(index) vs cyclic(phase) ===")
    print(f"  {'L':>3s} | {'en lin':>7s} {'en cyc':>7s} | {'en2 lin':>7s} {'en2 cyc':>7s} | winner")
    for L in range(nL):
        l1, c1 = loo_recover(acts["en"][L])
        l2, c2 = loo_recover(acts["en2"][L])
        win = "CYC" if (c1 > l1 and c2 > l2) else ("lin" if (l1 > c1 and l2 > c2) else "mix")
        if L % 2 == 0 or L == nL - 1:
            print(f"  {L:>3d} | {l1:>+7.3f} {c1:>+7.3f} | {l2:>+7.3f} {c2:>+7.3f} | {win}", flush=True)

    print("\n=== B. cross-lingual transfer (EN-fit -> project): semantic if high & uniform ===")
    print(f"  {'L':>3s} | {'es lin':>7s} {'es cyc':>7s} | {'fr lin':>7s} {'fr cyc':>7s}")
    bestcyc = (-9, -1)
    for L in range(nL):
        esl, esc = transfer(acts["en"][L], acts["es"][L])
        frl, frc = transfer(acts["en"][L], acts["fr"][L])
        mcyc = (esc + frc) / 2
        if mcyc > bestcyc[0]:
            bestcyc = (mcyc, L)
        if L % 2 == 0 or L == nL - 1:
            print(f"  {L:>3d} | {esl:>+7.3f} {esc:>+7.3f} | {frl:>+7.3f} {frc:>+7.3f}", flush=True)
    print(f"  >>> best mean cross-lingual CYCLIC r={bestcyc[0]:+.3f} @L{bestcyc[1]}")


if __name__ == "__main__":
    main()
