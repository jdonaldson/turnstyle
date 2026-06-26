"""Ring test: does SmolLM2 encode cyclic calendar units (months, weekdays) on a RING,
or as an ordinal LINE? Decisive contrast = circular vs linear distance structure.

For a ring, the wrap pair (December-January, Sunday-Monday) is CLOSE; for a line it is
maximally far. We compare the pairwise activation-distance matrix against two models:
  linear   dist(i,j) = |i-j|
  circular dist(i,j) = min(|i-j|, P-|i-j|)        # P = period (12 months, 7 days)
Higher Pearson r = better fit. Number words one..twelve are a KNOWN-LINEAR control
(NUMBER frame is r~0.95 linear magnitude) and should prefer the linear model — if the
test can't tell numbers from months, it's not measuring cyclicity.

Per direction-frame lesson: z-score per dim across the point set BEFORE distances
(SmolLM2 has massive-activation rogue dims that dominate raw cosine). Last-subword readout.

  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python experiments/time_ring.py
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import numpy as np

MONTHS = ["January", "February", "March", "April", "May", "June",
          "July", "August", "September", "October", "November", "December"]
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
NUMS = ["one", "two", "three", "four", "five", "six",
        "seven", "eight", "nine", "ten", "eleven", "twelve"]

SETS = {
    "months": (MONTHS, 12, "The event happened in {w}."),
    "weekdays": (DAYS, 7, "The event happened on {w}."),
    "numbers(ctrl)": (NUMS, 12, "The value is {w}."),
}


def collect_last(model, tok, device, words, template):
    """Last-subword hidden states per word, all layers. -> {w: [L, H]}."""
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
        out[w] = stk[:, idxs[-1], :].float().cpu().numpy()
    return out


def models(P):
    """linear & circular distance matrices over P equally-spaced indices."""
    i = np.arange(P)
    raw = np.abs(i[:, None] - i[None, :])
    lin = raw.astype(float)
    circ = np.minimum(raw, P - raw).astype(float)
    return lin, circ


def upper(M):
    iu = np.triu_indices_from(M, k=1)
    return M[iu]


def analyze(acts, words, P, layer):
    """z-score per dim, euclidean distance matrix, fit linear vs circular."""
    X = np.array([acts[w][layer] for w in words])
    X = (X - X.mean(0)) / (X.std(0) + 1e-6)
    D = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(-1))
    lin, circ = models(P)
    r_lin = np.corrcoef(upper(D), upper(lin))[0, 1]
    r_circ = np.corrcoef(upper(D), upper(circ))[0, 1]
    # wrap pair = (P-1, 0): Dec-Jan / Sun-Mon. rank of its distance among all pairs (0=closest)
    wrap = D[P - 1, 0]
    allpairs = np.sort(upper(D))
    wrap_rank = int(np.searchsorted(allpairs, wrap))
    n_pairs = len(allpairs)
    # nearest-neighbor calendar-adjacency: for each point, is argmin (excl self) at +-1 mod P?
    adj = 0
    for k in range(P):
        d = D[k].copy(); d[k] = np.inf
        nn = int(np.argmin(d))
        if (nn - k) % P in (1, P - 1):
            adj += 1
    return r_lin, r_circ, wrap_rank, n_pairs, adj / P


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    mid = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(mid)
    mdl = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.float16).to(dev)
    print(f"model loaded on {dev}", flush=True)

    for name, (words, P, tmpl) in SETS.items():
        acts = collect_last(mdl, tok, dev, words, tmpl)
        nL = acts[words[0]].shape[0]
        print(f"\n=== {name}  (P={P}, n={len(words)}, tmpl={tmpl!r}) ===", flush=True)
        print(f"  {'L':>3s} {'r_lin':>7s} {'r_circ':>7s} {'winner':>8s} "
              f"{'wrap_rank':>10s} {'nn_adj':>7s}", flush=True)
        best = (-9, -1)
        for L in range(nL):
            r_lin, r_circ, wr, npair, adj = analyze(acts, words, P, L)
            win = "CIRC" if r_circ > r_lin else "lin"
            mark = "*" if r_circ > r_lin and r_circ > best[0] else " "
            if r_circ > best[0]:
                best = (r_circ, L)
            if L % 3 == 0 or L == nL - 1:
                print(f"  {L:>3d} {r_lin:>+7.3f} {r_circ:>+7.3f} {win:>8s} "
                      f"{wr:>3d}/{npair:<3d}    {adj:>6.2f}{mark}", flush=True)
        print(f"  >>> best circular r={best[0]:+.3f} @L{best[1]}", flush=True)


if __name__ == "__main__":
    main()
