"""Top-k selection accuracy for the per-option date probe: is the correct date usually the
probe's runner-up? The error analysis showed ~1/3 of misses are +-1-3 days (near-misses),
so the correct option is likely rank 2. If top-2/top-3 are much higher than top-1 (63%),
the probe is an excellent 'narrow the candidates' stage to hand to symbolic arithmetic.

Grouped 5-fold OOF ranking, per layer. Reports top-1/2/3 vs per-k chance (mean min(k,n)/n).

  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python experiments/date_option_topk.py
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import re
import numpy as np
from turnstyle.bbh import load_task

N = 120


def parse_options(text):
    sec = text.split("Options:")[-1]
    return [(l, v.strip()) for l, v in re.findall(r'\(([A-Z])\)\s+([^\n(]+)', sec)]


def opt_idxs(chat, offs, opts):
    start = chat.find("Options:"); cur = start if start >= 0 else 0
    out = []
    for letter, dt in opts:
        pos = chat.find(dt, cur)
        if pos < 0:
            out.append((letter, None)); continue
        end = pos + len(dt); cur = end
        t = [k for k, (s, e) in enumerate(offs) if e > pos and s < end]
        out.append((letter, t[-1] if t else None))
    return out


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import GroupKFold
    mid = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(mid)
    mdl = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.float16).to(dev)
    print(f"model on {dev}", flush=True)

    exs = load_task("date_understanding")[:N]
    V, Y, G = [], [], []
    nopts = []
    for gid, ex in enumerate(exs):
        opts = parse_options(ex["input"]); tgt = ex["target"].strip()
        ct = tok.apply_chat_template([{"role": "user", "content": ex["input"]}],
                                     tokenize=False, add_generation_prompt=True)
        enc = tok(ct, return_offsets_mapping=True, return_tensors="pt")
        offs = enc.pop("offset_mapping")[0].tolist()
        enc = {k: v.to(dev) for k, v in enc.items()}
        with torch.no_grad():
            hs = mdl(**enc, output_hidden_states=True).hidden_states
        stk = torch.stack(hs, 0)[:, 0, :, :].float().cpu().numpy()
        cnt = 0
        for letter, ti in opt_idxs(ct, offs, opts):
            if ti is None:
                continue
            V.append(stk[:, ti, :].astype(np.float16))
            Y.append(1 if f"({letter})" == tgt else 0); G.append(gid); cnt += 1
        nopts.append(cnt)
        if gid % 40 == 0:
            print(f"  {gid+1}/{N}", flush=True)

    A = np.stack(V).astype(np.float32); Y = np.array(Y); G = np.array(G)
    nL = A.shape[1]
    ch = {k: float(np.mean([min(k, c) / c for c in nopts if c])) for k in (1, 2, 3)}
    print(f"\nper-k chance: top1={ch[1]:.2f} top2={ch[2]:.2f} top3={ch[3]:.2f}", flush=True)

    def topk(L):
        X = A[:, L, :]
        hit = {1: 0, 2: 0, 3: 0}; tot = 0
        for tr, te in GroupKFold(5).split(X, Y, G):
            sc = StandardScaler().fit(X[tr])
            clf = LogisticRegression(C=0.3, max_iter=2000, class_weight="balanced").fit(sc.transform(X[tr]), Y[tr])
            prob = clf.predict_proba(sc.transform(X[te]))[:, 1]
            teG = G[te]
            for g in set(teG):
                m = np.where(teG == g)[0]
                order = m[np.argsort(-prob[m])]          # option rows best->worst
                rank = int(np.where(Y[te][order] == 1)[0][0])  # rank of correct (0-based)
                for k in (1, 2, 3):
                    hit[k] += int(rank < k)
                tot += 1
        return {k: hit[k] / tot for k in (1, 2, 3)}

    print("\n=== top-k selection accuracy (grouped 5-fold OOF) ===")
    print(f"  {'L':>3s} | {'top1':>5s} {'top2':>5s} {'top3':>5s}")
    best = (-9, -1)
    for L in range(nL):
        t = topk(L)
        if t[1] > best[0]:
            best = (t[1], L)
        if L % 2 == 0 or L == nL - 1:
            print(f"  {L:>3d} | {t[1]:>5.2f} {t[2]:>5.2f} {t[3]:>5.2f}", flush=True)
    bt = topk(best[1])
    print(f"  >>> @L{best[1]}: top1={bt[1]:.2f} top2={bt[2]:.2f} top3={bt[3]:.2f}  "
          f"(chance {ch[1]:.2f}/{ch[2]:.2f}/{ch[3]:.2f})")


if __name__ == "__main__":
    main()
