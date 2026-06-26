"""Does 'recognition >> generation' on date_understanding replicate across models, or is
it a SmolLM2 weak-generation artifact? For a given model: greedy GENERATION accuracy vs the
per-option recognition PROBE (top-1/top-3, grouped CV layer sweep). Run one model per process.

Each run prints Wilson 95% CIs and saves per-problem hit vectors to
experiments/results/date_crossmodel/<model>.json . After running all models, run

  .venv/bin/python experiments/date_crossmodel.py --compare

to get PAIRED McNemar tests of the 'invariance' claim (same problems across models, so the
paired test is far more powerful than eyeballing three point estimates).

  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python experiments/date_crossmodel.py <model_id> [n_probe] [n_gen]
  models: HuggingFaceTB/SmolLM2-1.7B-Instruct | Qwen/Qwen2.5-1.5B-Instruct | microsoft/Phi-4-mini-instruct
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import os, sys, re, json
import numpy as np
from turnstyle.bbh import load_task, answer_matches

RESDIR = os.path.join(os.path.dirname(__file__), "results", "date_crossmodel")


def wilson(k, n, z=1.96):
    """Wilson score 95% CI for a binomial proportion. Returns (p, lo, hi)."""
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z / d * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5)
    return p, max(0.0, c - h), min(1.0, c + h)


def mcnemar(a_hits, b_hits):
    """Exact two-sided McNemar on two {gid: 0/1} dicts over shared gids.
    Returns (b, c, p) where b = a-right/b-wrong, c = b-right/a-wrong."""
    from scipy.stats import binomtest
    b = c = 0
    for g in a_hits:
        if g not in b_hits:
            continue
        if a_hits[g] and not b_hits[g]:
            b += 1
        elif b_hits[g] and not a_hits[g]:
            c += 1
    n = b + c
    p = binomtest(min(b, c), n, 0.5).pvalue if n else 1.0
    return b, c, p


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


def compare():
    """Load every saved per-model result and run paired McNemar on top-1 hits."""
    if not os.path.isdir(RESDIR):
        print(f"no results dir {RESDIR}; run per-model first"); return
    runs = {}
    for fn in sorted(os.listdir(RESDIR)):
        if not fn.endswith(".json"):
            continue
        with open(os.path.join(RESDIR, fn)) as f:
            r = json.load(f)
        runs[r["model"]] = r
    if len(runs) < 2:
        print(f"need >=2 model results in {RESDIR}, have {len(runs)}"); return
    names = list(runs)
    print(f"=== paired McNemar on probe top-1 ({len(names)} models) ===")
    for r in runs.values():
        k = sum(r["top1_hits"].values()); n = len(r["top1_hits"])
        p, lo, hi = wilson(k, n)
        print(f"  {r['model'].split('/')[-1]:30s} top-1 {p:.3f}  95% CI [{lo:.3f},{hi:.3f}]  (N={n}, L{r['best_layer']})")
    print("  --- pairwise (b=A-right/B-wrong, c=B-right/A-wrong) ---")
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b_ = runs[names[i]], runs[names[j]]
            ah = {int(k): v for k, v in a["top1_hits"].items()}
            bh = {int(k): v for k, v in b_["top1_hits"].items()}
            bb, cc, pv = mcnemar(ah, bh)
            sig = "  DIFFERENT (p<0.05)" if pv < 0.05 else "  n.s."
            print(f"  {names[i].split('/')[-1]:24s} vs {names[j].split('/')[-1]:24s}  "
                  f"b={bb} c={cc}  p={pv:.3f}{sig}")


def main():
    if "--compare" in sys.argv:
        compare(); return
    MODEL = sys.argv[1] if len(sys.argv) > 1 else "HuggingFaceTB/SmolLM2-1.7B-Instruct"
    N_PROBE = int(sys.argv[2]) if len(sys.argv) > 2 else 250
    N_GEN = int(sys.argv[3]) if len(sys.argv) > 3 else 250

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import GroupKFold
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL)
    mdl = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16).to(dev)
    print(f"== {MODEL} on {dev} (N_PROBE={N_PROBE}, N_GEN={N_GEN}) ==", flush=True)

    exs = load_task("date_understanding")[:N_PROBE]
    V, Y, G, nopts = [], [], [], []
    gen_hits = {}
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
            V.append(stk[:, ti, :].astype(np.float16)); Y.append(1 if f"({letter})" == tgt else 0)
            G.append(gid); cnt += 1
        nopts.append(cnt)
        if gid < N_GEN:
            with torch.no_grad():
                out = mdl.generate(**enc, max_new_tokens=50, do_sample=False)
            g = tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
            gen_hits[gid] = int(bool(answer_matches(g, tgt)))
        if gid % 25 == 0:
            print(f"  {gid+1}/{N_PROBE}", flush=True)

    A = np.stack(V).astype(np.float32); Y = np.array(Y); G = np.array(G)
    nL = A.shape[1]
    ch = {k: float(np.mean([min(k, c) / c for c in nopts if c])) for k in (1, 3)}

    def topk(L):
        X = A[:, L, :]; h1, h3 = {}, {}
        for tr, te in GroupKFold(5).split(X, Y, G):
            sc = StandardScaler().fit(X[tr])
            clf = LogisticRegression(C=0.3, max_iter=2000, class_weight="balanced").fit(sc.transform(X[tr]), Y[tr])
            p = clf.predict_proba(sc.transform(X[te]))[:, 1]
            for g in set(G[te]):
                m = np.where(G[te] == g)[0]
                order = m[np.argsort(-p[m])]
                rank = int(np.where(Y[te][order] == 1)[0][0])
                h1[int(g)] = int(rank < 1); h3[int(g)] = int(rank < 3)
        n = len(h1)
        return sum(h1.values()) / n, sum(h3.values()) / n, h1, h3

    best = (-9, -1, -1, None, None)
    for L in range(0, nL, 2):
        t1, t3, h1, h3 = topk(L)
        if t1 > best[0]:
            best = (t1, t3, L, h1, h3)
    t1, t3, bL, top1_hits, top3_hits = best

    gk = sum(gen_hits.values()); gn = len(gen_hits)
    gp, glo, ghi = wilson(gk, gn)
    p1, lo1, hi1 = wilson(sum(top1_hits.values()), len(top1_hits))
    p3, lo3, hi3 = wilson(sum(top3_hits.values()), len(top3_hits))
    short = MODEL.split('/')[-1]
    print(f"\n=== {short} ===")
    print(f"  generation:   {gp:.3f}  95% CI [{glo:.3f},{ghi:.3f}]  (N={gn})")
    print(f"  probe top-1:  {p1:.3f}  95% CI [{lo1:.3f},{hi1:.3f}]  @L{bL}  (N={len(top1_hits)}, chance {ch[1]:.3f})")
    print(f"  probe top-3:  {p3:.3f}  95% CI [{lo3:.3f},{hi3:.3f}]  (chance {ch[3]:.3f})")
    print(f"  >>> recognition-minus-generation gap = {p1 - gp:+.3f}")

    os.makedirs(RESDIR, exist_ok=True)
    out_path = os.path.join(RESDIR, short + ".json")
    with open(out_path, "w") as f:
        json.dump({
            "model": MODEL, "n_probe": N_PROBE, "n_gen": N_GEN, "best_layer": bL,
            "top1": p1, "top3": p3, "gen": gp,
            "top1_ci": [lo1, hi1], "top3_ci": [lo3, hi3], "gen_ci": [glo, ghi],
            "chance1": ch[1], "chance3": ch[3],
            "top1_hits": top1_hits, "top3_hits": top3_hits, "gen_hits": gen_hits,
        }, f)
    print(f"  saved {out_path}")


if __name__ == "__main__":
    main()
