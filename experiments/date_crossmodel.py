"""Does 'recognition >> generation' on date_understanding replicate across models, or is
it a SmolLM2 weak-generation artifact? For a given model: greedy GENERATION accuracy vs the
per-option recognition PROBE (top-1/top-3, grouped CV layer sweep). Run one model per process.

  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python experiments/date_crossmodel.py <model_id>
  models: HuggingFaceTB/SmolLM2-1.7B-Instruct | Qwen/Qwen2.5-1.5B-Instruct | microsoft/Phi-4-mini-instruct
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import sys, re
import numpy as np
from turnstyle.bbh import load_task, answer_matches

MODEL = sys.argv[1] if len(sys.argv) > 1 else "HuggingFaceTB/SmolLM2-1.7B-Instruct"
N_PROBE = 100
N_GEN = 60


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
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL)
    mdl = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16).to(dev)
    print(f"== {MODEL} on {dev} ==", flush=True)

    exs = load_task("date_understanding")[:N_PROBE]
    V, Y, G, nopts = [], [], [], []
    gen_ok = gen_n = 0
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
            gen_ok += answer_matches(g, tgt); gen_n += 1
        if gid % 25 == 0:
            print(f"  {gid+1}/{N_PROBE}", flush=True)

    A = np.stack(V).astype(np.float32); Y = np.array(Y); G = np.array(G)
    nL = A.shape[1]
    ch = {k: float(np.mean([min(k, c) / c for c in nopts if c])) for k in (1, 3)}

    def topk(L):
        X = A[:, L, :]; hit1 = hit3 = tot = 0
        for tr, te in GroupKFold(5).split(X, Y, G):
            sc = StandardScaler().fit(X[tr])
            clf = LogisticRegression(C=0.3, max_iter=2000, class_weight="balanced").fit(sc.transform(X[tr]), Y[tr])
            p = clf.predict_proba(sc.transform(X[te]))[:, 1]
            for g in set(G[te]):
                m = np.where(G[te] == g)[0]
                order = m[np.argsort(-p[m])]
                rank = int(np.where(Y[te][order] == 1)[0][0])
                hit1 += int(rank < 1); hit3 += int(rank < 3); tot += 1
        return hit1 / tot, hit3 / tot

    best = (-9, -1, -1)
    for L in range(0, nL, 2):
        t1, t3 = topk(L)
        if t1 > best[0]:
            best = (t1, t3, L)
    print(f"\n=== {MODEL.split('/')[-1]} ===")
    print(f"  generation:   {gen_ok}/{gen_n} = {gen_ok/gen_n:.3f}")
    print(f"  probe top-1:  {best[0]:.3f} @L{best[2]}   (chance {ch[1]:.3f})")
    print(f"  probe top-3:  {best[1]:.3f}              (chance {ch[3]:.3f})")
    print(f"  >>> recognition-minus-generation gap = {best[0]-gen_ok/gen_n:+.3f}")


if __name__ == "__main__":
    main()
