"""Per-option date-SELECTION probe: can SmolLM2 RECOGNIZE the correct date among the MC
options even though it GENERATES it at ~20%? Tests the mc_selection_two_stage thesis
('interrupt argmax and solve') on the one MC task never probed.

For each date_understanding example: one forward pass over the full prompt (stem+Options),
read each option's LAST-TOKEN hidden state, train a binary 'is this the correct option?'
probe with example-GROUPED 5-fold CV, argmax over an example's options = selection. Sweep
layers. Baselines: chance (1/n_opts), and greedy GENERATION on a subset (the bar to beat).

  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python experiments/date_option_probe.py
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import re
import numpy as np
from turnstyle.bbh import load_task, answer_matches

N_PROBE = 120
N_GEN = 60          # generation baseline subset (slower)


def parse_options(text):
    sec = text.split("Options:")[-1]
    return [(letter, val.strip())
            for letter, val in re.findall(r'\(([A-Z])\)\s+([^\n(]+)', sec)]


def option_last_token_idxs(chat_text, offsets, opts):
    """For each (letter, date), the last token index of its occurrence in the Options section."""
    start = chat_text.find("Options:")
    cursor = start if start >= 0 else 0
    idxs = []
    for letter, date in opts:
        pos = chat_text.find(date, cursor)
        if pos < 0:
            idxs.append((letter, None)); continue
        end = pos + len(date)
        cursor = end
        toks = [k for k, (s, e) in enumerate(offsets) if e > pos and s < end]
        idxs.append((letter, toks[-1] if toks else None))
    return idxs


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

    exs = load_task("date_understanding")[:N_PROBE]
    rows = []   # (group_id, opt_vectors[L,H] float16, is_correct)
    nL = None
    gen_ok = gen_n = 0
    for gid, ex in enumerate(exs):
        opts = parse_options(ex["input"])
        tgt = ex["target"].strip()
        ct = tok.apply_chat_template([{"role": "user", "content": ex["input"]}],
                                     tokenize=False, add_generation_prompt=True)
        enc = tok(ct, return_offsets_mapping=True, return_tensors="pt")
        offs = enc.pop("offset_mapping")[0].tolist()
        enc = {k: v.to(dev) for k, v in enc.items()}
        with torch.no_grad():
            hs = mdl(**enc, output_hidden_states=True).hidden_states
        stk = torch.stack(hs, 0)[:, 0, :, :].float().cpu().numpy()   # [L,seq,H]
        nL = stk.shape[0]
        for letter, ti in option_last_token_idxs(ct, offs, opts):
            if ti is None:
                continue
            rows.append((gid, stk[:, ti, :].astype(np.float16),
                         1 if f"({letter})" == tgt else 0))
        # generation baseline on a subset
        if gid < N_GEN:
            with torch.no_grad():
                out = mdl.generate(**enc, max_new_tokens=50, do_sample=False)
            g = tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
            gen_ok += answer_matches(g, tgt); gen_n += 1
        if gid % 30 == 0:
            print(f"  collected {gid+1}/{len(exs)}", flush=True)

    groups = np.array([r[0] for r in rows])
    Y = np.array([r[2] for r in rows])
    A = np.stack([r[1] for r in rows]).astype(np.float32)   # [nrows, L, H]
    n_ex = len(set(groups))
    chance = np.mean([1.0 / max(1, (groups == g).sum()) for g in set(groups)])
    print(f"\n{len(rows)} option-rows over {n_ex} examples; "
          f"chance(selection)={chance:.3f}  gen-baseline={gen_ok}/{gen_n}={gen_ok/max(1,gen_n):.3f}", flush=True)

    def selection_acc(L):
        X = A[:, L, :]
        gkf = GroupKFold(n_splits=5)
        correct = total = 0
        for tr, te in gkf.split(X, Y, groups):
            sc = StandardScaler().fit(X[tr])
            clf = LogisticRegression(C=0.3, max_iter=2000, class_weight="balanced")
            clf.fit(sc.transform(X[tr]), Y[tr])
            prob = clf.predict_proba(sc.transform(X[te]))[:, 1]
            for g in set(groups[te]):
                mask = groups[te] == g
                gi = np.where(mask)[0]
                pick = gi[np.argmax(prob[mask])]
                correct += int(Y[te][pick] == 1); total += 1
        return correct / total

    print("\n=== per-option date-selection probe (grouped 5-fold CV) ===")
    print(f"  {'L':>3s} | {'sel-acc':>7s}")
    best = (-9, -1)
    for L in range(nL):
        acc = selection_acc(L)
        if acc > best[0]:
            best = (acc, L)
        if L % 2 == 0 or L == nL - 1:
            print(f"  {L:>3d} | {acc:>7.3f}", flush=True)
    print(f"  >>> best selection acc = {best[0]:.3f} @L{best[1]}  "
          f"(chance {chance:.3f}, gen {gen_ok/max(1,gen_n):.3f})")


if __name__ == "__main__":
    main()
