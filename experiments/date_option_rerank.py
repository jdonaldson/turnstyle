"""Squeeze the top-3: can a SYMBOLIC offset re-ranker pick the correct date among the
probe's top-3? The probe gives the ballpark; the offset is ~language-free arithmetic.

Hybrid: for each example take the probe's top-3 option-dates; compute reachable targets
{ _apply_offset(sd + framing, question) : sd in stem_dates, framing in {-1,0,+1} days }
(framing covers today/tomorrow/yesterday); pick the top-3 option whose date is a reachable
target (highest-prob if several); else keep probe top-1. Compare:
  probe top-1   |  symbolic-over-ALL-options  |  symbolic-rerank-of-top3  |  top-3 ceiling
Also reports option-position bias (does the probe over-pick a slot?).

  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python experiments/date_option_rerank.py
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import re
import numpy as np
from datetime import timedelta
from turnstyle.bbh import load_task
from turnstyle.dates import _parse_date_str, _apply_offset

N = 120
LAYER = 19


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


def stem_dates(pre):
    out = []
    for c in re.findall(r'\d{1,2}/\d{1,2}/\d{2,4}', pre):
        d = _parse_date_str(c)
        if d:
            out.append(d)
    for m in re.finditer(r'[A-Z][a-z]+\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4}', pre):
        d = _parse_date_str(m.group(0))
        if d:
            out.append(d)
    return out


def reachable(text):
    """Set of target dates reachable by stem_date + framing(-1/0/+1) + question offset."""
    pre = text.split("Options:")[0]
    q = re.search(r'what is the date (.+?) in mm/dd/yyyy', text.lower())
    if not q:
        return set()
    qx = q.group(1)
    out = set()
    for sd in stem_dates(pre):
        for f in (-1, 0, 1):
            try:
                t = _apply_offset(sd + timedelta(days=f), qx)
            except Exception:
                t = None
            if t:
                out.add(t)
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
    print(f"model on {dev}  layer={LAYER}", flush=True)

    exs = load_task("date_understanding")[:N]
    per_ex = []   # gid -> (letters[(L,datestr,dateobj)], correct_letter, targets)
    V, Y, G = [], [], []
    for gid, ex in enumerate(exs):
        opts = parse_options(ex["input"]); tgt = ex["target"].strip()
        ct = tok.apply_chat_template([{"role": "user", "content": ex["input"]}],
                                     tokenize=False, add_generation_prompt=True)
        enc = tok(ct, return_offsets_mapping=True, return_tensors="pt")
        offs = enc.pop("offset_mapping")[0].tolist()
        enc = {k: v.to(dev) for k, v in enc.items()}
        with torch.no_grad():
            hs = mdl(**enc, output_hidden_states=True).hidden_states
        h = hs[LAYER][0].float().cpu().numpy()
        letters = []
        for letter, ti in opt_idxs(ct, offs, opts):
            if ti is None:
                continue
            V.append(h[ti]); Y.append(1 if f"({letter})" == tgt else 0); G.append(gid)
            letters.append((letter, dict(opts)[letter], _parse_date_str(dict(opts)[letter])))
        per_ex.append((letters, tgt, reachable(ex["input"])))
        if gid % 40 == 0:
            print(f"  {gid+1}/{N}", flush=True)

    X = np.array(V); Y = np.array(Y); G = np.array(G)
    prob_by_ex = {}
    for tr, te in GroupKFold(5).split(X, Y, G):
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(C=0.3, max_iter=2000, class_weight="balanced").fit(sc.transform(X[tr]), Y[tr])
        p = clf.predict_proba(sc.transform(X[te]))[:, 1]
        teG = G[te]
        for g in set(teG):
            prob_by_ex[g] = p[teG == g]

    top1 = sym_all = hybrid = top3 = tot = 0
    pos_pick = {}; pos_corr = {}
    for gid, (letters, tgt, targets) in enumerate(per_ex):
        if gid not in prob_by_ex:
            continue
        prob = prob_by_ex[gid]
        n = len(letters)
        order = np.argsort(-prob)
        corr_j = next(j for j, (l, _, _) in enumerate(letters) if f"({l})" == tgt)
        tot += 1
        top1 += int(order[0] == corr_j)
        top3 += int(corr_j in order[:3])
        pos_pick[order[0]] = pos_pick.get(order[0], 0) + 1
        pos_corr[corr_j] = pos_corr.get(corr_j, 0) + 1
        # symbolic over ALL options: pick option whose date is a reachable target (prob tie-break)
        match_all = [j for j in range(n) if letters[j][2] in targets] if targets else []
        if match_all:
            pick = max(match_all, key=lambda j: prob[j])
            sym_all += int(pick == corr_j)
        else:
            sym_all += int(order[0] == corr_j)
        # symbolic rerank within top-3
        top3set = list(order[:3])
        match3 = [j for j in top3set if letters[j][2] in targets] if targets else []
        if match3:
            pick = max(match3, key=lambda j: prob[j])
            hybrid += int(pick == corr_j)
        else:
            hybrid += int(order[0] == corr_j)

    print(f"\n=== n={tot} ===")
    print(f"  probe top-1:              {top1/tot:.3f}")
    print(f"  symbolic over ALL opts:   {sym_all/tot:.3f}")
    print(f"  symbolic rerank of top-3: {hybrid/tot:.3f}")
    print(f"  top-3 ceiling:            {top3/tot:.3f}")
    print(f"\n  position bias (slot: picked / correct):")
    for s in sorted(set(pos_pick) | set(pos_corr)):
        print(f"    slot {s}: picked {pos_pick.get(s,0):3d}   correct {pos_corr.get(s,0):3d}")


if __name__ == "__main__":
    main()
