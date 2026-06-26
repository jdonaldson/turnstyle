"""Inspect the per-option date-selection probe's ERRORS. Out-of-fold (grouped 5-fold)
predictions at L19 over the EN set; dump every miss with stem+options, and categorize:
  PICKED_STEM_DATE  - chose a date literally in the stem (no/!wrong offset applied)
  NEAR_MISS(+-d)    - chose a date within a few days of the correct answer
  WRONG_YEAR        - right month+day, wrong year
  WRONG_MONTH       - right day+year, wrong month
  OTHER

  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python experiments/date_option_errors.py
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import re
import numpy as np
from turnstyle.bbh import load_task
from turnstyle.dates import _parse_date_str

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


def categorize(pre, correct_s, picked_s):
    c, p = _parse_date_str(correct_s), _parse_date_str(picked_s)
    if c is None or p is None:
        return "OTHER"
    if p in stem_dates(pre):
        return "PICKED_STEM_DATE"
    dd = abs((p - c).days)
    if dd <= 3:
        return f"NEAR_MISS({(p-c).days:+d}d)"
    if (p.month, p.day) == (c.month, c.day):
        return "WRONG_YEAR"
    if (p.year, p.day) == (c.year, c.day):
        return "WRONG_MONTH"
    return "OTHER"


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
    per_ex = []   # (idx, options[(letter,date)], target, vecs[n_opt,H])
    rows_v, rows_y, rows_g = [], [], []
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
        vecs, letters = [], []
        for letter, ti in opt_idxs(ct, offs, opts):
            if ti is None:
                continue
            vecs.append(h[ti]); letters.append((letter, dict(opts)[letter]))
            rows_v.append(h[ti]); rows_y.append(1 if f"({letter})" == tgt else 0); rows_g.append(gid)
        per_ex.append((gid, letters, tgt, np.array(vecs)))
        if gid % 40 == 0:
            print(f"  {gid+1}/{N}", flush=True)

    X = np.array(rows_v); Y = np.array(rows_y); G = np.array(rows_g)
    # OOF predictions
    oof = {}
    for tr, te in GroupKFold(5).split(X, Y, G):
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(C=0.3, max_iter=2000, class_weight="balanced").fit(sc.transform(X[tr]), Y[tr])
        prob = clf.predict_proba(sc.transform(X[te]))[:, 1]
        for g in set(G[te]):
            idx = np.where(G[te] == g)[0]
            oof[g] = idx[np.argmax(prob[idx])]   # row index of picked option within fold-te
        # remap local te indices -> need per-example; store picked letter
    # redo cleanly: recompute picked per example
    picks = {}
    for tr, te in GroupKFold(5).split(X, Y, G):
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(C=0.3, max_iter=2000, class_weight="balanced").fit(sc.transform(X[tr]), Y[tr])
        prob = clf.predict_proba(sc.transform(X[te]))[:, 1]
        teG = G[te]
        for g in set(teG):
            mask = teG == g
            local = np.where(mask)[0]
            picks[g] = int(np.argmax(prob[local]))   # which option-position within example g

    cats = {}
    n_err = n_tot = 0
    print("\n=== ERRORS ===", flush=True)
    for gid, letters, tgt, vecs in per_ex:
        if gid not in picks:
            continue
        n_tot += 1
        pj = picks[gid]
        pick_letter, pick_date = letters[pj]
        if f"({pick_letter})" == tgt:
            continue
        n_err += 1
        corr_date = next((d for l, d in letters if f"({l})" == tgt), "?")
        pre = exs[gid]["input"].split("Options:")[0].strip()
        cat = categorize(pre, corr_date, pick_date)
        cats[re.sub(r'\(.*', '', cat)] = cats.get(re.sub(r'\(.*', '', cat), 0) + 1
        print(f"\n[{gid}] {cat}", flush=True)
        print(f"   {pre[:150]}", flush=True)
        print(f"   correct {tgt}={corr_date}   PICKED ({pick_letter})={pick_date}", flush=True)

    print(f"\n=== {n_err}/{n_tot} errors ({n_err/n_tot*100:.0f}%) | categories: {cats} ===")


if __name__ == "__main__":
    main()
