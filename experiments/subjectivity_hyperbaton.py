"""Does the subjectivity META-axis solve hyperbaton on its own?

The current hyperbaton solver clusters adjectives into ordering categories (L8) and
counts inversions against a HARDCODED category order (96%). This tests a leaner
hypothesis: a single continuous subjectivity axis suffices — the correct adjective
order is the one sorted by DECREASING subjectivity (most subjective farthest from the
noun). No category clustering, no hardcoded order — just project + sort.

  axis: fit from the subjectivity_order cache (English opinion[high] vs material[low])
  score: per option, count pairs out of decreasing-subjectivity order (inversions);
         pick the option with fewer; compare to gold.

Reuses the subjectivity_order axis; collects hyperbaton adjective activations once.
Usage:  python experiments/subjectivity_hyperbaton.py [--collect]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, "experiments")
sys.path.insert(0, "src")
import pole_probe as PP
import subjectivity_order as SO

BBH = "/Users/jdonaldson/Projects/swollm/data/bbh_cache"
CACHE = "experiments/data/hyperbaton_adj_acts.npz"
TMPL = "It is a {a} object."


def _examples():
    data = json.load(open(f"{BBH}/hyperbaton.json"))
    out = []
    for ex in data:
        opts = dict(re.findall(r"\(([A-Z])\)\s+(.+)", ex["input"]))
        seqs = {L: txt.split()[:-1] for L, txt in opts.items()}   # drop the noun
        out.append((seqs, ex["target"].strip().strip("()")))
    return out


def collect():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    vocab = sorted({w for seqs, _ in _examples() for s in seqs.values() for w in s})
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(PP.MODEL_ID)
    mdl = AutoModelForCausalLM.from_pretrained(
        PP.MODEL_ID, dtype=torch.float16).to(dev).eval()
    acts, words = [], []
    for i, w in enumerate(vocab):
        sent = TMPL.format(a=w)
        cs = sent.rfind(w); ce = cs + len(w)
        enc = tok(sent, return_offsets_mapping=True, return_tensors="pt")
        offs = enc.pop("offset_mapping")[0].tolist()
        enc = {k: v.to(dev) for k, v in enc.items()}
        with torch.no_grad():
            out = mdl(**enc, output_hidden_states=True)
        hs = torch.stack(out.hidden_states, 0)[:, 0].float().cpu().numpy()
        tk = next((k for k, (s, e) in enumerate(offs) if e > cs and s < ce), None)
        acts.append(hs[:, tk, :].astype(np.float16)); words.append(w)
        print(f"  [{i+1}/{len(vocab)}] {w}", end="\r", flush=True)
    print()
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    np.savez_compressed(CACHE, acts=np.stack(acts), words=np.array(words))
    print(f"saved {len(acts)} adjective vectors → {CACHE}")


def _inversions(scores):
    """# pairs (i<j) where score increases (violates decreasing-subjectivity order)."""
    n = len(scores)
    return sum(1 for i in range(n) for j in range(i + 1, n) if scores[i] < scores[j])


def analyze():
    so = np.load(SO.CACHE, allow_pickle=True)
    soA = so["acts"].astype(np.float32); soCat = so["cat"]; soLang = so["lang"]
    hb = np.load(CACHE, allow_pickle=True)
    hbA = hb["acts"].astype(np.float32)
    wi = {w: i for i, w in enumerate(hb["words"].tolist())}
    exs = _examples()
    nL = soA.shape[1]
    print(f"hyperbaton N={len(exs)}  adj vocab={len(wi)}")
    print(f"{'L':>3} {'acc':>7} {'decided':>8}")
    best = (0.0, -1)
    for L in range(nL):
        en = soLang == "en"
        mu = soA[en, L].mean(0); sd = soA[en, L].std(0) + 1e-6
        Z = (soA[:, L] - mu) / sd
        hi = Z[en & (soCat == "opinion")].mean(0)
        lo = Z[en & (soCat == "material")].mean(0)
        dirn = (hi - lo) / (np.linalg.norm(hi - lo) + 1e-12)
        score = {w: float(((hbA[wi[w], L] - mu) / sd) @ dirn) for w in wi}
        correct = decided = 0
        for seqs, gold in exs:
            inv = {Lk: _inversions([score[w] for w in s if w in score])
                   for Lk, s in seqs.items()}
            best_opt = min(inv, key=inv.get)
            tie = sum(1 for v in inv.values() if v == inv[best_opt]) > 1
            if not tie:
                decided += 1
                if best_opt == gold:
                    correct += 1
        acc = correct / len(exs)
        if acc > best[0]:
            best = (acc, L)
        print(f"{L:>3} {acc:>7.3f} {decided/len(exs):>8.2f}")
    print(f"\nbest L{best[1]} acc={best[0]:.3f}  (solver baseline 0.96, chance 0.50)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--collect", action="store_true")
    args = ap.parse_args()
    if args.collect or not os.path.exists(CACHE):
        collect()
    analyze()
