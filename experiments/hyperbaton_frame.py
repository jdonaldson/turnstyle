"""Rebuild hyperbaton on the frame family: classify each adjective to its ordering
CATEGORY (nearest-centroid over exemplar word-sets per rung), sort by the canonical
rung order, break ties with model fluency (sequence logprob). Validate vs the 85%
subjectivity-axis solver before wiring into src. Sweeps the classifier layer.
"""
from __future__ import annotations
import sys
import numpy as np
sys.path.insert(0, "experiments")
from turnstyle.bbh import load_task
from turnstyle.hyperbaton import _OPT_RE, _adjectives, _TMPL
from turnstyle.semantic_frame import _word_vectors

# canonical adjective-ordering rungs (index = position; opinion first, material/purpose last)
RUNGS = ["opinion", "size", "age", "shape", "color", "origin", "material", "purpose"]
EXEMPLARS = {
    "opinion": "lovely nice ugly horrible wonderful terrible good awful comfortable".split(),
    "size": "big small large tiny huge enormous little gigantic".split(),
    "age": "old new young ancient modern brand-new antique aged".split(),
    "shape": "round square triangular rectangular oval pyramidal circular".split(),
    "color": "red blue green yellow black white grey silver brown purple".split(),
    "origin": "American Indian Japanese Egyptian Russian Chinese French German Mexican".split(),
    "material": "iron glass wooden cloth metallic plastic golden paper leather steel".split(),
    "purpose": "smoking driving hiking hunting cooking sleeping snorkeling whittling".split(),
}


def fit_centroids(mdl, tok, dev, layer):
    allw = [w for r in RUNGS for w in EXEMPLARS[r]]
    V = _word_vectors(mdl, tok, dev, allw, layer, _TMPL)
    mu, sd = V.mean(0), V.std(0) + 1e-6
    cents, i = [], 0
    for r in RUNGS:
        n = len(EXEMPLARS[r])
        cents.append(((V[i:i+n] - mu) / sd).mean(0)); i += n
    return np.array(cents), mu, sd


def seq_logprob(mdl, tok, dev, text):
    import torch
    enc = tok(text, return_tensors="pt").to(dev)
    ids = enc["input_ids"]
    with torch.no_grad():
        lg = mdl(**enc).logits[0, :-1].float().log_softmax(-1)
    return float(lg[range(ids.shape[1]-1), ids[0, 1:]].sum())


def solve(prompt, mdl, tok, dev, cents, mu, sd, layer):
    opts = _OPT_RE.findall(prompt)
    if len(opts) != 2:
        return None
    (la, ta), (lb, tb) = opts
    aa, ab = _adjectives(ta), _adjectives(tb)
    if len(aa) < 2 or sorted(ta.split()) != sorted(tb.split()):
        return None

    def rungs_of(adjs):
        Z = (_word_vectors(mdl, tok, dev, adjs, layer, _TMPL) - mu) / sd
        return [int(np.argmin(((cents - z) ** 2).sum(1))) for z in Z]

    def inv(rs):  # pairs out of ascending rung order
        return sum(1 for i in range(len(rs)) for j in range(i+1, len(rs)) if rs[i] > rs[j])
    ia, ib = inv(rungs_of(aa)), inv(rungs_of(ab))
    if ia != ib:
        return f"({la})" if ia < ib else f"({lb})"
    # tie -> model fluency
    return f"({la})" if seq_logprob(mdl, tok, dev, ta) >= seq_logprob(mdl, tok, dev, tb) else f"({lb})"


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    mid = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(mid)
    mdl = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.float16).to(dev).eval()
    ex = load_task("hyperbaton")
    for L in (6, 8, 10, 12, 14):
        cents, mu, sd = fit_centroids(mdl, tok, dev, L)
        ok = n = 0
        for e in ex:
            a = solve(e["input"], mdl, tok, dev, cents, mu, sd, L)
            if a is not None:
                n += 1; ok += (a.lower() == e["target"].strip().lower())
        print(f"L{L:<2d}  acc={ok/len(ex)*100:.1f}%  committed={n}/{len(ex)}", flush=True)


if __name__ == "__main__":
    main()
