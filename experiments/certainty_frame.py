"""
certainty_frame.py — is epistemic CERTAINTY a recoverable scalar frame?

The highest project-value frame candidate: a readable certainty dial connects
to turnstyle's abstain machinery (and to hallucination work generally). Words
span impossible → certain, template "It is {w} that it happened." (predicative
epistemic adjective), last-subword readout.

Confound columns (the audit standards, both by construction and by test):
  * zipf frequency — the standing rule after space died of it.
  * MORPHOLOGICAL NEGATION — the certainty-specific trap: most low-certainty
    words wear a negative prefix (IMpossible, UNlikely, INconceivable), so a
    probe could read the prefix. The lexicon breaks this deliberately:
    UNdeniable / INdisputable / UNquestionable are negative-prefixed words at
    MAXIMUM certainty, and doubtful / dubious / questionable are prefix-free
    words at low certainty. We still report corr(labels, prefix) and a
    prefix-residualized refit.
  * Orthogonality vs the opinion (Evaluation) axis — is certainty just
    goodness? — and vs size (unrelated control), in a shared standardized space.

Writes experiments/results/certainty_frame.json.
Run:  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python experiments/certainty_frame.py
"""
from __future__ import annotations
import json, os, sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from ordering_frames import cv_r, ridge_dir
from freq_audit_family import collect_last_t, residualize
from turnstyle.frame_library import CANONICAL_FRAMES

from wordfreq import zipf_frequency

TEMPLATE = "It is {w} that it happened."

# certainty scalar, -3 (impossible) .. +3 (certain).
# Prefix-balanced by construction: neg-prefix at BOTH ends, prefix-free at both.
CERTAINTY = {
    "impossible": -3, "inconceivable": -3,
    "unlikely": -2, "implausible": -2, "doubtful": -2, "dubious": -2,
    "questionable": -1, "uncertain": -1, "unclear": -1,
    "possible": 0, "conceivable": 0,
    "plausible": 1, "credible": 1,
    "likely": 2, "probable": 2, "evident": 2, "obvious": 2,
    "certain": 3, "definite": 3, "undeniable": 3, "indisputable": 3,
    "unquestionable": 3,
}
NEG_PREFIXES = ("im", "in", "un", "dis")

def has_neg_prefix(w: str) -> bool:
    return any(w.startswith(p) for p in NEG_PREFIXES)


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    mid = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(mid)
    mdl = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.float16).to(dev)

    words = list(CERTAINTY)
    y = np.array([CERTAINTY[w] for w in words], dtype=float)
    z = np.array([zipf_frequency(w, "en") for w in words])
    pfx = np.array([float(has_neg_prefix(w)) for w in words])
    lab_zipf = float(np.corrcoef(y, z)[0, 1])
    lab_pfx = float(np.corrcoef(y, pfx)[0, 1])
    print(f"{len(words)} words | label corr: zipf {lab_zipf:+.3f}, neg-prefix {lab_pfx:+.3f}",
          flush=True)

    acts = collect_last_t(mdl, tok, dev, words, TEMPLATE)
    n_layers = acts[words[0]].shape[0]

    # recoverability + both residualizations, layer sweep
    best = {"raw": (-9, -1), "zipf_res": (-9, -1), "pfx_res": (-9, -1)}
    y_zres, y_pres = residualize(y, z), residualize(y, pfx)
    for L in range(n_layers):
        X = np.array([acts[w][L] for w in words])
        for key, yy in (("raw", y), ("zipf_res", y_zres), ("pfx_res", y_pres)):
            r = cv_r(X, yy)
            if r > best[key][0]:
                best[key] = (r, L)
    for k, (r, L) in best.items():
        print(f"  {k:9s} r={r:+.3f} @L{L}", flush=True)

    # orthogonality vs opinion (Evaluation) and size, shared standardized space
    ortho = {}
    for L0 in (8, 12):
        pools = {"certainty": (words, y, TEMPLATE)}
        for fam in ("opinion", "size"):
            spec = CANONICAL_FRAMES[fam]
            fw = list(spec["data"])
            pools[fam] = (fw, np.array([spec["data"][w] for w in fw], dtype=float),
                          spec.get("template", "It is a {w} object."))
        acts_all = dict(acts)
        for fam in ("opinion", "size"):
            fw, _, tmpl = pools[fam]
            acts_all.update(collect_last_t(mdl, tok, dev, fw, tmpl))
        stack = np.concatenate([np.array([acts_all[w][L0] for w in pools[n][0]])
                                for n in pools])
        mu, sd = stack.mean(0), stack.std(0) + 1e-6
        dirs = {n: ridge_dir((np.array([acts_all[w][L0] for w in pools[n][0]]) - mu) / sd,
                             pools[n][1]) for n in pools}
        ortho[f"L{L0}"] = {n: round(abs(float(dirs["certainty"] @ dirs[n])), 3)
                           for n in ("opinion", "size")}
        print(f"  |cos| @L{L0}: vs opinion {ortho[f'L{L0}']['opinion']:.3f}, "
              f"vs size {ortho[f'L{L0}']['size']:.3f}", flush=True)

    results = {"model": mid, "template": TEMPLATE, "n_words": len(words),
               "label_zipf_r": round(lab_zipf, 3), "label_prefix_r": round(lab_pfx, 3),
               "recoverability": {k: {"r": round(r, 3), "layer": L}
                                  for k, (r, L) in best.items()},
               "orthogonality": ortho}
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "results", "certainty_frame.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
