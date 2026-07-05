"""
concreteness_frame.py — concreteness (abstract <-> concrete) from the Brysbaert
et al. (2014) norms: the largest free label source in psycholinguistics.

Stratified sample of well-known nouns across the 1-5 concreteness range;
template "They talked about the {w}." (works for chair and freedom alike).
Audit columns: zipf (standing rule) + orthogonality vs opinion/size/certainty.
Writes experiments/results/concreteness_frame.json.
"""
from __future__ import annotations
import csv, json, os, sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from ordering_frames import cv_r, ridge_dir
from freq_audit_family import collect_last_t, residualize
from turnstyle.frame_library import CANONICAL_FRAMES

from wordfreq import zipf_frequency

TEMPLATE = "They talked about the {w}."
N_SAMPLE = 80


def load_words():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "data", "brysbaert_concreteness.txt")
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f, delimiter="\t"):
            w = r["Word"]
            if (r["Dom_Pos"] == "Noun" and w.isalpha() and w.islower()
                    and float(r["Percent_known"]) > 0.97
                    and zipf_frequency(w, "en") >= 3.7):
                rows.append((w, float(r["Conc.M"])))
    rows.sort(key=lambda t: t[1])
    step = max(1, len(rows) // N_SAMPLE)
    sample = rows[::step][:N_SAMPLE]
    return dict(sample)


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    mid = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(mid)
    mdl = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.float16).to(dev)

    data = load_words()
    words = list(data)
    y = np.array([data[w] for w in words])
    z = np.array([zipf_frequency(w, "en") for w in words])
    lab_zipf = float(np.corrcoef(y, z)[0, 1])
    print(f"concreteness: {len(words)} nouns, range {y.min():.2f}-{y.max():.2f}, "
          f"label|zipf {lab_zipf:+.3f}", flush=True)
    print("  e.g.", words[:4], "->", words[-4:], flush=True)

    acts = collect_last_t(mdl, tok, dev, words, TEMPLATE)
    n_layers = acts[words[0]].shape[0]
    y_res = residualize(y, z)
    best_raw = best_res = (-9, -1)
    for L in range(n_layers):
        X = np.array([acts[w][L] for w in words])
        r1, r2 = cv_r(X, y), cv_r(X, y_res)
        if r1 > best_raw[0]: best_raw = (r1, L)
        if r2 > best_res[0]: best_res = (r2, L)
    print(f"  raw r={best_raw[0]:+.3f} @L{best_raw[1]}  "
          f"zipf-res r={best_res[0]:+.3f} @L{best_res[1]}", flush=True)

    ortho = {}
    for L0 in (8, 12):
        pools = {"concreteness": (words, y, TEMPLATE)}
        for fam in ("opinion", "size", "certainty"):
            spec = CANONICAL_FRAMES[fam]
            fw = list(spec["data"])
            pools[fam] = (fw, np.array([spec["data"][w] for w in fw], dtype=float),
                          spec.get("template", "It is a {w} object."))
        acts_all = dict(acts)
        for fam in ("opinion", "size", "certainty"):
            fw, _, tmpl = pools[fam]
            acts_all.update(collect_last_t(mdl, tok, dev, fw, tmpl))
        stack = np.concatenate([np.array([acts_all[w][L0] for w in pools[n][0]])
                                for n in pools])
        mu, sd = stack.mean(0), stack.std(0) + 1e-6
        dirs = {n: ridge_dir((np.array([acts_all[w][L0] for w in pools[n][0]]) - mu) / sd,
                             pools[n][1]) for n in pools}
        ortho[f"L{L0}"] = {n: round(abs(float(dirs["concreteness"] @ dirs[n])), 3)
                           for n in ("opinion", "size", "certainty")}
        print(f"  |cos| @L{L0}:", ortho[f"L{L0}"], flush=True)

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "results", "concreteness_frame.json")
    with open(out, "w") as f:
        json.dump({"model": mid, "n": len(words), "label_zipf_r": round(lab_zipf, 3),
                   "raw": {"r": round(best_raw[0], 3), "layer": best_raw[1]},
                   "zipf_res": {"r": round(best_res[0], 3), "layer": best_res[1]},
                   "orthogonality": ortho}, f, indent=2)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
