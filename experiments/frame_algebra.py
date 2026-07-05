"""
frame_algebra.py — do frame directions COMPOSE? weight ≈ α·size + β·density?

If the panel is a vector algebra rather than just a coordinate system, a
physically composite property (weight ~ size x density) should have its
direction largely inside span{size, density}. Either outcome is a result:
high projection = frames compose linearly; low = weight is its own primitive.

Method: fit weight / density / size (canonical) directions in one shared
standardized space; project the weight direction onto span{size, density}
(least squares) and report |projection| = cos(weight, span). Controls:
cos(opinion, span{size,density}) should be LOW (composition isn't generic),
and each lexicon passes the zipf column. "airy"-type words assigned to density
only (never both lexicons). Writes experiments/results/frame_algebra.json.
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

T_OBJ = "It is a {w} object."

WEIGHT = {"weightless": -3, "featherlight": -3, "feathery": -2, "light": -1,
          "lightweight": -1, "portable": -1, "heavy": 1, "hefty": 2, "weighty": 2,
          "cumbersome": 2, "leaden": 3, "ponderous": 3}
# v2 density: my adjective lexicon (airy/foamy <-> dense/solid) DIED the frequency
# death (label|zipf +0.70, residualized -0.002). Replaced with the material-noun
# density from material_investigate.py, which recovered 0.93 there: light vs dense
# MATERIALS, template "It is made of {w}." — label|zipf near zero by construction
# (foam/paper common-light, lead/steel common-dense).
DENSITY = {"foam": 0, "feather": 0, "paper": 0, "cork": 0, "balsa": 0, "straw": 0,
           "lead": 1, "steel": 1, "gold": 1, "iron": 1, "granite": 1, "concrete": 1}
T_MATERIAL = "It is made of {w}."


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    mid = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(mid)
    mdl = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.float16).to(dev)

    size = CANONICAL_FRAMES["size"]
    opinion = CANONICAL_FRAMES["opinion"]
    pools = {
        "weight": (list(WEIGHT), np.array(list(WEIGHT.values()), dtype=float), T_OBJ),
        "density": (list(DENSITY), np.array(list(DENSITY.values()), dtype=float), T_MATERIAL),
        "size": (list(size["data"]), np.array(list(size["data"].values()), dtype=float),
                 size.get("template", T_OBJ)),
        "opinion": (list(opinion["data"]),
                    np.array(list(opinion["data"].values()), dtype=float),
                    opinion.get("template", T_OBJ)),
    }

    acts = {}
    results = {"model": mid, "recoverability": {}, "label_zipf": {}, "algebra": {}}
    for name, (words, y, tmpl) in pools.items():
        acts.update(collect_last_t(mdl, tok, dev, words, tmpl))
        z = np.array([zipf_frequency(w, "en") for w in words])
        lab = float(np.corrcoef(y, z)[0, 1])
        y_res = residualize(y, z)
        n_layers = acts[words[0]].shape[0]
        best_raw = best_res = (-9, -1)
        for L in range(n_layers):
            X = np.array([acts[w][L] for w in words])
            r1, r2 = cv_r(X, y), cv_r(X, y_res)
            if r1 > best_raw[0]: best_raw = (r1, L)
            if r2 > best_res[0]: best_res = (r2, L)
        results["recoverability"][name] = {
            "raw": {"r": round(best_raw[0], 3), "layer": best_raw[1]},
            "zipf_res": {"r": round(best_res[0], 3), "layer": best_res[1]}}
        results["label_zipf"][name] = round(lab, 3)
        print(f"  {name:8s} n={len(words):2d} zipf {lab:+.2f}  "
              f"raw {best_raw[0]:+.3f}@L{best_raw[1]}  res {best_res[0]:+.3f}@L{best_res[1]}",
              flush=True)

    for L0 in (8, 12):
        stack = np.concatenate([np.array([acts[w][L0] for w in pools[n][0]])
                                for n in pools])
        mu, sd = stack.mean(0), stack.std(0) + 1e-6
        dirs = {n: ridge_dir((np.array([acts[w][L0] for w in pools[n][0]]) - mu) / sd,
                             pools[n][1]) for n in pools}
        B = np.stack([dirs["size"], dirs["density"]], axis=1)     # d x 2 basis
        def span_cos(v):
            coef, *_ = np.linalg.lstsq(B, v, rcond=None)
            return float(np.linalg.norm(B @ coef)), coef
        w_span, w_coef = span_cos(dirs["weight"])
        o_span, _ = span_cos(dirs["opinion"])
        row = {"cos_weight_span": round(w_span, 3),
               "alpha_size": round(float(w_coef[0]), 3),
               "beta_density": round(float(w_coef[1]), 3),
               "cos_weight_size": round(abs(float(dirs["weight"] @ dirs["size"])), 3),
               "cos_weight_density": round(abs(float(dirs["weight"] @ dirs["density"])), 3),
               "cos_size_density": round(abs(float(dirs["size"] @ dirs["density"])), 3),
               "control_cos_opinion_span": round(o_span, 3)}
        results["algebra"][f"L{L0}"] = row
        print(f"  @L{L0}: cos(weight, span{{size,density}})={w_span:.3f} "
              f"(α_size={w_coef[0]:+.2f}, β_density={w_coef[1]:+.2f})  "
              f"[pairwise w·s={row['cos_weight_size']:.2f} w·d={row['cos_weight_density']:.2f} "
              f"s·d={row['cos_size_density']:.2f}]  control opinion→span={o_span:.3f}",
              flush=True)

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "results", "frame_algebra.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
