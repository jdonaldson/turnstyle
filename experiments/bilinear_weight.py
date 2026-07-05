"""
bilinear_weight.py — the multiplicative loophole in the algebra test.

frame_algebra showed weight's DIRECTION lies outside span{size, density}
(cos 0.22) — no LINEAR composition. But weight ~ size x density is
multiplicative, invisible to a span test. This tests composition at the
COORDINATE level on a 2x2 object factorial:

  24 object nouns crossing size {small, large} x density {light, dense}.
  Project each onto the fitted size / density / weight directions -> model
  coordinates (s_i, d_i, w_i). Then:
    sanity:      s_i separates small|large objects; d_i light|dense (AUC-ish)
    additive:    w ~ s + d                (R2_add)
    bilinear:    w ~ s + d + s*d          (R2_bil; the delta is the finding)
    factorial:   mean w by cell — does model-weight order feather < coin ~
                 balloon < anvil the way physics does?

Writes experiments/results/bilinear_weight.json.
Run:  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python experiments/bilinear_weight.py
"""
from __future__ import annotations
import json, os, sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from ordering_frames import ridge_dir
from freq_audit_family import collect_last_t
from turnstyle.frame_library import CANONICAL_FRAMES
from frame_algebra import WEIGHT, DENSITY, T_OBJ, T_MATERIAL

# 2x2 factorial: (size, density) -> nouns. Object template for projection.
OBJECTS = {  # word: (size +/-1, density +/-1)
    "feather": (-1, -1), "cork": (-1, -1), "bubble": (-1, -1),
    "petal": (-1, -1), "crumb": (-1, -1), "cotton": (-1, -1),
    "coin": (-1, +1), "bullet": (-1, +1), "pebble": (-1, +1),
    "marble": (-1, +1), "magnet": (-1, +1), "ingot": (-1, +1),
    "balloon": (+1, -1), "mattress": (+1, -1), "tent": (+1, -1),
    "haystack": (+1, -1), "kite": (+1, -1), "canoe": (+1, -1),
    "anvil": (+1, +1), "boulder": (+1, +1), "statue": (+1, +1),
    "engine": (+1, +1), "safe": (+1, +1), "girder": (+1, +1),
}
T_NOUN = "They looked at the {w}."


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    mid = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(mid)
    mdl = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.float16).to(dev)

    size = CANONICAL_FRAMES["size"]
    pools = {
        "weight": (list(WEIGHT), np.array(list(WEIGHT.values()), dtype=float), T_OBJ),
        "density": (list(DENSITY), np.array(list(DENSITY.values()), dtype=float), T_MATERIAL),
        "size": (list(size["data"]), np.array(list(size["data"].values()), dtype=float),
                 size.get("template", T_OBJ)),
    }
    objs = list(OBJECTS)

    acts = {}
    for name, (words, _, tmpl) in pools.items():
        acts.update(collect_last_t(mdl, tok, dev, words, tmpl))
    obj_acts = collect_last_t(mdl, tok, dev, objs, T_NOUN)

    results = {"model": mid, "layers": {}}
    for L0 in (8, 12, 16):
        stack = np.concatenate([np.array([acts[w][L0] for w in pools[n][0]])
                                for n in pools] +
                               [np.array([obj_acts[w][L0] for w in objs])])
        mu, sd = stack.mean(0), stack.std(0) + 1e-6
        dirs = {n: ridge_dir((np.array([acts[w][L0] for w in pools[n][0]]) - mu) / sd,
                             pools[n][1]) for n in pools}
        X = (np.array([obj_acts[w][L0] for w in objs]) - mu) / sd
        s = X @ dirs["size"]; d = X @ dirs["density"]; w = X @ dirs["weight"]
        gt_s = np.array([OBJECTS[o][0] for o in objs], dtype=float)
        gt_d = np.array([OBJECTS[o][1] for o in objs], dtype=float)
        gt_w = gt_s + gt_d          # physics-in-logs ground truth ordering

        def r(a, b): return float(np.corrcoef(a, b)[0, 1])
        def fit_r2(feats, target):
            A = np.column_stack(feats + [np.ones(len(target))])
            coef, *_ = np.linalg.lstsq(A, target, rcond=None)
            pred = A @ coef
            ss = 1 - ((target - pred) ** 2).sum() / ((target - target.mean()) ** 2).sum()
            return float(ss)

        r2_add = fit_r2([s, d], w)
        r2_bil = fit_r2([s, d, s * d], w)
        row = {"sanity_size_r": round(r(s, gt_s), 3),
               "sanity_density_r": round(r(d, gt_d), 3),
               "weight_vs_gt_r": round(r(w, gt_w), 3),
               "R2_additive": round(r2_add, 3), "R2_bilinear": round(r2_bil, 3),
               "delta_R2": round(r2_bil - r2_add, 3),
               "cell_means_w": {f"s{si:+d}d{di:+d}": round(float(
                   np.mean([w[i] for i, o in enumerate(objs) if OBJECTS[o] == (si, di)])), 2)
                   for si in (-1, 1) for di in (-1, 1)}}
        results["layers"][f"L{L0}"] = row
        print(f"@L{L0}: sanity s={row['sanity_size_r']:+.2f} d={row['sanity_density_r']:+.2f} "
              f"| w vs gt {row['weight_vs_gt_r']:+.2f} | R2 add {r2_add:.3f} -> bil {r2_bil:.3f} "
              f"(Δ{r2_bil - r2_add:+.3f}) | cells {row['cell_means_w']}", flush=True)

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "results", "bilinear_weight.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
