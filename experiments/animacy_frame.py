"""
animacy_frame.py — the animacy hierarchy (inanimate -> plant -> invertebrate ->
vertebrate -> human) as a scalar frame. Grammatically load-bearing across
languages (case, agreement, word order) — the adjective-ordering success
pattern applied to nouns.

Lexicon size-balanced by construction (mountain/boulder big-inanimate,
whale/elephant big-animate; coin/pebble small-inanimate, mouse/baby small-
animate) so the frame can't be size in disguise; |cos| vs size reported anyway,
plus zipf. Template "They looked at the {w}."
Writes experiments/results/animacy_frame.json.
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

TEMPLATE = "They looked at the {w}."

ANIMACY = {
    # 0 inanimate (big and small)
    "rock": 0, "stone": 0, "hammer": 0, "table": 0, "bottle": 0, "mountain": 0,
    "coin": 0, "boulder": 0, "pebble": 0,
    # 1 plants
    "tree": 1, "flower": 1, "moss": 1, "fern": 1, "bush": 1, "vine": 1,
    # 2 invertebrates
    "worm": 2, "snail": 2, "beetle": 2, "spider": 2, "jellyfish": 2, "clam": 2,
    # 3 vertebrate animals (big and small)
    "dog": 3, "cat": 3, "horse": 3, "dolphin": 3, "eagle": 3, "rabbit": 3,
    "whale": 3, "mouse": 3, "elephant": 3,
    # 4 humans
    "child": 4, "woman": 4, "man": 4, "farmer": 4, "teacher": 4, "doctor": 4,
    "baby": 4,
}


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    mid = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(mid)
    mdl = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.float16).to(dev)

    words = list(ANIMACY)
    y = np.array([ANIMACY[w] for w in words], dtype=float)
    z = np.array([zipf_frequency(w, "en") for w in words])
    lab_zipf = float(np.corrcoef(y, z)[0, 1])
    print(f"animacy: {len(words)} nouns, label|zipf {lab_zipf:+.3f}", flush=True)

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
        pools = {"animacy": (words, y, TEMPLATE)}
        for fam in ("size", "opinion"):
            spec = CANONICAL_FRAMES[fam]
            fw = list(spec["data"])
            pools[fam] = (fw, np.array([spec["data"][w] for w in fw], dtype=float),
                          spec.get("template", "It is a {w} object."))
        acts_all = dict(acts)
        for fam in ("size", "opinion"):
            fw, _, tmpl = pools[fam]
            acts_all.update(collect_last_t(mdl, tok, dev, fw, tmpl))
        stack = np.concatenate([np.array([acts_all[w][L0] for w in pools[n][0]])
                                for n in pools])
        mu, sd = stack.mean(0), stack.std(0) + 1e-6
        dirs = {n: ridge_dir((np.array([acts_all[w][L0] for w in pools[n][0]]) - mu) / sd,
                             pools[n][1]) for n in pools}
        ortho[f"L{L0}"] = {n: round(abs(float(dirs["animacy"] @ dirs[n])), 3)
                           for n in ("size", "opinion")}
        print(f"  |cos| @L{L0}:", ortho[f"L{L0}"], flush=True)

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "results", "animacy_frame.json")
    with open(out, "w") as f:
        json.dump({"model": mid, "n": len(words), "label_zipf_r": round(lab_zipf, 3),
                   "raw": {"r": round(best_raw[0], 3), "layer": best_raw[1]},
                   "zipf_res": {"r": round(best_res[0], 3), "layer": best_res[1]},
                   "orthogonality": ortho}, f, indent=2)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
