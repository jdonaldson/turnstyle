"""Build + validate the canonical FrameLibrary on a model, then persist it.

Fits the standard frame family (adjective-ordering rungs + number + time), prints
recoverability + the orthogonality matrix, projects a few probe words to sanity-check
the coordinates, and saves the library JSON (fingerprint-addressed under data/frames/).
"""
from __future__ import annotations

import sys

import numpy as np

from turnstyle.frame_library import FrameLibrary


def main():
    import torch
    from pathlib import Path
    from transformers import AutoModelForCausalLM, AutoTokenizer
    mid = sys.argv[1] if len(sys.argv) > 1 else "HuggingFaceTB/SmolLM2-1.7B-Instruct"
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(mid)
    mdl = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.float16).to(dev)

    lib = FrameLibrary().fit_canonical(mdl, tok, dev)

    print("\n=== recoverability (best held-out CV r per frame) ===")
    for n, r in lib.recoverability().items():
        print(f"  {n:9s} r={r:+.3f} @L{lib.frames[n].layer}")

    names, M = lib.orthogonality(mdl, tok, dev, layer=8)
    print("\n=== orthogonality |cos| @L8 ===")
    print("           " + " ".join(f"{n[:7]:>7s}" for n in names))
    for i, a in enumerate(names):
        print(f"  {a:9s} " + " ".join(f"{M[i, j]:7.3f}" for j in range(len(names))))
    off = M[~np.eye(len(names), dtype=bool)]
    print(f"  mean off-diagonal |cos| = {off.mean():.3f}  max = {off.max():.3f}")

    print("\n=== sample projections (coordinate per frame) ===")
    for w in ("enormous", "ancient", "wonderful", "metallic", "round", "blue"):
        c = lib.project_word(w, mdl, tok, dev)
        top = sorted(c.items(), key=lambda kv: -abs(kv[1]))[:3]
        print(f"  {w:10s} " + "  ".join(f"{n}={v:+.2f}" for n, v in top))

    from turnstyle.profile import model_fingerprint
    fp = model_fingerprint(mdl)
    out = Path("data/frames"); out.mkdir(parents=True, exist_ok=True)
    p = lib.save(out / f"{fp}.json")
    print(f"\nsaved -> {p}  (fingerprint {fp})")
    # round-trip check
    lib2 = FrameLibrary.load(p)
    assert lib2.names == lib.names
    print(f"reload OK: {len(lib2)} frames")


if __name__ == "__main__":
    main()
