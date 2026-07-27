"""Does the PLE table itself carry semantic frame structure? (bridge form (b))

The Per-Layer Embedding table is a pure LOOKUP: token id + layer -> a 256-dim row,
fetched with no compute ([[ple_flash_tiering]]).  If a frame scalar (size, animacy,
certainty...) is linearly recoverable from those rows, then frame structure lives in
flash-resident memory rather than in the forward pass -- which is exactly the
premise of [[flash_resident_probe_library]].

Reads the tables straight out of the safetensors shards.  NO model load, NO forward
passes -- the whole point is that this information is available without compute.

Sources compared per frame:
  base       `embed_tokens[t]`               2048   ordinary input embedding
  ple[l]     `embed_tokens_per_layer[t]`      256   the layer-l PLE row
  ple_all    the same, flattened             7680   all 30 layers at once
  acts       (from gemma3n_readout_sweep)    2048   stream0, for scale

`base` is the control that decides the whole thing.  A PLE row that predicts `size`
is only interesting if it beats the ordinary embedding -- otherwise it is just
"static token identity predicts a lexical scalar", which we already knew.

Two facts make the depth curve here the sharp version of the question:
  - PLE is written to AltUp streams 1-3, never stream 0 (`modeling_gemma3n.py:1348`),
    so this is genuinely a different channel from the activation sweep.
  - Gemma's 262k vocab gives high single-token coverage, where a 32k vocab would
    have fragmented most of the lexicon ([[color_semantic_frame]]).

    uv run python experiments/gemma3n_ple_bridge.py
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from turnstyle.frame_library import CANONICAL_FRAMES, _cv_r  # noqa: E402

MODEL = "google/gemma-3n-E2B-it"
N_LAYER, PLE_DIM = 30, 256
OUT = Path(__file__).parent / "results" / "gemma3n_ple_bridge.json"


def load_tables():
    """mmap the two embedding tables out of the shards — no model instantiation.

    Returns row-fetch closures rather than slices: the safetensors handles must stay
    open for the lifetime of use, and the stored dtype is bf16, which numpy cannot
    consume directly.
    """
    from safetensors import safe_open

    snap = glob.glob(str(Path.home() / ".cache/huggingface/hub/"
                         "models--google--gemma-3n-E2B-it/snapshots/*/"))[0]
    idx = json.load(open(Path(snap) / "model.safetensors.index.json"))["weight_map"]
    keys = {"base": "model.language_model.embed_tokens.weight",
            "ple": "model.language_model.embed_tokens_per_layer.weight"}
    handles, fetch = [], {}
    for short, key in keys.items():
        f = safe_open(Path(snap) / idx[key], framework="pt")
        handles.append(f)                      # keep alive
        sl = f.get_slice(key)

        def get(rows, _sl=sl):
            return np.stack([_sl[int(i), :].float().numpy() for i in rows])

        fetch[short] = get
    return fetch, handles


def token_ids(tok, words):
    """Last-subword id of each word as it appears mid-sentence (leading space).

    Meaning assembles at the FINAL subword ([[color_semantic_frame]]), so for
    multi-token words that is the row to look up.
    """
    ids, single = {}, 0
    for w in words:
        enc = tok(" " + w, add_special_tokens=False)["input_ids"]
        ids[w] = enc[-1]
        single += (len(enc) == 1)
    return ids, single


def main():
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL)
    tables, _handles = load_tables()

    # activation-side numbers for scale, if the sweep has been run
    acts_r = {}
    sweep = OUT.with_name("gemma3n_readout_sweep.json")
    if sweep.exists():
        acts_r = {k: v["stream0"]["r"]
                  for k, v in json.loads(sweep.read_text())["results"].items()}

    results = {}
    print(f"{'frame':14s} {'cov':>7s} {'base':>7s} {'ple_all':>8s} {'ple_best':>9s} "
          f"{'@L':>4s} {'acts':>7s}   verdict")
    print("-" * 78)
    for name, spec in CANONICAL_FRAMES.items():
        data = spec["data"]
        words = list(data)
        y = np.asarray([data[w] for w in words], float)
        ids, single = token_ids(tok, words)
        rows = [ids[w] for w in words]

        base = tables["base"](rows)
        ple = tables["ple"](rows).reshape(len(rows), N_LAYER, PLE_DIM)

        r_base = _cv_r(base, y)
        r_all = _cv_r(ple.reshape(len(rows), -1), y)
        curve = [_cv_r(ple[:, L, :], y) for L in range(N_LAYER)]
        curve = [(-9.0 if c != c else c) for c in curve]
        bl = int(np.argmax(curve))

        r_act = acts_r.get(name, float("nan"))
        lift = curve[bl] - r_base
        verd = ("PLE > base" if lift > 0.05 else
                "≈ base" if lift > -0.05 else "PLE < base")
        results[name] = {"coverage": f"{single}/{len(words)}", "base": r_base,
                         "ple_all": r_all, "ple_best": curve[bl], "ple_layer": bl,
                         "acts_stream0": r_act, "curve": curve}
        print(f"{name:14s} {single:>3d}/{len(words):<3d} {r_base:>+7.3f} {r_all:>+8.3f} "
              f"{curve[bl]:>+9.3f} {'L' + str(bl):>4s} {r_act:>+7.3f}   {verd}")

    ks = list(results)
    print("-" * 78)
    for lab, key in (("MEAN base", "base"), ("MEAN ple_best", "ple_best"),
                     ("MEAN acts", "acts_stream0")):
        vals = [results[k][key] for k in ks]
        vals = [v for v in vals if v == v]
        print(f"  {lab:16s} {np.mean(vals):+.3f}")

    lifts = [results[k]["ple_best"] - results[k]["base"] for k in ks]
    print(f"\n  PLE beats base on {sum(l > 0.05 for l in lifts)}/{len(ks)} frames "
          f"(mean lift {np.mean(lifts):+.3f})")

    # NOT a depth-reorganization readout.  A 256-dim fit at N=11-15 words is badly
    # overparameterized, so CV r swings deep negative and the layer-to-layer spread
    # measures ESTIMATOR VARIANCE, not structure in the table.  Report the negative
    # excursion so the instability is visible rather than dressed up as a depth curve.
    print(f"\n  PLE per-layer curve — instability check (NOT a depth gradient):")
    for k in ks:
        c = np.array(results[k]["curve"])
        n = len(CANONICAL_FRAMES[k]["data"])
        print(f"    {k:14s} N={n:<3d} range [{c.min():+.2f}, {c.max():+.2f}]  "
              f"{'UNSTABLE — CV r goes negative' if c.min() < -0.1 else 'stable-ish'}")
    print("    (256 dims vs N words: negative CV r = overfit, not signal. Any depth"
          "\n     claim from these curves would be reading noise.)")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"model": MODEL, "results": results}, indent=2))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
