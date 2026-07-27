"""Gemma 3n E2B: layer x readout sweep for the canonical frame family.

Gemma 3n's `output_hidden_states` returns the PRE-reduction AltUp stack,
`(altup_num_inputs, B, T, H)` -- see [[ple_flash_tiering]].  Turnstyle's
`_collect` assumes `(B, T, H)`, so there is no single obvious vector to project a
frame onto.  Three candidates, and no a-priori winner:

  stream0   the ACTIVE stream -- the only one attention+MLP actually run on, and
            what the AltUp router conditions on.  But only cos 0.62 with the
            model's own output, and PLE never writes to it
            (`modeling_gemma3n.py:1348` adds to `corrected_predictions[1:]`).
  reduced   unembed-project streams 1-3, magnitude-match, mean -- reproduces the
            model's real reduction (verified cos 1.0000 vs last_hidden_state).
            CAVEAT: `altup_unembed_projections` are trained for the END of the
            stack; applying them mid-stack is an approximation with no guarantee.
  concat    all four streams, 8192-dim.  Let the ridge decide.  Nearly free
            because the fit is DUAL ridge (cost scales with N words, not D).

Reports best held-out CV r and its layer for every (frame, readout), plus the
depth pattern per readout -- the lexical-early / relational-late gradient from
[[frame_sweep_patterns]] is only cross-model comparable once a readout is pinned.

    uv run python experiments/gemma3n_readout_sweep.py            # all 14 frames
    uv run python experiments/gemma3n_readout_sweep.py --frames size,number,time

Must run OUTSIDE the sandbox (torch misreports MPS as unavailable inside it).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from turnstyle.frame_library import CANONICAL_FRAMES, _cv_r, _DEFAULT_TEMPLATE  # noqa: E402

MODEL = "google/gemma-3n-E2B-it"
READOUTS = ("stream0", "reduced", "concat")
OUT = Path(__file__).parent / "results" / "gemma3n_readout_sweep.json"
CACHE = Path(__file__).parent / "results" / "_gemma3n_acts"

# concat is 4x the dims of stream0/reduced at the same ridge alpha, so a fixed alpha
# under-regularizes it -- the readout comparison is confounded unless alpha is tuned
# per (readout, layer).  Sweeping it is pure numpy once activations are cached.
ALPHAS = (1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0)


def collect(model, tok, dev, words, template):
    """{word: (raw (L+1, S, H), reduced (L+1, H))} at the word's LAST subword.

    One forward per word; both views derived from it.  Last-subword pooling is
    mandatory for multi-token words -- meaning assembles at the final subword
    ([[color_semantic_frame]]).
    """
    inner = model.model
    n_alt = model.config.get_text_config().altup_num_inputs
    out = {}
    for w in words:
        sent = template.format(w=w)
        cs = sent.rfind(w)
        ce = cs + len(w)
        enc = tok(sent, return_offsets_mapping=True, return_tensors="pt")
        offs = enc.pop("offset_mapping")[0].tolist()
        enc = {k: v.to(dev) for k, v in enc.items()}
        with torch.no_grad():
            hs = model(**enc, output_hidden_states=True).hidden_states
            stk = torch.stack(hs, 0)                     # (L+1, S, B, T, H)
            idxs = [k for k, (s, e) in enumerate(offs) if e > cs and s < ce] or [-1]
            tokpos = idxs[-1]
            raw = stk[:, :, 0, tokpos, :]                # (L+1, S, H)

            # Reproduce modeling_gemma3n.py:1717-1729 at EVERY layer.  The final
            # RMSNorm is skipped deliberately: it is a per-channel rescale and
            # _cv_r standardizes features per fold anyway.
            tgt = torch.mean(raw[:, 0] ** 2, dim=-1, keepdim=True) ** 0.5
            eps = torch.tensor(1e-5, device=raw.device)
            temp = [raw[:, 0]]
            for i in range(1, n_alt):
                p = inner.altup_unembed_projections[i - 1](raw[:, i])
                mag = torch.sqrt(torch.maximum(torch.mean(p**2, dim=-1, keepdim=True), eps))
                temp.append(p * tgt / mag)
            red = torch.mean(torch.stack(temp), dim=0)   # (L+1, H)

        out[w] = (raw.float().cpu().numpy(), red.float().cpu().numpy())
    return out


def view(acts, words, readout, layer):
    """Feature matrix (N, D) for one readout at one layer."""
    if readout == "stream0":
        return np.array([acts[w][0][layer, 0, :] for w in words])
    if readout == "reduced":
        return np.array([acts[w][1][layer] for w in words])
    if readout == "concat":
        return np.array([acts[w][0][layer].reshape(-1) for w in words])
    raise ValueError(readout)


def load_or_collect(name, spec, model, tok, dev):
    """Cached activations. Collection needs a 10GB model load + one forward per word;
    everything downstream is numpy, so cache it and make re-analysis free."""
    words = list(spec["data"])
    path = CACHE / f"{name}.npz"
    if path.exists():
        z = np.load(path)
        if list(z["words"]) == words:
            return {w: (z[f"raw::{w}"], z[f"red::{w}"]) for w in words}
    if model is None:
        raise SystemExit(f"no cache for '{name}' and --no-model given; run once without it")
    acts = collect(model, tok, dev, words, spec.get("template", _DEFAULT_TEMPLATE))
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, words=np.array(words),
                        **{f"raw::{w}": acts[w][0] for w in words},
                        **{f"red::{w}": acts[w][1] for w in words})
    return acts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", default="", help="comma-separated subset")
    ap.add_argument("--alpha-sweep", action="store_true",
                    help="tune ridge alpha per (readout, layer) — removes the "
                         "dimension/regularization confound on concat")
    ap.add_argument("--no-model", action="store_true", help="cache-only, skip the model load")
    args = ap.parse_args()

    dev, tok, model = "cpu", None, None
    if not args.no_model:
        from transformers import AutoTokenizer, Gemma3nForCausalLM

        dev = "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"device: {dev}", flush=True)
        tok = AutoTokenizer.from_pretrained(MODEL)
        model = Gemma3nForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(dev).eval()

    specs = dict(CANONICAL_FRAMES)
    if args.frames:
        want = [f.strip() for f in args.frames.split(",")]
        specs = {k: v for k, v in specs.items() if k in want}
    alphas = ALPHAS if args.alpha_sweep else (10.0,)
    print(f"frames: {len(specs)}  words: {sum(len(s['data']) for s in specs.values())}  "
          f"alphas: {list(alphas)}\n", flush=True)

    results = {}
    for name, spec in specs.items():
        data = spec["data"]
        words = list(data)
        y = np.asarray([data[w] for w in words], float)
        acts = load_or_collect(name, spec, model, tok, dev)
        n_layers = acts[words[0]][0].shape[0]

        row = {}
        for ro in READOUTS:
            curve, best_a = [], []
            for L in range(n_layers):
                X = view(acts, words, ro, L)
                rs = [_cv_r(X, y, alpha=a) for a in alphas]
                rs = [(-9.0 if r != r else r) for r in rs]
                j = int(np.argmax(rs))
                curve.append(rs[j])
                best_a.append(alphas[j])
            best = int(np.argmax(curve))
            row[ro] = {"r": curve[best], "layer": best, "curve": curve,
                       "alpha": best_a[best]}
        results[name] = row

        cells = "  ".join(f"{ro}={row[ro]['r']:+.3f}@L{row[ro]['layer']:<2d}" for ro in READOUTS)
        win = max(READOUTS, key=lambda r: row[r]["r"])
        print(f"  {name:14s} {cells}   win={win}", flush=True)

    # ---- summary ----
    print(f"\n{'=' * 74}\nBEST CV r BY READOUT\n{'=' * 74}")
    print(f"  {'frame':14s} " + "".join(f"{ro:>12s}" for ro in READOUTS) + "   winner")
    for name, row in results.items():
        win = max(READOUTS, key=lambda r: row[r]["r"])
        print(f"  {name:14s} " + "".join(f"{row[ro]['r']:>+12.3f}" for ro in READOUTS)
              + f"   {win}")

    print(f"\n  {'MEAN':14s} " + "".join(
        f"{np.mean([results[n][ro]['r'] for n in results]):>+12.3f}" for ro in READOUTS))
    print(f"  {'MIN (floor)':14s} " + "".join(
        f"{min(results[n][ro]['r'] for n in results):>+12.3f}" for ro in READOUTS))
    wins = {ro: sum(max(READOUTS, key=lambda r: results[n][r]["r"]) == ro for n in results)
            for ro in READOUTS}
    print(f"  {'wins':14s} " + "".join(f"{wins[ro]:>12d}" for ro in READOUTS))

    # A peak layer is only meaningful if the curve HAS a peak.  Report plateau width
    # and spread alongside it, always -- see [[peak_layer_not_load_bearing]].
    print(f"\n{'=' * 74}\nPEAK LAYER — stream0, WITH stability\n{'=' * 74}")
    print(f"  {'frame':14s} {'peak':>5s} {'≤.01':>6s} {'≤.02':>6s} {'spread':>8s}   verdict")
    for name, row in results.items():
        c = np.array(row["stream0"]["curve"], float)
        c[c < -1] = np.nan
        mx = np.nanmax(c)
        w1, w2 = int(np.nansum(c >= mx - 0.01)), int(np.nansum(c >= mx - 0.02))
        spread = float(np.nanmax(c[1:]) - np.nanmin(c[1:]))
        verdict = "FLAT — peak is noise" if spread < 0.20 else (
            "soft" if w2 > 6 else "real")
        print(f"  {name:14s} {'L' + str(row['stream0']['layer']):>5s} {w1:>6d} {w2:>6d} "
              f"{spread:>8.3f}   {verdict}")

    out = OUT.with_name(OUT.stem + ("_alphatuned" if args.alpha_sweep else "") + ".json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"model": MODEL, "readouts": list(READOUTS),
                               "alphas": list(alphas), "results": results}, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
