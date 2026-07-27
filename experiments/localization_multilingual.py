"""Does MULTILINGUAL training drive frame localization?

The six-backbone audit found peak frame recoverability nearly constant
(.915-.938) while DEGRADATION away from the peak varies ~2x -- Mistral-7B .469
down to Phi-4-mini .221, with no relation to parameter count
([[peak_layer_not_load_bearing]]).  Ranking the six by language regime almost
sorts them: the three multilingual models take the top three slots, the two
English-only SmolLM2s sit below them -- and then Phi-4-mini, multilingual, is
LAST.  Correlating a model-card label across n=6 cannot settle that.

The mechanism the hypothesis implies is testable WITHIN a model.  If multilingual
training forces a language-agnostic bottleneck, and frames are readable in the
bottleneck but less so outside it, then cross-lingual transfer should be sharply
PEAKED in a multilingual model and coincide with where that model localizes.

Critically, the test cannot be about the LEVEL of transfer.  SmolLM2 -- nominally
English-only -- already poles Spanish at 1.00 and French at 1.00 with an
English-trained probe (`cross_language.py`, [[adjective_polarity_primitive]]).
So transfer is not the multilingual model's private property; the question is
whether it is LOCALIZED.

Per layer, on the matched-template en/es/fr/de polarity lexicon (only the
language varies by construction):

  mono      leave-one-axis-out EN pole accuracy -- generalization within English
  transfer  EN-fit direction applied to es/fr/de -- the language-agnostic signal

and then: is `transfer` peaked or flat, and does its peak sit where `mono` peaks?

Both curves come from ONE lexicon so the comparison is self-consistent; `mono`
also serves as a check that localization measured on this lexicon tracks the
localization measured on the frame family.

    uv run python experiments/localization_multilingual.py --model Qwen/Qwen2.5-1.5B-Instruct

Run OUTSIDE the sandbox (torch misreports MPS as unavailable inside it).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cross_language import LANGS  # noqa: E402

RESULTS = Path(__file__).parent / "results"


def collect(model, tok, dev, lang):
    """[(axis, pole, (L+1, H) last-subword acts, n_subwords)] for one language.

    n_subwords is carried so transfer can be split by target-word fragmentation:
    a single-token foreign word needs no assembly and its meaning is present at
    the embedding, whereas a 2-3 token word must be composed across the stack.
    That predicts different transfer DEPTH for the two groups within one model.
    """
    spec = LANGS[lang]
    out = []
    for axis, adj, pole in spec["adj"]:
        n_sub = len(tok(" " + adj, add_special_tokens=False)["input_ids"])
        for noun in spec["nouns"]:
            sent = spec["tmpl"].format(n=noun, adj=adj)
            cs = sent.rfind(adj)
            ce = cs + len(adj)
            enc = tok(sent, return_offsets_mapping=True, return_tensors="pt")
            offs = enc.pop("offset_mapping")[0].tolist()
            enc = {k: v.to(dev) for k, v in enc.items()}
            with torch.no_grad():
                hs = model(**enc, output_hidden_states=True).hidden_states
            stk = torch.stack(hs, 0)[:, 0, :, :]
            idxs = [k for k, (s, e) in enumerate(offs) if e > cs and s < ce] or [-1]
            out.append((axis, pole, stk[:, idxs[-1], :].float().cpu().numpy(), n_sub))
    return out


def direction(X, y):
    """Unit diff-in-means direction. Standardization-free so it transfers across
    languages without refitting any scale on the target language."""
    p, n = X[y > 0].mean(0), X[y < 0].mean(0)
    d = p - n
    return d / (np.linalg.norm(d) + 1e-12)


def acc(X, y, d, thr):
    return float(np.mean(((X @ d) > thr).astype(int) * 2 - 1 == y))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--dtype", default="float32", choices=("float32", "bfloat16"))
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=getattr(torch, args.dtype)).to(dev).eval()
    print(f"{args.model}  dtype={args.dtype}  device={dev}", flush=True)

    data = {lg: collect(model, tok, dev, lg) for lg in ("en", "es", "fr", "de")}
    n_layers = data["en"][0][2].shape[0]
    print(f"layers={n_layers}  " + " ".join(f"{lg}={len(v)}" for lg, v in data.items()),
          flush=True)

    en_axes = np.array([a for a, _, _, _ in data["en"]])
    en_y = np.array([p for _, p, _, _ in data["en"]], float)

    # Split the non-English evaluation set by target-word fragmentation. A
    # single-token foreign word carries its meaning at the embedding; a
    # multi-token one must be assembled across the stack. If that is what sets
    # transfer DEPTH, the two groups peak at different layers WITHIN one model --
    # a within-model test, so no cross-model confound.
    frag = {lg: np.array([n for _, _, _, n in data[lg]]) for lg in ("es", "fr", "de")}
    mono, xfer = [], {lg: [] for lg in ("es", "fr", "de")}
    split = {"single": [], "multi": []}
    for L in range(n_layers):
        Xen = np.array([v[L] for _, _, v, _ in data["en"]])

        # mono: leave-one-AXIS-out. Word-level holdout leaks through neighbours
        # (the lesson from the polarity primitive), so hold out whole axes.
        preds = np.zeros(len(en_y))
        for ax in set(en_axes):
            te = en_axes == ax
            tr = ~te
            d = direction(Xen[tr], en_y[tr])
            thr = float(np.median(Xen[tr] @ d))
            preds[te] = np.where((Xen[te] @ d) > thr, 1, -1)
        mono.append(float(np.mean(preds == en_y)))

        # transfer: fit on ALL of English, apply to each other language unchanged
        d = direction(Xen, en_y)
        thr = float(np.median(Xen @ d))
        pooled = {"single": [], "multi": []}
        for lg in xfer:
            Xl = np.array([v[L] for _, _, v, _ in data[lg]])
            yl = np.array([p for _, p, _, _ in data[lg]], float)
            xfer[lg].append(acc(Xl, yl, d, thr))
            one = frag[lg] == 1
            for key, m in (("single", one), ("multi", ~one)):
                if m.sum():
                    pooled[key].append((Xl[m], yl[m]))
        for key in split:
            if pooled[key]:
                X = np.concatenate([a for a, _ in pooled[key]])
                y = np.concatenate([b for _, b in pooled[key]])
                split[key].append(acc(X, y, d, thr))
            else:
                split[key].append(float("nan"))

    mono = np.array(mono)
    xall = np.mean([xfer[lg] for lg in xfer], axis=0)

    def stats(c, label):
        pk = int(np.argmax(c))
        return {"label": label, "peak_layer": pk, "peak_depth": pk / (len(c) - 1),
                "max": float(c.max()), "min": float(c[1:].min()),
                "spread": float(c[1:].max() - c[1:].min())}

    sm, sx = stats(mono, "mono_en"), stats(xall, "transfer")
    print(f"\n{'=' * 74}\nPER-LAYER (mono = LOO-axis EN, transfer = EN-fit -> es/fr/de)\n{'=' * 74}")
    print(f"  {'layer':>5s} {'depth':>6s} {'mono':>7s} {'xfer':>7s}   " +
          "  ".join(f"{lg:>5s}" for lg in xfer))
    for L in range(n_layers):
        star = "  <-mono pk" if L == sm["peak_layer"] else ""
        star += "  <-xfer pk" if L == sx["peak_layer"] else ""
        print(f"  {L:>5d} {L / (n_layers - 1):>6.0%} {mono[L]:>7.3f} {xall[L]:>7.3f}   " +
              "  ".join(f"{xfer[lg][L]:>5.2f}" for lg in xfer) + star)

    print(f"\n{'=' * 74}\nVERDICT\n{'=' * 74}")
    for s in (sm, sx):
        print(f"  {s['label']:9s} peak L{s['peak_layer']} ({s['peak_depth']:.0%})  "
              f"max {s['max']:.3f}  min {s['min']:.3f}  spread {s['spread']:.3f}")
    print(f"\n  peak coincidence: mono L{sm['peak_layer']} vs transfer L{sx['peak_layer']}"
          f"  (|delta depth| = {abs(sm['peak_depth'] - sx['peak_depth']):.0%})")
    print(f"  curve correlation mono vs transfer = "
          f"{np.corrcoef(mono[1:], xall[1:])[0, 1]:+.3f}")
    print(f"\n  transfer is {'PEAKED' if sx['spread'] > 0.20 else 'FLAT'} "
          f"(spread {sx['spread']:.3f}); "
          f"{'localized language-agnostic signal' if sx['spread'] > 0.20 else 'readable throughout — no bottleneck'}")

    # Within-model fragmentation control: does transfer depth track how many
    # subwords the FOREIGN word costs?
    n_one = int(sum((frag[lg] == 1).sum() for lg in frag))
    n_many = int(sum((frag[lg] > 1).sum() for lg in frag))
    print(f"\n{'=' * 74}\nFRAGMENTATION CONTROL (within model)\n{'=' * 74}")
    print(f"  target words: single-token {n_one}, multi-token {n_many}")
    ss = {}
    for key in ("single", "multi"):
        c = np.array(split[key], float)
        if np.isnan(c).all() or (key == "single" and n_one < 5) or (key == "multi" and n_many < 5):
            print(f"  {key:7s} n too small to read ({n_one if key == 'single' else n_many})")
            continue
        ss[key] = stats(c, key)
        print(f"  {key:7s} peak L{ss[key]['peak_layer']} ({ss[key]['peak_depth']:.0%})  "
              f"max {ss[key]['max']:.3f}  spread {ss[key]['spread']:.3f}")
    if len(ss) == 2:
        dd = ss["multi"]["peak_depth"] - ss["single"]["peak_depth"]
        print(f"\n  multi peaks {dd:+.0%} deeper than single")
        print(f"  => {'SUPPORTS assembly: fragmented words need more stack' if dd > 0.10 else 'NO fragmentation effect on transfer depth'}")

    extra = {"transfer_split": {k: list(map(float, v)) for k, v in split.items()},
             "split_stats": ss, "n_single": n_one, "n_multi": n_many,
             "frag_mean": {lg: float(frag[lg].mean()) for lg in frag}}

    tag = args.model.split("/")[-1]
    p = RESULTS / f"localization_multilingual_{tag}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"model": args.model, "dtype": args.dtype,
                             "mono": mono.tolist(),
                             "transfer_mean": xall.tolist(),
                             "transfer_per_lang": {k: list(map(float, v))
                                                   for k, v in xfer.items()},
                             "mono_stats": sm, "transfer_stats": sx, **extra}, indent=2))
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
