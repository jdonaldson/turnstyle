"""Which ternary affect theory does SmolLM2 agree with strongest? (phase 1, intrinsic)

The major 3-factor theories agree on two axes (valence-ish, arousal-ish) and
differ mainly on the THIRD. We fit each theory's 3 axes as signed directions in a
SHARED standardized activation space (so directions are cross-comparable) and
score them on three intrinsic, falsifiable criteria per layer:

  1. independence  — mean pairwise |cos| of a theory's 3 axes (Osgood's own claim:
                     the factors are independent). lower = the model represents
                     them as genuine independent factors.
  2. encodability  — leave-one-word-out sign accuracy of each axis's pole contrast
                     (the polarity-loo method). higher = the model encodes it.
  3. third-axis horse race — hold a SHARED valence + arousal fixed, swap only the
                     third axis (Potency / Dominance / Tension / Attention-rejection);
                     score each on independence-from-{V,A}, encodability, and the
                     UNIQUE direction it carries beyond the V–A plane.

Plus a dimensionality check: does a 4th axis (novelty/unpredictability, Fontaine)
add an independent, encodable direction — i.e. is "ternary" even right?

Anti-circularity caveat: axes are fit AND scored on theory-defined anchor words,
so this measures internal consistency of the model's geometry, NOT external human
agreement. Phase 2 adds NRC-VAD / Affect-Control-Theory lexicons for that.

Surface variance dominates raw geometry (concept_geometry_dyf_measurement), so we
project axes orthogonal to the top-K activation PCs (the validated SemanticFrame
default, K=3).

    python -m experiments.epa_theory_horse_race            # all layers
    python -m experiments.epa_theory_horse_race --layers 8,11,14
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
TEMPLATES = ["It is {w}.", "They felt {w}."]   # average two framings to hedge template bias
SURFACE_SUPPRESS = 3

# ── anchor sets: each axis = (high-pole words, low-pole words) ────────────────
# Per-theory native axes (theories define axis 1 & 2 slightly differently — that
# difference is itself part of what we test).
THEORIES = {
    "EPA": {  # Osgood: Evaluation / Potency / Activity
        "Evaluation": (["good", "nice", "pleasant", "beautiful", "kind", "helpful"],
                       ["bad", "awful", "unpleasant", "ugly", "cruel", "harmful"]),
        "Potency":    (["strong", "powerful", "big", "mighty", "heavy", "large"],
                       ["weak", "powerless", "small", "feeble", "light", "tiny"]),
        "Activity":   (["active", "fast", "lively", "energetic", "quick", "dynamic"],
                       ["passive", "slow", "still", "sluggish", "quiet", "inert"]),
    },
    "PAD": {  # Mehrabian/Russell: Pleasure / Arousal / Dominance
        "Pleasure":   (["happy", "pleased", "content", "satisfied", "joyful", "delighted"],
                       ["unhappy", "displeased", "miserable", "frustrated", "sad", "annoyed"]),
        "Arousal":    (["aroused", "excited", "stimulated", "frantic", "jittery", "alert"],
                       ["relaxed", "calm", "sluggish", "dull", "sleepy", "drowsy"]),
        "Dominance":  (["dominant", "controlling", "commanding", "influential", "powerful", "assertive"],
                       ["submissive", "controlled", "helpless", "dependent", "meek", "powerless"]),
    },
    "Wundt": {  # pleasure-displeasure / excitation-calm / tension-relief
        "Pleasure":   (["pleasurable", "agreeable", "gratifying", "pleasant", "delightful", "enjoyable"],
                       ["displeasing", "disagreeable", "distressing", "unpleasant", "painful", "miserable"]),
        "Excitation": (["excited", "stirred", "animated", "aroused", "energized", "lively"],
                       ["inhibited", "subdued", "restrained", "calm", "still", "quiet"]),
        "Tension":    (["tense", "strained", "anxious", "edgy", "nervous", "uptight"],
                       ["relaxed", "loose", "untroubled", "serene", "easygoing", "carefree"]),
    },
    "Schlosberg": {  # pleasantness / activation / attention-rejection
        "Pleasantness": (["pleasant", "agreeable", "enjoyable", "nice", "lovely", "delightful"],
                         ["unpleasant", "disagreeable", "distasteful", "nasty", "horrible", "repugnant"]),
        "Activation":   (["activated", "aroused", "excited", "alert", "energetic", "stimulated"],
                         ["deactivated", "calm", "sleepy", "relaxed", "drowsy", "still"]),
        "AttentionRejection": (["attentive", "interested", "accepting", "receptive", "engaged", "curious"],
                               ["rejecting", "dismissive", "avoidant", "repelled", "disgusted", "indifferent"]),
    },
}

# Shared base for the third-axis horse race (one canonical valence + arousal).
SHARED = {
    "Valence": (["good", "pleasant", "positive", "nice", "wonderful", "lovely"],
                ["bad", "unpleasant", "negative", "nasty", "awful", "horrible"]),
    "Arousal": (["excited", "aroused", "energetic", "frantic", "alert", "stimulated"],
                ["calm", "relaxed", "sluggish", "sleepy", "quiet", "still"]),
}
THIRD_CANDIDATES = {
    "Potency":   THEORIES["EPA"]["Potency"],
    "Dominance": THEORIES["PAD"]["Dominance"],
    "Tension":   THEORIES["Wundt"]["Tension"],
    "AttentionRejection": THEORIES["Schlosberg"]["AttentionRejection"],
}
FOURTH = {  # Fontaine: is a 4th (novelty/unpredictability) axis warranted?
    "Novelty": (["unpredictable", "surprising", "novel", "unexpected", "strange", "sudden"],
                ["predictable", "familiar", "expected", "ordinary", "routine", "usual"]),
}


def _all_words():
    ws = set()
    for fac in THEORIES.values():
        for hi, lo in fac.values():
            ws.update(hi); ws.update(lo)
    for d in (SHARED, THIRD_CANDIDATES, FOURTH):
        for hi, lo in d.values():
            ws.update(hi); ws.update(lo)
    return sorted(ws)


def collect_acts(model, tok, device, words):
    """word -> (n_layers+1, H) hidden states, averaged over templates."""
    acts = {}
    for w in words:
        per_tmpl = []
        for tmpl in TEMPLATES:
            sent = tmpl.format(w=w)
            cs = sent.rfind(w); ce = cs + len(w)
            enc = tok(sent, return_offsets_mapping=True, return_tensors="pt")
            offs = enc.pop("offset_mapping")[0].tolist()
            enc = {k: v.to(device) for k, v in enc.items()}
            with torch.no_grad():
                hs = model(**enc, output_hidden_states=True).hidden_states
            tk = next((k for k, (s, e) in enumerate(offs) if e > cs and s < ce), -1)
            per_tmpl.append(torch.stack(hs, 0)[:, 0, tk, :].float().cpu().numpy())
        acts[w] = np.mean(per_tmpl, axis=0)   # (L+1, H)
    return acts


class LayerSpace:
    """Shared standardization + surface suppression at one layer."""
    def __init__(self, acts, layer, words, k_surf=SURFACE_SUPPRESS):
        M = np.vstack([acts[w][layer] for w in words]).astype(float)
        self.mean = M.mean(0)
        self.scale = M.std(0) + 1e-6
        Z = (M - self.mean) / self.scale
        self.surf = None
        if k_surf > 0:
            _, _, Vt = np.linalg.svd(Z - Z.mean(0), full_matrices=False)
            self.surf = Vt[:k_surf]
        self._z = {w: Z[i] for i, w in enumerate(words)}
        self._acts, self._layer = acts, layer

    def z(self, w):
        return self._z[w]

    def _suppress(self, d):
        if self.surf is not None:
            d = d - (d @ self.surf.T) @ self.surf
        return d / (np.linalg.norm(d) + 1e-12)

    def direction(self, hi, lo):
        d = np.mean([self.z(w) for w in hi], 0) - np.mean([self.z(w) for w in lo], 0)
        return self._suppress(d)

    def encodability(self, hi, lo):
        """leave-one-word-out sign accuracy."""
        words = [(w, +1) for w in hi] + [(w, -1) for w in lo]
        correct = 0
        for w, sign in words:
            rhi = [x for x in hi if x != w]
            rlo = [x for x in lo if x != w]
            d = self.direction(rhi, rlo)
            center = 0.5 * (np.mean([self.z(x) for x in rhi], 0)
                            + np.mean([self.z(x) for x in rlo], 0)) @ d
            proj = self.z(w) @ d - center
            correct += (proj > 0) == (sign > 0)
        return correct / len(words)


def mean_abs_cos(dirs):
    n = len(dirs)
    vals = [abs(float(dirs[i] @ dirs[j])) for i in range(n) for j in range(i + 1, n)]
    return float(np.mean(vals))


def analyze_layer(space):
    out = {"theories": {}, "third_race": {}, "fourth": {}}

    # per-theory: independence + encodability
    for tname, facs in THEORIES.items():
        dirs = [space.direction(*facs[f]) for f in facs]
        enc = {f: space.encodability(*facs[f]) for f in facs}
        out["theories"][tname] = {
            "independence_abscos": mean_abs_cos(dirs),
            "encodability": {f: round(v, 3) for f, v in enc.items()},
            "encodability_mean": round(float(np.mean(list(enc.values()))), 3),
        }

    # third-axis horse race: shared V + A, swap third
    vdir = space.direction(*SHARED["Valence"])
    adir = space.direction(*SHARED["Arousal"])
    VA = np.vstack([vdir, adir])
    Q, _ = np.linalg.qr(VA.T)            # orthonormal basis of the V–A plane
    for tname, (hi, lo) in THIRD_CANDIDATES.items():
        td = space.direction(hi, lo)
        proj = (td @ Q) @ Q.T            # projection onto V–A plane
        unique = float(np.linalg.norm(td - proj))   # fraction orthogonal to V,A
        out["third_race"][tname] = {
            "abscos_valence": round(abs(float(td @ vdir)), 3),
            "abscos_arousal": round(abs(float(td @ adir)), 3),
            "unique_beyond_VA": round(unique, 3),
            "encodability": round(space.encodability(hi, lo), 3),
        }

    # dimensionality: is a 4th (novelty) axis independent + encodable vs EPA's 3?
    epa_dirs = [space.direction(*THEORIES["EPA"][f]) for f in THEORIES["EPA"]]
    ndir = space.direction(*FOURTH["Novelty"])
    out["fourth"] = {
        "novelty_max_abscos_vs_EPA": round(max(abs(float(ndir @ d)) for d in epa_dirs), 3),
        "novelty_encodability": round(space.encodability(*FOURTH["Novelty"]), 3),
    }
    return out


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=MODEL)
    p.add_argument("--layers", help="comma-separated layer indices (default: all)")
    p.add_argument("--output", default="experiments/data/epa_theory_horse_race.json")
    args = p.parse_args(argv)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model)
    mdl = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float16).to(device)
    words = _all_words()
    print(f"device={device}  unique anchor words={len(words)}  collecting acts…", flush=True)
    acts = collect_acts(mdl, tok, device, words)
    n_layers = acts[words[0]].shape[0]
    layers = ([int(x) for x in args.layers.split(",")] if args.layers
              else list(range(n_layers)))

    report = {}
    for L in layers:
        r = analyze_layer(LayerSpace(acts, L, words))
        report[L] = r
        # inspectable per-layer line: theory independence + best third axis
        ind = {t: r["theories"][t]["independence_abscos"] for t in r["theories"]}
        best_theory = min(ind, key=ind.get)
        print(f"L{L:2d} indep(|cos|↓): "
              + "  ".join(f"{t}={ind[t]:.2f}" for t in ind)
              + f"  | most-independent={best_theory}", flush=True)

    # ── summary across layers ───────────────────────────────────────────────
    print("\n=== THIRD-AXIS HORSE RACE (mean over layers; shared V+A) ===")
    print(f"{'third axis':20s} {'enc↑':>6} {'unique↑':>8} {'cos|V↓':>7} {'cos|A↓':>7}")
    for t in THIRD_CANDIDATES:
        e = np.mean([report[L]["third_race"][t]["encodability"] for L in layers])
        u = np.mean([report[L]["third_race"][t]["unique_beyond_VA"] for L in layers])
        cv = np.mean([report[L]["third_race"][t]["abscos_valence"] for L in layers])
        ca = np.mean([report[L]["third_race"][t]["abscos_arousal"] for L in layers])
        print(f"{t:20s} {e:6.2f} {u:8.2f} {cv:7.2f} {ca:7.2f}")

    print("\n=== THEORY INDEPENDENCE + ENCODABILITY (mean over layers) ===")
    print(f"{'theory':12s} {'indep|cos↓':>10} {'enc_mean↑':>10}")
    for t in THEORIES:
        ind = np.mean([report[L]["theories"][t]["independence_abscos"] for L in layers])
        enc = np.mean([report[L]["theories"][t]["encodability_mean"] for L in layers])
        print(f"{t:12s} {ind:10.2f} {enc:10.2f}")

    nov_cos = np.mean([report[L]["fourth"]["novelty_max_abscos_vs_EPA"] for L in layers])
    nov_enc = np.mean([report[L]["fourth"]["novelty_encodability"] for L in layers])
    print(f"\n4th-axis (novelty): max|cos| vs EPA = {nov_cos:.2f} (↓=independent), "
          f"encodability = {nov_enc:.2f}  → 4D warranted if BOTH low-cos AND high-enc")

    import os
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(report, f, indent=1)
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
