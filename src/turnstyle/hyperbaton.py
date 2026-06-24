"""hyperbaton via frame-category routing — the principled replacement for both the
swollm L8 k-means solver and the earlier single-subjectivity-axis solver.

English adjective ordering is opinion > size > age > shape > color > origin > material >
(purpose) > noun. Of two candidate orderings (which are permutations of each other), the
correct one sorts adjectives by ASCENDING canonical rung. We classify each adjective to
its rung by NEAREST-CENTROID over per-rung exemplar word-sets (calibration-only; runtime
uses activation centroids, so it's language-agnostic — category structure transfers), then
pick the option with fewer out-of-order pairs. Ties (equal inversions — e.g. same-rung
adjectives) are broken by the model's own fluency (sequence logprob of each option).

This fixes the single-axis failure (which mis-ranked the mid-hierarchy — origin floated
above shape/color): discrete category centroids separate origin/shape/color cleanly.
Validated SmolLM2 @L6: 85% (axis) -> 95.2% (this), committed 250/250, no abstains.

`fit_subjectivity_axis` is kept for back-compat (the old ModelProfile slot); the solver
no longer uses it.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from turnstyle.semantic_frame import fit_axis_from_vectors, _word_vectors

_TMPL = "It is a {w} object."
DEFAULT_LAYER = 6
_OPT_RE = re.compile(r"\(([A-Z])\)\s+(.+)")

# canonical ordering rungs (index = position); opinion first, material/purpose last.
RUNGS = ["opinion", "size", "age", "shape", "color", "origin", "material", "purpose"]
EXEMPLARS = {
    "opinion": "lovely nice ugly horrible wonderful terrible good awful comfortable".split(),
    "size": "big small large tiny huge enormous little gigantic".split(),
    "age": "old new young ancient modern brand-new antique aged".split(),
    "shape": "round square triangular rectangular oval pyramidal circular".split(),
    "color": "red blue green yellow black white grey silver brown purple".split(),
    "origin": "American Indian Japanese Egyptian Russian Chinese French German Mexican".split(),
    "material": "iron glass wooden cloth metallic plastic golden paper leather steel".split(),
    "purpose": "smoking driving hiking hunting cooking sleeping snorkeling whittling".split(),
}


@dataclass
class OrderingClassifier:
    """Nearest-centroid classifier mapping an adjective activation → canonical rung index."""
    rungs: list
    centroids: np.ndarray       # (R, H) standardized rung centroids
    mean: np.ndarray
    scale: np.ndarray
    layer: int

    def rung(self, vec: np.ndarray) -> int:
        z = (np.asarray(vec, float) - self.mean) / self.scale
        return int(np.argmin(((self.centroids - z) ** 2).sum(1)))


def fit_ordering_classifier(model, tokenizer, device,
                            layer: int = DEFAULT_LAYER) -> OrderingClassifier:
    """Build per-rung centroids from the exemplar word-sets (one shared standardization)."""
    allw = [w for r in RUNGS for w in EXEMPLARS[r]]
    V = _word_vectors(model, tokenizer, device, allw, layer, _TMPL).astype(float)
    mean, scale = V.mean(0), V.std(0) + 1e-6
    cents, i = [], 0
    for r in RUNGS:
        n = len(EXEMPLARS[r])
        cents.append(((V[i:i + n] - mean) / scale).mean(0))
        i += n
    return OrderingClassifier(RUNGS, np.array(cents), mean, scale, layer)


def _adjectives(option_text: str) -> list:
    """Adjective sequence of an option = its words minus the trailing noun."""
    return option_text.strip().rstrip(".").split()[:-1]


def is_hyperbaton(prompt: str) -> bool:
    """Cheap structural gate (no model): two options that are permutations of each other,
    each with >=2 adjectives. Lets the dispatcher avoid fitting the classifier otherwise."""
    opts = _OPT_RE.findall(prompt)
    if len(opts) != 2:
        return False
    (_, ta), (_, tb) = opts
    return len(_adjectives(ta)) >= 2 and sorted(ta.split()) == sorted(tb.split())


def _inversions(rungs: list) -> int:
    return sum(1 for i in range(len(rungs)) for j in range(i + 1, len(rungs))
               if rungs[i] > rungs[j])


def _seq_logprob(model, tokenizer, device, text: str) -> float:
    import torch
    enc = tokenizer(text, return_tensors="pt").to(device)
    ids = enc["input_ids"]
    with torch.no_grad():
        lg = model(**enc).logits[0, :-1].float().log_softmax(-1)
    return float(lg[range(ids.shape[1] - 1), ids[0, 1:]].sum())


def solve_hyperbaton(prompt, model, tokenizer, device, classifier):
    """Return the option letter of the correctly-ordered sequence, or None if not
    hyperbaton / no classifier. Always commits on a real hyperbaton pair (ties → fluency)."""
    if not is_hyperbaton(prompt) or classifier is None or model is None:
        return None
    (la, ta), (lb, tb) = _OPT_RE.findall(prompt)
    adj_a, adj_b = _adjectives(ta), _adjectives(tb)

    def rungs_of(adjs):
        V = _word_vectors(model, tokenizer, device, adjs, classifier.layer, _TMPL)
        return [classifier.rung(v) for v in V]

    ia, ib = _inversions(rungs_of(adj_a)), _inversions(rungs_of(adj_b))
    if ia != ib:
        return f"({la})" if ia < ib else f"({lb})"
    # tie (e.g. same-rung adjectives): the model's own fluency decides
    return (f"({la})" if _seq_logprob(model, tokenizer, device, ta)
            >= _seq_logprob(model, tokenizer, device, tb) else f"({lb})")


# ── back-compat: the old subjectivity axis (no longer used by the solver) ──────
_OPINION = ["lovely", "nice", "ugly", "horrible"]
_MATERIAL = ["metallic", "plastic", "golden", "wooden"]


def fit_subjectivity_axis(model, tokenizer, device, layer: int = 14):
    """Deprecated: the single opinion↔material axis. Kept for the ModelProfile slot;
    superseded by fit_ordering_classifier (frame-category routing)."""
    hi = _word_vectors(model, tokenizer, device, _OPINION, layer, _TMPL)
    lo = _word_vectors(model, tokenizer, device, _MATERIAL, layer, _TMPL)
    return fit_axis_from_vectors("subjectivity", "material", "opinion", layer, hi, lo)


__all__ = ["OrderingClassifier", "fit_ordering_classifier", "solve_hyperbaton",
           "is_hyperbaton", "fit_subjectivity_axis", "RUNGS", "EXEMPLARS", "DEFAULT_LAYER"]
