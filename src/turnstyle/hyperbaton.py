"""hyperbaton via the subjectivity axis — built on the adjective-polarity work.

English adjective ordering (opinion → size → age → shape → color → origin →
material → noun) is predicted by SUBJECTIVITY: the more subjective an adjective,
the farther it sits from the noun. So of two candidate orderings, the correct one
is sorted by DECREASING subjectivity. We reuse the polarity machinery's
`BipolarAxis`: fit ONE axis from the two extremes (opinion = high/subjective vs
material = low/intrinsic), project each option's adjectives, and pick the option
with fewer decreasing-order inversions.

The axis transfers cross-lingually (same finding as the polarity primitive — see
memory subjectivity_cross_lingual_ordering), so this replaces the swollm L8
k-means + hardcoded-category-order solver with the principled, language-agnostic
path. The seed lists below are CALIBRATION-ONLY (training-time); runtime uses the
projected axis, not the words.
"""
from __future__ import annotations

import re

from turnstyle.semantic_frame import fit_axis_from_vectors, _word_vectors

# calibration extremes (the two ends of the subjectivity hierarchy)
_OPINION = ["lovely", "nice", "ugly", "horrible"]        # subjective → high
_MATERIAL = ["metallic", "plastic", "golden", "wooden"]  # intrinsic  → low
_TMPL = "It is a {w} object."
DEFAULT_LAYER = 14

_OPT_RE = re.compile(r"\(([A-Z])\)\s+(.+)")


def fit_subjectivity_axis(model, tokenizer, device, layer: int = DEFAULT_LAYER):
    """Fit the subjectivity BipolarAxis (opinion=high vs material=low)."""
    hi = _word_vectors(model, tokenizer, device, _OPINION, layer, _TMPL)
    lo = _word_vectors(model, tokenizer, device, _MATERIAL, layer, _TMPL)
    return fit_axis_from_vectors("subjectivity", "material", "opinion", layer, hi, lo)


def _inversions(scores) -> int:
    """# pairs (i<j) where subjectivity increases — violates decreasing order."""
    return sum(1 for i in range(len(scores)) for j in range(i + 1, len(scores))
               if scores[i] < scores[j])


def _adjectives(option_text: str) -> list[str]:
    """The adjective sequence of an option = its words minus the trailing noun."""
    return option_text.strip().rstrip(".").split()[:-1]


def solve_hyperbaton(prompt: str, model, tokenizer, device, axis):
    """Return the option letter "(X)" of the correctly-ordered sequence, or None.

    Structural gate (cheap, before any model forward): hyperbaton's two options are
    PERMUTATIONS of each other (same words, different order). If they aren't, this
    isn't hyperbaton — bail so an unrelated 2-option MC prompt can't mis-commit."""
    opts = _OPT_RE.findall(prompt)
    if len(opts) != 2:
        return None
    (la, ta), (lb, tb) = opts[0], opts[1]
    adj_a, adj_b = _adjectives(ta), _adjectives(tb)
    if len(adj_a) < 2 or sorted(ta.split()) != sorted(tb.split()):
        return None  # not a permutation pair → not hyperbaton

    if axis is None or model is None:
        return None
    sa = [axis.project(v) for v in
          _word_vectors(model, tokenizer, device, adj_a, axis.layer, _TMPL)]
    sb = [axis.project(v) for v in
          _word_vectors(model, tokenizer, device, adj_b, axis.layer, _TMPL)]
    ia, ib = _inversions(sa), _inversions(sb)
    if ia == ib:
        return None
    return f"({la})" if ia < ib else f"({lb})"


__all__ = ["fit_subjectivity_axis", "solve_hyperbaton", "DEFAULT_LAYER"]
