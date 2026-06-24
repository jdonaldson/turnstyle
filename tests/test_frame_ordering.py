"""No-model tests for the frame-as-column ordering path: superlative parsing, membership
routing, rank/route math on synthetic frames, and graceful abstention."""
from __future__ import annotations

import numpy as np

from turnstyle.frame_library import FrameLibrary, Frame, _superlative_roots
from turnstyle.frame_ordering import solve_frame_ordering, _candidates
from turnstyle.semantic_frame import BipolarAxis


def test_superlative_roots():
    assert "big" in _superlative_roots("biggest")
    assert "small" in _superlative_roots("smallest")
    assert "heavy" in _superlative_roots("heaviest")
    assert "old" in _superlative_roots("oldest")


def test_candidates_parses_superlatives_and_most_least():
    cands = dict((a, l) for a, l in _candidates("which is the biggest and most ancient?"))
    assert cands.get("biggest") is False
    assert cands.get("ancient") is False
    least = dict((a, l) for a, l in _candidates("the least old one"))
    assert least.get("old") is True


def _toy_library():
    """A library with one 'size' frame whose direction is e0; values keyed in .data so
    membership routing works, projection deterministic."""
    H = 4
    ax = BipolarAxis("size", "tiny", "huge", 0, np.zeros(H), np.ones(H), np.eye(H)[0], 0.0)
    f = Frame(ax, data={"tiny": -3, "small": -2, "big": 2, "huge": 3}, coord_scale=1.0)
    return FrameLibrary().add(f)


def test_route_membership_no_model():
    lib = _toy_library()
    assert lib.route("biggest") == ("size", 1)     # big -> high pole
    assert lib.route("smallest") == ("size", -1)   # small -> low pole
    assert lib.route("funniest") is None           # not a frame member -> abstain


def test_solve_abstains_without_library_or_superlative():
    assert solve_frame_ordering("Which is the biggest?\nOptions:\n(A) ant\n(B) cow",
                                None, None, None, None) is None
    lib = _toy_library()
    # no superlative that routes -> abstain (model args unused on this path)
    assert solve_frame_ordering("Which one do you like?\nOptions:\n(A) ant\n(B) cow",
                                lib, None, None, None) is None


def test_rank_orders_by_projection():
    # vectors whose e0 component encodes size; rank should sort by it
    H = 4
    ax = BipolarAxis("size", "tiny", "huge", 0, np.zeros(H), np.ones(H), np.eye(H)[0], 0.0)
    f = Frame(ax, data={"tiny": -3, "huge": 3})
    # project is e0 component; build a fake _collect via direct projection check
    vecs = {"ant": np.array([0.1, 0, 0, 0]), "whale": np.array([5.0, 0, 0, 0]),
            "cat": np.array([1.0, 0, 0, 0])}
    scored = sorted(((c, f.project(v)) for c, v in vecs.items()),
                    key=lambda kv: kv[1], reverse=True)
    assert [c for c, _ in scored] == ["whale", "cat", "ant"]
