"""No-model tests for the tokenization control on frames.

A frame's depth profile is confounded by how badly its lexicon fragments: a word
costing 2-3 subwords has its meaning ASSEMBLED across layers, so it reads as
"localizing" for a tokenizer reason rather than a conceptual one. Mistral fragments
the frame lexicon at 1.47 subwords/word against ~1.11 elsewhere and is the most
"localized" model — plausibly for that reason alone, which is why fragmentation is
recorded on every Frame rather than being an optional analysis step.

Uses a stub tokenizer so these stay model-free.
"""
from __future__ import annotations

import numpy as np

from turnstyle.frame_library import (Frame, FrameLibrary, fragmentation,
                                     shared_single_token_words)
from turnstyle.semantic_frame import BipolarAxis


class StubTok:
    """Splits designated words into N pieces; everything else is one token.

    Returns real character offsets so the span logic under test is the same one
    `_collect` uses to pool the readout.
    """

    def __init__(self, splits=None):
        self.splits = splits or {}

    def __call__(self, sent, return_offsets_mapping=False):
        offs, i = [], 0
        for word in sent.split(" "):
            bare = word.strip(".,!?")
            n = self.splits.get(bare, 1)
            start = sent.index(bare, i)
            # carve the bare word into n contiguous character spans
            step = max(1, len(bare) // n)
            for k in range(n):
                s = start + k * step
                e = start + len(bare) if k == n - 1 else start + (k + 1) * step
                offs.append((s, e))
            i = start + len(bare)
        return {"offset_mapping": offs}


def _mk_frame(name="size", frag=None, single_only=False, H=4):
    ax = BipolarAxis(name, "-", "+", 0, np.zeros(H), np.ones(H), np.eye(H)[0], 0.0)
    return Frame(ax, data={"a": 0, "b": 1}, frag=frag, single_token_only=single_only)


def test_fragmentation_counts_in_context_spans():
    tok = StubTok({"enormous": 3, "tiny": 1})
    got = fragmentation(tok, ["enormous", "tiny"], "It is a {w}.")
    assert got == {"enormous": 3, "tiny": 1}


def test_fragmentation_never_returns_zero():
    # a word the tokenizer somehow yields no span for must floor at 1, not 0 --
    # a 0 would silently divide-by-zero downstream and read as "perfectly clean"
    tok = StubTok()
    got = fragmentation(tok, ["thing"], "It is a {w}.")
    assert got["thing"] >= 1


def test_shared_single_token_words_is_the_intersection():
    a = StubTok({"gigantic": 2})               # a splits gigantic
    b = StubTok({"minuscule": 2})              # b splits minuscule
    words = ["big", "gigantic", "minuscule"]
    assert shared_single_token_words([a, b], words, "It is a {w}.") == ["big"]
    # single tokenizer -> just its own clean words
    assert shared_single_token_words([a], words, "It is a {w}.") == ["big", "minuscule"]


def test_frame_reports_fragmentation_stats():
    f = _mk_frame(frag={"a": 1, "b": 1, "c": 3})
    assert f.frag_mean == 5 / 3
    assert f.single_token_frac == 2 / 3
    assert f.single_token_words == ["a", "b"]


def test_depth_claim_is_safe_gates_on_fragmentation():
    clean = _mk_frame(frag={w: 1 for w in "abcdefghij"})
    dirty = _mk_frame(frag={"a": 1, "b": 2, "c": 3})
    assert clean.depth_claim_is_safe() is True
    assert dirty.depth_claim_is_safe() is False
    # threshold is tunable but the default is strict
    assert dirty.depth_claim_is_safe(min_single=0.3) is True


def test_missing_frag_reports_unknown_not_clean():
    """A frame fitted before the control must NOT masquerade as tokenization-clean."""
    old = _mk_frame(frag=None)
    assert old.frag_mean is None
    assert old.single_token_frac is None
    assert old.depth_claim_is_safe() is False        # unknown is not safe


def test_frag_survives_json_and_npz_round_trip(tmp_path):
    lib = FrameLibrary(fingerprint="fp").add(
        _mk_frame(frag={"a": 1, "b": 2}, single_only=True))
    back = FrameLibrary.from_dict(lib.to_dict())
    assert back.frames["size"].frag == {"a": 1, "b": 2}
    assert back.frames["size"].single_token_only is True

    p = lib.save_npz(tmp_path / "f.npz")
    back2 = FrameLibrary.load_npz(p)
    assert back2.frames["size"].frag == {"a": 1, "b": 2}
    assert back2.frames["size"].single_token_only is True


def test_pre_control_npz_still_loads(tmp_path):
    """Bundled artifacts predate `frag`; they must load with frag=None, not crash."""
    lib = FrameLibrary(fingerprint="fp").add(_mk_frame(frag={"a": 1}))
    p = lib.save_npz(tmp_path / "g.npz")
    # strip frag from the stored metadata, mimicking an older file
    import json
    z = dict(np.load(p, allow_pickle=False))
    meta = json.loads(bytes(z["__meta__"]).decode())
    for m in meta["frames"].values():
        m.pop("frag", None)
        m.pop("single_token_only", None)
    z["__meta__"] = np.frombuffer(json.dumps(meta).encode(), dtype=np.uint8)
    np.savez_compressed(p, **z)

    back = FrameLibrary.load_npz(p)
    assert back.frames["size"].frag is None
    assert back.frames["size"].single_token_frac is None


def test_tokenization_report_flags_unknown_and_dirty():
    lib = FrameLibrary()
    lib.add(_mk_frame("clean", frag={w: 1 for w in "abcdefghij"}))
    lib.add(_mk_frame("dirty", frag={"a": 1, "b": 3}))
    lib.add(_mk_frame("unknown", frag=None))
    rep = lib.tokenization_report()
    assert rep["clean"]["depth_safe"] is True
    assert rep["dirty"]["depth_safe"] is False
    assert rep["unknown"]["depth_safe"] is None       # unknown, distinct from False
    assert rep["dirty"]["single_token_frac"] == 0.5
