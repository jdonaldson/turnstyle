"""No-model tests for position-marginalized choice scoring (dispatch).

The ChoiceProbe reads a contextualized last-token hidden state, so its per-option
score carries a POSITION component (snarks: 100% in natural order, 37.5% reordered).
`_score_choice_marginalized` averages each option over the slots it occupies (cyclic
shifts) so position cancels. These tests inject a fake position-biased scorer (no
model) and assert: (1) single-pass picks the WRONG option when position dominates a
small content gap, (2) marginalization recovers the content winner, and (3) the
marginalized answer is identical under input reordering (binary = fully invariant).
"""
import pytest

import turnstyle.dispatch as D


class _Artifact:
    def _format(self, label):           # mirrors ProbeArtifact._format
        return f"({label})"


def _ctx():
    c = D.Ctx(model=object(), choice_artifact=_Artifact())
    return c


# content gap (0.2) is SMALLER than the slot-A position bias (0.3): single-pass is
# position-dominated, marginalization is not.
def _fake_scores(prompt, ctx):
    head, contents = D._split_canonical_options(prompt)
    out = {}
    for i, c in enumerate(contents):
        L = chr(ord("A") + i)
        base = 0.6 if "SARC" in c else 0.4
        if L == "A":
            base += 0.3                 # positional bias toward the first slot
        out[L] = base
    return out


PROMPT = (
    "Which statement is sarcastic?\n"
    "Options:\n"
    "(A) plain statement\n"
    "(B) SARC statement"          # the correct (sarcastic) content sits in slot B
)
PROMPT_SWAPPED = (
    "Which statement is sarcastic?\n"
    "Options:\n"
    "(A) SARC statement\n"
    "(B) plain statement"
)


def test_split_canonical_options():
    head, contents = D._split_canonical_options(PROMPT)
    assert head.endswith("Options:\n")
    assert contents == ["plain statement", "SARC statement"]


def test_single_pass_is_position_dominated(monkeypatch):
    # the fake: slot A gets +0.3, so the plain option in slot A (0.7) beats the
    # sarcastic option in slot B (0.6) — a single pass picks the WRONG letter.
    s = _fake_scores(PROMPT, None)
    assert max(s, key=lambda k: s[k]) == "A"      # wrong (correct content is in B)


def test_marginalization_recovers_content(monkeypatch):
    monkeypatch.setattr(D, "_choice_scores", _fake_scores)
    ans, scores = D._score_choice_marginalized(PROMPT, _ctx())
    assert ans == "(B)"                            # the SARC content's slot


def test_marginalization_is_order_invariant_binary(monkeypatch):
    monkeypatch.setattr(D, "_choice_scores", _fake_scores)
    a0, _ = D._score_choice_marginalized(PROMPT, _ctx())
    a1, _ = D._score_choice_marginalized(PROMPT_SWAPPED, _ctx())
    # the SARC content wins regardless of which slot it is presented in
    assert a0 == "(B)" and a1 == "(A)"            # both name the sarcastic content


def test_marginalized_scores_average_over_slots(monkeypatch):
    monkeypatch.setattr(D, "_choice_scores", _fake_scores)
    _, scores = D._score_choice_marginalized(PROMPT, _ctx())
    # each option appears once in slot A (+0.3) and once in slot B over the 2 shifts
    # SARC: (0.6 + 0.9)/2 = 0.75 ; plain: (0.7 + 0.4)/2 = 0.55
    assert scores["B"] == pytest.approx(0.75)
    assert scores["A"] == pytest.approx(0.55)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
