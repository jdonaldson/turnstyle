"""No-model tests for the zero-shot content-PMI MC floor (dispatch).

When an MC prompt has no calibrated probe, `_score_options_pmi` scores each option's
CONTENT by domain-conditional PMI: logP(content|question) - logP(content|neutral),
argmax. These inject a fake LM (no model) and assert: (1) PMI picks the question-
relevant option even when raw likelihood would pick a high-frequency distractor
(prior correction matters), (2) the answer is order-invariant (content scored as an
isolated continuation), and (3) the floor declines on non-MC prompts.
"""
import pytest

import turnstyle.dispatch as D


class _FakeTok:
    """apply_chat_template returns the bare user content so the fake LM can tell the
    question context from the neutral-prior context by substring."""
    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True):
        return msgs[0]["content"]


# (logP|question, logP|neutral) per content. COMMON is high-frequency (wins on RAW
# question likelihood) but question-irrelevant; CORRECT wins on PMI after the prior
# (its high intrinsic rarity) is subtracted.
_TABLE = {"CORRECT": (-1.0, -2.0), "COMMON": (-0.5, -0.5)}


def _fake_logprob(model, tok, device, context, continuation):
    neutral = D._PMI_NEUTRAL in context
    q, n = _TABLE[continuation.strip()]
    return n if neutral else q


def _ctx():
    return D.Ctx(model=object(), tokenizer=_FakeTok(), device="cpu")


PROMPT = "Which is correct?\nOptions:\n(A) CORRECT\n(B) COMMON"
PROMPT_SWAPPED = "Which is correct?\nOptions:\n(A) COMMON\n(B) CORRECT"


def test_raw_question_likelihood_would_pick_the_distractor():
    # sanity on the fixture: on raw logP(content|question), COMMON (-0.5) beats
    # CORRECT (-1.0) — so a non-prior-corrected scorer picks the wrong option.
    assert _TABLE["COMMON"][0] > _TABLE["CORRECT"][0]


def test_pmi_picks_question_relevant_option(monkeypatch):
    monkeypatch.setattr(D, "_lm_logprob", _fake_logprob)
    ans, pmis = D._score_options_pmi(PROMPT, _ctx())
    # PMI: CORRECT = -1-(-2) = +1.0 ; COMMON = -0.5-(-0.5) = 0.0
    assert ans == "(A)"
    assert pmis["A"] == pytest.approx(1.0) and pmis["B"] == pytest.approx(0.0)


def test_pmi_is_order_invariant(monkeypatch):
    monkeypatch.setattr(D, "_lm_logprob", _fake_logprob)
    a0, _ = D._score_options_pmi(PROMPT, _ctx())          # CORRECT in slot A
    a1, _ = D._score_options_pmi(PROMPT_SWAPPED, _ctx())  # CORRECT in slot B
    assert a0 == "(A)" and a1 == "(B)"                    # winner follows content


def test_floor_declines_on_non_mc():
    assert D._score_options_pmi("What is 3 + 4 * 2?", _ctx()) is None


def test_solve_choice_uses_floor_when_no_probe(monkeypatch):
    monkeypatch.setattr(D, "_lm_logprob", _fake_logprob)
    ctx = _ctx()                       # choice_artifact is None ⇒ floor path
    res = D._solve_choice(PROMPT, ["A", "B"], None, None, ctx)
    assert isinstance(res, D.Answer)
    assert res.source == "pmi_floor" and res.text == "(A)"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
