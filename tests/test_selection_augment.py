"""No-model tests for the single-mode selection-probe order augmentation."""
import re

from turnstyle.dispatch_turnstyle import _augment_option_orderings

LET = re.compile(r"\(([A-Z])\)")

EX = {
    "input": "Pick the closest movie. Options:\n(A) Alpha\n(B) Bravo\n(C) Charlie\n(D) Delta",
    "target": "(C)",  # Charlie is correct
}


def _correct_content(ex):
    """The option text the target points at, for the given example."""
    body = ex["input"].split("Options:")[-1]
    opts = dict(re.findall(r"\(([A-Z])\)\s*([^\n(]+)", body))
    return opts[LET.search(ex["target"]).group(1)].strip()


def test_augment_count_and_identity():
    aug = _augment_option_orderings([EX], k=4)
    assert len(aug) == 4                     # identity + 3 perms
    assert aug[0]["input"] == EX["input"]    # identity first
    assert aug[0]["target"] == EX["target"]


def test_augment_target_tracks_correct_option():
    # Across every ordering the remapped target must still point at "Charlie".
    for a in _augment_option_orderings([EX], k=4):
        assert _correct_content(a) == "Charlie"


def test_augment_orderings_are_distinct():
    inputs = [a["input"] for a in _augment_option_orderings([EX], k=4)]
    assert len(set(inputs)) == len(inputs)


def test_non_mc_passes_through():
    plain = {"input": "What is 2+2?", "target": "4"}
    out = _augment_option_orderings([plain], k=4)
    assert out == [plain]


def test_deterministic_seed():
    a = _augment_option_orderings([EX], k=4, seed=0)
    b = _augment_option_orderings([EX], k=4, seed=0)
    assert [x["input"] for x in a] == [x["input"] for x in b]
