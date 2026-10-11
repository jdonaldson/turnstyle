"""Symbolic witness for snarks: SenticNet-9 polarity-incongruity scorer.

The ⊢ leg of the two-witness protocol. Given two candidate sentences, predict
which is sarcastic by scoring internal polarity incongruity against the
SenticNet 9 lexicon (Cambria et al., IEEE TCSS 2026). Fit on NOTHING — no
snarks labels touch this code. That task-externality is the whole point:
this witness's errors should be uncorrelated with a hidden-state probe's.

Heuristic (stated plainly so it can be argued with):
  sarcasm ≈ strong positive and strong negative polarity co-occurring in one
  sentence (praise-shaped words in blame-shaped company). Score each sentence
  by the span between its most positive and most negative matched concept,
  requiring both signs to be present; the option with the larger span is
  called sarcastic. Confidence = |score_A - score_B|.

This is a lexicon-based approximation of the symbolic approach — NOT the full
SenticNet pipeline (no conceptual-primitive normalization, no HAN). It is the
cheapest honest ⊢ witness, per the pre-flight "cheapest ablation" rule.
"""

import re
import sys
from pathlib import Path

STOP = {
    "the", "a", "an", "and", "or", "but", "if", "then", "than", "so", "such",
    "is", "are", "was", "were", "be", "been", "being", "am",
    "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them",
    "my", "your", "his", "its", "our", "their",
    "to", "of", "in", "on", "at", "by", "for", "with", "as", "from",
    "this", "that", "these", "those", "there", "here",
    "do", "does", "did", "have", "has", "had", "will", "would", "can", "could",
    "not", "no", "n't", "what", "which", "who", "how", "when", "where",
}


LINE_RE = re.compile(r"^senticnet\['(.*)'\] = \[(.*)\]\s*$")


def load_senticnet(path):
    """Parse senticnet.py line-by-line.

    NOT exec'd: the distributed file contains emoticon concepts with unescaped
    quotes (e.g. senticnet[':'-)'] at line ~292k) and is not valid Python.
    We only need concept -> polarity_value (8th field).
    """
    lex, skipped = {}, 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            m = LINE_RE.match(line)
            if not m:
                continue
            concept, body = m.group(1), m.group(2)
            fields = body.split(", ")
            if len(fields) < 8:
                skipped += 1
                continue
            try:
                lex[concept] = float(fields[7])
            except ValueError:
                skipped += 1
    if skipped:
        print(f"note: skipped {skipped} unparseable lexicon lines", file=sys.stderr)
    return lex


def tokens(sentence):
    return [t for t in re.findall(r"[a-z']+", sentence.lower()) if t not in STOP]


def matched_polarities(sentence, lex):
    """Look up unigrams, bigrams, trigrams (underscore-joined) in the lexicon.

    Longest-match-first: a token consumed by an n-gram concept is not
    double-counted as a unigram.
    """
    toks = tokens(sentence)
    used = [False] * len(toks)
    out = []
    for n in (3, 2, 1):
        for i in range(len(toks) - n + 1):
            if any(used[i:i + n]):
                continue
            concept = "_".join(toks[i:i + n])
            if concept in lex:
                out.append((concept, lex[concept]))
                for j in range(i, i + n):
                    used[j] = True
    return out


def incongruity(sentence, lex):
    """Polarity span if both signs present; 0 otherwise."""
    pols = [p for _, p in matched_polarities(sentence, lex)]
    pos = [p for p in pols if p > 0.1]
    neg = [p for p in pols if p < -0.1]
    if not pos or not neg:
        return 0.0
    return max(pos) - min(neg)


def predict(option_a, option_b, lex):
    """Return (prediction 'A'/'B', confidence, score_a, score_b)."""
    sa, sb = incongruity(option_a, lex), incongruity(option_b, lex)
    pred = "A" if sa > sb else "B" if sb > sa else None  # None = symbolic abstain (tie)
    return pred, abs(sa - sb), sa, sb


if __name__ == "__main__":
    lex = load_senticnet(sys.argv[1] if len(sys.argv) > 1 else "senticnet/senticnet.py")
    print(f"lexicon: {len(lex)} concepts with polarity")
    demo = [
        ("I love being ignored by customer service for three hours.",
         "I dislike being ignored by customer service for three hours."),
    ]
    for a, b in demo:
        pred, conf, sa, sb = predict(a, b, lex)
        print(f"A={sa:.3f} B={sb:.3f} -> sarcastic: {pred} (conf {conf:.3f})")
