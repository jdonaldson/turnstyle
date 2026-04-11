"""Comparison ordering turnstyle — grounds logical deduction in constraint satisfaction.

Parses ordering constraints from the problem text via LLM extraction, finds the
unique permutation consistent with all constraints, then maps the result to the
correct answer option.

Handles BBH logical_deduction_three/five/seven_objects.

Extraction schema
-----------------
constraint → pairwise:   {"lo": "lower item", "hi": "higher item"}
constraint → positional: {"item": "item name", "pos": N}
  pos=1  lowest/leftmost/oldest/worst
  pos=-1 highest/rightmost/newest/best
  pos=0  middle
query:                   {"ask": "item_at", "pos": N}
                      or {"ask": "arrangement"}
"""

from __future__ import annotations

import re
from itertools import permutations

from turnstyle.core import SequenceLogitsProcessor, Turnstyle
from turnstyle.ir import SentenceIRSpec, SentenceRecord


# ── few-shot extraction prompt ───────────────────────────────────────────────

_EXTRACT_PROMPT = """\
Extract ordering information as JSON.

constraint → pairwise: {{"lo": "lower item", "hi": "higher item"}}
  lo is older/worse/lighter/leftward of hi
constraint → positional: {{"item": "item name", "pos": N}}
  pos=1 lowest/leftmost/oldest/worst, pos=-1 highest/rightmost/newest/best
query → {{"ask": "item_at", "pos": N}} or {{"ask": "arrangement"}}

Examples:
sentence: The green book is to the left of the red book.
type: constraint
{{"lo": "green book", "hi": "red book"}}

sentence: The red car is newer than the blue car.
type: constraint
{{"lo": "blue car", "hi": "red car"}}

sentence: Alice finished above Bob.
type: constraint
{{"lo": "Bob", "hi": "Alice"}}

sentence: The oak tree is the tallest.
type: constraint
{{"item": "oak tree", "pos": -1}}

sentence: The pine tree is the shortest.
type: constraint
{{"item": "pine tree", "pos": 1}}

sentence: Tom finished first.
type: constraint
{{"item": "Tom", "pos": -1}}

sentence: Tom finished second-to-last.
type: constraint
{{"item": "Tom", "pos": 2}}

sentence: The yellow envelope is second from the left.
type: constraint
{{"item": "yellow envelope", "pos": 2}}

sentence: Which book is the leftmost?
type: query
{{"ask": "item_at", "pos": 1}}

sentence: Which runner finished last?
type: query
{{"ask": "item_at", "pos": 1}}

sentence: Which is the second-newest?
type: query
{{"ask": "item_at", "pos": -2}}

sentence: Which competitor finished first?
type: query
{{"ask": "item_at", "pos": -1}}

sentence: Which is in the middle?
type: query
{{"ask": "item_at", "pos": 0}}

sentence: Which of the following is a valid arrangement from left to right?
type: query
{{"ask": "arrangement"}}

sentence: {sentence}
type: {type}
"""


# ── sentence classifier (syntactic, no keyword vocabulary) ───────────────────


def _classify_comparison(sentence: str) -> str:
    return "query" if "?" in sentence else "constraint"


# ── constraint solver ────────────────────────────────────────────────────────


def _aggregate_comparison(
    records: list[SentenceRecord],
    _question: str | None,
    options: dict[str, str],
) -> str | None:
    """Collect lo/hi and item/pos constraints from records, find unique ordering."""
    pairs:  list[tuple[str, str]] = []  # (lo, hi) — lo ranks below hi
    pinned: list[tuple[str, int]] = []  # (name, raw_pos)
    query:  dict | None = None

    for rec in records:
        d = rec.data
        if not isinstance(d, dict):
            continue
        if rec.record_type == "query" and "ask" in d:
            query = d
        elif "lo" in d and "hi" in d:
            pairs.append((str(d["lo"]).strip().lower(), str(d["hi"]).strip().lower()))
        elif "item" in d and "pos" in d:
            try:
                pinned.append((str(d["item"]).strip().lower(), int(d["pos"])))
            except (TypeError, ValueError):
                pass

    if not pairs and not pinned:
        return None

    # Unique items in first-seen order
    items = list(dict.fromkeys(
        [name for lo, hi in pairs for name in (lo, hi)]
        + [name for name, _ in pinned]
    ))
    n = len(items)

    def abs_pos(raw: int) -> int:
        """Map raw pos (signed, 0=middle) to 1-indexed absolute position."""
        if raw < 0:  return n + raw + 1   # -1 → n, -2 → n-1
        if raw == 0: return (n + 1) // 2  # middle
        return raw

    pinned_abs = [(name, abs_pos(pos)) for name, pos in pinned
                  if 1 <= abs_pos(pos) <= n]

    # Find the unique permutation satisfying all constraints
    ordering = None
    for perm in permutations(items):
        rank = {item: i for i, item in enumerate(perm)}
        if (all(rank[lo] < rank[hi] for lo, hi in pairs)
                and all(rank[name] + 1 == pos for name, pos in pinned_abs)):
            if ordering is not None:
                return None  # ambiguous — more than one valid ordering
            ordering = list(perm)

    if ordering is None:
        return None

    if query is None:
        return None

    ask = query.get("ask")

    if ask == "arrangement":
        for letter, opt in options.items():
            if [o.strip().lower() for o in re.split(r",\s*", opt)] == ordering:
                return f"({letter})"

    elif ask == "item_at":
        try:
            idx = abs_pos(int(query.get("pos", 1))) - 1
        except (TypeError, ValueError):
            return None
        if 0 <= idx < n:
            target = ordering[idx]
            for letter, opt in options.items():
                if target in opt.lower():
                    return f"({letter})"

    return None


# ── SentenceIRSpec ────────────────────────────────────────────────────────────

COMPARISON_ORDERING_SPEC = SentenceIRSpec(
    sentence_types=["constraint", "query"],
    extract_prompt=_EXTRACT_PROMPT,
    aggregate=_aggregate_comparison,
    classify_fn=_classify_comparison,
    max_tokens=40,
)


# ── Turnstyle subclass ────────────────────────────────────────────────────────


class ComparisonOrderingTurnstyle(Turnstyle):
    """Grounds comparison/ordering tasks in deterministic constraint satisfaction.

    LLM extracts (lo, hi) pairwise and (item, pos) positional constraints from
    each sentence; the unique consistent permutation is found and mapped to the
    correct option letter — no adjective vocabulary lists.

    Handles BBH logical_deduction_three/five/seven_objects.

        t = ComparisonOrderingTurnstyle(model, tokenizer, device)
        text, proof = t.generate("The following paragraphs each describe a set of three objects...")
    """

    probe_label = "comparison_ordering"
    sentence_ir_spec = COMPARISON_ORDERING_SPEC
    examples = [
        "The following paragraphs each describe a set of three objects kept in order. The objects are ordered left to right.\nThe green book is to the left of the red book. The red book is to the left of the blue book.\nWhich of the following options is a valid arrangement of the books from left to right?\nOptions:\n(A) green, blue, red\n(B) red, green, blue\n(C) green, red, blue",
        "The following paragraphs each describe a set of three objects kept in order. The objects are ordered left to right.\nThe white envelope is newer than the yellow envelope. The yellow envelope is newer than the pink envelope.\nWhich is the newest?\nOptions:\n(A) white envelope\n(B) yellow envelope\n(C) pink envelope",
        "The following paragraphs each describe a set of three objects kept in order. The objects are ordered from oldest to newest.\nThe brown box is older than the gray box. The gray box is older than the black box.\nWhich is the oldest?\nOptions:\n(A) brown box\n(B) gray box\n(C) black box",
        "The following paragraphs each describe a set of five objects kept in order. The objects are ordered left to right.\nThe red shoe is to the left of the blue shoe. The blue shoe is to the left of the green shoe. The green shoe is to the left of the yellow shoe. The yellow shoe is to the left of the purple shoe.\nWhich shoe is second from the left?\nOptions:\n(A) red shoe\n(B) blue shoe\n(C) green shoe\n(D) yellow shoe\n(E) purple shoe",
        "The following paragraphs each describe a set of three objects kept in order. The objects are ordered by price.\nThe silver watch is more expensive than the gold watch. The gold watch is more expensive than the bronze watch.\nWhich watch is the cheapest?\nOptions:\n(A) silver watch\n(B) gold watch\n(C) bronze watch",
        "The following paragraphs each describe a set of three objects kept in order. The objects are ordered from lightest to heaviest.\nThe small ball is lighter than the medium ball. The medium ball is lighter than the large ball.\nWhich ball is the heaviest?\nOptions:\n(A) small ball\n(B) medium ball\n(C) large ball",
        "The following paragraphs each describe a set of three objects kept in order. The objects are ordered from tallest to shortest.\nThe oak tree is taller than the maple tree. The maple tree is taller than the pine tree.\nWhich tree is the shortest?\nOptions:\n(A) oak tree\n(B) maple tree\n(C) pine tree",
        "The following paragraphs each describe a set of three golfers kept in order. They were each given a score from 1 to 3 so that no two of them have the same score.\nTom finished above Jerry. Jerry finished above Spike.\nWhich golfer finished last?\nOptions:\n(A) Tom\n(B) Jerry\n(C) Spike",
        "The following paragraphs each describe a set of three competitors kept in order. They were each given a score from 1 to 3 so that no two of them have the same score.\nAlice finished above Bob. Bob finished above Carol.\nWhich competitor finished first?\nOptions:\n(A) Alice\n(B) Bob\n(C) Carol",
        "The following paragraphs each describe a set of three objects kept in order. The objects are ordered from newest to oldest.\nThe red car is newer than the blue car. The blue car is newer than the green car.\nWhich is the second-newest?\nOptions:\n(A) red car\n(B) blue car\n(C) green car",
        "The following paragraphs each describe a set of five objects kept in order. The objects are ordered left to right.\nThe apple is to the left of the banana. The banana is to the left of the cherry. The cherry is to the left of the date. The elderberry is the rightmost.\nWhich fruit is the leftmost?\nOptions:\n(A) apple\n(B) banana\n(C) cherry\n(D) date\n(E) elderberry",
        "The following paragraphs each describe a set of three books kept in order. They are ordered from best to worst.\nThe mystery book is better than the romance book. The romance book is better than the sci-fi book.\nWhich is the second-best?\nOptions:\n(A) mystery book\n(B) romance book\n(C) sci-fi book",
        "The following paragraphs each describe a set of three objects kept in order. The objects are ordered from fastest to slowest.\nThe red car is faster than the blue car. The blue car is faster than the green car.\nWhich car is the slowest?\nOptions:\n(A) red car\n(B) blue car\n(C) green car",
        "The following paragraphs each describe a set of three colored pencils kept in order. They are ordered from left to right.\nThe orange pencil is to the left of the purple pencil. The purple pencil is to the left of the brown pencil.\nWhich pencil is in the middle?\nOptions:\n(A) orange pencil\n(B) purple pencil\n(C) brown pencil",
        "The following paragraphs each describe a set of three students kept in order. They are ranked from best to worst student.\nMaria is a better student than John. John is a better student than Paul.\nWhich student has the worst rank?\nOptions:\n(A) Maria\n(B) John\n(C) Paul",
        "The following paragraphs each describe a set of three objects kept in order. The objects are ordered from heaviest to lightest.\nThe iron block is heavier than the wooden block. The wooden block is heavier than the plastic block.\nWhich is the second-lightest?\nOptions:\n(A) iron block\n(B) wooden block\n(C) plastic block",
        "The following paragraphs each describe a set of five runners in a race. They were each given a rank from 1 to 5, with 1 being fastest.\nAnn finished above Beth. Beth finished above Carol. Carol finished above Diane. Diane finished above Ella.\nWhich runner finished third?\nOptions:\n(A) Ann\n(B) Beth\n(C) Carol\n(D) Diane\n(E) Ella",
        "The following paragraphs each describe a set of three objects kept in order. The objects are ordered from left to right.\nThe tall lamp is to the right of the short lamp. The short lamp is to the right of the medium lamp.\nWhich lamp is in the middle?\nOptions:\n(A) tall lamp\n(B) short lamp\n(C) medium lamp",
        "The following paragraphs each describe a set of three colored balls. They are ordered from largest to smallest.\nThe blue ball is larger than the red ball. The red ball is larger than the green ball.\nWhich ball is the smallest?\nOptions:\n(A) blue ball\n(B) red ball\n(C) green ball",
        "The following paragraphs each describe a set of three boxes kept in order. The boxes are ordered from most expensive to least expensive.\nThe gold box is more expensive than the silver box. The silver box is more expensive than the copper box.\nWhich box is second-most-expensive?\nOptions:\n(A) gold box\n(B) silver box\n(C) copper box",
        "The following paragraphs each describe a set of three objects kept in order. The objects are ordered from left to right on a shelf.\nThe dictionary is to the left of the encyclopedia. The encyclopedia is to the left of the atlas.\nWhich book is on the far right?\nOptions:\n(A) dictionary\n(B) encyclopedia\n(C) atlas",
        "The following paragraphs each describe a set of three people kept in order. They are ordered by age from youngest to oldest.\nSam is younger than Pat. Pat is younger than Chris.\nWho is the oldest?\nOptions:\n(A) Sam\n(B) Pat\n(C) Chris",
        "The following paragraphs each describe a set of three objects. They are ordered from left to right.\nThe wooden chair is to the left of the plastic chair. The plastic chair is to the left of the metal chair.\nWhich chair is the leftmost?\nOptions:\n(A) wooden chair\n(B) plastic chair\n(C) metal chair",
        "The following paragraphs each describe a set of three swimmers. They are ranked by their finishing time, fastest to slowest.\nMike finished above Dave. Dave finished above Steve.\nWhich swimmer finished last?\nOptions:\n(A) Mike\n(B) Dave\n(C) Steve",
        "The following paragraphs each describe a set of three chocolates. They are ordered from sweetest to least sweet.\nMilk chocolate is sweeter than dark chocolate. Dark chocolate is sweeter than white chocolate.\nWhich chocolate is the least sweet?\nOptions:\n(A) milk chocolate\n(B) dark chocolate\n(C) white chocolate",
        "The following paragraphs each describe a set of three flowers kept in order. They are ordered from tallest to shortest.\nThe sunflower is taller than the daisy. The daisy is taller than the violet.\nWhich flower is in the middle?\nOptions:\n(A) sunflower\n(B) daisy\n(C) violet",
        "The following paragraphs each describe a set of three objects kept in order. The objects are ordered from newest to oldest.\nThe laptop is newer than the tablet. The tablet is newer than the phone.\nWhat is the second-newest object?\nOptions:\n(A) laptop\n(B) tablet\n(C) phone",
        "The following paragraphs each describe a set of three cities. They are ordered by population from largest to smallest.\nCity A has more residents than City B. City B has more residents than City C.\nWhich city is the smallest?\nOptions:\n(A) City A\n(B) City B\n(C) City C",
        "The following paragraphs each describe a set of three horses in a race. They were given places 1 (first) through 3 (last).\nStorm finished above Thunder. Thunder finished above Lightning.\nWhich horse finished last?\nOptions:\n(A) Storm\n(B) Thunder\n(C) Lightning",
        "The following paragraphs each describe a set of three objects kept in order. The objects are ordered from left to right.\nThe square is to the left of the circle. The circle is to the left of the triangle.\nWhich shape is in the middle?\nOptions:\n(A) square\n(B) circle\n(C) triangle",
    ]

    def parse(self, prompt: str):
        """No regex fast path — all solving via sentence_ir_spec."""
        return None

    def make_processor(self, parsed, max_new_tokens: int):
        (answer_letter,) = parsed
        answer_ids = self.tokenizer.encode(answer_letter, add_special_tokens=False)
        return SequenceLogitsProcessor(
            self.tokenizer,
            answer_ids,
            expression="comparison_ordering",
            answer_str=answer_letter,
            bias_strength=self.bias_strength,
            max_new_tokens=max_new_tokens,
        )
