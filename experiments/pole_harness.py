"""Full harness for the probe-locality logical_deduction solver.

Decomposes the solver into three swappable stages so we can measure each in
isolation (the thing the earlier 70% prototype could not):

  1. FRAMES      — structural extraction of constraints from the paragraph.
                   Spatial (left/right) and finished/rank (above/below/Nth) are
                   closed-class directional → emitted with a known direction.
                   Scalar-adjective frames (X is <PHRASE> than Y / X is the
                   <PHRASE>) capture the raw adjective phrase for a resolver.
  2. POLE        — a pluggable resolver maps a scalar-adjective phrase to a pole
                   (HIGH end vs LOW end). Variants: oracle / regex / morphology /
                   probe. This is the only stage that touches open-class vocab.
  3. SOLVER      — exact permutation search over the constraints, then map the
                   unique ordering to an option letter (option/query superlatives
                   resolved by the same pole resolver).

Convention: higher index = HIGH pole end (rightmost / newest / most-expensive /
finished-first). Pole HIGH means the subject moves toward the higher index.

Stage 1+3 are language-/vocab-agnostic structure; stage 2 is where the
"probe local facts" idea lives. Run with --resolver to compare.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from itertools import permutations

BBH = "/Users/jdonaldson/Projects/swollm/data/bbh_cache"
TASKS = ["logical_deduction_three_objects",
         "logical_deduction_five_objects",
         "logical_deduction_seven_objects"]

ORDINALS = {"second": 2, "third": 3, "fourth": 4, "fifth": 5,
            "sixth": 6, "seventh": 7}


# ── item extraction ──────────────────────────────────────────────────────────

def extract_items(text: str) -> list[str]:
    m = re.search(r"(?:three|four|five|six|seven)\s+[\w ]+?:\s*(.+?)\.",
                  text, re.I)
    if not m:
        return []
    out = []
    for p in re.split(r",\s*(?:and\s+)?|\s+and\s+", m.group(1)):
        p = re.sub(r"^(a|an|the)\s+", "", p.strip(), flags=re.I).strip().rstrip(".")
        if p:
            out.append(p)
    return out


# ── stage 1: frame extraction ────────────────────────────────────────────────
# Each constraint is one of:
#   ("before", i, j)            item i strictly before item j (lower index)
#   ("at",     i, pos)          item i pinned to 1-indexed absolute pos
#   ("cmp",    i, j, phrase)    "item i is <phrase> than item j" — pole-resolved
#   ("sup",    i, phrase)       "item i is the <phrase>"          — pole-resolved
#   ("supord", i, k, phrase)    "item i is the <ord>-<phrase>"    — pole-resolved

_SUP_PHRASE = r"(?:most|least)\s+\w+|\w+est|\w+most"


def _sentences(body: str) -> list[str]:
    return [s for s in re.split(r"(?<=[.?])\s+|\n", body) if s.strip()]


def extract_frames(text: str, items: list[str]) -> list[tuple]:
    body = text.split("Options:")[0]
    n = len(items)
    item_lo = [it.lower() for it in items]
    pat = "|".join(re.escape(it) for it in items)
    art = r"(?:(?:a|an|the)\s+)?"
    be = r"(?:is|are|was|were)"
    cons: list[tuple] = []

    def idx(name: str):
        lo = name.lower()
        for k, it in enumerate(item_lo):
            if it == lo:
                return k
        return None

    def before(a, b):
        ia, ib = idx(a), idx(b)
        if ia is not None and ib is not None and ia != ib:
            cons.append(("before", ia, ib))

    def at(a, pos):
        ia = idx(a)
        if ia is not None and 1 <= pos <= n:
            cons.append(("at", ia, pos))

    # one sentence at a time — frames never span a sentence boundary
    for s in _sentences(body):
        # spatial (structural)
        for m in re.finditer(rf"({pat})\s+{be}\s+to\s+the\s+right\s+of\s+{art}({pat})", s, re.I):
            before(m.group(2), m.group(1))
        for m in re.finditer(rf"({pat})\s+{be}\s+to\s+the\s+left\s+of\s+{art}({pat})", s, re.I):
            before(m.group(1), m.group(2))

        # finished / rank (structural)
        for m in re.finditer(rf"({pat})\s+finished\s+(?:above|ahead of)\s+{art}({pat})", s, re.I):
            before(m.group(2), m.group(1))
        for m in re.finditer(rf"({pat})\s+finished\s+(?:below|behind)\s+{art}({pat})", s, re.I):
            before(m.group(1), m.group(2))

        # scalar-adjective ordinal-superlative: "X is the <ord>-<phrase>" (before plain cmp/sup)
        ord_hit = False
        for word, k in ORDINALS.items():
            for m in re.finditer(rf"({pat})\s+{be}\s+the\s+{word}-({_SUP_PHRASE})", s, re.I):
                ia = idx(m.group(1))
                if ia is not None:
                    cons.append(("supord", ia, k, m.group(2).strip().lower()))
                    ord_hit = True

        # scalar-adjective comparative: "X is <phrase> than Y"
        for m in re.finditer(rf"({pat})\s+{be}\s+([^.]+?)\s+than\s+{art}({pat})", s, re.I):
            ia, ib = idx(m.group(1)), idx(m.group(3))
            if ia is not None and ib is not None and ia != ib:
                cons.append(("cmp", ia, ib, m.group(2).strip().lower()))

        # spatial superlative (structural)
        for m in re.finditer(rf"({pat})\s+{be}\s+the\s+rightmost", s, re.I):
            at(m.group(1), n)
        for m in re.finditer(rf"({pat})\s+{be}\s+the\s+leftmost", s, re.I):
            at(m.group(1), 1)

        # finished absolute (structural)
        for m in re.finditer(rf"({pat})\s+finished\s+(?:first|1st)\b", s, re.I):
            at(m.group(1), n)
        for m in re.finditer(rf"({pat})\s+finished\s+last\b", s, re.I):
            at(m.group(1), 1)

        # ordinal spatial / finished (structural, number-words are closed-class)
        for word, k in ORDINALS.items():
            for m in re.finditer(rf"({pat})\s+{be}\s+(?:the\s+)?{word}\s+from\s+the\s+left", s, re.I):
                at(m.group(1), k)
            for m in re.finditer(rf"({pat})\s+{be}\s+(?:the\s+)?{word}\s+from\s+the\s+right", s, re.I):
                at(m.group(1), n - k + 1)
            for m in re.finditer(rf"({pat})\s+finished\s+{word}(?!-to)\b", s, re.I):
                at(m.group(1), n - k + 1)
        for word, k in [("second", 2), ("third", 3), ("fourth", 4), ("fifth", 5)]:
            for m in re.finditer(rf"({pat})\s+finished\s+{word}-to-last\b", s, re.I):
                at(m.group(1), k)

        # scalar-adjective superlative: "X is the <phrase>" (skip if ordinal already matched)
        if not ord_hit:
            for m in re.finditer(rf"({pat})\s+{be}\s+the\s+({_SUP_PHRASE})", s, re.I):
                ia = idx(m.group(1))
                if ia is not None:
                    cons.append(("sup", ia, m.group(2).strip().lower()))

    return cons


# ── stage 2: pole resolvers ──────────────────────────────────────────────────
# A resolver maps a scalar-adjective phrase to +1 (HIGH end) or -1 (LOW end),
# or None if it cannot decide.  Closed-class modifiers (more/less/most/least)
# and -er/-est morphology are stripped structurally first; the resolver only
# sees the open-class ROOT.

HIGH, LOW = +1, -1

_ORD_PREFIX = re.compile(r"^(?:second|third|fourth|fifth|sixth|seventh)-")


def normalize_phrase(phrase: str) -> tuple[str, int]:
    """Strip closed-class modifiers/morphology. Return (root, sign_flip).

    sign_flip = -1 if a 'less/least' negation is present, else +1. Closed-class
    modifiers (more/less/most/least) and -er/-est morphology are removed here so
    the pole resolver only sees the open-class ROOT (new/old/expensive/cheap...).
    """
    p = re.sub(r"^the\s+", "", phrase.strip().lower())
    p = _ORD_PREFIX.sub("", p)
    flip = 1
    m = re.match(r"(most|least|more|less)\s+(.+)", p)
    if m:
        if m.group(1) in ("least", "less"):
            flip = -1
        p = m.group(2)
    if p.endswith("est"):
        p = p[:-3]
    elif p.endswith("er"):
        p = p[:-2]
    return p.strip(), flip


# -- oracle: labeled lexicon (ceiling for plumbing) --
_ORACLE = {
    "new": HIGH, "old": LOW, "expensive": HIGH, "cheap": LOW,
    "newer": HIGH, "older": LOW, "cheaper": LOW,
    "tall": HIGH, "short": LOW, "large": HIGH, "small": LOW,
    "heavy": HIGH, "light": LOW, "fast": HIGH, "slow": LOW,
    "high": HIGH, "low": LOW, "good": HIGH, "bad": LOW,
    "big": HIGH, "long": HIGH, "young": LOW,
}


def resolve_oracle(phrase: str):
    root, flip = normalize_phrase(phrase)
    base = _ORACLE.get(root)
    if base is None:
        # try the raw (e.g. 'pricey' family or already-root)
        base = _ORACLE.get(phrase.strip().lower())
    return None if base is None else base * flip


# -- regex: original adj_h/adj_l lists from comparison_ordering.py --
_ADJ_H = re.compile(r"\b(newer|larger|heavier|taller|faster|more expensive|later|"
                    r"better|higher|pricier|more costly|newest|largest|heaviest|"
                    r"tallest|fastest|most expensive|latest|best|highest)\b", re.I)
_ADJ_L = re.compile(r"\b(older|smaller|lighter|shorter|slower|cheaper|earlier|"
                    r"worse|lower|less expensive|less costly|oldest|smallest|"
                    r"lightest|shortest|slowest|cheapest|earliest|worst|lowest)\b", re.I)


def resolve_regex(phrase: str):
    if _ADJ_H.search(phrase):
        return HIGH
    if _ADJ_L.search(phrase):
        return LOW
    return None


# -- morphology: cannot give absolute pole; returns None (root-consistency is
#    handled by a different solver path, not a pole). Kept for completeness. --
def resolve_morphology(_phrase: str):
    return None


RESOLVERS = {
    "oracle": resolve_oracle,
    "regex": resolve_regex,
    "morphology": resolve_morphology,
}


# ── stage 3: symbolic solver ─────────────────────────────────────────────────

def constraints_to_pairs(cons, n, resolver):
    """Lower the typed constraints to ('before', i, j) and ('at', i, pos)."""
    pairs, pins = [], []
    unresolved = 0
    for c in cons:
        if c[0] == "before":
            pairs.append((c[1], c[2]))
        elif c[0] == "at":
            pins.append((c[1], c[2]))
        elif c[0] == "cmp":
            _, i, j, phrase = c
            pole = resolver(phrase)
            if pole is None:
                unresolved += 1
                continue
            if pole == HIGH:
                pairs.append((j, i))   # i higher index
            else:
                pairs.append((i, j))
        elif c[0] == "sup":
            _, i, phrase = c
            pole = resolver(phrase)
            if pole is None:
                unresolved += 1
                continue
            pins.append((i, n if pole == HIGH else 1))
        elif c[0] == "supord":
            _, i, k, phrase = c
            pole = resolver(phrase)
            if pole is None:
                unresolved += 1
                continue
            pins.append((i, n - k + 1 if pole == HIGH else k))
    return pairs, pins, unresolved


def solve_ordering(items, pairs, pins):
    n = len(items)
    valid = []
    for perm in permutations(range(n)):
        rank = {item: pos for pos, item in enumerate(perm)}
        if all(rank[a] < rank[b] for a, b in pairs) and \
           all(rank[i] + 1 == pos for i, pos in pins):
            valid.append(perm)
            if len(valid) > 1:
                return None
    return valid[0] if valid else None


# ── answer mapping (option/query superlatives, same resolver) ────────────────

def parse_options(text: str):
    sec = text.split("Options:")[-1]
    return {L: v.strip() for L, v in
            re.findall(r"\(([A-Z])\)\s+(.+?)(?=\n\([A-Z]\)|\Z)", sec, re.S)}


def map_answer(order, items, text, resolver):
    """order = perm of item indices (low→high). Return option letter or None."""
    n = len(items)
    pos_of = {item: p for p, item in enumerate(order)}  # 0-indexed low→high
    options = parse_options(text)
    body = text.split("Options:")[0]
    # the question = the sentence that ends in '?'  (NOT the constraint statements)
    qline = next((s for s in reversed(_sentences(body)) if "?" in s), "")
    item_lo = [it.lower() for it in items]

    # full arrangement options: comma list of item names, left→right == low→high
    for L, opt in options.items():
        parts = [p.strip().lower() for p in re.split(r",\s*", opt)]
        if len(parts) == n and all(p in item_lo for p in parts):
            want = [item_lo.index(p) for p in parts]
            if want == list(order):
                return f"({L})"

    # locate the positional claim — in the query or in each option
    def claim_pos(s):
        """Return the 1-indexed position a positional phrase refers to, or None."""
        s = s.lower()
        for word, k in ORDINALS.items():
            if re.search(rf"\b{word}\s+from\s+the\s+left", s):
                return k
            if re.search(rf"\b{word}\s+from\s+the\s+right", s):
                return n - k + 1
        if re.search(r"\bleftmost\b", s):
            return 1
        if re.search(r"\brightmost\b", s):
            return n
        if re.search(r"\bin\s+the\s+middle\b|\bcenter\b", s):
            return (n + 1) // 2
        if re.search(r"finished\s+first\b", s):
            return n
        if re.search(r"finished\s+last\b", s):
            return 1
        for word, k in ORDINALS.items():
            if re.search(rf"finished\s+{word}-to-last\b", s):
                return k
            if re.search(rf"finished\s+{word}\b", s):
                return n - k + 1
        # ordinal-superlative: "the second-newest"
        mo = re.search(r"\bthe\s+(second|third|fourth|fifth|sixth|seventh)-([\w ]+)", s)
        if mo:
            k = ORDINALS[mo.group(1)]
            pole = resolver(mo.group(2))
            if pole is not None:
                return (n - k + 1) if pole == HIGH else k
        # plain superlative: "the newest" / "the most expensive"
        ms = re.search(r"\bthe\s+((?:most|least)\s+\w+|\w+est)\b", s)
        if ms:
            pole = resolver(ms.group(1))
            if pole is not None:
                return n if pole == HIGH else 1
        return None

    # query-style: position claim in the question, answer = item at that pos
    qpos = claim_pos(qline)
    if qpos is not None and 1 <= qpos <= n:
        target = order[qpos - 1]
        for L, opt in options.items():
            if item_lo[target] in opt.lower():
                return f"({L})"

    # option-style: each option claims a position; pick the true one
    for L, opt in options.items():
        cp = claim_pos(opt)
        if cp is None:
            continue
        # which item does the option name?
        for it_i, it in enumerate(item_lo):
            if it in opt.lower():
                if pos_of[it_i] + 1 == cp:
                    return f"({L})"
                break
    return None


# ── full pipeline ────────────────────────────────────────────────────────────

def solve(text, resolver):
    items = extract_items(text)
    if not items:
        return None, "no_items"
    cons = extract_frames(text, items)
    pairs, pins, unresolved = constraints_to_pairs(cons, len(items), resolver)
    if not pairs and not pins:
        return None, ("unresolved_pole" if unresolved else "no_frames")
    order = solve_ordering(items, pairs, pins)
    if order is None:
        return None, ("unresolved_pole" if unresolved else "no_unique_order")
    ans = map_answer(order, items, text, resolver)
    if ans is None:
        return None, "no_answer_map"
    return ans, "ok"


def run(resolver_name, verbose=0):
    resolver = RESOLVERS[resolver_name]
    print(f"\n=== resolver: {resolver_name} ===")
    grand_c = grand_n = 0
    for t in TASKS:
        data = json.load(open(f"{BBH}/{t}.json"))
        c = 0
        fails = Counter()
        for ex in data:
            ans, status = solve(ex["input"], resolver)
            ok = ans == ex["target"].strip()
            if ok:
                c += 1
            else:
                fails[status if status != "ok" else "wrong_answer"] += 1
                if verbose and fails[status if status != "ok" else "wrong_answer"] <= verbose:
                    print(f"  [{t[18:].split('_')[0]}] gold={ex['target'].strip()} "
                          f"got={ans} ({status})")
        n = len(data)
        grand_c += c
        grand_n += n
        fail_str = "  ".join(f"{k}={v}" for k, v in fails.most_common())
        print(f"  {t[18:]:16s} {c}/{n} = {c/n:5.1%}   {fail_str}")
    print(f"  {'TOTAL':16s} {grand_c}/{grand_n} = {grand_c/grand_n:5.1%}")
    return grand_c, grand_n


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--resolver", default="oracle",
                    choices=list(RESOLVERS) + ["all"])
    ap.add_argument("-v", "--verbose", type=int, default=0)
    args = ap.parse_args()
    names = list(RESOLVERS) if args.resolver == "all" else [args.resolver]
    for nm in names:
        run(nm, args.verbose)
