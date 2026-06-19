"""ComparisonSolver — logical_deduction via probe-local + symbolic-global.

The principled, generalizing replacement for the adjective-list regexes in
comparison_ordering.py. Decomposition (validated in experiments/pole_harness.py):

  structural frames  — spatial (left/right) and rank (finished above/Nth) are
                       closed-class directional → extracted exactly, no pole.
  scalar frames      — "X is <adj> than Y" / "X is the <adj>" capture the adjective;
                       its HIGH/LOW pole is supplied by the adjective-polarity probe
                       ([[adjective_polarity_primitive]]) — cross-lingual, no word list.
  symbolic solver    — exact permutation search over the constraints (transitivity).

Each stage does only what it is reliable at: the model perceives local lexical
poles (probe ~1.0 in-axis, cross-language ~0.98), code reasons the global ordering.

Pole resolution order per adjective root: polarity probe (if a model + shipped
probe are given) → regex lexicon fallback (offline / English). normalize_phrase
strips closed-class more/less/most/least + -er/-est so only the open-class root
is asked of the probe.
"""
from __future__ import annotations

import re
from itertools import permutations
from typing import Optional

HIGH, LOW = +1, -1
ORDINALS = {"second": 2, "third": 3, "fourth": 4, "fifth": 5,
            "sixth": 6, "seventh": 7}
_SUP_PHRASE = r"(?:most|least)\s+\w+|\w+est|\w+most"
_ORD_PREFIX = re.compile(r"^(?:second|third|fourth|fifth|sixth|seventh)-")


# ── item + sentence helpers ──────────────────────────────────────────────────

def extract_items(text: str) -> list[str]:
    m = re.search(r"(?:three|four|five|six|seven)\s+[\w ]+?:\s*(.+?)\.", text, re.I)
    if not m:
        return []
    out = []
    for p in re.split(r",\s*(?:and\s+)?|\s+and\s+", m.group(1)):
        p = re.sub(r"^(a|an|the)\s+", "", p.strip(), flags=re.I).strip().rstrip(".")
        if p:
            out.append(p)
    return out


def _sentences(body: str) -> list[str]:
    return [s for s in re.split(r"(?<=[.?])\s+|\n", body) if s.strip()]


def normalize_phrase(phrase: str) -> tuple[str, int]:
    """Strip closed-class modifiers/morphology → (open-class root, sign_flip).

    sign_flip = -1 for a less/least negation. The probe is only asked about the
    root (new/old/expensive/...), never the modifier."""
    p = re.sub(r"^the\s+", "", phrase.strip().lower())
    p = _ORD_PREFIX.sub("", p)
    flip = 1
    m = re.match(r"(most|least|more|less)\s+(.+)", p)
    if m:
        if m.group(1) in ("least", "less"):
            flip = -1
        p = m.group(2)
    # naive -er/-est removal — covers regular stems (new/old/cheap, most/least X).
    # Irregular morphology (y→i "shinier", doubling "biggest") is not lemmatized;
    # for the probe path the polarity direction still covers many such words, but
    # the resolved key is the stripped stem, so supply pole_map keys to match.
    if p.endswith("est"):
        p = p[:-3]
    elif p.endswith("er"):
        p = p[:-2]
    return p.strip(), flip


# ── stage 1: frames ──────────────────────────────────────────────────────────
# constraints: ("before",i,j) | ("at",i,pos) | ("cmp",i,j,phrase)
#            | ("sup",i,phrase) | ("supord",i,k,phrase)

def extract_frames(text: str, items: list[str]) -> list[tuple]:
    body = text.split("Options:")[0]
    n = len(items)
    item_lo = [it.lower() for it in items]
    pat = "|".join(re.escape(it) for it in items)
    art = r"(?:(?:a|an|the)\s+)?"
    be = r"(?:is|are|was|were)"
    cons: list[tuple] = []

    def idx(name):
        lo = name.lower()
        return next((k for k, it in enumerate(item_lo) if it == lo), None)

    def before(a, b):
        ia, ib = idx(a), idx(b)
        if ia is not None and ib is not None and ia != ib:
            cons.append(("before", ia, ib))

    def at(a, pos):
        ia = idx(a)
        if ia is not None and 1 <= pos <= n:
            cons.append(("at", ia, pos))

    for s in _sentences(body):
        for m in re.finditer(rf"({pat})\s+{be}\s+to\s+the\s+right\s+of\s+{art}({pat})", s, re.I):
            before(m.group(2), m.group(1))
        for m in re.finditer(rf"({pat})\s+{be}\s+to\s+the\s+left\s+of\s+{art}({pat})", s, re.I):
            before(m.group(1), m.group(2))
        for m in re.finditer(rf"({pat})\s+finished\s+(?:above|ahead of)\s+{art}({pat})", s, re.I):
            before(m.group(2), m.group(1))
        for m in re.finditer(rf"({pat})\s+finished\s+(?:below|behind)\s+{art}({pat})", s, re.I):
            before(m.group(1), m.group(2))

        ord_hit = False
        for word, k in ORDINALS.items():
            for m in re.finditer(rf"({pat})\s+{be}\s+the\s+{word}-({_SUP_PHRASE})", s, re.I):
                ia = idx(m.group(1))
                if ia is not None:
                    cons.append(("supord", ia, k, m.group(2).strip().lower()))
                    ord_hit = True

        for m in re.finditer(rf"({pat})\s+{be}\s+([^.]+?)\s+than\s+{art}({pat})", s, re.I):
            ia, ib = idx(m.group(1)), idx(m.group(3))
            if ia is not None and ib is not None and ia != ib:
                cons.append(("cmp", ia, ib, m.group(2).strip().lower()))

        for m in re.finditer(rf"({pat})\s+{be}\s+the\s+rightmost", s, re.I):
            at(m.group(1), n)
        for m in re.finditer(rf"({pat})\s+{be}\s+the\s+leftmost", s, re.I):
            at(m.group(1), 1)
        for m in re.finditer(rf"({pat})\s+finished\s+(?:first|1st)\b", s, re.I):
            at(m.group(1), n)
        for m in re.finditer(rf"({pat})\s+finished\s+last\b", s, re.I):
            at(m.group(1), 1)

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

        if not ord_hit:
            for m in re.finditer(rf"({pat})\s+{be}\s+the\s+({_SUP_PHRASE})", s, re.I):
                ia = idx(m.group(1))
                if ia is not None:
                    cons.append(("sup", ia, m.group(2).strip().lower()))
    return cons


# ── stage 2: pole resolution ─────────────────────────────────────────────────

# offline / English fallback lexicon (used only when no probe is available)
_ADJ_H = re.compile(r"\b(new|newer|newest|larg|larger|largest|heavy|heavier|heaviest|"
                    r"tall|taller|tallest|fast|faster|fastest|expensive|late|later|latest|"
                    r"good|better|best|high|higher|highest|big|bigger|biggest|"
                    r"long|longer|longest|strong|stronger|strongest|rich|richer|richest)\b", re.I)
_ADJ_L = re.compile(r"\b(old|older|oldest|small|smaller|smallest|light|lighter|lightest|"
                    r"short|shorter|shortest|slow|slower|slowest|cheap|cheaper|cheapest|"
                    r"early|earlier|earliest|bad|worse|worst|low|lower|lowest|"
                    r"weak|weaker|weakest|poor|poorer|poorest)\b", re.I)


def regex_pole(root: str) -> Optional[int]:
    if _ADJ_H.fullmatch(root):
        return HIGH
    if _ADJ_L.fullmatch(root):
        return LOW
    return None


def _adjective_roots(text: str) -> set[str]:
    """All scalar-adjective roots appearing anywhere (body + options + query)."""
    roots: set[str] = set()
    for m in re.finditer(r"\b(?:is|are|was|were)\s+([^.]+?)\s+than\b", text, re.I):
        roots.add(normalize_phrase(m.group(1))[0])
    for m in re.finditer(rf"\bthe\s+({_SUP_PHRASE})", text, re.I):
        roots.add(normalize_phrase(m.group(1))[0])
    # ordinal-superlatives: "second-cheapest", "third-most expensive" (hyphen breaks
    # the _SUP_PHRASE scan, so the adjective after the ordinal needs its own pattern)
    for m in re.finditer(r"\b(?:second|third|fourth|fifth|sixth|seventh)-"
                         r"((?:most|least)\s+\w+|\w+est|\w+most)", text, re.I):
        roots.add(normalize_phrase(m.group(1))[0])
    return {r for r in roots if r}


def make_pole_map(text, model=None, tokenizer=None, device=None, probe=None) -> dict:
    """Resolve every adjective root in the prompt to a pole. Probe first
    (cross-lingual, generalizing), regex lexicon as offline fallback."""
    roots = _adjective_roots(text)
    pole_map: dict[str, int] = {}
    if probe is not None and model is not None:
        from turnstyle.polarity import word_poles
        # ask the probe in a neutral template; pole is a lexical property
        try:
            pole_map.update(word_poles(model, tokenizer, device, list(roots), probe))
        except Exception:  # noqa: BLE001 — fall back to regex on any probe failure
            pass
    for r in roots:
        if r not in pole_map:
            p = regex_pole(r)
            if p is not None:
                pole_map[r] = p
    return pole_map


# ── stage 3: symbolic solver + answer mapping ────────────────────────────────

def _resolver(pole_map):
    def res(phrase):
        root, flip = normalize_phrase(phrase)
        base = pole_map.get(root)
        return None if base is None else base * flip
    return res


def constraints_to_pairs(cons, n, resolver):
    pairs, pins = [], []
    for c in cons:
        if c[0] == "before":
            pairs.append((c[1], c[2]))
        elif c[0] == "at":
            pins.append((c[1], c[2]))
        elif c[0] == "cmp":
            pole = resolver(c[3])
            if pole is None:
                continue
            pairs.append((c[2], c[1]) if pole == HIGH else (c[1], c[2]))
        elif c[0] == "sup":
            pole = resolver(c[2])
            if pole is not None:
                pins.append((c[1], n if pole == HIGH else 1))
        elif c[0] == "supord":
            pole = resolver(c[3])
            if pole is not None:
                pins.append((c[1], n - c[2] + 1 if pole == HIGH else c[2]))
    return pairs, pins


def solve_ordering(n, pairs, pins):
    valid = None
    for perm in permutations(range(n)):
        rank = {it: pos for pos, it in enumerate(perm)}
        if all(rank[a] < rank[b] for a, b in pairs) and \
           all(rank[i] + 1 == pos for i, pos in pins):
            if valid is not None:
                return None
            valid = perm
    return valid


def parse_options(text):
    sec = text.split("Options:")[-1]
    return {L: v.strip() for L, v in
            re.findall(r"\(([A-Z])\)\s+(.+?)(?=\n\([A-Z]\)|\Z)", sec, re.S)}


def map_answer(order, items, text, resolver):
    n = len(items)
    pos_of = {it: p for p, it in enumerate(order)}
    options = parse_options(text)
    body = text.split("Options:")[0]
    qline = next((s for s in reversed(_sentences(body)) if "?" in s), "")
    item_lo = [it.lower() for it in items]

    for L, opt in options.items():
        parts = [p.strip().lower() for p in re.split(r",\s*", opt)]
        if len(parts) == n and all(p in item_lo for p in parts):
            if [item_lo.index(p) for p in parts] == list(order):
                return f"({L})"

    def claim_pos(s):
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
        mo = re.search(r"\bthe\s+(second|third|fourth|fifth|sixth|seventh)-([\w ]+)", s)
        if mo:
            pole = resolver(mo.group(2))
            if pole is not None:
                k = ORDINALS[mo.group(1)]
                return (n - k + 1) if pole == HIGH else k
        ms = re.search(rf"\bthe\s+({_SUP_PHRASE})", s)
        if ms:
            pole = resolver(ms.group(1))
            if pole is not None:
                return n if pole == HIGH else 1
        return None

    qpos = claim_pos(qline)
    if qpos is not None and 1 <= qpos <= n:
        target = order[qpos - 1]
        for L, opt in options.items():
            if item_lo[target] in opt.lower():
                return f"({L})"

    for L, opt in options.items():
        cp = claim_pos(opt)
        if cp is None:
            continue
        for it_i, it in enumerate(item_lo):
            if it in opt.lower():
                if pos_of[it_i] + 1 == cp:
                    return f"({L})"
                break
    return None


def solve_comparison(text, pole_map: Optional[dict] = None,
                     model=None, tokenizer=None, device=None, probe=None
                     ) -> Optional[str]:
    """Solve a logical_deduction prompt → option letter, or None.

    Supply a `pole_map` (root→pole) directly, or a (model, probe) pair to resolve
    poles via the polarity primitive. With neither, poles fall back to the regex
    lexicon (offline / English)."""
    items = extract_items(text)
    if not items:
        return None
    if pole_map is None:
        pole_map = make_pole_map(text, model, tokenizer, device, probe)
    resolver = _resolver(pole_map)
    cons = extract_frames(text, items)
    pairs, pins = constraints_to_pairs(cons, len(items), resolver)
    if not pairs and not pins:
        return None
    order = solve_ordering(len(items), pairs, pins)
    if order is None:
        return None
    return map_answer(order, items, text, resolver)
