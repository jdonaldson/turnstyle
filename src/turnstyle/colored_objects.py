"""reasoning_about_colored_objects: spatial/attribute reasoning over a row of
colored objects — the principled (keyword-list-free) way.

swollm's solver hardcoded a color vocabulary (`_CO_COLORS`) AND a color→option-letter
map. Both are unnecessary. The color is *positional*: every scene item is
"a <color> <object>" (or "<n> <color> <objects>"), so the color is structurally the
first descriptor word after the article/number and the object is the remainder — no
color list needed, and the observed color set falls out of the scene. The value→letter
map is derived from the prompt's own options (color answers match a color option;
count answers match a numeric/number-word option; yes/no match a yes/no option). What
remains are structural query operators (leftmost/right-of/count/what-color/neither-nor/
remove) — spatial relations and set operations with no natural-language synonym lists.

solve_colored_objects(prompt) → option letter "(A)" or None (abstain on shapes we
don't model, e.g. novel query phrasings).
"""
from __future__ import annotations

import re

_NUMBER_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
}
_OPT_RE = re.compile(r"^\(([A-Z])\)\s*(.+?)\s*$", re.MULTILINE)


def _options(prompt: str) -> dict[str, str]:
    return {letter: val.strip().lower() for letter, val in _OPT_RE.findall(prompt)}


def _value_to_letter(options: dict[str, str], value: str):
    value = str(value).strip().lower()
    for letter, val in options.items():
        if val == value:
            return f"({letter})"
    return None


def _count_to_letter(options: dict[str, str], n: int):
    """Match an integer count to a numeric option, whether written as a digit or a
    number word (BBH uses both '(A) 3' and '(A) three')."""
    for letter, val in options.items():
        iv = int(val) if val.isdigit() else _NUMBER_WORDS.get(val)
        if iv == n:
            return f"({letter})"
    return None


def _strip_articles(text: str) -> str:
    return re.sub(r"^(?:a|an|the)\s+", "", text.strip(), flags=re.IGNORECASE)


_GENERIC_NOUNS = {"object", "objects", "item", "items", "thing", "things"}


def _normalize_obj(name: str) -> str:
    name = name.strip().rstrip(".?")
    # "X of Y" container phrases pluralize the head ("pairs of sunglasses").
    name = re.sub(r"\b(pair|sheet|bunch|set|piece)s of\b", r"\1 of", name)
    if name.endswith("es") and name[:-2].endswith(("sh", "ch")):
        return name[:-2]
    if name.endswith("s") and not name.endswith("ss"):
        return name[:-1]
    return name


def _parse_filter(filt: str):
    """Split a count/remove filter into (color, object). At most one is set; both
    None means 'all objects'. A color qualifier is positional — the word before a
    generic noun (object/item/thing) — so an absent-from-scene color still filters
    (yielding zero), no color list needed."""
    words = filt.split()
    if words and words[-1] in _GENERIC_NOUNS:
        return (words[0], None) if len(words) >= 2 else (None, None)
    return None, filt


def _obj_matches(a: str, b: str) -> bool:
    return _normalize_obj(a) == _normalize_obj(b)


def _scene_body(prompt: str) -> str:
    body = prompt.split("Options:")[0]
    sentences = re.split(r"\.\s+(?=[A-Z])", body)
    parts = []
    for s in sentences:
        if re.match(r"(?:What|How|Is|If)\b", s.strip()):
            break
        parts.append(s)
    return ". ".join(parts) + "."


def _parse_scene(prompt: str):
    """Return (items, is_inventory). items: [(color, obj)] or [(count, color, obj)].

    Color is positional — the first descriptor word after the article (and optional
    leading number word for inventory rows). No color vocabulary needed."""
    scene = _scene_body(prompt)
    lower = scene.lower()

    if ":" in scene and "arranged in a row" in lower:
        items_text = scene.split(":", 1)[1].strip()
    else:
        items_text = re.sub(
            r"^On the \w+,?\s*(?:you see|there (?:is|are)|i see)\s*", "",
            scene, flags=re.IGNORECASE).strip()
        items_text = re.sub(
            r"(?:a set of things|a bunch of things|a bunch of objects|"
            r"several (?:things|items|objects))\s*(?:arranged in a row)?[,:]\s*",
            "", items_text, flags=re.IGNORECASE).strip()

    items_text = re.sub(r",?\s+and\s+", ", ", items_text)
    raw = []           # (count|None, color, obj)
    for part in items_text.split(","):
        part = part.strip().rstrip(".")
        if not part:
            continue
        words = part.split()
        count = None
        if words and words[0].lower() in _NUMBER_WORDS:
            count = _NUMBER_WORDS[words[0].lower()]
            words = words[1:]
        else:
            words = _strip_articles(" ".join(words)).split()
        if len(words) < 2:
            continue
        raw.append((count, words[0].lower(), " ".join(words[1:]).strip()))

    # inventory = every item carried an explicit leading count word.
    is_inventory = bool(raw) and all(c is not None for c, _, _ in raw)
    items = [(c if c is not None else 1, col, ob) for c, col, ob in raw]
    return items, is_inventory


def _find(items, reference, colors):
    """Locate a scene item by a color reference or an object name. items are
    (count, color, obj)."""
    ref = reference.lower().strip().rstrip("?")
    for color in colors:
        if ref == color or ref.startswith(color + " "):
            for i, (_, col, _ob) in enumerate(items):
                if col == color:
                    return i
            return None
    for i, (_, _col, ob) in enumerate(items):
        if _obj_matches(ob, ref):
            return i
    return None


def solve_colored_objects(prompt: str) -> str | None:
    options = _options(prompt)
    if len(options) < 2:
        return None

    items, _is_inventory = _parse_scene(prompt)
    if not items:
        return None
    colors = {col for _, col, _ in items}        # observed color set (from scene)
    lower = prompt.lower()

    def count_letter(n):
        return _count_to_letter(options, n)

    def color_letter(c):
        return _value_to_letter(options, c)

    # remove + count (items carry counts; default 1 for non-inventory rows)
    if "remove" in lower and "remain" in lower:
        m = re.search(r"if i remove all the\s+(.+?)\s+from", lower)
        m2 = re.search(r"how many\s+(.+?)\s+remain", lower)
        if not (m and m2):
            return None
        rem_color, rem_obj = _parse_filter(m.group(1).strip())
        kept = [(c, col, ob) for c, col, ob in items
                if not (rem_color and col == rem_color)
                and not (rem_obj and _obj_matches(ob, rem_obj))]
        cnt_color, cnt_obj = _parse_filter(m2.group(1).strip())
        total = sum(c for c, col, ob in kept
                    if (not cnt_color or col == cnt_color)
                    and (not cnt_obj or _obj_matches(ob, cnt_obj)))
        return count_letter(total)

    # directly to the left/right of X
    if "directly to the" in lower:
        m = re.search(r"directly to the\s+(left|right)\s+of the\s+(.+?)(?:\?|$)", lower)
        if not m:
            return None
        idx = _find(items, m.group(2), colors)
        if idx is None:
            return None
        j = idx - 1 if m.group(1) == "left" else idx + 1
        if 0 <= j < len(items):
            return color_letter(items[j][1])
        return None

    # how many non-<color> items to the left/right of X
    if re.search(r"non[- ]\w+", lower) and ("left" in lower or "right" in lower):
        m = re.search(r"how many non[- ](\w+)\s+(?:items|things|objects)\s+.*?"
                      r"(?:to the\s+)?(left|right)\s+of the\s+(.+?)(?:\?|$)", lower)
        if not m:
            return None
        excl, direction = m.group(1).lower(), m.group(2)
        idx = _find(items, m.group(3), colors)
        if idx is None:
            return None
        subset = items[:idx] if direction == "left" else items[idx + 1:]
        return count_letter(sum(1 for _, col, _ob in subset if col != excl))

    # furthest/farthest from X
    if "furthest" in lower or "farthest" in lower:
        m = re.search(r"(?:furthest|farthest) from the\s+(.+?)(?:\?|$)", lower)
        if not m:
            return None
        idx = _find(items, m.group(1), colors)
        if idx is None:
            return None
        j = 0 if idx >= len(items) - 1 - idx else len(items) - 1
        return color_letter(items[j][1])

    if "left-most" in lower or "leftmost" in lower:
        return color_letter(items[0][1])
    if "right-most" in lower or "rightmost" in lower:
        return color_letter(items[-1][1])

    # neither X nor Y count
    if "neither" in lower and "nor" in lower:
        m = re.search(r"neither\s+(\w+)\s+nor\s+(\w+)", lower)
        if m:
            c1, c2 = m.group(1).lower(), m.group(2).lower()
            return count_letter(sum(1 for _, col, _ob in items if col not in (c1, c2)))

    # how many <color> / <object> (plain count)
    if lower.count("how many") and ("object" in lower or "item" in lower or "thing" in lower
                                    or any(c in lower for c in colors)):
        m = re.search(r"how many\s+(.+?)\s+(?:are there|do you see|are on)", lower)
        if m:
            filt = m.group(1).strip()
            f_color = next((c for c in colors if filt.startswith(c)), None)
            generic = any(w in filt for w in ("object", "item", "thing"))
            if f_color:
                return count_letter(sum(c for c, col, _ob in items if col == f_color))
            if generic:
                return count_letter(sum(c for c, _col, _ob in items))

    # yes/no color check: "is the <obj> <color>?" — only when options are yes/no
    # (so "what color is the X?" isn't hijacked by the embedded "is the").
    if _value_to_letter(options, "yes") and re.search(r"\bis the\b", lower) and "?" in lower:
        m = re.search(r"is the\s+(.+?)\s+(\w+)\?", lower)
        if m:
            cand = m.group(2).lower()
            for _, color, obj in items:
                if _obj_matches(obj, m.group(1).strip()):
                    return (_value_to_letter(options, "yes") if color == cand
                            else _value_to_letter(options, "no"))
        return None

    # what color is the X
    if "what color" in lower or "what is the color" in lower:
        m = re.search(r"what (?:color is|is the color of) the\s+(.+?)(?:\?|$)", lower)
        if m:
            for _, color, obj in items:
                if _obj_matches(obj, m.group(1).strip().rstrip("?")):
                    return color_letter(color)

    return None


__all__ = ["solve_colored_objects"]
