"""Colored objects utilities — scene parsing and string matching.

Parses scenes with colored objects (inventory or row) into structured data.
Used with SQLTurnstyle via colored_objects_parse_tables().

Scene types:
  1. Inventory: "there is/are one red X, two blue Y..." — unordered, with counts
  2. Row: "arranged in a row: a red X, a blue Y..." — ordered, positional
"""

from __future__ import annotations

import re


# ════════════════════════════════════════════════════════════════════════
# Number word parsing
# ════════════════════════════════════════════════════════════════════════

_NUM_WORDS = {
    "a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
    "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
    "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20,
}

COLORS = {
    "red", "orange", "yellow", "green", "blue", "brown", "magenta", "fuchsia",
    "mauve", "teal", "turquoise", "burgundy", "silver", "gold", "black", "grey",
    "purple", "pink", "white",
}


# ════════════════════════════════════════════════════════════════════════
# String matching helpers
# ════════════════════════════════════════════════════════════════════════

def _stem(word):
    """Minimal stemmer: strip trailing 's'/'es' for plural matching."""
    if len(word) > 3:
        for suffix in ("shes", "ches", "xes", "zes", "ses"):
            if word.endswith(suffix):
                return word[:-2]
    if word.endswith("s") and len(word) > 2:
        return word[:-1]
    return word


def _stem_phrase(phrase):
    """Stem each word in a phrase."""
    return " ".join(_stem(w) for w in phrase.split())


# ════════════════════════════════════════════════════════════════════════
# Scene parsing
# ════════════════════════════════════════════════════════════════════════

def parse_scene(text):
    """Parse the scene description into a list of (color, object_type) items.

    For inventory scenes (with counts), items are repeated.
    For row scenes, items are in order.

    Returns:
        items: list of (color, object_type) tuples, in order if row scene
        is_row: bool
    """
    # Get scene description (before the question mark)
    parts = text.split("?")
    scene_text = parts[0] if len(parts) > 1 else text

    # Detect row vs inventory
    is_row = "arranged in a row" in scene_text or "in a row" in scene_text

    # Find the items list - after the colon for row scenes, or parse "there is/are"
    if ":" in scene_text and is_row:
        items_text = scene_text.split(":")[-1].strip()
    else:
        m = re.search(
            r'(?:there (?:is|are)|you see|I see)\s+(.+)', scene_text, re.IGNORECASE,
        )
        if m:
            items_text = m.group(1).strip()
        else:
            items_text = scene_text

    # Stop at first sentence boundary — question text must not bleed in
    items_text = re.split(r'\.\s+(?=[A-Z])', items_text)[0]
    items_text = items_text.rstrip(".")

    # Split on comma + and
    parts = re.split(r',\s*(?:and\s+)?|\s+and\s+', items_text)

    items = []
    for part in parts:
        part = part.strip()
        if not part:
            continue

        words = part.split()
        if not words:
            continue

        # Parse quantity
        qty = 1
        start = 0
        if words[0].lower() in _NUM_WORDS:
            qty = _NUM_WORDS[words[0].lower()]
            start = 1

        # Find color
        color = None
        obj_words = []
        for i in range(start, len(words)):
            if words[i].lower() in COLORS and color is None:
                color = words[i].lower()
            else:
                obj_words.append(words[i].lower())

        if color is None:
            continue

        obj_type = " ".join(obj_words)
        for _ in range(qty):
            items.append((color, obj_type))

    return items, is_row


# ════════════════════════════════════════════════════════════════════════
# SQLTurnstyle integration
# ════════════════════════════════════════════════════════════════════════

def colored_objects_parse_tables(text: str) -> dict | None:
    """Parse a colored-objects scene into SQLite-ready table dict.

    Returns {table_name: (columns, rows)} or None if scene can't be parsed.
    Suitable for SQLTurnstyle(parse_tables_fn=colored_objects_parse_tables).
    """
    items, is_row = parse_scene(text)
    if not items:
        return None
    if is_row:
        return {"objects": (
            ["position", "color", "type"],
            [(i + 1, c, _stem_phrase(t)) for i, (c, t) in enumerate(items)],
        )}
    return {"objects": (
        ["color", "type"],
        [(c, _stem_phrase(t)) for c, t in items],
    )}
