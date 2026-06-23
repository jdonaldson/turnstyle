"""object_counting: count items belonging to a category, the principled way.

BBH format: "I have a flute, a piano, a trumpet, four stoves, a violin, and a
drum. How many musical instruments do I have?"

The hard part is *category membership* (is a flute a musical instrument?). swollm's
solver answered this with hardcoded category sets (animals/fruits/…) — a semantic
keyword list, exactly what the project bans in solvers. Here membership is decided
by the MODEL's world knowledge (a yes/no scorer over the item), and only the
counting is deterministic. The item list is parsed structurally (split on commas/
and, leading number-words expanded) — number words are a closed structural vocab,
not content semantics.

solve_object_counting(prompt, model, tok, device) → the integer count as a string,
or None (so the dispatcher abstains rather than mis-committing).
"""
from __future__ import annotations

import re

_NUMBER_WORDS = {
    "a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
    "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
    "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20,
}

_QUESTION_RE = re.compile(r"how many (\w+(?:\s\w+)*) do i have", re.IGNORECASE)


def _split_items(text: str) -> list[str]:
    """Structural split of an item list on commas and a trailing 'and'."""
    text = re.sub(r",?\s+and\s+", ", ", text.strip())
    return [p.strip() for p in text.split(",") if p.strip()]


def _singular(word: str) -> str:
    w = word.strip().lower()
    if w.endswith("ies") and len(w) > 4:
        return w[:-3] + "y"
    if w.endswith("es") and w[:-2].endswith(("s", "x", "z", "ch", "sh")):
        return w[:-2]
    if w.endswith("s") and not w.endswith("ss"):
        return w[:-1]
    return w


def parse_item_list(prompt: str):
    """Structurally parse (category, [(qty, item_name), ...]) or None.

    No semantic vocabulary: items are delimiter-split and quantities come from the
    closed number-word set; the bare noun is whatever follows the quantity."""
    m = _QUESTION_RE.search(prompt)
    if not m:
        return None
    category = m.group(1).lower()

    items_text = prompt[: m.start()].strip()
    items_text = re.sub(r"^i have\s+", "", items_text, flags=re.IGNORECASE).rstrip(". ")
    parsed = []
    for part in _split_items(items_text):
        words = part.split()
        if not words:
            continue
        qty = 1
        if words[0].lower() in _NUMBER_WORDS:
            qty = _NUMBER_WORDS[words[0].lower()]
            words = words[1:]
        if not words:
            continue
        parsed.append((qty, " ".join(words)))
    if not parsed:
        return None
    return category, parsed


def _member_scorer(model, tokenizer, device):
    """Return f(item, category)->bool via a yes/no logit comparison (model world
    knowledge). Cached per call site so each unique item is asked once."""
    import torch

    def yes_ids(words):
        ids = []
        for w in words:
            enc = tokenizer.encode(w, add_special_tokens=False)
            if enc:
                ids.append(enc[0])
        return ids

    YES = yes_ids([" yes", " Yes", "yes", "Yes"])
    NO = yes_ids([" no", " No", "no", "No"])
    cache: dict[tuple[str, str], bool] = {}

    def is_member(item: str, category: str) -> bool:
        key = (item, category)
        if key in cache:
            return cache[key]
        q = f"Is a {item} a kind of {category}? Answer yes or no."
        msgs = [{"role": "user", "content": q}]
        text = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True)
        enc = tokenizer(text, return_tensors="pt").to(device)
        with torch.no_grad():
            logits = model(**enc).logits[0, -1]
        yes = max(float(logits[i]) for i in YES) if YES else -1e9
        no = max(float(logits[i]) for i in NO) if NO else -1e9
        cache[key] = yes > no
        return cache[key]

    return is_member


def solve_object_counting(prompt: str, model, tokenizer, device,
                          *, scorer=None) -> str | None:
    """Structural parse + model-classified category membership + deterministic count."""
    parsed = parse_item_list(prompt)
    if parsed is None:
        return None
    category, items = parsed

    # "objects"/"things" = count everything; no membership question needed.
    if _singular(category) in ("object", "thing", "item"):
        return str(sum(qty for qty, _ in items))

    if model is None:
        return None
    is_member = scorer or _member_scorer(model, tokenizer, device)
    cat_sing = _singular(category)
    total = 0
    for qty, name in items:
        if is_member(_singular(name), cat_sing):
            total += qty
    return str(total)


__all__ = ["parse_item_list", "solve_object_counting"]
