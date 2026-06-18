#!/usr/bin/env python3
"""Test object_counting accumulator-side fix.

Same prompt as baseline. Fix the consumer:
1. Recurse into nested {items: [...]} wrappers
2. Accept type "object" or any dict with a "name" field as an item
3. {type: number, value: N} → recover name from segment text by stripping
   the leading number word
4. When category lookup fails on extracted name, retry with full segment text
   (handles "stalks of celery" → name="stalk" losing "celery")

Compare baseline vs fixed on object_counting.
"""

import json
import os
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

BBH_CACHE = "/Users/jdonaldson/Projects/swollm/data/bbh_cache"
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
DEVICE = "mps"
N_EVAL = 250

# ── Same baseline prompt ───────────────────────────────────────────
_OBJ_EXTRACT_PROMPT = (
    'You are a text transcription tool. Extract the item as JSON.\n\n'
    'Statement: "I have a flute"\n'
    'JSON: {{"type": "item", "name": "flute", "count": 1}}\n\n'
    'Statement: "I have three trombones"\n'
    'JSON: {{"type": "item", "name": "trombone", "count": 3}}\n\n'
    'Statement: "I have two cats"\n'
    'JSON: {{"type": "item", "name": "cat", "count": 2}}\n\n'
    'Statement: "How many musical instruments do I have?"\n'
    'JSON: {{"type": "query", "category": "musical instruments"}}\n\n'
    'Statement: "How many animals do I have?"\n'
    'JSON: {{"type": "query", "category": "animals"}}\n\n'
    'Statement: "How many objects do I have?"\n'
    'JSON: {{"type": "query", "category": "objects"}}\n\n'
    'Statement: "{segment}"\n'
    'JSON:'
)

_NUMBER_WORDS = {
    "a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
    "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
}

_MUSICAL_INSTRUMENTS = {
    'flute', 'piano', 'trombone', 'violin', 'accordion', 'clarinet', 'drum',
    'trumpet', 'guitar', 'banjo', 'harmonica', 'saxophone', 'ukulele', 'cello',
    'harp', 'oboe', 'bassoon', 'tuba', 'french horn', 'organ', 'fiddle',
    'mandolin', 'piccolo', 'bongo', 'tambourine', 'cymbal', 'xylophone',
    'recorder', 'bugle', 'lute', 'sitar', 'bagpipe', 'maracas', 'viola',
}
_FRUITS = {
    'apple', 'banana', 'strawberry', 'peach', 'orange', 'plum', 'raspberry',
    'grape', 'nectarine', 'blackberry', 'blueberry', 'cherry', 'mango',
    'pineapple', 'watermelon', 'kiwi', 'lemon', 'lime', 'pear', 'papaya',
    'coconut', 'fig', 'grapefruit', 'pomegranate', 'tangerine', 'cantaloupe',
    'apricot', 'guava', 'passion fruit', 'dragonfruit', 'persimmon',
    'cranberry', 'melon',
}
_VEGETABLES = {
    'carrot', 'potato', 'tomato', 'onion', 'broccoli', 'lettuce', 'spinach',
    'corn', 'pepper', 'cucumber', 'cabbage', 'celery', 'garlic', 'mushroom',
    'eggplant', 'zucchini', 'pumpkin', 'squash', 'beet', 'radish', 'turnip',
    'artichoke', 'asparagus', 'cauliflower', 'pea', 'bean', 'yam',
    'sweet potato', 'kale', 'okra', 'leek',
}
_ANIMALS = {
    'cat', 'dog', 'fish', 'bird', 'rabbit', 'hamster', 'snake', 'turtle',
    'frog', 'mouse', 'rat', 'duck', 'chicken', 'cow', 'pig', 'horse',
    'sheep', 'goat', 'bear', 'wolf', 'fox', 'deer', 'elephant', 'lion',
    'tiger', 'monkey', 'parrot', 'eagle', 'owl', 'penguin', 'whale',
    'dolphin', 'shark', 'butterfly', 'ant', 'bee', 'spider', 'snail',
    'worm', 'crab', 'lobster', 'octopus', 'starfish', 'camel', 'giraffe',
    'zebra', 'hippo', 'rhino', 'gorilla', 'cheetah', 'leopard', 'panda',
    'koala', 'kangaroo', 'donkey', 'mule', 'llama', 'alpaca',
}
_CATEGORY_SETS = {
    'musical instruments': _MUSICAL_INSTRUMENTS,
    'fruits': _FRUITS, 'vegetables': _VEGETABLES, 'animals': _ANIMALS,
}
_IRREGULAR_PLURALS = {
    'mice': 'mouse', 'geese': 'goose', 'oxen': 'ox',
    'teeth': 'tooth', 'feet': 'foot',
}


def _word_variants(word):
    w = word.lower()
    variants = {w}
    if w in _IRREGULAR_PLURALS:
        variants.add(_IRREGULAR_PLURALS[w])
    if w.endswith('s') and not w.endswith('ss'):
        variants.add(w[:-1])
    if w.endswith('es'):
        variants.add(w[:-2])
    if w.endswith('ies') and len(w) > 4:
        variants.add(w[:-3] + 'y')
    variants.add(w + 's')
    return variants


def _in_category(name, cat_set):
    for w in name.lower().split():
        if _word_variants(w) & cat_set:
            return True
    return False


def _in_category_text(text, cat_set):
    """Like _in_category but for arbitrary text — checks every word."""
    for w in re.findall(r"[a-zA-Z]+", text.lower()):
        if _word_variants(w) & cat_set:
            return True
    return False


# ── Helpers ────────────────────────────────────────────────────────
def load_task(name):
    with open(os.path.join(BBH_CACHE, f"{name}.json")) as f:
        return json.load(f)


def _extract_body_question(text):
    if text.startswith("Question:"):
        text = text[len("Question:"):].strip()
    q_idx = text.rfind("?")
    if q_idx < 0:
        return text, ""
    period_idx = text.rfind(".", 0, q_idx)
    if period_idx >= 0:
        return text[:period_idx + 1].strip(), text[period_idx + 1:q_idx + 1].strip()
    return text, text[:q_idx + 1].strip()


def _parse_json(text):
    text = re.sub(r'```(?:json)?\s*', '', text).replace('```', '')
    obj_start = text.find("{")
    obj_end = text.rfind("}")
    if obj_start >= 0 and obj_end > obj_start:
        try:
            return json.loads(text[obj_start:obj_end + 1])
        except (json.JSONDecodeError, ValueError):
            pass
    return None


def generate(model, tokenizer, device, prompt, max_tokens=40):
    messages = [{"role": "user", "content": prompt}]
    chat_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(chat_text, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def deterministic_segment(body, question):
    parts = re.split(r',\s*(?:and\s+)?|\s+and\s+', body)
    parts = [p.strip().rstrip('.') for p in parts if p.strip()]
    parts = [re.sub(r'^I have\s+', '', p, flags=re.IGNORECASE) for p in parts]
    if question:
        parts.append(question)
    return parts


# ── Accumulators ───────────────────────────────────────────────────
def consume_baseline(data, items_list, segment_text):
    """Original: only type=item makes it through."""
    if not isinstance(data, dict):
        return None
    t = data.get("type", "")
    if t == "item":
        items_list.append(data)
    elif t == "query":
        return data.get("category", "").lower()
    return None


def consume_fixed(data, items_list, segment_text):
    """Fixed: recurse into items list, accept object type, recover bare number,
    fall back to segment text on missing name.
    """
    if not isinstance(data, dict):
        return None

    t = data.get("type", "")

    # Query → return category
    if t == "query":
        return data.get("category", "").lower()

    # Recurse into nested items list
    nested = data.get("items")
    if isinstance(nested, list) and len(nested) > 0 and isinstance(nested[0], dict):
        for item in nested:
            consume_fixed(item, items_list, segment_text)
        return None

    # Accept type=item directly
    if t == "item":
        items_list.append({
            "name": data.get("name", segment_text),
            "count": data.get("count", 1),
            "_segment": segment_text,
        })
        return None

    # Accept type=object as item
    if t == "object" and "name" in data:
        items_list.append({
            "name": data["name"],
            "count": data.get("count", 1),
            "_segment": segment_text,
        })
        return None

    # type=number → recover name from segment text
    if t == "number":
        value = data.get("value", 1)
        try:
            value = int(value)
        except (ValueError, TypeError):
            value = 1
        # Strip leading number word from segment
        words = segment_text.lower().split()
        if words and words[0] in _NUMBER_WORDS:
            name = " ".join(words[1:])
        else:
            name = segment_text
        items_list.append({"name": name, "count": value, "_segment": segment_text})
        return None

    # Anything with a name field
    if "name" in data:
        items_list.append({
            "name": data["name"],
            "count": data.get("count", 1),
            "_segment": segment_text,
        })
        return None

    # Total bailout — recover entirely from segment
    words = segment_text.lower().split()
    if words and words[0] in _NUMBER_WORDS:
        items_list.append({
            "name": " ".join(words[1:]),
            "count": _NUMBER_WORDS[words[0]],
            "_segment": segment_text,
        })
    elif words and words[0] in ("a", "an"):
        items_list.append({
            "name": " ".join(words[1:]),
            "count": 1,
            "_segment": segment_text,
        })
    return None


def compute_total_baseline(items, category):
    cat_set = _CATEGORY_SETS.get(category)
    total = 0
    for item in items:
        name = item.get("name", "")
        count = item.get("count", 1)
        try:
            count = int(count)
        except (ValueError, TypeError):
            count = 1
        if category == "objects":
            total += count
        elif cat_set and _in_category(name, cat_set):
            total += count
    return total


def compute_total_fixed(items, category):
    """Same as baseline plus segment-text fallback when name doesn't match."""
    cat_set = _CATEGORY_SETS.get(category)
    total = 0
    for item in items:
        name = item.get("name", "")
        count = item.get("count", 1)
        try:
            count = int(count)
        except (ValueError, TypeError):
            count = 1
        if category == "objects":
            total += count
            continue
        if not cat_set:
            continue
        if _in_category(name, cat_set):
            total += count
        else:
            # Fallback: check the original segment text for any category word
            seg = item.get("_segment", "")
            if seg and _in_category_text(seg, cat_set):
                total += count
    return total


# ── Eval ───────────────────────────────────────────────────────────
def eval_objcount(model, tokenizer, examples, consume_fn, compute_fn, label):
    correct = 0
    total = 0

    for ex_idx, ex in enumerate(examples[:N_EVAL]):
        text = ex["input"]
        gt = ex["target"].strip()

        body, question = _extract_body_question(text)
        if not body or not question:
            continue

        segments = deterministic_segment(body, question)
        items = []
        category = None

        for seg in segments:
            response = generate(model, tokenizer, DEVICE,
                                _OBJ_EXTRACT_PROMPT.format(segment=seg),
                                max_tokens=40)
            data = _parse_json(response)
            cat = consume_fn(data, items, seg)
            if cat:
                category = cat

        if category:
            predicted = str(compute_fn(items, category))
        else:
            predicted = None

        if predicted == gt:
            correct += 1
        total += 1

        if total % 50 == 0:
            print(f"  [{label}] {total}/{N_EVAL} correct={correct}/{total} "
                  f"({100*correct/total:.1f}%)", flush=True)

    return correct, total


def main():
    print(f"Device: {DEVICE}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16).to(DEVICE)
    model.eval()

    examples = load_task('object_counting')

    print(f"\n{'='*70}")
    print("FIXED accumulator (recurse + object type + number recovery + segment fallback)")
    print(f"{'='*70}", flush=True)
    f_correct, f_total = eval_objcount(
        model, tokenizer, examples, consume_fixed, compute_total_fixed, "FIXED")
    print(f"\n  FIXED: {f_correct}/{f_total} ({100*f_correct/f_total:.1f}%)", flush=True)

    print(f"\n{'='*70}")
    print("BASELINE accumulator (control)")
    print(f"{'='*70}", flush=True)
    b_correct, b_total = eval_objcount(
        model, tokenizer, examples, consume_baseline, compute_total_baseline, "BASE")
    print(f"\n  BASELINE: {b_correct}/{b_total} ({100*b_correct/b_total:.1f}%)", flush=True)

    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"  Baseline: {b_correct}/{b_total} ({100*b_correct/b_total:.1f}%)")
    print(f"  Fixed:    {f_correct}/{f_total} ({100*f_correct/f_total:.1f}%)")
    print(f"  Delta: {100*(f_correct-b_correct)/b_total:+.1f}pp")


if __name__ == "__main__":
    main()
