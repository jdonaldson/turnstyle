#!/usr/bin/env python3
"""Diagnose object_counting failures.

For each example, run the existing extraction pipeline and categorize failures:
1. extraction_lost_item — fewer items extracted than appear in input
2. wrong_count — right items, wrong per-item count
3. category_miss — item extracted but not matched in category set
4. wrong_category — query extracted wrong category
5. compute_only — extraction perfect, ground-truth lookup wrong (bug in lookup tables)

Print 3-5 failure examples per category for analysis.
"""

import json
import os
import re
from collections import defaultdict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

BBH_CACHE = "/Users/jdonaldson/Projects/swollm/data/bbh_cache"
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
DEVICE = "mps"
N_EVAL = 250
N_SHOW_PER_CAT = 4

# ── Number words ───────────────────────────────────────────────────
_NUMBER_WORDS = {
    "a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
    "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
}

# ── Existing extraction prompt + lookup tables ─────────────────────
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


# ── Helpers ────────────────────────────────────────────────────────
def load_task(name):
    with open(os.path.join(BBH_CACHE, f"{name}.json")) as f:
        return json.load(f)


def _extract_body_question(text):
    """Split into body (item list) + question."""
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
    """Comma-split + 'and ' split for the body, then append the question."""
    parts = re.split(r',\s*(?:and\s+)?|\s+and\s+', body)
    parts = [p.strip().rstrip('.') for p in parts if p.strip()]
    # Drop the leading "I have" from the first part
    parts = [re.sub(r'^I have\s+', '', p, flags=re.IGNORECASE) for p in parts]
    if question:
        parts.append(question)
    return parts


def parse_ground_truth(body):
    """Parse the input regex-style to get ground-truth (name, count) pairs.

    Used to compare against LLM extraction.
    """
    # Strip "I have"
    body_clean = re.sub(r'^I have\s+', '', body.strip().rstrip('.'), flags=re.IGNORECASE)
    parts = re.split(r',\s*(?:and\s+)?|\s+and\s+', body_clean)
    items = []
    for part in parts:
        part = part.strip().rstrip('.')
        if not part:
            continue
        words = part.split()
        if not words:
            continue
        first = words[0].lower()
        if first in _NUMBER_WORDS:
            count = _NUMBER_WORDS[first]
            name = " ".join(words[1:])
        else:
            count = 1
            name = part
        # Strip leading article if any leftover
        name = re.sub(r'^(?:an?\s+|the\s+)', '', name, flags=re.IGNORECASE).strip()
        # Singularize trailing 's' for matching
        items.append((name.lower(), count))
    return items


def parse_question_category(question):
    m = re.search(r'how many\s+(.+?)\s+do', question, re.IGNORECASE)
    if m:
        return m.group(1).strip().lower()
    return None


# ── Main ───────────────────────────────────────────────────────────
def main():
    print(f"Device: {DEVICE}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16).to(DEVICE)
    model.eval()

    examples = load_task('object_counting')[:N_EVAL]

    counters = defaultdict(int)
    failure_examples = defaultdict(list)
    correct = 0
    total = 0

    for ex_idx, ex in enumerate(examples):
        text = ex["input"]
        gt = ex["target"].strip()

        body, question = _extract_body_question(text)
        if not body or not question:
            counters["malformed"] += 1
            continue

        gt_items = parse_ground_truth(body)
        gt_category = parse_question_category(question)

        # Run extraction pipeline
        segments = deterministic_segment(body, question)
        extracted_items = []
        extracted_category = None
        raw_responses = []

        for seg in segments:
            response = generate(model, tokenizer, DEVICE,
                                _OBJ_EXTRACT_PROMPT.format(segment=seg),
                                max_tokens=40)
            raw_responses.append((seg, response))
            data = _parse_json(response)
            if not isinstance(data, dict):
                continue
            t = data.get("type", "")
            if t == "item":
                extracted_items.append(data)
            elif t == "query":
                extracted_category = data.get("category", "").lower()

        # Compute predicted answer
        if not extracted_category:
            predicted = None
        else:
            cat_set = _CATEGORY_SETS.get(extracted_category)
            tot = 0
            for item in extracted_items:
                name = item.get("name", "")
                count = item.get("count", 1)
                try:
                    count = int(count)
                except (ValueError, TypeError):
                    count = 1
                if extracted_category == "objects":
                    tot += count
                elif cat_set and _in_category(name, cat_set):
                    tot += count
            predicted = str(tot)

        is_correct = (predicted == gt)
        if is_correct:
            correct += 1
            total += 1
            continue
        total += 1

        # ── Categorize failure ─────────────────────────────────
        cat = "unknown"

        if extracted_category is None:
            cat = "no_category_extracted"
        elif extracted_category != gt_category:
            cat = "wrong_category"
        elif len(extracted_items) < len(gt_items):
            cat = "extraction_lost_item"
        elif len(extracted_items) > len(gt_items):
            cat = "extraction_extra_item"
        else:
            # Same number of items — check counts
            count_mismatches = 0
            for ex_item, gt_item in zip(extracted_items, gt_items):
                ex_count = ex_item.get("count", 1)
                try:
                    ex_count = int(ex_count)
                except (ValueError, TypeError):
                    ex_count = 1
                if ex_count != gt_item[1]:
                    count_mismatches += 1
            if count_mismatches > 0:
                cat = "wrong_count"
            else:
                # Counts match — must be category lookup
                # Compute oracle answer using parsed ground-truth items + extracted category
                cat_set = _CATEGORY_SETS.get(extracted_category)
                oracle_total = 0
                for name, count in gt_items:
                    if extracted_category == "objects":
                        oracle_total += count
                    elif cat_set:
                        if _in_category(name, cat_set):
                            oracle_total += count
                if str(oracle_total) == gt:
                    cat = "category_lookup_failed"
                else:
                    cat = "compute_or_lookup_bug"

        counters[cat] += 1
        if len(failure_examples[cat]) < N_SHOW_PER_CAT:
            failure_examples[cat].append({
                "idx": ex_idx,
                "body": body,
                "question": question,
                "gt": gt,
                "gt_items": gt_items,
                "gt_category": gt_category,
                "extracted_items": extracted_items,
                "extracted_category": extracted_category,
                "predicted": predicted,
                "raw_responses": raw_responses,
            })

        if (ex_idx + 1) % 50 == 0:
            print(f"  [{ex_idx+1}/{N_EVAL}] correct={correct}/{total} ({100*correct/total:.1f}%)",
                  flush=True)

    # ── Print summary ─────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"FINAL: {correct}/{total} ({100*correct/total:.1f}%)")
    print(f"{'='*70}")
    print("\nFailure categories:")
    for cat, n in sorted(counters.items(), key=lambda x: -x[1]):
        print(f"  {cat:30s} {n:3d}")

    print(f"\n{'='*70}")
    print("FAILURE EXAMPLES")
    print(f"{'='*70}")
    for cat, exs in failure_examples.items():
        print(f"\n--- {cat} ({counters[cat]} total) ---")
        for ex in exs:
            print(f"\n  ex {ex['idx']}: gt={ex['gt']}, predicted={ex['predicted']}")
            print(f"    body: {ex['body'][:160]}{'...' if len(ex['body']) > 160 else ''}")
            print(f"    question: {ex['question']}")
            print(f"    gt_items ({len(ex['gt_items'])}): {ex['gt_items']}")
            print(f"    gt_category: {ex['gt_category']}")
            print(f"    extr_items ({len(ex['extracted_items'])}):")
            for it in ex['extracted_items'][:8]:
                print(f"      {it}")
            print(f"    extr_category: {ex['extracted_category']}")
            # Show first 3 raw responses where the parse failed or returned non-item
            shown = 0
            for seg, resp in ex['raw_responses']:
                if shown >= 3:
                    break
                d = _parse_json(resp)
                if not isinstance(d, dict) or d.get("type") not in ("item", "query"):
                    print(f"    BAD seg: '{seg[:60]}' -> {resp[:120]}")
                    shown += 1


if __name__ == "__main__":
    main()
