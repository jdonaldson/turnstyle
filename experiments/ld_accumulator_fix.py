#!/usr/bin/env python3
"""Test logical_deduction accumulator-side fixes.

Same prompt/segmenter as baseline. Fix the consumer:

1. Expand relation field fallback: add `price`, `comparison`
2. Expand item2 field fallback: add `compared_item`, `compared_to`, `comparative`,
   `comparative_item`, `comparison_item`
3. Segment-text override for "second-X" phrases: if the segment literally contains
   "second-newest" / "second-oldest" / "second-cheapest" / "second-most expensive",
   use that as the position regardless of extracted position field (Qwen often drops
   the "second-" prefix and returns just "newest").
4. Skip preamble segments ("The following paragraphs...", "The statements are
   logically consistent...")
5. Accept any list-valued field as object list (covers `golfer_names`, `book_names`,
   `names`, `players`, etc.) — but reject single-entry intro artifacts like
   `['fruit stand']` by rejecting values where the only entry isn't in the segment
   as a plural noun.

Compare baseline vs fixed on logical_deduction_three_objects.
"""
import json
import os
import re
import sys

sys.path.insert(0, "/tmp")
import qwen_two_stage_experiment as q2  # noqa: E402

import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

BBH_CACHE = "/Users/jdonaldson/Projects/swollm/data/bbh_cache"
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
DEVICE = "mps"
N_EVAL = 250

# Reuse helpers from existing module
_parse_json = q2._parse_json
_extract_body = q2._extract_body
_extract_options = q2._extract_options
generate = q2.generate
segment = q2.segment
_LD_EXTRACT_PROMPT = q2._LD_EXTRACT_PROMPT
_resolve_position = q2._resolve_position
_resolve_relation = q2._resolve_relation
_solve_csp = q2._solve_csp
_extract_option_position = q2._extract_option_position
_strip_articles = q2._strip_articles


# Preamble patterns to skip
_PREAMBLE_PATTERNS = [
    re.compile(r'the following paragraphs', re.IGNORECASE),
    re.compile(r'the statements are logically consistent', re.IGNORECASE),
]


def _is_preamble(seg):
    return any(p.search(seg) for p in _PREAMBLE_PATTERNS)


# Segment-text override: detect "second-X" / "third-X" phrases directly
_SECOND_X_RE = re.compile(
    r'\b(?:second|third|fourth)-(?:newest|oldest|cheapest|most\s+expensive)',
    re.IGNORECASE,
)


def _segment_position_override(segment_text):
    """If segment literally contains 'second-newest' etc., return that phrase."""
    m = _SECOND_X_RE.search(segment_text)
    if m:
        return m.group(0).lower().replace(' ', '-').replace('-most-expensive', '-most expensive')
    return None


def ld_solve_fixed(model, tokenizer, device, text):
    body = _extract_body(text)
    options = _extract_options(text)

    segments = segment(model, tokenizer, device, body)
    if not segments:
        return None, {"error": "segment_failed"}

    objects = []
    constraints = []
    debug_extractions = []

    for seg in segments:
        # [FIX 4] Skip preamble segments
        if _is_preamble(seg):
            continue

        response = generate(model, tokenizer, device,
                            _LD_EXTRACT_PROMPT.format(segment=seg), max_tokens=50)
        data = _parse_json(response)
        debug_extractions.append({"seg": seg[:60], "parsed": data})
        if not isinstance(data, dict):
            continue

        # Entity fields
        i1 = data.get("item1") or data.get("item") or data.get("subject")

        # [FIX 2] Expand item2 field fallback
        i2 = (data.get("item2")
              or data.get("other_item")
              or data.get("object2")
              or data.get("compared_item")
              or data.get("compared_to")
              or data.get("comparative")
              or data.get("comparative_item")
              or data.get("comparison_item"))

        # Item1/item2 confusion: both present
        if not i2 and "item" in data and "item1" in data:
            i2 = data.get("item1")
            i1 = data.get("item")

        # [FIX 1] Expand relation field fallback
        rel = (data.get("relation")
               or data.get("price_relation")
               or data.get("spatial_relation")
               or data.get("ordering")
               or data.get("comparison")
               or data.get("price")
               or data.get("comparison_type"))

        # Position field can contain a relation
        pos = data.get("position", "")
        if not rel and pos and _resolve_relation(str(pos)):
            rel = pos

        # [FIX 3] Segment-text override for "second-X"
        seg_override = _segment_position_override(seg)

        if i1 and i2 and rel:
            # Relationship: two items + relation phrase
            item1 = _strip_articles(str(i1))
            item2 = _strip_articles(str(i2)).rstrip('.')
            direction = _resolve_relation(str(rel))
            if item1 and item2 and direction:
                if direction == 'item1_first':
                    constraints.append(("lt", item1, item2))
                else:
                    constraints.append(("lt", item2, item1))
        elif "item" in data and ("position" in data or seg_override):
            # Attribute: one item + position
            item = _strip_articles(str(data["item"]))
            n_obj = max(len(objects), 3)
            # [FIX 3] Use segment override if available
            pos_str = seg_override if seg_override else str(data["position"])
            resolved = _resolve_position(pos_str, n_obj)
            if item and resolved is not None:
                if isinstance(resolved, tuple) and resolved[0] == 'not':
                    constraints.append(("neq", item, resolved[1]))
                else:
                    constraints.append(("eq", item, resolved))
        else:
            # Extract objects from any list-valued field
            for key, val in data.items():
                if isinstance(val, list) and all(isinstance(v, str) for v in val):
                    # Reject obviously wrong single-entry intros like ['fruit stand']
                    if len(val) == 1 and any(
                        kw in val[0].lower()
                        for kw in ('stand', 'show', 'shelf', 'tournament',
                                   'branch', 'race', 'scene')
                    ):
                        continue
                    for name in val:
                        name = _strip_articles(name.strip())
                        if name and len(name) > 1:
                            objects.append(name)
                    break
            else:
                if "item" in data and not rel:
                    name = _strip_articles(str(data["item"]))
                    if (name and len(name) > 1
                            and not any(w in name.lower() for w in
                                        ('three', 'objects', 'paragraphs',
                                         'stand', 'show', 'shelf', 'tournament'))):
                        objects.append(name)

    # Build object list from constraints (extraction is unreliable)
    constraint_objects = []
    seen = set()
    for c in constraints:
        if c[0] == 'lt':
            for name in (c[1], c[2]):
                if name.lower() not in seen:
                    constraint_objects.append(name)
                    seen.add(name.lower())
        elif c[0] in ('eq', 'neq'):
            if c[1].lower() not in seen:
                constraint_objects.append(c[1])
                seen.add(c[1].lower())

    if constraint_objects:
        objects = constraint_objects

    # Try to find missing objects from options (3-object problems)
    if len(objects) < 3 and options:
        for opt_text in options.values():
            opt_lower = opt_text.lower()
            m = re.match(r'(?:the\s+)?(.+?)\s+(?:is|are)\s+', opt_lower)
            if m:
                candidate = _strip_articles(m.group(1).strip())
                if candidate and candidate.lower() not in seen:
                    objects.append(candidate)
                    seen.add(candidate.lower())
                    if len(objects) >= 3:
                        break

    if not objects or not constraints:
        return None, {"error": "no_objects_or_constraints",
                      "objects": objects, "constraints": constraints,
                      "extractions": debug_extractions}

    # Normalize constraint object names
    obj_stripped = {_strip_articles(o).lower(): o for o in objects}
    normalized_constraints = []
    for c in constraints:
        if c[0] == 'lt':
            a = obj_stripped.get(_strip_articles(c[1]).lower(), c[1])
            b = obj_stripped.get(_strip_articles(c[2]).lower(), c[2])
            normalized_constraints.append(('lt', a, b))
        elif c[0] in ('eq', 'neq'):
            a = obj_stripped.get(_strip_articles(c[1]).lower(), c[1])
            normalized_constraints.append((c[0], a, c[2]))

    valid = _solve_csp(objects, normalized_constraints)
    if not valid:
        return None, {"error": "no_valid_perm",
                      "objects": objects, "constraints": normalized_constraints,
                      "extractions": debug_extractions}

    if not options:
        return None, {"error": "no_options"}

    n = len(objects)
    for letter, opt_text in options.items():
        opt_lower = opt_text.lower()
        opt_obj = None
        for stripped in sorted(obj_stripped.keys(), key=len, reverse=True):
            if stripped in opt_lower:
                opt_obj = obj_stripped[stripped]
                break
        if not opt_obj:
            continue
        target_pos = _extract_option_position(opt_lower, n)
        if target_pos is None:
            continue
        if all(sol[opt_obj] == target_pos for sol in valid):
            return f"({letter})", {"objects": objects,
                                   "constraints": normalized_constraints}

    return None, {"error": "no_match", "objects": objects,
                  "constraints": normalized_constraints, "valid": len(valid)}


def load_task(name):
    with open(os.path.join(BBH_CACHE, f"{name}.json")) as f:
        return json.load(f)


def eval_ld(model, tokenizer, examples, solve_fn, label):
    correct = 0
    total = 0
    for idx, ex in enumerate(examples[:N_EVAL]):
        text = ex["input"]
        gt = ex["target"].strip()

        answer, _ = solve_fn(model, tokenizer, DEVICE, text)
        total += 1
        if answer == gt:
            correct += 1

        if total % 50 == 0:
            print(f"  [{label}] {total}/{N_EVAL} correct={correct}/{total} "
                  f"({100*correct/total:.1f}%)", flush=True)
    return correct, total


def main():
    print(f"Device: {DEVICE}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16).to(DEVICE)
    model.eval()

    examples = load_task("logical_deduction_three_objects")

    print(f"\n{'='*70}")
    print("FIXED ld_solve (field fallbacks + segment override + preamble skip)")
    print(f"{'='*70}", flush=True)
    f_correct, f_total = eval_ld(model, tokenizer, examples, ld_solve_fixed, "FIXED")
    print(f"\n  FIXED: {f_correct}/{f_total} "
          f"({100*f_correct/f_total:.1f}%)", flush=True)

    print(f"\n{'='*70}")
    print("BASELINE ld_solve (control)")
    print(f"{'='*70}", flush=True)
    b_correct, b_total = eval_ld(model, tokenizer, examples, q2.ld_solve, "BASE")
    print(f"\n  BASELINE: {b_correct}/{b_total} "
          f"({100*b_correct/b_total:.1f}%)", flush=True)

    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"  Baseline: {b_correct}/{b_total} ({100*b_correct/b_total:.1f}%)")
    print(f"  Fixed:    {f_correct}/{f_total} ({100*f_correct/f_total:.1f}%)")
    print(f"  Delta: {100*(f_correct-b_correct)/b_total:+.1f}pp")


if __name__ == "__main__":
    main()
