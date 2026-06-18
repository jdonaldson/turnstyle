#!/usr/bin/env python3
"""Test accumulator-side fix for tracking_shuffled extraction failures.

Keep baseline prompt unchanged. Extend the accumulator to:
1. Fall back to person1/player1 when no person/player/name AND no person2/player2
2. Recurse into nested items/assignments lists
3. Filter out placeholder names (unknown, person, anyone, etc.)

Compare baseline vs fixed accumulator on tracking_shuffled.
"""

import json
import os
import re

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

BBH_CACHE = "/Users/jdonaldson/Projects/swollm/data/bbh_cache"
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
DEVICE = "mps"
N_EVAL = 250

def load_task(name):
    with open(os.path.join(BBH_CACHE, f"{name}.json")) as f:
        return json.load(f)

def _extract_body(text):
    if text.startswith("Question:"):
        text = text[len("Question:"):].strip()
    parts = re.split(r'\bOptions\b:?\s*', text, maxsplit=1)
    body = parts[0].strip()
    q_idx = body.rfind("?")
    if q_idx >= 0:
        period_idx = body.rfind(".", 0, q_idx)
        if period_idx >= 0:
            return body[:period_idx + 1].strip()
    return body

def _extract_options(text):
    options = {}
    for m in re.finditer(r"\(([A-R])\)\s+(.+?)(?=\s*\([A-R]\)|\s*$)", text):
        options[m.group(1)] = m.group(2).strip()
    return options

def _segment_tracking_shuffled(body):
    segments = []
    for sent in [s.strip() for s in body.split('.') if s.strip()]:
        if re.search(r'\w+\s+(?:has|gets|buys|is dancing|is playing)', sent):
            parts = re.split(r',\s*(?:and\s+)?|\.\s*(?:and\s+)?', sent)
            parts = [p.strip() for p in parts if p.strip()]
            if len(parts) > 1:
                segments.extend(parts)
                continue
        segments.append(sent)
    return segments

TASK_SEGMENTERS = {
    'navigate': lambda body: [s.strip() for s in body.split('.') if s.strip()],
    'web_of_lies': lambda body: [s.strip() for s in body.split('.') if s.strip()],
    'tracking_shuffled_objects_three_objects': _segment_tracking_shuffled,
    'object_counting': lambda body: [p.strip().rstrip('.') for p in re.split(r',\s*(?:and\s+)?|\s+and\s+', body) if p.strip()],
    'logical_deduction_three_objects': lambda body: [s.strip() for s in body.split('.') if s.strip()],
}

def extract_hidden_l1(body, model, tokenizer, device):
    inputs = tokenizer(body, return_tensors="pt").to(device)
    captured = {}
    def hook_fn(module, input, output):
        h = output[0] if isinstance(output, tuple) else output
        captured['h'] = h.detach().cpu().float().numpy()[0]
    handle = model.model.layers[1].register_forward_hook(hook_fn)
    with torch.no_grad():
        model(**inputs)
    handle.remove()
    return captured['h']

def label_tokens(body, segments, tokenizer):
    enc = tokenizer(body, return_offsets_mapping=True, add_special_tokens=False)
    offsets = enc["offset_mapping"]
    n_tokens = len(offsets)
    labels = [0] * n_tokens
    if n_tokens == 0 or not segments:
        return labels
    boundary_chars, search_start = [], 0
    for seg in segments:
        idx = body.find(seg.strip(), search_start)
        if idx < 0:
            idx = body.lower().find(seg.strip().lower(), search_start)
        if idx >= 0:
            boundary_chars.append(idx)
            search_start = idx + len(seg.strip())
    for tok_idx, (char_start, char_end) in enumerate(offsets):
        for bc in boundary_chars:
            if char_start <= bc < char_end:
                labels[tok_idx] = 1
                break
    return labels

def probe_segment(body, hidden_states, clf, scaler, tokenizer):
    enc = tokenizer(body, return_offsets_mapping=True, add_special_tokens=False)
    offsets = enc["offset_mapping"]
    n_tok = min(len(offsets), hidden_states.shape[0])
    if n_tok == 0:
        return [body]
    X = scaler.transform(hidden_states[:n_tok])
    probs = clf.predict_proba(X)[:, 1]
    boundaries = [0]
    for i in range(1, n_tok):
        if probs[i] > 0.5:
            boundaries.append(i)
    segments = []
    for k, tok_idx in enumerate(boundaries):
        char_start = offsets[tok_idx][0]
        char_end = offsets[boundaries[k + 1]][0] if k + 1 < len(boundaries) else len(body)
        seg = body[char_start:char_end].strip().rstrip('.,;')
        if seg:
            segments.append(seg)
    return segments if segments else [body]

# ── Same baseline prompt ───────────────────────────────────────────

_PROMPT = (
    'You are a text transcription tool. Extract ONLY what is literally stated.\n\n'
    'Statement: "Alice has a yellow ball"\n'
    'JSON: {{"type": "assignment", "person": "Alice", "item": "yellow ball"}}\n\n'
    'Statement: "Bob buys Moby Dick"\n'
    'JSON: {{"type": "assignment", "person": "Bob", "item": "Moby Dick"}}\n\n'
    'Statement: "Bob is dancing with Karl"\n'
    'JSON: {{"type": "assignment", "person": "Bob", "item": "Karl"}}\n\n'
    'Statement: "Alice and Bob swap balls"\n'
    'JSON: {{"type": "swap", "person1": "Alice", "person2": "Bob"}}\n\n'
    'Statement: "Bob and Claire switch partners"\n'
    'JSON: {{"type": "swap", "person1": "Bob", "person2": "Claire"}}\n\n'
    'Statement: "{segment}"\n'
    'JSON:'
)

# ── Accumulators ───────────────────────────────────────────────────

_META_KEYS = {"type", "person", "person1", "person2", "people", "players",
              "activities", "content", "items", "start", "end",
              "start_of_semester", "end_of_period", "end_of_dance",
              "end_of_game", "end_of_match"}

_SWAP_TYPES = {"swap", "switch", "trade", "exchange"}

_PLACEHOLDER_NAMES = {"unknown", "person", "anyone", "someone", "user",
                      "player", "people", "they", "them", "anybody", "everyone"}

def _is_real_name(name):
    if not isinstance(name, str) or not name:
        return False
    if name.lower() in _PLACEHOLDER_NAMES:
        return False
    return True

def _extract_entity_value_baseline(data):
    """Original flexible accumulator: only person/player/name."""
    entity = data.get("person") or data.get("player") or data.get("name")
    if not entity:
        return None, None
    value = None
    for k, v in data.items():
        if k in _META_KEYS:
            continue
        if isinstance(v, str) and v != entity:
            value = v
            break
    return entity, value

def _extract_entity_value_fixed(data):
    """Extended: fall back to person1/player1 when no person2/player2."""
    entity = data.get("person") or data.get("player") or data.get("name")
    if not entity:
        # Fall back to person1 ONLY if there's no second person (not a swap)
        if not data.get("person2") and not data.get("player2"):
            entity = data.get("person1") or data.get("player1")
    if not _is_real_name(entity):
        return None, None
    value = None
    for k, v in data.items():
        if k in _META_KEYS:
            continue
        if isinstance(v, str) and v != entity:
            value = v
            break
    return entity, value

def consume_baseline(data, state, swaps):
    if not isinstance(data, dict):
        return
    t = (data.get("type") or "").lower()
    if t in _SWAP_TYPES:
        p1 = data.get("person1") or data.get("player1")
        p2 = data.get("person2") or data.get("player2")
        if p1 and p2:
            swaps.append((p1, p2))
        return
    entity, value = _extract_entity_value_baseline(data)
    if entity and value:
        state[entity] = value

def consume_fixed(data, state, swaps):
    """Fixed accumulator: person1 fallback + nested items recursion + name filter."""
    if not isinstance(data, dict):
        return

    t = (data.get("type") or "").lower()
    # Swap detection (must come first to not be confused with assignment)
    if t in _SWAP_TYPES:
        p1 = data.get("person1") or data.get("player1")
        p2 = data.get("person2") or data.get("player2")
        if p1 and p2 and _is_real_name(p1) and _is_real_name(p2):
            swaps.append((p1, p2))
        return

    # Recurse into nested items list
    items = data.get("items")
    if isinstance(items, list) and len(items) > 0 and isinstance(items[0], dict):
        for item in items:
            consume_fixed(item, state, swaps)
        return

    # Single-tuple extraction with extended entity fallback
    entity, value = _extract_entity_value_fixed(data)
    if entity and value:
        state[entity] = value

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

def eval_tracking(model, tokenizer, clf, scaler, examples, consume_fn, label):
    correct = 0
    total = 0
    persons_in_state = []

    for ex in examples[:N_EVAL]:
        body = _extract_body(ex["input"])
        options = _extract_options(ex["input"])
        gt = ex["target"].strip()
        if not body or len(body) < 10:
            continue

        h = extract_hidden_l1(body, model, tokenizer, DEVICE)
        segments = probe_segment(body, h, clf, scaler, tokenizer)

        state = {}
        swaps = []
        for seg in segments:
            response = generate(model, tokenizer, DEVICE,
                                _PROMPT.format(segment=seg), max_tokens=40)
            data = _parse_json(response)
            consume_fn(data, state, swaps)

        for n1, n2 in swaps:
            if n1 in state and n2 in state:
                state[n1], state[n2] = state[n2], state[n1]

        persons_in_state.append(len(state))

        query_person = None
        body_tail = body.strip().rstrip('.')
        for pat in [r'(\w+)\s+has\s*$', r'(\w+)\s+has\s+the\s*$',
                    r'(\w+)\s+is\s+playing\s*$', r'(\w+)\s+is\s+dancing\s+with\s*$']:
            m = re.search(pat, body_tail)
            if m:
                query_person = m.group(1)
                break

        ans = None
        if query_person and query_person in state:
            result = state[query_person]
            for letter, opt_value in options.items():
                opt_clean = opt_value.rstrip('.')
                if result.lower() == opt_clean.lower():
                    ans = f"({letter})"
                    break
                if result.lower() in opt_clean.lower() or opt_clean.lower() in result.lower():
                    ans = f"({letter})"
                    break

        if ans == gt:
            correct += 1
        total += 1

        if total % 50 == 0:
            print(f"  [{label}] {total}/{N_EVAL} correct={correct}/{total} ({100*correct/total:.1f}%) avg_p={np.mean(persons_in_state):.2f}", flush=True)

    avg_persons = float(np.mean(persons_in_state)) if persons_in_state else 0.0
    return correct, total, avg_persons


def main():
    print(f"Device: {DEVICE}")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16).to(DEVICE)
    model.eval()

    print("\nTraining L1 boundary probe on all 5 tasks...", flush=True)
    features_list = []
    labels_list = []
    for task_name, segmenter in TASK_SEGMENTERS.items():
        examples = load_task(task_name)[:250]
        for ex in examples:
            body = _extract_body(ex["input"])
            if not body or len(body) < 10:
                continue
            segs = segmenter(body)
            if len(segs) < 2:
                continue
            labs = label_tokens(body, segs, tokenizer)
            try:
                h = extract_hidden_l1(body, model, tokenizer, DEVICE)
            except Exception:
                continue
            n = min(len(labs), h.shape[0])
            features_list.append(h[:n])
            labels_list.append(np.array(labs[:n]))

    X = np.concatenate(features_list)
    y = np.concatenate(labels_list)
    sc = StandardScaler()
    clf = LogisticRegression(solver='lbfgs', max_iter=1000, C=1.0, class_weight='balanced')
    clf.fit(sc.fit_transform(X), y)
    print(f"  Trained on {len(X)} tokens, {y.sum()} boundaries")

    track_examples = load_task('tracking_shuffled_objects_three_objects')

    # Run FIXED first (the experiment)
    print(f"\n{'='*70}")
    print("FIXED accumulator (person1 fallback + items recursion + name filter)")
    print(f"{'='*70}", flush=True)
    f_correct, f_total, f_avg = eval_tracking(
        model, tokenizer, clf, sc, track_examples, consume_fixed, "FIXED")
    print(f"\n  FIXED: {f_correct}/{f_total} ({100*f_correct/f_total:.1f}%)  avg_persons={f_avg:.2f}", flush=True)

    print(f"\n{'='*70}")
    print("BASELINE accumulator (control)")
    print(f"{'='*70}", flush=True)
    b_correct, b_total, b_avg = eval_tracking(
        model, tokenizer, clf, sc, track_examples, consume_baseline, "BASE")
    print(f"\n  BASELINE: {b_correct}/{b_total} ({100*b_correct/b_total:.1f}%)  avg_persons={b_avg:.2f}", flush=True)

    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"  Baseline: {b_correct}/{b_total} ({100*b_correct/b_total:.1f}%)  avg_persons={b_avg:.2f}")
    print(f"  Fixed:    {f_correct}/{f_total} ({100*f_correct/f_total:.1f}%)  avg_persons={f_avg:.2f}")
    print(f"  Delta: {100*(f_correct-b_correct)/b_total:+.1f}pp  avg_persons: {f_avg-b_avg:+.2f}")


if __name__ == "__main__":
    main()
