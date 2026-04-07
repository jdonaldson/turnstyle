"""IR extraction — replace per-task regex with LLM-based structured extraction.

Single-pass JSON extraction via LLM generation, followed by deterministic
compute on the parsed IR. Each task defines an IRSpec (prompt + compute);
the extraction machinery is shared.

Two usage patterns:
    1. IRSolver(model, tokenizer, device, spec) — as llm_fallback for solvers
    2. ir_solve(model, tokenizer, device, text, spec) — standalone function
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Callable

import torch


@dataclass
class IRSpec:
    """Declarative specification for extracting a typed IR from text.

    extraction_prompt: few-shot prompt with {body} placeholder. The LLM should
        output a JSON object or array.
    compute: (ir_data, question, options) → answer string or None.
        ir_data is the parsed JSON (list or dict).
        question is the extracted question text (may be None for free-form tasks).
        options is a dict of letter→text (empty for non-MC tasks).
    max_tokens: maximum tokens for LLM generation.
    """
    extraction_prompt: str
    compute: Callable[[Any, str | None, dict], str | None]
    max_tokens: int = 300


def _parse_json(text: str) -> Any | None:
    """Extract JSON from model output, tolerant of surrounding text.

    Tries object first (most IRSpecs expect a dict), then array.
    If the outer object fails to parse (e.g. concatenated fragments),
    wraps in an array and retries — lets compute functions merge fragments.
    """
    # Try object first — expected top-level type for most IRSpecs
    obj_start = text.find("{")
    obj_end = text.rfind("}")
    if obj_start >= 0 and obj_end > obj_start:
        candidate = text[obj_start:obj_end + 1]
        try:
            return json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            # Might be concatenated objects: {...}, {...}, ...
            # Wrap in array brackets and retry
            try:
                return json.loads("[" + candidate + "]")
            except (json.JSONDecodeError, ValueError):
                pass

    # Try array
    arr_start = text.find("[")
    arr_end = text.rfind("]")
    if arr_start >= 0 and arr_end > arr_start:
        try:
            return json.loads(text[arr_start:arr_end + 1])
        except (json.JSONDecodeError, ValueError):
            pass

    return None


def _extract_body(text: str) -> str:
    """Extract the scene/body portion before the question."""
    # Strip "Question:" prefix if present
    if text.startswith("Question:"):
        text = text[len("Question:"):].strip()

    # Take everything before "Options:" if present
    parts = re.split(r'\bOptions\b:?\s*', text, maxsplit=1)
    body = parts[0].strip()

    # Try to separate body from question (last sentence ending with ?)
    q_idx = body.rfind("?")
    if q_idx >= 0:
        # Find sentence start before the question
        period_idx = body.rfind(".", 0, q_idx)
        if period_idx >= 0:
            return body[:period_idx + 1].strip()
    return body


def _extract_question(text: str) -> str | None:
    """Extract the question sentence (last ? before Options:)."""
    q_text = text.split("Options:")[0]
    q_idx = q_text.rfind("?")
    if q_idx < 0:
        return None
    q_start = q_text.rfind(".", 0, q_idx)
    if q_start < 0:
        q_start = q_text.rfind("  ", 0, q_idx)
    raw = q_text[q_start + 1:q_idx + 1].strip()
    if "\n" in raw:
        for line in reversed(raw.split("\n")):
            line = line.strip()
            if line and "?" in line:
                return line
    return raw


def _extract_options(text: str) -> dict:
    """Extract BBH-style options: (A) foo (B) bar → {A: foo, B: bar}."""
    options = {}
    for m in re.finditer(r"\(([A-R])\)\s+(.+?)(?=\s*\([A-R]\)|\s*$)", text):
        options[m.group(1)] = m.group(2).strip()
    return options


def ir_solve(
    model, tokenizer, device,
    text: str,
    spec: IRSpec,
    diag: dict | None = None,
) -> str | None:
    """Extract structured IR via LLM → compute answer.

    1. Extract body from BBH text
    2. Format extraction prompt with body
    3. LLM generates JSON
    4. Parse JSON
    5. Compute answer via spec.compute(ir_data, question, options)
    """
    body = _extract_body(text)
    question = _extract_question(text)
    options = _extract_options(text)

    if diag is not None:
        diag["body"] = body
        diag["question"] = question
        diag["options"] = options

    # LLM generates JSON from body
    prompt = spec.extraction_prompt.format(body=body)
    messages = [{"role": "user", "content": prompt}]
    chat_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(chat_text, return_tensors="pt").to(device)

    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=spec.max_tokens,
            do_sample=False, temperature=1.0)
    response = tokenizer.decode(
        out[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True).strip()

    if diag is not None:
        diag["raw_response"] = response

    ir_data = _parse_json(response)
    if ir_data is None:
        if diag is not None:
            diag["error"] = "json_parse_failed"
        return None

    if diag is not None:
        diag["ir_data"] = ir_data

    # Compute answer
    try:
        answer = spec.compute(ir_data, question, options)
    except Exception as e:
        if diag is not None:
            diag["error"] = f"compute_failed: {e}"
        return None

    if diag is not None:
        diag["answer"] = answer

    return answer


class IRSolver:
    """Lightweight wrapper for IR extraction, used as llm_fallback.

    Exposes .model/.tokenizer/.device for compatibility with existing
    solver patterns that access llm_fallback attributes.
    """

    def __init__(self, model, tokenizer, device, spec: IRSpec):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.spec = spec

    def solve(self, text: str, diag: dict | None = None) -> str | None:
        """Extract IR and compute answer."""
        return ir_solve(
            self.model, self.tokenizer, self.device,
            text, self.spec, diag=diag)


# ═══════════════════════════════════════════════════════════════════════
# Per-sentence IR extraction
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class SentenceRecord:
    """One extracted record from a single sentence."""
    sentence: str
    record_type: str
    data: dict
    confidence: float = 1.0


@dataclass
class SentenceIRSpec:
    """Specification for per-sentence extraction + symbolic aggregation.

    sentence_types: valid type labels for classification.
    extract_prompt: template with {sentence} and {type} placeholders.
    aggregate: (records, question, options) → answer string or None.
    classify_fn: (sentence) → type string. None = use classify_token.
    split_fn: (body) → list[str]. None = period-split.
    max_tokens: maximum tokens per sentence extraction.
    """
    sentence_types: list[str]
    extract_prompt: str
    aggregate: Callable
    classify_fn: Callable | None = None
    split_fn: Callable | None = None
    segment_prompt: str | None = None
    segment_max_tokens: int = 200
    max_tokens: int = 60


def _default_split(body: str) -> list[str]:
    """Split text on periods, strip whitespace, filter empty."""
    return [s.strip() for s in body.split(".") if s.strip()]


def _segment_via_llm(model, tokenizer, device, body, spec):
    """LLM-based segmentation. Returns [(text, type), ...] or None on failure."""
    from turnstyle.extract import generate_short

    prompt = spec.segment_prompt.format(
        body=body, types=", ".join(spec.sentence_types))
    response, _ = generate_short(
        model, tokenizer, device, prompt,
        max_tokens=spec.segment_max_tokens)
    parsed = _parse_json(response)
    if not isinstance(parsed, list):
        return None
    segments = []
    valid_types = set(spec.sentence_types)
    for item in parsed:
        if not isinstance(item, dict):
            continue
        text = item.get("text", "").strip()
        seg_type = item.get("type", "").strip()
        if not text:
            continue
        if seg_type not in valid_types:
            if spec.classify_fn is not None:
                seg_type = spec.classify_fn(text)
            else:
                continue
        segments.append((text, seg_type))
    return segments if segments else None


def sentence_ir_solve(
    model, tokenizer, device,
    text: str,
    spec: SentenceIRSpec,
    diag: dict | None = None,
) -> str | None:
    """Per-sentence extraction → symbolic aggregation.

    1. Extract body/question/options from BBH text
    2. Segment body into typed chunks (LLM or deterministic fallback)
    3. Append question segment
    4. Extract structured data from each segment via LLM
    5. Aggregate records and compute answer
    """
    from turnstyle.extract import generate_short

    body = _extract_body(text)
    question = _extract_question(text)
    options = _extract_options(text)

    # Segmentation: try LLM first, fall back to deterministic
    segments = None  # list of (text, type) pairs
    segment_method = "deterministic"

    if spec.segment_prompt is not None:
        segments = _segment_via_llm(model, tokenizer, device, body, spec)
        if segments is not None:
            segment_method = "llm"
        else:
            segment_method = "llm_failed"

    if segments is None:
        # Deterministic fallback: split + classify
        split_fn = spec.split_fn or _default_split
        sentences = split_fn(body)
        segments = []
        for sentence in sentences:
            if spec.classify_fn is not None:
                record_type = spec.classify_fn(sentence)
            else:
                from turnstyle.extract import classify_token
                idx, _ = classify_token(
                    model, tokenizer, device, sentence, spec.sentence_types)
                record_type = spec.sentence_types[idx]
            segments.append((sentence, record_type))

    # Append question segment if present and not already in LLM output
    if question:
        has_query = any(t == "query" for _, t in segments)
        if not has_query:
            if spec.classify_fn is not None:
                q_type = spec.classify_fn(question)
            else:
                q_type = "query"
            segments.append((question, q_type))

    if diag is not None:
        diag["body"] = body
        diag["question"] = question
        diag["options"] = options
        diag["segments"] = [(t, tp) for t, tp in segments]
        diag["segment_method"] = segment_method

    records = []
    sentence_diags = []

    for seg_text, seg_type in segments:
        # Extract via LLM
        prompt = spec.extract_prompt.format(sentence=seg_text, type=seg_type)
        response, confidence = generate_short(
            model, tokenizer, device, prompt, max_tokens=spec.max_tokens)

        data = _parse_json(response)
        s_diag = {
            "sentence": seg_text,
            "type": seg_type,
            "response": response,
            "confidence": confidence,
            "parsed": data is not None,
        }
        sentence_diags.append(s_diag)

        if data is not None:
            records.append(SentenceRecord(
                sentence=seg_text,
                record_type=seg_type,
                data=data,
                confidence=confidence,
            ))

    if diag is not None:
        diag["sentence_extractions"] = sentence_diags
        diag["n_parsed"] = len(records)
        diag["n_segments"] = len(segments)

    # Aggregate
    try:
        answer = spec.aggregate(records, question, options)
    except Exception as e:
        if diag is not None:
            diag["error"] = f"aggregate_failed: {e}"
        return None

    if diag is not None:
        diag["answer"] = answer

    return answer


class SentenceIRSolver:
    """Wrapper for per-sentence IR extraction, same interface as IRSolver."""

    def __init__(self, model, tokenizer, device, spec: SentenceIRSpec):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.spec = spec

    def solve(self, text: str, diag: dict | None = None) -> str | None:
        """Extract per-sentence IR and compute answer."""
        return sentence_ir_solve(
            self.model, self.tokenizer, self.device,
            text, self.spec, diag=diag)
