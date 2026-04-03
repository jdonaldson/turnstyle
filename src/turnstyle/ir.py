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

    Tries to find a JSON array [...] or object {...} in the output.
    Returns parsed JSON or None.
    """
    # Try array first
    start = text.find("[")
    end = text.rfind("]")
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except (json.JSONDecodeError, ValueError):
            pass

    # Try object
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end + 1])
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
