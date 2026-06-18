"""Extraction module — regex fast path, LLM fallback for input parsing.

Turnstyle regex parsers are tuned to specific phrasings. This module
generalizes: try regex first, then use the LLM itself to extract
structured fields from arbitrary input.

Two core primitives:
    classify_token — single forward pass, N-way classification from next-token logits
    generate_short — short greedy generation for open-ended fields
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, Any, Callable

import torch

if TYPE_CHECKING:
    from turnstyle.core import Turnstyle


class ExtractionMethod(Enum):
    """How the input was parsed."""
    REGEX = auto()
    LLM = auto()
    FAILED = auto()


@dataclass
class FieldSpec:
    """Specification for a single field to extract.

    If options is set, uses classify_token (finite field).
    If options is None, uses generate_short (open-ended field).
    """
    name: str
    prompt_template: str         # {input} placeholder
    options: list[str] | None = None
    postprocess: Callable[[str], str] | None = None
    max_tokens: int = 30


@dataclass
class ExtractionSpec:
    """Full extraction specification for a turnstyle.

    fields: what to extract from the input
    assemble: combine extracted fields into the same tuple format parse() returns
    min_confidence: minimum average confidence to accept extraction
    """
    fields: list[FieldSpec]
    assemble: Callable[[dict[str, Any]], Any]
    min_confidence: float = 0.3


@dataclass
class ExtractionResult:
    """Result of an extraction attempt."""
    parsed: Any
    method: ExtractionMethod
    confidence: float = 1.0
    raw_fields: dict[str, tuple[str, float]] = field(default_factory=dict)


def classify_token(
    model, tokenizer, device, prompt: str, options: list[str],
) -> tuple[int, float]:
    """Single forward pass N-way classification.

    Sums probabilities across all tokens that encode each option,
    returns (best_option_index, best_probability).
    """
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits[0, -1]  # last token logits
    probs = torch.softmax(logits, dim=-1)

    # Sum probability mass for each option across its token variants
    option_probs = []
    for opt in options:
        # Encode with and without leading space
        variants = set()
        for prefix in ("", " "):
            ids = tokenizer.encode(prefix + opt, add_special_tokens=False)
            if ids:
                variants.add(ids[0])
        prob = sum(probs[tid].item() for tid in variants)
        option_probs.append(prob)

    best_idx = max(range(len(option_probs)), key=lambda i: option_probs[i])
    return best_idx, option_probs[best_idx]


def generate_short(
    model, tokenizer, device, prompt: str, max_tokens: int = 30,
) -> tuple[str, float]:
    """Short greedy generation for open-ended fields.

    Returns (generated_text, average_token_confidence).
    """
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            output_scores=True,
            return_dict_in_generate=True,
        )

    # Extract generated tokens (exclude prompt)
    prompt_len = inputs["input_ids"].shape[1]
    gen_ids = out.sequences[0, prompt_len:]
    generated = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

    # Compute average confidence from scores
    if out.scores:
        confidences = []
        for i, score in enumerate(out.scores):
            if i >= len(gen_ids):
                break
            token_id = gen_ids[i].item()
            # Skip EOS
            if token_id == tokenizer.eos_token_id:
                break
            prob = torch.softmax(score[0], dim=-1)[token_id].item()
            confidences.append(prob)
        avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
    else:
        avg_conf = 0.5  # no scores available

    return generated, avg_conf


def extract(
    prompt: str,
    turnstyle: Turnstyle,
    spec: ExtractionSpec | None = None,
) -> ExtractionResult | None:
    """Extract structured input: regex fast path, LLM fallback.

    1. Try turnstyle.parse(prompt) — if it works, return REGEX result
    2. If spec provided: extract each field via LLM
    3. If confidence sufficient: assemble → return LLM result
    4. Else: return FAILED result
    """
    # Fast path: regex
    parsed = turnstyle.parse(prompt)
    if parsed is not None:
        return ExtractionResult(
            parsed=parsed, method=ExtractionMethod.REGEX, confidence=1.0)

    if spec is None:
        return None

    # LLM extraction: field by field
    raw_fields: dict[str, tuple[str, float]] = {}
    field_values: dict[str, Any] = {}

    for fs in spec.fields:
        field_prompt = fs.prompt_template.replace("{input}", prompt)

        if fs.options is not None:
            idx, prob = classify_token(
                turnstyle.model, turnstyle.tokenizer, turnstyle.device,
                field_prompt, fs.options)
            value = fs.options[idx]
            raw_fields[fs.name] = (value, prob)
        else:
            value, prob = generate_short(
                turnstyle.model, turnstyle.tokenizer, turnstyle.device,
                field_prompt, max_tokens=fs.max_tokens)
            raw_fields[fs.name] = (value, prob)

        if fs.postprocess is not None:
            value = fs.postprocess(value)
        field_values[fs.name] = value

    # Check confidence
    avg_confidence = (
        sum(prob for _, prob in raw_fields.values()) / len(raw_fields)
        if raw_fields else 0.0
    )

    if avg_confidence < spec.min_confidence:
        return ExtractionResult(
            parsed=None, method=ExtractionMethod.FAILED,
            confidence=avg_confidence, raw_fields=raw_fields)

    # Assemble
    try:
        assembled = spec.assemble(field_values)
    except Exception:
        return ExtractionResult(
            parsed=None, method=ExtractionMethod.FAILED,
            confidence=avg_confidence, raw_fields=raw_fields)

    return ExtractionResult(
        parsed=assembled, method=ExtractionMethod.LLM,
        confidence=avg_confidence, raw_fields=raw_fields)
