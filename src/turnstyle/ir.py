"""IR extraction — replace per-task regex with LLM-based structured extraction.

Single-pass JSON extraction via LLM generation, followed by deterministic
compute on the parsed IR. Each task defines an IRSpec (prompt + compute);
the extraction machinery is shared.

Two usage patterns:
    1. IRSolver(model, tokenizer, device, spec) — as llm_fallback for solvers
    2. ir_solve(model, tokenizer, device, text, spec) — standalone function

Scene parsing is sentences-first: parse_scene() splits the full text into
sentences and classifies each as body / question / option — no dependency on
"Options:" keyword.
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


@dataclass
class Scene:
    """Parsed scene: body sentences, question, and MCQ options.

    Produced by parse_scene(). Used by RoutingTurnstyle for routing
    and by solvers that need structured scene access.
    """
    body: list[str]
    question: str | None
    options: dict[str, str]


def parse_scene(text: str) -> Scene:
    """Sentences-first scene parsing — no "Options:" keyword dependency.

    Splits text into individual sentences, classifies each as:
      - option: line starting with (A)/(B)/... pattern
      - question: chunk ending with '?'
      - body: everything else

    Returns (body_sentences, question, options) where:
      body_sentences — list of fact/premise/instruction sentences
      question       — the query sentence, or None
      options        — {letter: text} for MCQ choices
    """
    text = re.sub(r'^Question:\s*', '', text.strip())

    body: list[str] = []
    question: str | None = None
    options: dict[str, str] = {}

    for line in text.split('\n'):
        line = line.strip()
        if not line or re.match(r'^Options:?\s*$', line, re.I):
            continue

        # Option line: "(A) text"
        m = re.match(r'^\(([A-R])\)\s*(.+)$', line)
        if m:
            options[m.group(1)] = m.group(2).strip()
            continue

        # Split line into sentences, keeping delimiter attached to each chunk
        for chunk in re.split(r'(?<=[.?!])\s+', line):
            chunk = chunk.strip()
            if not chunk:
                continue
            if chunk.endswith('?'):
                question = chunk
            else:
                body.append(chunk)

    return Scene(body=body, question=question, options=options)


def _default_split(body: str) -> list[str]:
    """Split text on periods, strip whitespace, filter empty.

    Available as SentenceIRSpec.split_fn override for tasks that need
    period-stripped sentences (no trailing '.').
    """
    return [s.strip() for s in body.split(".") if s.strip()]


def ir_solve(
    model, tokenizer, device,
    text: str,
    spec: IRSpec,
    diag: dict | None = None,
) -> str | None:
    """Extract structured IR via LLM → compute answer.

    1. Parse body/question/options from text (sentences-first)
    2. Format extraction prompt with body
    3. LLM generates JSON
    4. Parse JSON
    5. Compute answer via spec.compute(ir_data, question, options)
    """
    scene = parse_scene(text)
    body = " ".join(scene.body)
    question = scene.question
    options = scene.options

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
    aggregate: (records, options) → answer string or None.
    classify_fn: (sentence) → type string. None = use classify_token.
    split_fn: (body_str) → list[str]. Overrides parse_scene sentence splitting
        when set — useful for tasks needing period-stripped sentences.
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

    1. Parse body sentences/question/options from text (sentences-first)
    2. Optionally re-split body via spec.split_fn if set
    3. Segment body into typed chunks (LLM or classify_fn fallback)
    4. Append question segment
    5. Extract structured data from each segment via LLM
    6. Aggregate records and compute answer
    """
    from turnstyle.extract import generate_short

    scene = parse_scene(text)
    body_sentences = scene.body
    question = scene.question
    options = scene.options

    # Allow spec.split_fn to override parse_scene sentence splitting
    if spec.split_fn is not None:
        body_sentences = spec.split_fn(" ".join(body_sentences))

    # Segmentation: try LLM first, fall back to classify-each
    segments = None
    segment_method = "deterministic"

    if spec.segment_prompt is not None:
        segments = _segment_via_llm(model, tokenizer, device, " ".join(body_sentences), spec)
        if segments is not None:
            segment_method = "llm"
        else:
            segment_method = "llm_failed"

    if segments is None:
        segments = []
        for sentence in body_sentences:
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
        diag["body"] = " ".join(body_sentences)
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
        answer = spec.aggregate(records, options)
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


# ═══════════════════════════════════════════════════════════════════════
# Turnstyle subclasses for probe-routed deterministic solvers
# ═══════════════════════════════════════════════════════════════════════

# ── direction tables (navigate) ─────────────────────────────────────────────
_ABS_DIR: dict[str, tuple[int, int]] = {
    "forward":  ( 0,  1), "backward": ( 0, -1),
    "left":     (-1,  0), "right":    ( 1,  0),
    "north":    ( 0,  1), "south":    ( 0, -1),
    "west":     (-1,  0), "east":     ( 1,  0),
}
_FACING = [(0, 1), (1, 0), (0, -1), (-1, 0)]
_STEP_ABS_RE = re.compile(
    r"Take\s+(\d+)\s+steps?\s+(forward|backward|left|right|north|south|east|west)", re.I)
_STEP_FWD_RE = re.compile(r"Take\s+(\d+)\s+steps?\.?\s*$", re.I)
_STEP_REL_RE = re.compile(r"Take\s+(\d+)\s+steps?\s+(forward|backward)", re.I)
_TURN_RE = re.compile(r"Turn\s+(left|right|around)", re.I)


def _navigate_solve(body_sentences: list[str]) -> str | None:
    """Deterministic navigate solver. Returns 'Yes'/'No' or None."""
    body = " ".join(body_sentences)
    x, y = 0, 0
    if "Always face forward" in body:
        for n_str, direction in _STEP_ABS_RE.findall(body):
            dx, dy = _ABS_DIR[direction.lower()]
            x += int(n_str) * dx
            y += int(n_str) * dy
    else:
        facing = 0
        fwd_map = {"forward": 0, "backward": 2}
        for sent in body_sentences:
            sent = re.sub(r'[.!]\s*$', '', sent.strip())
            t = _TURN_RE.match(sent)
            if t:
                d = t.group(1).lower()
                if d == "left":    facing = (facing - 1) % 4
                elif d == "right": facing = (facing + 1) % 4
                else:              facing = (facing + 2) % 4
                continue
            sr = _STEP_REL_RE.match(sent)
            if sr:
                n = int(sr.group(1))
                offset = fwd_map.get(sr.group(2).lower(), 0)
                fx, fy = _FACING[(facing + offset) % 4]
                x += n * fx; y += n * fy
                continue
            sf = _STEP_FWD_RE.match(sent)
            if sf:
                n = int(sf.group(1))
                fx, fy = _FACING[facing]
                x += n * fx; y += n * fy

    return "Yes" if (x == 0 and y == 0) else "No"


# ── truth chain patterns (web_of_lies) ─────────────────────────────────────
_WOL_BASE_RE = re.compile(r"^(\w+)\s+(tells\s+the\s+truth|lies)\.?\s*$", re.I)
_WOL_SAYS_RE = re.compile(r"(\w+)\s+says\s+(\w+)\s+(tells\s+the\s+truth|lies)", re.I)
_WOL_QUERY_RE = re.compile(r"Does\s+(\w+)\s+tell\s+the\s+truth\?", re.I)


def _wol_solve(body_sentences: list[str], question: str | None) -> str | None:
    """Deterministic web_of_lies solver. Returns 'Yes'/'No' or None."""
    truth: dict[str, bool] = {}

    for sent in body_sentences:
        sent = sent.strip()
        if re.search(r'\bsays\b', sent, re.I):
            continue
        m = _WOL_BASE_RE.match(sent)
        if m:
            truth[m.group(1)] = m.group(2).strip().lower() == "tells the truth"

    full_body = " ".join(body_sentences)
    prev_size = -1
    while len(truth) != prev_size:
        prev_size = len(truth)
        for speaker, subject, claim in _WOL_SAYS_RE.findall(full_body):
            asserted = claim.strip().lower() == "tells the truth"
            if subject in truth and speaker not in truth:
                truth[speaker] = (asserted == truth[subject])
            elif speaker in truth and subject not in truth:
                truth[subject] = asserted if truth[speaker] else not asserted

    if not question:
        return None
    m = _WOL_QUERY_RE.search(question)
    if not m:
        return None
    queried = m.group(1)
    if queried not in truth:
        return None
    return "Yes" if truth[queried] else "No"


from turnstyle.core import SequenceLogitsProcessor, Turnstyle


class NavigateTurnstyle(Turnstyle):
    """Grounds spatial navigation in deterministic coordinate simulation.

    Parses movement instructions, simulates the path, and biases the
    model's answer toward 'Yes' (returns to origin) or 'No' (does not).

        t = NavigateTurnstyle(model, tokenizer, device)
        text, proof = t.generate("If you follow these instructions, do you return to the starting point?\\nAlways face forward. Take 3 steps right. Take 3 steps left.\\nOptions:\\n(A) Yes\\n(B) No")
    """

    probe_label = "spatial_navigation"
    examples = [
        "If you follow these instructions, do you return to the starting point? Always face forward. Take 3 steps right. Take 3 steps left.\nOptions:\n(A) Yes\n(B) No",
        "If you follow these instructions, do you return to the starting point? Always face forward. Take 5 steps forward. Take 5 steps backward.\nOptions:\n(A) Yes\n(B) No",
        "If you follow these instructions, do you return to the starting point? Always face forward. Take 1 step left. Take 1 step right.\nOptions:\n(A) Yes\n(B) No",
        "If you follow these instructions, do you return to the starting point? Always face forward. Take 7 steps north. Take 7 steps south.\nOptions:\n(A) Yes\n(B) No",
        "If you follow these instructions, do you return to the starting point? Always face forward. Take 2 steps east. Take 3 steps west.\nOptions:\n(A) Yes\n(B) No",
        "If you follow these instructions, do you return to the starting point? Always face forward. Take 10 steps forward. Take 5 steps backward. Take 5 steps backward.\nOptions:\n(A) Yes\n(B) No",
        "If you follow these instructions, do you return to the starting point? Take 4 steps. Turn around. Take 4 steps.\nOptions:\n(A) Yes\n(B) No",
        "If you follow these instructions, do you return to the starting point? Take 3 steps. Turn left. Take 3 steps. Turn left. Take 3 steps. Turn left. Take 3 steps.\nOptions:\n(A) Yes\n(B) No",
        "If you follow these instructions, do you return to the starting point? Take 2 steps. Turn right. Take 2 steps. Turn right. Take 2 steps. Turn right. Take 2 steps.\nOptions:\n(A) Yes\n(B) No",
        "If you follow these instructions, do you return to the starting point? Always face forward. Take 3 steps left. Take 1 step right.\nOptions:\n(A) Yes\n(B) No",
        "If you follow these instructions, do you return to the starting point? Always face forward. Take 9 steps forward. Take 9 steps backward. Take 1 step left.\nOptions:\n(A) Yes\n(B) No",
        "If you follow these instructions, do you return to the starting point? Take 5 steps. Turn right. Take 5 steps. Turn right. Take 5 steps. Turn right. Take 5 steps.\nOptions:\n(A) Yes\n(B) No",
        "If you follow these instructions, do you return to the starting point? Always face forward. Take 8 steps right. Take 8 steps left. Take 3 steps forward.\nOptions:\n(A) Yes\n(B) No",
        "If you follow these instructions, do you return to the starting point? Take 1 step. Turn around. Take 1 step.\nOptions:\n(A) Yes\n(B) No",
        "If you follow these instructions, do you return to the starting point? Always face forward. Take 6 steps north. Take 3 steps south. Take 3 steps south.\nOptions:\n(A) Yes\n(B) No",
        "If you follow these instructions, do you return to the starting point? Always face forward. Take 4 steps east. Take 4 steps west. Take 2 steps north. Take 2 steps south.\nOptions:\n(A) Yes\n(B) No",
        "If you follow these instructions, do you return to the starting point? Take 7 steps. Turn left. Take 3 steps.\nOptions:\n(A) Yes\n(B) No",
        "If you follow these instructions, do you return to the starting point? Always face forward. Take 2 steps forward. Take 2 steps forward. Take 4 steps backward.\nOptions:\n(A) Yes\n(B) No",
        "If you follow these instructions, do you return to the starting point? Always face forward. Take 1 step right. Take 1 step forward. Take 1 step left. Take 1 step backward.\nOptions:\n(A) Yes\n(B) No",
        "If you follow these instructions, do you return to the starting point? Take 10 steps. Turn right. Take 10 steps. Turn right. Take 10 steps. Turn right. Take 10 steps.\nOptions:\n(A) Yes\n(B) No",
        "If you follow these instructions, do you return to the starting point? Always face forward. Take 5 steps left. Take 3 steps right. Take 2 steps right.\nOptions:\n(A) Yes\n(B) No",
        "If you follow these instructions, do you return to the starting point? Take 6 steps. Turn around. Take 3 steps.\nOptions:\n(A) Yes\n(B) No",
        "If you follow these instructions, do you return to the starting point? Always face forward. Take 12 steps north. Take 12 steps south.\nOptions:\n(A) Yes\n(B) No",
        "If you follow these instructions, do you return to the starting point? Always face forward. Take 3 steps west. Take 3 steps east. Take 5 steps north. Take 5 steps south.\nOptions:\n(A) Yes\n(B) No",
        "If you follow these instructions, do you return to the starting point? Take 4 steps. Turn left. Take 4 steps. Turn left. Take 4 steps. Turn left. Take 4 steps.\nOptions:\n(A) Yes\n(B) No",
        "If you follow these instructions, do you return to the starting point? Always face forward. Take 2 steps right. Take 5 steps left.\nOptions:\n(A) Yes\n(B) No",
        "If you follow these instructions, do you return to the starting point? Take 8 steps. Turn right. Take 8 steps. Turn around. Take 8 steps. Turn left. Take 8 steps.\nOptions:\n(A) Yes\n(B) No",
        "If you follow these instructions, do you return to the starting point? Always face forward. Take 1 step north. Take 1 step east. Take 1 step south. Take 1 step west.\nOptions:\n(A) Yes\n(B) No",
        "If you follow these instructions, do you return to the starting point? Always face forward. Take 15 steps forward. Take 7 steps backward. Take 8 steps backward.\nOptions:\n(A) Yes\n(B) No",
        "If you follow these instructions, do you return to the starting point? Take 3 steps. Turn right. Take 3 steps. Turn right. Take 6 steps. Turn right. Take 6 steps.\nOptions:\n(A) Yes\n(B) No",
    ]

    def parse(self, prompt: str):
        """Deterministic solve: parse scene, simulate navigation, match answer."""
        scene = parse_scene(prompt)
        body_sentences, question, options = scene.body, scene.question, scene.options
        answer = _navigate_solve(body_sentences)
        if answer is None:
            return None
        for letter, val in options.items():
            if val.strip().lower() == answer.lower():
                return (f"({letter})", answer)
        # If no options found, return the raw answer
        return (answer, answer)

    def make_processor(self, parsed, max_new_tokens: int):
        option_letter, answer_text = parsed
        answer_ids = self.tokenizer.encode(option_letter, add_special_tokens=False)
        return SequenceLogitsProcessor(
            self.tokenizer, answer_ids, expression="navigate",
            answer_str=option_letter, bias_strength=self.bias_strength,
            max_new_tokens=max_new_tokens)


class WebOfLiesTurnstyle(Turnstyle):
    """Grounds truth-chain reasoning in deterministic propagation.

    Parses base truth facts and derived statements, propagates truth
    values, and biases the model's answer toward 'Yes' or 'No'.

        t = WebOfLiesTurnstyle(model, tokenizer, device)
        text, proof = t.generate("Maeva tells the truth. Phoebe says Maeva lies. Does Phoebe tell the truth?\\nOptions:\\n(A) Yes\\n(B) No")
    """

    probe_label = "truth_chain"
    examples = [
        "Maeva tells the truth. Phoebe says Maeva lies. Does Phoebe tell the truth?\nOptions:\n(A) Yes\n(B) No",
        "Alice lies. Bob says Alice tells the truth. Does Bob tell the truth?\nOptions:\n(A) Yes\n(B) No",
        "Charlie tells the truth. Dana says Charlie lies. Eve says Dana tells the truth. Does Eve tell the truth?\nOptions:\n(A) Yes\n(B) No",
        "Frank lies. Grace says Frank lies. Does Grace tell the truth?\nOptions:\n(A) Yes\n(B) No",
        "Henry tells the truth. Iris says Henry tells the truth. Jack says Iris lies. Does Jack tell the truth?\nOptions:\n(A) Yes\n(B) No",
        "Kate lies. Leo says Kate tells the truth. Maya says Leo tells the truth. Does Maya tell the truth?\nOptions:\n(A) Yes\n(B) No",
        "Nora tells the truth. Oliver says Nora lies. Pete says Oliver lies. Does Pete tell the truth?\nOptions:\n(A) Yes\n(B) No",
        "Quinn lies. Rachel says Quinn lies. Sam says Rachel tells the truth. Does Sam tell the truth?\nOptions:\n(A) Yes\n(B) No",
        "Tina tells the truth. Uma says Tina tells the truth. Does Uma tell the truth?\nOptions:\n(A) Yes\n(B) No",
        "Victor lies. Wendy says Victor tells the truth. Xander says Wendy lies. Does Xander tell the truth?\nOptions:\n(A) Yes\n(B) No",
        "Yolanda tells the truth. Zach says Yolanda lies. Does Zach tell the truth?\nOptions:\n(A) Yes\n(B) No",
        "Alex lies. Blake says Alex lies. Does Blake tell the truth?\nOptions:\n(A) Yes\n(B) No",
        "Cameron tells the truth. Dylan says Cameron tells the truth. Erin says Dylan lies. Does Erin tell the truth?\nOptions:\n(A) Yes\n(B) No",
        "Finn lies. Georgia says Finn tells the truth. Hana says Georgia tells the truth. Does Hana tell the truth?\nOptions:\n(A) Yes\n(B) No",
        "Ivan tells the truth. Julia says Ivan lies. Karl says Julia tells the truth. Does Karl tell the truth?\nOptions:\n(A) Yes\n(B) No",
        "Lisa lies. Marcus says Lisa lies. Nat says Marcus lies. Does Nat tell the truth?\nOptions:\n(A) Yes\n(B) No",
        "Oscar tells the truth. Penny says Oscar tells the truth. Quinn says Penny lies. Does Quinn tell the truth?\nOptions:\n(A) Yes\n(B) No",
        "Rosa lies. Sam says Rosa tells the truth. Tara says Sam tells the truth. Does Tara tell the truth?\nOptions:\n(A) Yes\n(B) No",
        "Ulrich tells the truth. Vera says Ulrich lies. Will says Vera lies. Does Will tell the truth?\nOptions:\n(A) Yes\n(B) No",
        "Xena lies. Yoshi says Xena lies. Zoe says Yoshi tells the truth. Does Zoe tell the truth?\nOptions:\n(A) Yes\n(B) No",
        "Abel tells the truth. Beth says Abel tells the truth. Carl says Beth tells the truth. Does Carl tell the truth?\nOptions:\n(A) Yes\n(B) No",
        "Diana lies. Ethan says Diana lies. Faye says Ethan lies. Does Faye tell the truth?\nOptions:\n(A) Yes\n(B) No",
        "Gary tells the truth. Hana says Gary tells the truth. Ida says Hana tells the truth. Does Ida tell the truth?\nOptions:\n(A) Yes\n(B) No",
        "Jake lies. Kim says Jake tells the truth. Leo says Kim lies. Does Leo tell the truth?\nOptions:\n(A) Yes\n(B) No",
        "Mia tells the truth. Nick says Mia lies. Olga says Nick lies. Does Olga tell the truth?\nOptions:\n(A) Yes\n(B) No",
        "Paul lies. Queenie says Paul lies. Rob says Queenie tells the truth. Does Rob tell the truth?\nOptions:\n(A) Yes\n(B) No",
        "Sara tells the truth. Todd says Sara tells the truth. Uma says Todd lies. Does Uma tell the truth?\nOptions:\n(A) Yes\n(B) No",
        "Vlad lies. Wren says Vlad tells the truth. Xavier says Wren tells the truth. Does Xavier tell the truth?\nOptions:\n(A) Yes\n(B) No",
        "Yara tells the truth. Zeus says Yara lies. Does Zeus tell the truth?\nOptions:\n(A) Yes\n(B) No",
        "Adam lies. Barb says Adam lies. Carol says Barb tells the truth. Does Carol tell the truth?\nOptions:\n(A) Yes\n(B) No",
    ]

    def parse(self, prompt: str):
        """Deterministic solve: parse scene, propagate truth values, match answer."""
        scene = parse_scene(prompt)
        body_sentences, question, options = scene.body, scene.question, scene.options
        answer = _wol_solve(body_sentences, question)
        if answer is None:
            return None
        for letter, val in options.items():
            if val.strip().lower() == answer.lower():
                return (f"({letter})", answer)
        return (answer, answer)

    def make_processor(self, parsed, max_new_tokens: int):
        option_letter, answer_text = parsed
        answer_ids = self.tokenizer.encode(option_letter, add_special_tokens=False)
        return SequenceLogitsProcessor(
            self.tokenizer, answer_ids, expression="web_of_lies",
            answer_str=option_letter, bias_strength=self.bias_strength,
            max_new_tokens=max_new_tokens)
