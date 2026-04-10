"""Base conversion turnstyle — grounds number base conversions in exact computation.

The LLM extracts three fields (number, from-base, to-base) from free-form text;
exact arithmetic computes the result.  No keyword patterns required.

    t = BaseConversionTurnstyle(model, tokenizer, device)
    text, proof = t.generate("Express the binary value 1010 as a decimal number")
    text, proof = t.generate("Hex ff to decimal")

Optionally train a hidden-state probe for faster / more accurate from-base
inference (replaces one classify_token call with a single forward pass):

    t.build_from_base_probe()
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from turnstyle.arithmetic import ArithmeticLogitsProcessor
from turnstyle.core import Turnstyle
from turnstyle.extract import ExtractionSpec, FieldSpec

if TYPE_CHECKING:
    from turnstyle.probe import TurnstyleProbe

# ── base name lookup ──────────────────────────────────────────────────

_BASE_NAMES: dict[str, int] = {
    'binary': 2, 'bin': 2, 'base 2': 2, 'base2': 2,
    'octal': 8, 'oct': 8, 'base 8': 8, 'base8': 8,
    'decimal': 10, 'dec': 10, 'base 10': 10, 'base10': 10,
    'hex': 16, 'hexadecimal': 16, 'base 16': 16, 'base16': 16,
}

_BASE_OPTIONS = ["binary", "octal", "decimal", "hexadecimal"]



def _format_result(value: int, base: int) -> str:
    if base == 2:
        return bin(value)[2:]
    elif base == 8:
        return oct(value)[2:]
    elif base == 16:
        return hex(value)[2:]
    else:
        return str(value)


def _parse_number(text: str, base: int | None = None) -> tuple[int, int] | None:
    """Parse a number with optional prefix. Returns (value, detected_base)."""
    text = text.strip()
    if text.startswith(('0x', '0X')):
        try:
            return int(text, 16), 16
        except ValueError:
            return None
    if text.startswith(('0b', '0B')):
        try:
            return int(text, 2), 2
        except ValueError:
            return None
    if text.startswith(('0o', '0O')):
        try:
            return int(text, 8), 8
        except ValueError:
            return None
    if base and base != 10:
        try:
            return int(text, base), base
        except ValueError:
            return None
    try:
        return int(text), 10
    except ValueError:
        return None


# ── LLM extraction spec ───────────────────────────────────────────────

def _assemble_base_conversion(fields: dict[str, Any]):
    number_text = re.sub(r'\s+', '', fields["number_text"])  # strip whitespace
    from_base = _BASE_NAMES.get(fields["from_base"])
    to_base = _BASE_NAMES.get(fields["to_base"])
    if not from_base or not to_base or from_base == to_base:
        return None
    parsed = _parse_number(number_text, from_base)
    if not parsed:
        return None
    value, _ = parsed
    result_str = _format_result(value, to_base)
    expr = f"{number_text}(base{from_base})\u2192base{to_base}"
    return value, from_base, to_base, result_str, expr


BASE_CONVERSION_EXTRACTION_SPEC = ExtractionSpec(
    fields=[
        FieldSpec(
            name="number_text",
            prompt_template=(
                "Number conversion problem: {input}\n"
                "What is the number to be converted? "
                "Reply with only the number itself, no explanation."
            ),
            options=None,
            max_tokens=20,
        ),
        FieldSpec(
            name="from_base",
            prompt_template=(
                "Number conversion problem: {input}\n"
                "What base (number system) is the input number written in?"
            ),
            options=_BASE_OPTIONS,
        ),
        FieldSpec(
            name="to_base",
            prompt_template=(
                "Number conversion problem: {input}\n"
                "What base (number system) should the answer be expressed in?"
            ),
            options=_BASE_OPTIONS,
        ),
    ],
    assemble=_assemble_base_conversion,
)


# ── from-base probe training data ─────────────────────────────────────
# Prompts where from-base is unambiguous (explicit context or prefix).
# Probe trained on last-token hidden state at layer L; generalises to
# implicit cases (e.g. "What is 1010 in decimal?" → infers binary).

_FROM_BASE_TRAINING: dict[str, list[str]] = {
    "binary": [
        "Convert 1010 from binary to decimal",
        "Convert 11111111 from binary to decimal",
        "Convert 0b1010 to decimal",
        "What is 0b11111111 in decimal?",
        "What is 0b101 in decimal?",
        "Express the binary number 1100 in decimal",
        "The binary representation 10101010 equals what in decimal?",
        "Convert the binary value 110011 to decimal",
        "0b1111 to decimal",
        "What decimal value does binary 11001100 represent?",
        "Convert 1010 from binary to hex",
        "Convert 11110000 from binary to octal",
        "0b11001 to hex",
        "binary 1000 to decimal",
    ],
    "octal": [
        "Convert 17 from octal to decimal",
        "Convert 0o17 to decimal",
        "What is 0o52 in decimal?",
        "What is 0o755 in decimal?",
        "Express the octal number 77 in decimal",
        "The octal value 123 is what in decimal?",
        "Convert 0o377 to binary",
        "What decimal number equals octal 644?",
        "Convert 17 from octal to hex",
        "0o100 to decimal",
        "Convert the octal 755 to binary",
        "What is 0o17 in binary?",
        "0o644 to decimal",
        "octal 52 to decimal",
    ],
    "decimal": [
        "Convert 255 from decimal to binary",
        "Convert 42 from decimal to hex",
        "What is 255 in binary?",
        "What is 10 in binary?",
        "Express decimal 255 as hexadecimal",
        "The decimal number 100 in binary is?",
        "Convert decimal 65 to hex",
        "What is the binary of 8?",
        "Convert 255 to binary",
        "Convert 16 to hex",
        "What is 42 in octal?",
        "Express 127 in hexadecimal",
        "Convert the decimal number 256 to octal",
        "255 to hex",
    ],
    "hexadecimal": [
        "Convert ff from hexadecimal to decimal",
        "Convert 0xff to decimal",
        "What is 0xFF in decimal?",
        "What is 0xAB in decimal?",
        "Express the hex value ff as decimal",
        "The hexadecimal 1A equals what in decimal?",
        "Convert hex ff to binary",
        "0x1F to decimal",
        "What decimal value is hex 2A?",
        "Convert 1A from hexadecimal to octal",
        "0xDEAD to decimal",
        "Convert the hex number FF to binary",
        "hex 100 to decimal",
        "What is 0xF0 in binary?",
    ],
}


# ── processor with hex support ────────────────────────────────────────

class BaseConversionProcessor(ArithmeticLogitsProcessor):
    """Extends digit biasing to include hex characters a-f."""

    def __init__(self, tokenizer, answer_digits: list[int], expression: str,
                 answer_value: int, bias_strength: float = 15.0,
                 max_new_tokens: int = 50):
        super().__init__(tokenizer, answer_digits, expression, answer_value,
                         bias_strength, max_new_tokens)
        for i, ch in enumerate('abcdef'):
            val = 10 + i
            ids = tokenizer.encode(ch, add_special_tokens=False)
            if ids:
                self.digit_to_token[val] = ids[0]
                self.token_to_digit[ids[0]] = val


class BaseConversionTurnstyle(Turnstyle):
    """Grounds number base conversions in exact computation.

    The LLM extracts number, from-base, and to-base from free-form text;
    Python computes the exact result and biases generation.

        t = BaseConversionTurnstyle(model, tokenizer, device)
        text, proof = t.generate("Express the binary value 1010 as decimal")

    Optionally train a hidden-state probe for faster from-base inference:

        t.build_from_base_probe()
        text, proof = t.generate("What is 1010 in decimal?")
        # probe infers from_base=2 → result is 10
    """

    probe_label = "base_conversion"
    extraction_spec = BASE_CONVERSION_EXTRACTION_SPEC
    _from_base_probe: "TurnstyleProbe | None" = None
    _from_base_probe_layer: int = 2

    def parse(self, prompt: str):
        """Use probe + LLM for from-base when probe is trained; else defer to extraction_spec."""
        if self._from_base_probe is None:
            return None  # extraction_spec handles everything

        from turnstyle.extract import classify_token, generate_short

        from_base = self._probe_from_base(prompt)
        if from_base is None:
            return None

        to_idx, _ = classify_token(
            self.model, self.tokenizer, self.device,
            f"Number conversion problem: {prompt!r}\n"
            "What base should the answer be expressed in?",
            _BASE_OPTIONS,
        )
        to_base = _BASE_NAMES[_BASE_OPTIONS[to_idx]]

        number_text, _ = generate_short(
            self.model, self.tokenizer, self.device,
            f"Number conversion problem: {prompt!r}\n"
            "What is the number to be converted? Reply with only the number.",
            max_tokens=20,
        )
        number_text = re.sub(r'\s+', '', number_text)

        if not to_base or from_base == to_base or not number_text:
            return None

        parsed = _parse_number(number_text, from_base)
        if not parsed:
            return None
        value, _ = parsed
        result_str = _format_result(value, to_base)
        expr = f"{number_text}(base{from_base})\u2192base{to_base}"
        return value, from_base, to_base, result_str, expr

    def build_from_base_probe(self, layer: int = 2) -> None:
        """Train a from-base classifier and install it on this instance.

        Hooks the last-token hidden state at ``layer``, runs one forward pass
        per training prompt, fits a 4-class SGDClassifier, and stores a baked
        ``TurnstyleProbe`` (no scaler needed at inference time).
        """
        import numpy as np
        import torch
        from sklearn.linear_model import SGDClassifier
        from sklearn.preprocessing import StandardScaler

        from turnstyle.probe import TurnstyleProbe

        X: list[np.ndarray] = []
        y: list[str] = []
        captured: dict[str, torch.Tensor] = {}

        def _hook(module, input, output):
            captured["h"] = output[0][:, -1, :].detach().cpu()

        target_layer = self.model.model.layers[layer]
        handle = target_layer.register_forward_hook(_hook)
        try:
            for label, prompts in _FROM_BASE_TRAINING.items():
                for prompt in prompts:
                    enc = self.tokenizer(prompt, return_tensors="pt").to(self.device)
                    with torch.no_grad():
                        self.model(**enc)
                    X.append(captured["h"][0].numpy())
                    y.append(label)
        finally:
            handle.remove()

        X_arr = np.array(X)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_arr)

        clf = SGDClassifier(loss="log_loss", max_iter=2000, random_state=42)
        clf.fit(X_scaled, y)

        scale = torch.tensor(scaler.scale_, dtype=torch.float32)
        mean = torch.tensor(scaler.mean_, dtype=torch.float32)
        W = torch.tensor(clf.coef_, dtype=torch.float32) / scale
        b = torch.tensor(clf.intercept_, dtype=torch.float32) - (W @ mean)

        self._from_base_probe = TurnstyleProbe(
            weights=W, bias=b, labels=clf.classes_.tolist()
        )
        self._from_base_probe_layer = layer

    def _probe_from_base(self, prompt: str) -> int | None:
        """One forward pass → last-token hidden state → probe → base int."""
        import torch

        captured: dict[str, torch.Tensor] = {}

        def _hook(module, input, output):
            captured["h"] = output[0][:, -1, :].detach()

        target_layer = self.model.model.layers[self._from_base_probe_layer]
        handle = target_layer.register_forward_hook(_hook)
        try:
            enc = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            with torch.no_grad():
                self.model(**enc)
        finally:
            handle.remove()

        label, _conf = self._from_base_probe.predict_best(captured["h"][0])
        return _BASE_NAMES.get(label)

    def make_processor(self, parsed, max_new_tokens: int):
        decimal_value, from_base, to_base, result_str, expr = parsed

        answer_digits = []
        for ch in result_str:
            if ch.isdigit():
                answer_digits.append(int(ch))
            elif ch.lower() in 'abcdef':
                answer_digits.append(ord(ch.lower()) - ord('a') + 10)

        if to_base == 16:
            proc = BaseConversionProcessor(
                self.tokenizer, answer_digits, expr, decimal_value,
                self.bias_strength, max_new_tokens)
            proc.proof.answer_charset = "0123456789abcdef"
        else:
            proc = ArithmeticLogitsProcessor(
                self.tokenizer, answer_digits, expr, decimal_value,
                self.bias_strength, max_new_tokens)

        proc.proof.answer_str = result_str
        return proc
