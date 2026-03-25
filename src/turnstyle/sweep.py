"""Probe sweep — find the best layer and train a routing probe.

    from turnstyle.sweep import probe_sweep
    result = probe_sweep("HuggingFaceTB/SmolLM2-1.7B-Instruct")
    result.probe.save("routing_probe.pt")
    print(result.summary())

Supports two backends:

- **torch** (default): Uses ``register_forward_hook`` to capture hidden states.
  Works with any HuggingFace ``PreTrainedModel``.

- **mlx**: Iterates ``model.model.layers`` directly (no hooks needed).
  Works with standard ``mlx-lm`` architectures (LLaMA, Mistral, SmolLM, Qwen,
  Gemma, Phi). 2-5x faster on Apple Silicon for prefill. Use
  ``backend='torch'`` as fallback for non-standard architectures.
"""

from __future__ import annotations

import platform
import random
import sys
import warnings
from dataclasses import dataclass

import numpy as np
import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from turnstyle.probe import TurnstyleProbe

# ── default prompt labels ────────────────────────────────────────────

ALL_LABELS = [
    "arithmetic",
    "date",
    "unit",
    "currency",
    "percentage",
    "counting",
    "base_conversion",
    "sandbox",
    "number_theory",
]

# ── prompt templates ─────────────────────────────────────────────────
# Each label maps to a list of template strings with {placeholders}.
# Templates include both regex-matchable and non-regex phrasings.

_TEMPLATES: dict[str, list[str]] = {
    "arithmetic": [
        "What is {a} + {b}?",
        "What is {a} - {b}?",
        "What is {a} * {b}?",
        "Calculate {a} plus {b}",
        "How much is {a} times {b}?",
        "Can you compute {a} minus {b}?",
        "Add {a} and {b} together",
        "Subtract {b} from {a}",
        "Multiply {a} by {b}",
        "What do you get when you add {a} to {b}?",
        "I need the sum of {a} and {b}",
        "Tell me {a} multiplied by {b}",
        "Find the difference between {a} and {b}",
        "What's the product of {a} and {b}?",
    ],
    "date": [
        "How many days between {month1} {day1} and {month2} {day2}?",
        "How many days from {month1} {day1} to {month2} {day2}?",
        "How many weeks between {month1} {day1} and {month2} {day2}?",
        "How many days until Christmas?",
        "How many days until New Year?",
        "What's the number of days from {month1} {day1} to {month2} {day2}?",
        "Count the days between {month1} {day1} and {month2} {day2}",
        "How far apart are {month1} {day1} and {month2} {day2}?",
        "Days remaining until {month2} {day2}?",
        "How long from {month1} {day1} to {month2} {day2} in days?",
    ],
    "unit": [
        "How many km is {n} miles?",
        "Convert {n} fahrenheit to celsius",
        "What is {n} kg in pounds?",
        "{n} lbs to kg",
        "How many meters in {n} feet?",
        "What's {n} inches in centimeters?",
        "Convert {n} gallons to liters",
        "I need {n} miles converted to kilometers",
        "Express {n} ounces in grams",
        "How much is {n} cups in milliliters?",
        "Change {n} yards to meters",
        "What does {n} kg weigh in pounds?",
    ],
    "currency": [
        "How much is {n} USD in EUR?",
        "Convert {n} GBP to JPY",
        "{n} dollars to euros",
        "What is {n} EUR in USD?",
        "How many yen is {n} dollars?",
        "Convert {n} canadian dollars to USD",
        "What's {n} pounds in dollars?",
        "I need to convert {n} USD to GBP",
        "Exchange {n} euros for dollars",
        "How much would {n} USD be in japanese yen?",
    ],
    "percentage": [
        "What is {p}% of {n}?",
        "What percentage is {a} of {b}?",
        "{p}% tip on ${n}",
        "{p}% off {n}",
        "Calculate {p} percent of {n}",
        "How much is {p}% of {n}?",
        "Find {p}% of {n}",
        "If I take {p}% off {n}, what do I get?",
        "What's {p} percent of {n} dollars?",
        "I need to figure out {p}% of {n}",
    ],
    "counting": [
        "How many vowels in '{word}'?",
        "How many r's in '{word}'?",
        "How many words in '{phrase}'?",
        "How many letters in '{word}'?",
        "Count the vowels in '{word}'",
        "How many e's in '{word}'?",
        "What's the letter count in '{word}'?",
        "How many characters in '{phrase}'?",
        "Count the consonants in '{word}'",
        "Tell me how many s's are in '{word}'",
    ],
    "base_conversion": [
        "What is {n} in binary?",
        "What is {n} in hex?",
        "Convert {n} to octal",
        "What is 0x{h} in decimal?",
        "Convert {n} from decimal to binary",
        "{n} to hexadecimal",
        "Express {n} in base 2",
        "What's the binary representation of {n}?",
        "Show me {n} in hex format",
        "Turn {n} into binary",
    ],
    "sandbox": [
        "What does `sum(range({n}))` return?",
        "What is the output of `len('{word}')`?",
        "Evaluate: `{n} ** {e}`",
        "What does `sorted([{a}, {b}, {c}])` return?",
        "Run: `max([{a}, {b}, {c}])`",
        "What is the result of `{n} % {m}`?",
        "Execute: `list(range({a}, {b}))`",
        "What does `'{word}'.upper()` return?",
        "What does `{a} * {b} + {c}` evaluate to in Python?",
        "Compute `pow({a}, {b}, {m})` for me",
    ],
    "number_theory": [
        "What is the GCD of {a} and {b}?",
        "Find the greatest common divisor of {a} and {b}",
        "What is the LCM of {a} and {b}?",
        "Least common multiple of {a} and {b}",
        "Simplify {a}/{b}",
        "Reduce {a}/{b} to lowest terms",
        "What is gcd({a}, {b})?",
        "Find the lcm of {a} and {b}",
        "What is {a}/{b} in simplest form?",
        "Can you reduce the fraction {a}/{b}?",
    ],
}

_NEGATIVE_TEMPLATES = [
    "What is the capital of {country}?",
    "Tell me about {topic}",
    "Who wrote {book}?",
    "What year did {event} happen?",
    "Explain {concept} in simple terms",
    "What's the weather like today?",
    "Can you help me write an email?",
    "What is photosynthesis?",
    "Tell me a joke about {topic}",
    "Summarize the plot of {book}",
    "What are the symptoms of {condition}?",
    "How do you say hello in {language}?",
    "What's the tallest mountain in the world?",
    "Who is the current president of {country}?",
    "Explain quantum computing",
]

# ── fill values ──────────────────────────────────────────────────────

_MONTHS = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]
_WORDS = [
    "strawberry", "mississippi", "encyclopedia", "communication",
    "extraordinary", "hippopotamus", "beautiful", "concatenation",
    "supercalifragilistic", "antidisestablishmentarianism",
    "onomatopoeia", "serendipity", "abracadabra", "ratatouille",
]
_PHRASES = [
    "the quick brown fox", "hello world", "to be or not to be",
    "all that glitters is not gold", "the rain in spain",
]
_COUNTRIES = ["France", "Japan", "Brazil", "Canada", "Australia"]
_TOPICS = ["space", "history", "cooking", "music", "philosophy"]
_BOOKS = ["1984", "Hamlet", "The Odyssey", "Dune", "Pride and Prejudice"]
_EVENTS = [
    "the moon landing", "World War II end", "the Berlin Wall fall",
]
_CONCEPTS = ["gravity", "evolution", "democracy", "relativity"]
_CONDITIONS = ["the flu", "diabetes", "insomnia"]
_LANGUAGES = ["French", "Japanese", "Spanish", "German", "Mandarin"]


def _fill_template(template: str, rng: random.Random) -> str:
    """Fill a template with random values."""
    replacements = {
        "{a}": str(rng.randint(1, 999)),
        "{b}": str(rng.randint(1, 999)),
        "{c}": str(rng.randint(1, 999)),
        "{n}": str(rng.randint(1, 9999)),
        "{m}": str(rng.randint(2, 50)),
        "{e}": str(rng.randint(2, 8)),
        "{p}": str(rng.randint(1, 99)),
        "{h}": hex(rng.randint(16, 4095))[2:],
        "{day1}": str(rng.randint(1, 28)),
        "{day2}": str(rng.randint(1, 28)),
        "{month1}": rng.choice(_MONTHS),
        "{month2}": rng.choice(_MONTHS),
        "{word}": rng.choice(_WORDS),
        "{phrase}": rng.choice(_PHRASES),
        "{country}": rng.choice(_COUNTRIES),
        "{topic}": rng.choice(_TOPICS),
        "{book}": rng.choice(_BOOKS),
        "{event}": rng.choice(_EVENTS),
        "{concept}": rng.choice(_CONCEPTS),
        "{condition}": rng.choice(_CONDITIONS),
        "{language}": rng.choice(_LANGUAGES),
    }
    result = template
    for key, val in replacements.items():
        result = result.replace(key, val)
    return result


# ── public API ───────────────────────────────────────────────────────


def generate_prompts(
    labels: list[str] | None = None,
    per_label: int = 50,
    include_negative: bool = False,
    seed: int = 42,
) -> dict[str, list[str]]:
    """Generate training prompts for each turnstyle type.

    labels: subset of ALL_LABELS to generate for (default: all 8)
    per_label: number of prompts per label
    include_negative: if True, add a "_none" class with general prompts
    seed: random seed for reproducibility

    Returns {label: [prompt_strings]}.
    """
    rng = random.Random(seed)
    labels = labels or list(ALL_LABELS)
    prompts: dict[str, list[str]] = {}

    for label in labels:
        templates = _TEMPLATES.get(label)
        if templates is None:
            raise ValueError(
                f"Unknown label {label!r}. "
                f"Known labels: {', '.join(ALL_LABELS)}"
            )
        generated = []
        for i in range(per_label):
            tmpl = templates[i % len(templates)]
            generated.append(_fill_template(tmpl, rng))
        prompts[label] = generated

    if include_negative:
        neg = []
        for i in range(per_label):
            tmpl = _NEGATIVE_TEMPLATES[i % len(_NEGATIVE_TEMPLATES)]
            neg.append(_fill_template(tmpl, rng))
        prompts["_none"] = neg

    return prompts


# ── model layer detection ────────────────────────────────────────────


def _get_inner_model(model):
    """Resolve the inner transformer model that has embed_tokens + layers.

    Handles several layout patterns:
    - Standard: model.model (LLaMA, Mistral, SmolLM, Gemma text-only)
    - Multimodal: model.language_model.model (Gemma 3 4B+)
    - Direct: model itself has .layers (some MLX models)
    - GPT-2: model.transformer

    Returns (inner, layout) where layout is 'standard', 'multimodal',
    'direct', or 'gpt2'.
    """
    # Standard: model.model.layers (LLaMA, Mistral, Qwen, SmolLM, etc.)
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model, "standard"
    # Multimodal: model.language_model.model.layers (Gemma 3 4B+)
    if hasattr(model, "language_model") and hasattr(model.language_model, "model"):
        inner = model.language_model.model
        if hasattr(inner, "layers"):
            return inner, "multimodal"
    # GPT-2, GPT-Neo: model.transformer.h
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer, "gpt2"
    # Direct: model.layers (some MLX models)
    if hasattr(model, "layers"):
        return model, "direct"

    raise ValueError(
        f"Cannot detect layers for {type(model).__name__}. "
        "Pass layer_range=(start, end) to specify manually, "
        "or ensure model has .model.layers, .language_model.model.layers, "
        "or .transformer.h"
    )


def _get_layers(model) -> list:
    """Get the list of transformer layers from the model."""
    inner, layout = _get_inner_model(model)
    if layout == "gpt2":
        return list(inner.h)
    return list(inner.layers)


# ── hidden state extraction ──────────────────────────────────────────


def _extract_all_hidden_states(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompts: list[str],
    labels: list[int],
    device: str,
    pool: str,
    layer_indices: list[int],
) -> dict[int, torch.Tensor]:
    """Extract hidden states from all target layers in one forward pass per prompt.

    Returns {layer_idx: (num_prompts, hidden_dim) tensor}.
    """
    layers = _get_layers(model)

    # Storage: accumulate per-layer hidden states
    all_hidden: dict[int, list[torch.Tensor]] = {i: [] for i in layer_indices}

    for prompt in prompts:
        # Capture from all target layers simultaneously
        captured: dict[int, torch.Tensor] = {}

        handles = []
        for idx in layer_indices:
            def make_hook(layer_idx: int):
                def hook_fn(module, input, output):
                    h = output[0] if isinstance(output, tuple) else output
                    captured[layer_idx] = h.detach()
                return hook_fn
            handles.append(layers[idx].register_forward_hook(make_hook(idx)))

        try:
            # Tokenize with chat template
            messages = [{"role": "user", "content": prompt}]
            try:
                text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                )
            except Exception:
                text = prompt
            inputs = tokenizer(text, return_tensors="pt").to(device)

            with torch.no_grad():
                model(**inputs)

            # Pool and store
            for idx in layer_indices:
                h = captured[idx]  # (1, seq_len, hidden_dim)
                if pool == "last":
                    pooled = h[0, -1]
                else:  # mean
                    pooled = h[0].mean(dim=0)
                all_hidden[idx].append(pooled.cpu())
        finally:
            for handle in handles:
                handle.remove()

    # Stack into tensors → numpy for unified interface
    return {idx: torch.stack(vecs).numpy() for idx, vecs in all_hidden.items()}


# ── MLX hidden state extraction ─────────────────────────────────


def _extract_all_hidden_states_mlx(
    model,
    tokenizer,
    prompts: list[str],
    pool: str,
    layer_indices: list[int],
) -> dict[int, np.ndarray]:
    """Extract hidden states via direct layer iteration (MLX models).

    No hooks — iterates ``model.model.layers`` directly, same as standard
    mlx-lm model ``__call__`` implementations. Works for LLaMA, Mistral,
    SmolLM, Qwen, Gemma, Phi, and other architectures using the standard
    ``model.model.layers`` + ``model.model.embed_tokens`` pattern.
    """
    import mlx.core as mx
    from mlx_lm.models.base import create_attention_mask

    inner, _layout = _get_inner_model(model)
    layer_set = set(layer_indices)
    all_hidden: dict[int, list[np.ndarray]] = {i: [] for i in layer_indices}

    for prompt in prompts:
        messages = [{"role": "user", "content": prompt}]
        try:
            tokens = tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True,
            )
        except Exception:
            tokens = tokenizer.encode(prompt)
        tokens_mx = mx.array(tokens)[None]  # (1, seq_len)

        h = inner.embed_tokens(tokens_mx)
        cache = [None] * len(inner.layers)

        # Build causal mask
        fa_idx = getattr(inner, "fa_idx", 0)
        mask = create_attention_mask(h, cache[fa_idx])

        for i, (layer, c) in enumerate(zip(inner.layers, cache)):
            h = layer(h, mask, cache=c)
            if i in layer_set:
                # Cast to float32 before pooling — avoids fp16 overflow
                # and bfloat16 buffer errors. Do the cast in MLX space
                # then convert to numpy.
                h_slice = h[0]  # (seq_len, hidden_dim)
                if isinstance(h_slice, np.ndarray):
                    h_np = h_slice.astype(np.float32)
                else:
                    h_f32 = h_slice.astype(mx.float32)
                    mx.eval(h_f32)
                    h_np = np.array(h_f32, copy=False)
                if pool == "last":
                    pooled = h_np[-1]
                else:
                    pooled = h_np.mean(axis=0)
                all_hidden[i].append(pooled)

    return {idx: np.stack(vecs) for idx, vecs in all_hidden.items()}


# ── backend detection ────────────────────────────────────────────


def _is_mlx_module(model) -> bool:
    """Check if model is an MLX nn.Module."""
    try:
        import mlx.nn as nn
        return isinstance(model, nn.Module)
    except ImportError:
        return False


def _detect_backend(model, backend: str | None) -> str:
    """Resolve backend to 'torch' or 'mlx'.

    When backend is None:
    - MLX nn.Module → 'mlx'
    - PyTorch PreTrainedModel → 'torch'
    - str (model name): prefer 'mlx' on Apple Silicon if mlx_lm is
      importable, otherwise 'torch'
    """
    if backend is not None:
        if backend not in ("torch", "mlx"):
            raise ValueError(f"backend must be 'torch', 'mlx', or None, got {backend!r}")
        return backend

    # Already-instantiated model
    if _is_mlx_module(model):
        return "mlx"
    if isinstance(model, PreTrainedModel):
        return "torch"

    # String model name — auto-detect platform
    if isinstance(model, str):
        if platform.machine() == "arm64" and sys.platform == "darwin":
            try:
                import mlx_lm  # noqa: F401
                return "mlx"
            except ImportError:
                pass
        return "torch"

    return "torch"


# ── per-layer training ───────────────────────────────────────────────


def _train_probe_at_layer(
    X_train, y_train, X_test, y_test, label_names: list[str],
) -> tuple[float, dict[str, float], object, object]:
    """Train a logistic regression probe at one layer.

    Returns (accuracy, per_label_accuracy, sklearn_classifier, scaler).
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    clf = LogisticRegression(
        max_iter=1000,
        solver="lbfgs",
        C=1.0,
    )
    clf.fit(X_train_scaled, y_train)

    accuracy = clf.score(X_test_scaled, y_test)

    # Per-label accuracy
    preds = clf.predict(X_test_scaled)
    per_label: dict[str, float] = {}
    for i, name in enumerate(label_names):
        mask = y_test == i
        if mask.sum() > 0:
            per_label[name] = float((preds[mask] == i).mean())
        else:
            per_label[name] = 0.0

    return accuracy, per_label, clf, scaler


def _sklearn_to_probe(
    clf, scaler, label_names: list[str], threshold: float,
) -> TurnstyleProbe:
    """Convert a trained sklearn LogisticRegression to a TurnstyleProbe.

    The probe computes sigmoid(Wh + b). Sklearn's multinomial logistic
    regression computes softmax(Wh + b). For routing we only need the
    ranking (argmax), which is preserved. The threshold may need tuning
    since sigmoid and softmax produce different scales.

    When a StandardScaler is used, we fold the scaling into the weights:
    W_eff = W / scale, b_eff = b - W @ (mean / scale)
    """
    import numpy as np

    W = clf.coef_  # (num_classes, hidden_dim)
    b = clf.intercept_  # (num_classes,)

    # Fold scaler into weights
    scale = scaler.scale_
    mean = scaler.mean_
    W_eff = W / scale[np.newaxis, :]
    b_eff = b - (W_eff @ mean)

    weights = torch.tensor(W_eff, dtype=torch.float32)
    bias = torch.tensor(b_eff, dtype=torch.float32)

    return TurnstyleProbe(weights, bias, label_names, threshold)


# ── SweepResult ──────────────────────────────────────────────────────


@dataclass
class SweepResult:
    """Results from a probe sweep across model layers."""

    layer_accuracies: dict[int, float]
    best_layer: int
    best_accuracy: float
    labels: list[str]
    probe: TurnstyleProbe
    per_label_accuracy: dict[int, dict[str, float]]
    pool: str
    train_size: int
    test_size: int
    backend: str = "torch"

    def summary(self) -> str:
        """Human-readable summary table."""
        lines = [
            f"Probe sweep: {len(self.layer_accuracies)} layers, "
            f"{self.train_size} train / {self.test_size} test, "
            f"pool={self.pool}, backend={self.backend}",
            "",
            "Layer  Accuracy",
            "─────  ────────",
        ]
        for layer in sorted(self.layer_accuracies):
            acc = self.layer_accuracies[layer]
            marker = " ◀ best" if layer == self.best_layer else ""
            lines.append(f"  {layer:3d}   {acc:6.1%}{marker}")

        lines.append("")
        lines.append(f"Best layer: {self.best_layer} ({self.best_accuracy:.1%})")

        if self.best_layer in self.per_label_accuracy:
            lines.append("")
            lines.append("Per-label accuracy at best layer:")
            for label, acc in sorted(
                self.per_label_accuracy[self.best_layer].items()
            ):
                lines.append(f"  {label:20s} {acc:6.1%}")

        return "\n".join(lines)


# ── main entry point ─────────────────────────────────────────────────


def probe_sweep(
    model: PreTrainedModel | str,
    tokenizer: PreTrainedTokenizerBase | str | None = None,
    prompts: dict[str, list[str]] | None = None,
    layer_range: tuple[int, int] | None = None,
    pool: str = "mean",
    test_ratio: float = 0.2,
    threshold: float = 0.5,
    save_path: str | None = None,
    device: str | None = None,
    include_negative: bool = False,
    verbose: bool = True,
    backend: str | None = None,
) -> SweepResult:
    """Sweep all layers to find the best probe, then train it.

    model: HF model, mlx-lm model, or model name string
    tokenizer: HF tokenizer (auto-loaded if model is a string)
    prompts: {label: [prompt_strings]} or None for auto-generated
    layer_range: (start, end) inclusive, or None for all layers
    pool: "mean" or "last" hidden state pooling
    test_ratio: fraction of prompts held out for evaluation
    threshold: sigmoid threshold for the final TurnstyleProbe
    save_path: if set, save the best probe to this .pt path
    device: "cpu", "cuda", etc. (auto-detected if None; ignored for MLX)
    include_negative: add a "_none" class for non-turnstyle prompts
    verbose: print progress
    backend: "mlx", "torch", or None (auto-detect). MLX is preferred on
        Apple Silicon when mlx-lm is installed.

    Returns SweepResult with the best probe and per-layer accuracies.
    """
    # Resolve backend before loading model
    resolved_backend = _detect_backend(model, backend)

    # Load model if string
    if isinstance(model, str):
        if verbose:
            print(f"Loading model: {model} (backend={resolved_backend})")
        if resolved_backend == "mlx":
            from mlx_lm import load as mlx_load
            model, tokenizer = mlx_load(model)
        else:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            tokenizer = tokenizer or AutoTokenizer.from_pretrained(model)
            model = AutoModelForCausalLM.from_pretrained(
                model, torch_dtype=torch.float16,
            )

    if tokenizer is None:
        raise ValueError("tokenizer is required when model is not a string")

    # Device setup (torch only — MLX uses unified memory)
    if resolved_backend == "torch":
        if device is None:
            if hasattr(model, "device"):
                device = str(model.device)
            else:
                device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        model.eval()
    else:
        device = "mlx"  # placeholder — MLX has no device concept

    if resolved_backend == "mlx":
        warnings.warn(
            "MLX hidden state extraction iterates model.model.layers directly. "
            "This works for standard architectures (LLaMA, Mistral, SmolLM, etc.) "
            "but may break on non-standard models. Use backend='torch' as fallback.",
            stacklevel=2,
        )

    # Generate prompts
    if prompts is None:
        prompts = generate_prompts(include_negative=include_negative)
    label_names = sorted(prompts.keys())
    label_to_idx = {name: i for i, name in enumerate(label_names)}

    # Build flat prompt list + labels
    all_prompts: list[str] = []
    all_labels: list[int] = []
    for label in label_names:
        for p in prompts[label]:
            all_prompts.append(p)
            all_labels.append(label_to_idx[label])

    # Train/test split (stratified by shuffling within each label)
    rng = np.random.RandomState(42)
    train_idx: list[int] = []
    test_idx: list[int] = []
    offset = 0
    for label in label_names:
        n = len(prompts[label])
        indices = list(range(offset, offset + n))
        rng.shuffle(indices)
        split = max(1, int(n * test_ratio))
        test_idx.extend(indices[:split])
        train_idx.extend(indices[split:])
        offset += n

    train_prompts = [all_prompts[i] for i in train_idx]
    test_prompts = [all_prompts[i] for i in test_idx]
    train_labels = np.array([all_labels[i] for i in train_idx])
    test_labels = np.array([all_labels[i] for i in test_idx])

    # Determine layer range
    layers = _get_layers(model)
    num_layers = len(layers)
    if layer_range is not None:
        start, end = layer_range
        layer_indices = list(range(start, min(end + 1, num_layers)))
    else:
        layer_indices = list(range(num_layers))

    if verbose:
        print(
            f"Sweeping {len(layer_indices)} layers, "
            f"{len(train_prompts)} train / {len(test_prompts)} test prompts, "
            f"{len(label_names)} labels (backend={resolved_backend})"
        )

    # Extract hidden states — dispatch to backend
    # Both backends return dict[int, np.ndarray]
    if resolved_backend == "mlx":
        if verbose:
            print("Extracting train hidden states (MLX)...")
        train_hidden = _extract_all_hidden_states_mlx(
            model, tokenizer, train_prompts, pool, layer_indices,
        )
        if verbose:
            print("Extracting test hidden states (MLX)...")
        test_hidden = _extract_all_hidden_states_mlx(
            model, tokenizer, test_prompts, pool, layer_indices,
        )
    else:
        if verbose:
            print("Extracting train hidden states...")
        train_hidden = _extract_all_hidden_states(
            model, tokenizer, train_prompts, train_labels.tolist(),
            device, pool, layer_indices,
        )
        if verbose:
            print("Extracting test hidden states...")
        test_hidden = _extract_all_hidden_states(
            model, tokenizer, test_prompts, test_labels.tolist(),
            device, pool, layer_indices,
        )

    # Train probe at each layer
    # Both backends return numpy arrays — feed directly to sklearn
    layer_accuracies: dict[int, float] = {}
    per_label_acc: dict[int, dict[str, float]] = {}
    best_layer = -1
    best_accuracy = -1.0
    best_clf = None
    best_scaler = None

    for idx in layer_indices:
        X_train = train_hidden[idx]
        X_test = test_hidden[idx]

        acc, per_label, clf, scaler = _train_probe_at_layer(
            X_train, train_labels, X_test, test_labels, label_names,
        )
        layer_accuracies[idx] = acc
        per_label_acc[idx] = per_label

        if verbose:
            print(f"  Layer {idx:3d}: {acc:.1%}")

        if acc > best_accuracy:
            best_accuracy = acc
            best_layer = idx
            best_clf = clf
            best_scaler = scaler

    # Convert best sklearn model to TurnstyleProbe
    probe = _sklearn_to_probe(best_clf, best_scaler, label_names, threshold)

    if save_path:
        probe.save(save_path)
        if verbose:
            print(f"Saved probe to {save_path}")

    result = SweepResult(
        layer_accuracies=layer_accuracies,
        best_layer=best_layer,
        best_accuracy=best_accuracy,
        labels=label_names,
        probe=probe,
        per_label_accuracy=per_label_acc,
        pool=pool,
        train_size=len(train_prompts),
        test_size=len(test_prompts),
        backend=resolved_backend,
    )

    if verbose:
        print(f"\nBest layer: {best_layer} ({best_accuracy:.1%})")

    return result
