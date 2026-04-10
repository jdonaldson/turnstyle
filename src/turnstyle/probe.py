"""Linear probe routing for turnstyles.

When regex parsers miss novel phrasings, a trained linear probe on model
hidden states detects which turnstyle should handle the prompt. Regex-first,
probe-fallback — existing parse patterns are always tried first.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from turnstyle.core import Turnstyle

# Position constants for ExtractionPoint
LAST_TOKEN = -1
MEAN_POOL = -2


@dataclass
class ExtractionPoint:
    """Where to read a hidden state vector from a model forward pass.

    layer: transformer layer index (0 = embedding, 1+ = transformer blocks)
    position: token index (0, 1, 2, ...), or LAST_TOKEN (-1) for the final
              token, or MEAN_POOL (-2) to average across all positions.
    """

    layer: int
    position: int  # token index, LAST_TOKEN, or MEAN_POOL


class TurnstyleProbe:
    """Linear probe for turnstyle routing from model hidden states.

    A simple sigmoid(Wh + b) classifier. Each row of the weight matrix
    corresponds to one turnstyle type. Scores above threshold activate
    that turnstyle.
    """

    def __init__(
        self,
        weights: torch.Tensor,
        bias: torch.Tensor,
        labels: list[str],
        threshold: float = 0.5,
    ):
        """
        weights: (num_types, hidden_dim) — probe weight matrix
        bias: (num_types,) — probe bias vector
        labels: turnstyle type names matching weight rows
        threshold: sigmoid score above which a turnstyle type is activated
        """
        self.weights = weights
        self.bias = bias
        self.labels = labels
        self.threshold = threshold

    def predict(self, hidden_state: torch.Tensor) -> dict[str, float]:
        """Score turnstyle types from a pooled hidden state vector.

        hidden_state: (hidden_dim,) — mean-pooled or last-token hidden state
        Returns: {label: score} for all types above threshold
        """
        logits = hidden_state @ self.weights.T + self.bias
        scores = torch.sigmoid(logits)
        return {
            label: float(score)
            for label, score in zip(self.labels, scores)
            if score > self.threshold
        }

    def predict_all(self, hidden_state: torch.Tensor) -> dict[str, float]:
        """Score all turnstyle types (ignoring threshold).

        Useful for debugging and threshold tuning.
        """
        logits = hidden_state @ self.weights.T + self.bias
        scores = torch.sigmoid(logits)
        return {label: float(score) for label, score in zip(self.labels, scores)}

    def predict_best(self, hidden_state: torch.Tensor) -> tuple[str, float]:
        """Return the single highest-scoring label and its confidence."""
        logits = hidden_state @ self.weights.T + self.bias
        scores = torch.sigmoid(logits)
        idx = torch.argmax(scores).item()
        return self.labels[idx], float(scores[idx])

    def save(self, path: str):
        """Save probe weights to a .pt file."""
        torch.save(
            {
                "weights": self.weights,
                "bias": self.bias,
                "labels": self.labels,
                "threshold": self.threshold,
            },
            path,
        )

    @classmethod
    def load(cls, path: str) -> TurnstyleProbe:
        """Load probe weights from a .pt file."""
        data = torch.load(path, weights_only=False)
        return cls(data["weights"], data["bias"], data["labels"], data["threshold"])


class MultiPositionProbe:
    """Linear probe reading from multiple (layer, position) extraction points.

    Different token positions encode different information: position 0 carries
    the quantifier/command word, while the last token carries accumulated
    structural context. Concatenating hidden states from multiple points into
    a single feature vector dramatically improves classification.

    Example: pos0@L1 (2048d) + last@L23 (2048d) = 4096d feature vector,
    boosting 27-way pattern accuracy from 71% to 90%.
    """

    def __init__(
        self,
        weights: torch.Tensor,
        bias: torch.Tensor,
        labels: list[str],
        extraction_points: list[ExtractionPoint],
        threshold: float = 0.5,
    ):
        """
        weights: (num_labels, total_dim) where total_dim = len(points) * hidden_dim
        bias: (num_labels,)
        labels: class names matching weight rows
        extraction_points: where to read hidden states (order must match training)
        threshold: sigmoid score above which a label is activated
        """
        self.weights = weights
        self.bias = bias
        self.labels = labels
        self.extraction_points = extraction_points
        self.threshold = threshold

    @property
    def required_layers(self) -> set[int]:
        """Unique layer indices needed for extraction."""
        return {ep.layer for ep in self.extraction_points}

    def assemble(self, hidden_states: dict[int, torch.Tensor]) -> torch.Tensor:
        """Build feature vector from per-layer hidden states.

        hidden_states: {layer_index: (seq_len, hidden_dim)} — batch dim
                       must already be squeezed.
        Returns: (total_dim,) concatenated feature vector.
        """
        parts = []
        for ep in self.extraction_points:
            hs = hidden_states[ep.layer]  # (seq_len, hidden_dim)
            if ep.position == LAST_TOKEN:
                parts.append(hs[-1])
            elif ep.position == MEAN_POOL:
                parts.append(hs.mean(dim=0))
            else:
                idx = min(ep.position, hs.shape[0] - 1)
                parts.append(hs[idx])
        return torch.cat(parts)

    def predict(self, assembled: torch.Tensor) -> dict[str, float]:
        """Score labels from a pre-assembled feature vector."""
        logits = assembled @ self.weights.T + self.bias
        scores = torch.sigmoid(logits)
        return {
            label: float(score)
            for label, score in zip(self.labels, scores)
            if score > self.threshold
        }

    def predict_all(self, assembled: torch.Tensor) -> dict[str, float]:
        """Score all labels (ignoring threshold)."""
        logits = assembled @ self.weights.T + self.bias
        scores = torch.sigmoid(logits)
        return {label: float(score) for label, score in zip(self.labels, scores)}

    def predict_best(self, assembled: torch.Tensor) -> tuple[str, float]:
        """Return the single highest-scoring label and its confidence."""
        logits = assembled @ self.weights.T + self.bias
        scores = torch.sigmoid(logits)
        idx = torch.argmax(scores).item()
        return self.labels[idx], float(scores[idx])

    def predict_from_layers(
        self, hidden_states: dict[int, torch.Tensor],
    ) -> dict[str, float]:
        """Full pipeline: assemble from layer outputs, then predict."""
        return self.predict(self.assemble(hidden_states))

    def save(self, path: str):
        """Save probe weights and extraction spec to a .pt file."""
        torch.save(
            {
                "weights": self.weights,
                "bias": self.bias,
                "labels": self.labels,
                "threshold": self.threshold,
                "extraction_points": [
                    (ep.layer, ep.position) for ep in self.extraction_points
                ],
            },
            path,
        )

    @classmethod
    def load(cls, path: str) -> MultiPositionProbe:
        """Load probe weights and extraction spec from a .pt file."""
        data = torch.load(path, weights_only=False)
        points = [
            ExtractionPoint(layer=l, position=p)
            for l, p in data["extraction_points"]
        ]
        return cls(
            data["weights"], data["bias"], data["labels"],
            points, data.get("threshold", 0.5),
        )


class IntentProbe:
    """Multi-dimensional probe: extracts structured parameters from hidden states.

    Each dimension is a TurnstyleProbe trained on a different classification task.
    Example dimensions for arithmetic:
        "operation": labels=["add", "sub", "mul", "div"]
        "operand_a": labels=["0", "1", ..., "999"]
    """

    def __init__(self, dimensions: dict[str, TurnstyleProbe]):
        self.dimensions = dimensions

    def predict(self, hidden_state: torch.Tensor) -> dict[str, tuple[str, float]]:
        """Extract all dimensions. Returns {dim_name: (label, confidence)}."""
        return {
            name: probe.predict_best(hidden_state)
            for name, probe in self.dimensions.items()
        }

    def save(self, path: str):
        """Save all dimension probes to a directory."""
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        meta = {"dimensions": list(self.dimensions.keys())}
        torch.save(meta, p / "meta.pt")
        for name, probe in self.dimensions.items():
            probe.save(str(p / f"{name}.pt"))

    @classmethod
    def load(cls, path: str) -> "IntentProbe":
        """Load intent probe from a directory."""
        p = Path(path)
        meta = torch.load(p / "meta.pt", weights_only=False)
        dimensions = {}
        for name in meta["dimensions"]:
            dimensions[name] = TurnstyleProbe.load(str(p / f"{name}.pt"))
        return cls(dimensions)


class MetacognitiveProbe:
    """Binary gate: does the model need help on this example?

    Wraps a TurnstyleProbe with labels=["fail", "succeed"]. When the
    "succeed" score exceeds threshold, the model is expected to handle
    the task without intervention.
    """

    def __init__(self, probe: TurnstyleProbe, threshold: float = 0.7):
        self.probe = probe
        self.threshold = threshold

    def needs_intervention(
        self, hidden_state: torch.Tensor,
    ) -> tuple[bool, float]:
        """Check whether the model needs turnstyle intervention.

        Returns (should_intervene, confidence).
        Defaults to True (intervene) when uncertain.
        """
        scores = self.probe.predict_all(hidden_state)
        succeed_score = scores.get("succeed", 0.0)
        if succeed_score > self.threshold:
            return False, succeed_score  # model's got this
        return True, 1.0 - succeed_score  # model needs help

    def save(self, path: str):
        """Save to a .pt file (probe weights + threshold)."""
        data = {
            "weights": self.probe.weights,
            "bias": self.probe.bias,
            "labels": self.probe.labels,
            "threshold": self.threshold,
            "probe_threshold": self.probe.threshold,
        }
        torch.save(data, path)

    @classmethod
    def load(cls, path: str, threshold: float | None = None) -> "MetacognitiveProbe":
        """Load from a .pt file. Uses saved threshold unless overridden."""
        data = torch.load(path, weights_only=False)
        probe = TurnstyleProbe(
            data["weights"], data["bias"], data["labels"],
            data.get("probe_threshold", 0.5),
        )
        saved_threshold = data.get("threshold", 0.7)
        return cls(probe, threshold if threshold is not None else saved_threshold)


class StrategyRouter:
    """Routes between named strategies based on per-strategy success probes.

    Each strategy has a TurnstyleProbe predicting whether that strategy
    will succeed on the current input. The router picks the strategy with
    the highest "succeed" score.
    """

    def __init__(self, default_strategy: str = "baseline"):
        self.strategies: dict[str, TurnstyleProbe] = {}
        self.default_strategy = default_strategy

    def add_strategy(self, name: str, probe: TurnstyleProbe):
        """Register a strategy with its success-prediction probe."""
        self.strategies[name] = probe

    def route(self, hidden_state: torch.Tensor) -> tuple[str, float]:
        """Pick the best strategy for this input.

        Returns (strategy_name, confidence).
        Falls back to default_strategy when no probes are registered.
        """
        if not self.strategies:
            return self.default_strategy, 0.0

        best_name = self.default_strategy
        best_score = 0.0
        for name, probe in self.strategies.items():
            scores = probe.predict_all(hidden_state)
            succeed_score = scores.get("succeed", 0.0)
            if succeed_score > best_score:
                best_score = succeed_score
                best_name = name
        return best_name, best_score

    def save(self, path: str):
        """Save all strategy probes to a directory."""
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        meta = {
            "strategies": list(self.strategies.keys()),
            "default_strategy": self.default_strategy,
        }
        torch.save(meta, p / "meta.pt")
        for name, probe in self.strategies.items():
            probe.save(str(p / f"{name}.pt"))

    @classmethod
    def load(cls, path: str) -> "StrategyRouter":
        """Load all strategy probes from a directory."""
        p = Path(path)
        meta = torch.load(p / "meta.pt", weights_only=False)
        router = cls(default_strategy=meta["default_strategy"])
        for name in meta["strategies"]:
            router.add_strategy(name, TurnstyleProbe.load(str(p / f"{name}.pt")))
        return router


class RoutingTurnstyle(Turnstyle):
    """Routes prompts to turnstyles using regex-first, probe-fallback.

    Wraps multiple turnstyles. On ``generate()``:
    1. Try each turnstyle's ``parse()`` (fast regex path)
    2. If no regex matches, run a forward pass and probe hidden states
    3. Delegate to the highest-scoring turnstyle above threshold
    4. If nothing matches, model generates freely
    """

    def __init__(
        self,
        turnstyles: list[Turnstyle],
        probe: TurnstyleProbe | MultiPositionProbe,
        layer_index: int,
        pool: str = "mean",
        strategy_router: StrategyRouter | None = None,
    ):
        """
        turnstyles: list of turnstyle instances
        probe: TurnstyleProbe or MultiPositionProbe for routing
        layer_index: hidden layer for downstream consumers (metacognitive gate,
                     parse_from_hidden). For MultiPositionProbe, the probe
                     specifies its own extraction points; layer_index is only
                     used for downstream pooled hidden state.
        pool: "mean" or "last" — how to pool hidden states across positions
        strategy_router: optional StrategyRouter for multi-strategy routing
        """
        first = turnstyles[0]
        super().__init__(first.model, first.tokenizer, first.device, first.bias_strength)
        self.turnstyles = turnstyles
        self.probe = probe
        self.layer_index = layer_index
        self.pool = pool
        self.strategy_router = strategy_router
        self._hook_handles: list = []
        self._captured_hidden: dict[int, torch.Tensor] | None = None

        # Map probe labels → turnstyle instances
        # Convention: each turnstyle class has a probe_label class attribute
        self.label_to_turnstyle: dict[str, Turnstyle] = {}
        for t in turnstyles:
            label = getattr(t, "probe_label", "") or type(t).__name__.lower().replace("turnstyle", "")
            self.label_to_turnstyle[label] = t

    def _install_hooks(self, layer_indices: set[int] | None = None):
        """Register forward hooks on one or more layers.

        If layer_indices is None, hooks the single self.layer_index.
        Captured hidden states are stored in self._captured_hidden
        as {layer_index: (batch, seq_len, hidden_dim)}.
        """
        self._captured_hidden = {}
        indices = layer_indices or {self.layer_index}

        for layer_idx in indices:
            layer = self.model.model.layers[layer_idx]

            def make_hook(idx):
                def hook_fn(module, input, output):
                    if isinstance(output, tuple):
                        self._captured_hidden[idx] = output[0].detach()
                    else:
                        self._captured_hidden[idx] = output.detach()
                return hook_fn

            handle = layer.register_forward_hook(make_hook(layer_idx))
            self._hook_handles.append(handle)

    def _remove_hooks(self):
        """Remove all registered forward hooks."""
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles = []

    def _pool_hidden(self, hidden: torch.Tensor) -> torch.Tensor:
        """Pool hidden states across sequence positions.

        hidden: (batch=1, seq_len, hidden_dim)
        Returns: (hidden_dim,)
        """
        if self.pool == "last":
            return hidden[0, -1]
        return hidden[0].mean(dim=0)

    def parse(self, prompt: str):
        """Try regex parsers. Returns list of (turnstyle, parsed) or None."""
        matches = []
        for t in self.turnstyles:
            parsed = t.parse(prompt)
            if parsed is not None:
                matches.append((t, parsed))
        return matches if matches else None

    def _probe_route(
        self, prompt: str,
    ) -> tuple[list[tuple[Turnstyle, float]], torch.Tensor | None]:
        """Use probe to identify candidate turnstyles.

        Returns ([(turnstyle, score)], pooled_hidden_state).
        The pooled hidden state is reused by parse_from_hidden() and
        metacognitive gate checks.

        For MultiPositionProbe: hooks all required layers, assembles the
        multi-position feature vector for scoring, but returns a single-layer
        pooled vector for downstream consumers.
        """
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)

        if isinstance(self.probe, MultiPositionProbe):
            # Multi-position: hook all required layers + layer_index for downstream
            layers = self.probe.required_layers | {self.layer_index}
            self._install_hooks(layers)
            try:
                with torch.no_grad():
                    self.model(**inputs)
                assert self._captured_hidden, "Hooks did not capture hidden states"
                # Build per-layer dict with batch dim squeezed
                layer_hidden = {
                    idx: h[0] for idx, h in self._captured_hidden.items()
                }
                assembled = self.probe.assemble(layer_hidden)
                # Downstream hidden state: pool from self.layer_index
                h_downstream = self._pool_hidden(
                    self._captured_hidden[self.layer_index],
                )
            finally:
                self._remove_hooks()

            scores = self.probe.predict(assembled)
        else:
            # Single-position probe: existing behavior
            self._install_hooks()
            try:
                with torch.no_grad():
                    self.model(**inputs)
                assert self._captured_hidden, "Hook did not capture hidden states"
                h_downstream = self._pool_hidden(
                    self._captured_hidden[self.layer_index],
                )
            finally:
                self._remove_hooks()

            scores = self.probe.predict(h_downstream)

        candidates = []
        for label, score in sorted(scores.items(), key=lambda x: -x[1]):
            if label in self.label_to_turnstyle:
                candidates.append((self.label_to_turnstyle[label], score))
        return candidates, h_downstream

    def _extract_hidden(self, prompt: str) -> torch.Tensor:
        """Run a forward pass and return the pooled hidden state."""
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)

        self._install_hooks()
        try:
            with torch.no_grad():
                self.model(**inputs)
            assert self._captured_hidden
            return self._pool_hidden(
                self._captured_hidden[self.layer_index],
            )
        finally:
            self._remove_hooks()

    def _check_metacognitive_gate(
        self, turnstyle: Turnstyle, prompt: str, h: torch.Tensor | None = None,
    ) -> tuple[bool, float]:
        """Check if a turnstyle's metacognitive probe says to skip intervention.

        Returns (should_skip, confidence). If no probe, returns (False, 0.0).
        """
        gate = getattr(turnstyle, "metacognitive_probe", None)
        if gate is None:
            return False, 0.0
        if h is None:
            h = self._extract_hidden(prompt)
        needs_help, confidence = gate.needs_intervention(h)
        return not needs_help, confidence

    def generate(self, prompt: str, max_new_tokens: int = 50):
        """Generate with routing: regex first, probe fallback.

        Metacognitive gate: if a matched turnstyle has a metacognitive_probe
        and it predicts the model will succeed, skip the turnstyle.
        Strategy router: if available, pick the best strategy for the input.

        Returns: (text, diagnostic_or_None)
        """
        # Step 1: Try regex parsers
        matches = self.parse(prompt)

        if matches is not None:
            # Regex matched — use first match
            t, parsed = matches[0]

            # Metacognitive gate: check if model needs help
            should_skip, _ = self._check_metacognitive_gate(t, prompt)
            if should_skip:
                return self._free_generate(prompt, max_new_tokens)

            return t.generate(prompt, max_new_tokens=max_new_tokens)

        # Step 2: No regex match — use probe
        candidates, h = self._probe_route(prompt)

        if candidates:
            t, _score = candidates[0]

            # Strategy router: pick best strategy if available
            if self.strategy_router is not None and h is not None:
                strategy_name, _conf = self.strategy_router.route(h)
                if strategy_name in self.label_to_turnstyle:
                    t = self.label_to_turnstyle[strategy_name]

            # Metacognitive gate on routed turnstyle
            should_skip, _ = self._check_metacognitive_gate(t, prompt, h)
            if should_skip:
                return self._free_generate(prompt, max_new_tokens)

            # Step 3: Try probe-based parsing with captured hidden states
            parsed = t.parse_from_hidden(h)
            if parsed is not None:
                processor = t.make_processor(parsed, max_new_tokens)
                messages = [{"role": "user", "content": prompt}]
                text = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                )
                inputs = self.tokenizer(
                    text, return_tensors="pt",
                ).to(self.device)
                with torch.no_grad():
                    out = self.model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        logits_processor=[processor],
                    )
                text = self.tokenizer.decode(
                    out[0][inputs["input_ids"].shape[1]:],
                    skip_special_tokens=True,
                ).strip()
                return text, getattr(processor, "proof", None)

            # Step 3b: Try LLM extraction on routed turnstyle
            extraction_spec = getattr(t, 'extraction_spec', None)
            if extraction_spec is not None:
                from turnstyle.extract import extract
                result = extract(prompt, t, extraction_spec)
                if result is not None and result.parsed is not None:
                    processor = t.make_processor(result.parsed, max_new_tokens)
                    messages = [{"role": "user", "content": prompt}]
                    text = self.tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True,
                    )
                    inputs = self.tokenizer(
                        text, return_tensors="pt",
                    ).to(self.device)
                    with torch.no_grad():
                        out = self.model.generate(
                            **inputs,
                            max_new_tokens=max_new_tokens,
                            do_sample=False,
                            logits_processor=[processor],
                        )
                    text = self.tokenizer.decode(
                        out[0][inputs["input_ids"].shape[1]:],
                        skip_special_tokens=True,
                    ).strip()
                    return text, getattr(processor, "proof", None)

            # Step 4: Fall back to regex on routed turnstyle
            return t.generate(prompt, max_new_tokens=max_new_tokens)

        # Step 5: No match at all — free generation
        return self._free_generate(prompt, max_new_tokens)

    @classmethod
    def build(
        cls,
        solvers: "list[Turnstyle]",
        model,
        tokenizer,
        device: str | None = None,
        layer: int = 1,
        pool: str = "last",
        n_per_solver: int = 50,
        bias_strength: float = 15.0,
        verbose: bool = True,
    ) -> "RoutingTurnstyle":
        """Train a route probe from each solver's examples, return a fitted hub.

        Collects ``examples[:n_per_solver]`` from each solver, runs a single-layer
        probe sweep at ``layer`` with last-token pooling (optimal per 2026-04-10
        experiment: L1, last-token, 92-100% LOO on multi-task families), then
        returns a ready-to-use ``RoutingTurnstyle``.

        Args:
            solvers: list of Turnstyle instances, each with ``probe_label`` and
                ``examples`` class attributes.
            model: loaded HuggingFace PreTrainedModel (already on correct device).
            tokenizer: corresponding tokenizer.
            device: device string ("cpu", "cuda", etc.). Auto-detected if None.
            layer: transformer layer index for probe (default 1 = L1).
            pool: hidden-state pooling mode (default "last" = last-token).
            n_per_solver: maximum examples per solver for probe training.
            bias_strength: logit bias applied by this hub's turnstyles.
            verbose: print probe sweep progress.

        Returns:
            RoutingTurnstyle with a trained probe at the specified layer.

        Note:
            Requires sklearn (``pip install scikit-learn``). On macOS, lbfgs
            may trigger joblib multiprocessing — use n_per_solver <= 50 to keep
            training fast and avoid long joblib overhead.
        """
        from turnstyle.sweep import probe_sweep

        # Collect prompts keyed by probe_label
        prompts: dict[str, list[str]] = {}
        for s in solvers:
            label = getattr(s, "probe_label", "")
            examples = getattr(s, "examples", [])
            if label and examples:
                prompts[label] = list(examples[:n_per_solver])

        if not prompts:
            raise ValueError(
                "No solver has both probe_label and examples — "
                "cannot train route probe. Add examples to your solver classes."
            )

        # Auto-detect device
        if device is None:
            if hasattr(model, "device"):
                device = str(model.device)
            else:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"

        if verbose:
            label_counts = {k: len(v) for k, v in prompts.items()}
            print(f"RoutingTurnstyle.build(): training route probe")
            print(f"  labels: {list(label_counts.keys())}")
            print(f"  examples per label: {label_counts}")
            print(f"  layer={layer}, pool={pool}, device={device}")

        result = probe_sweep(
            model,
            tokenizer,
            prompts=prompts,
            layer_range=(layer, layer),
            pool=pool,
            device=device,
            verbose=verbose,
        )

        if verbose:
            print(f"  best layer: {result.best_layer} ({result.best_accuracy:.1%})")

        return cls(
            solvers,
            result.probe,
            layer_index=layer,
            pool=pool,
        )

    def _free_generate(
        self, prompt: str, max_new_tokens: int,
    ) -> tuple[str, None]:
        """Unbiased generation — model generates freely."""
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False,
            )
        text = self.tokenizer.decode(
            out[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        ).strip()
        return text, None
