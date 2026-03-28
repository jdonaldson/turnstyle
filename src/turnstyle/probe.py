"""Linear probe routing for turnstyles.

When regex parsers miss novel phrasings, a trained linear probe on model
hidden states detects which turnstyle should handle the prompt. Regex-first,
probe-fallback — existing parse patterns are always tried first.
"""

from __future__ import annotations

from pathlib import Path

import torch

from turnstyle.core import Turnstyle


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
        probe: TurnstyleProbe,
        layer_index: int,
        pool: str = "mean",
        strategy_router: StrategyRouter | None = None,
    ):
        """
        turnstyles: list of turnstyle instances
        probe: trained TurnstyleProbe with weights
        layer_index: which hidden layer to extract (e.g., 23 for SmolLM2)
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
        self._hook_handle = None
        self._captured_hidden = None

        # Map probe labels → turnstyle instances
        # Convention: each turnstyle class has a probe_label class attribute
        self.label_to_turnstyle: dict[str, Turnstyle] = {}
        for t in turnstyles:
            label = getattr(
                t, "probe_label",
                type(t).__name__.lower().replace("turnstyle", ""),
            )
            self.label_to_turnstyle[label] = t

    def _install_hook(self):
        """Register forward hook on the target layer."""
        layer = self.model.model.layers[self.layer_index]

        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                self._captured_hidden = output[0].detach()
            else:
                self._captured_hidden = output.detach()

        self._hook_handle = layer.register_forward_hook(hook_fn)

    def _remove_hook(self):
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None

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
        The pooled hidden state is reused by parse_from_hidden().
        """
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)

        self._install_hook()
        try:
            with torch.no_grad():
                self.model(**inputs)
            assert self._captured_hidden is not None, "Hook did not capture hidden states"
            h = self._pool_hidden(self._captured_hidden)
        finally:
            self._remove_hook()

        scores = self.probe.predict(h)

        candidates = []
        for label, score in sorted(scores.items(), key=lambda x: -x[1]):
            if label in self.label_to_turnstyle:
                candidates.append((self.label_to_turnstyle[label], score))
        return candidates, h

    def _extract_hidden(self, prompt: str) -> torch.Tensor:
        """Run a forward pass and return the pooled hidden state."""
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)

        self._install_hook()
        try:
            with torch.no_grad():
                self.model(**inputs)
            assert self._captured_hidden is not None
            return self._pool_hidden(self._captured_hidden)
        finally:
            self._remove_hook()

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
