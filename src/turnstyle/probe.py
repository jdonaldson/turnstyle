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
    ):
        """
        turnstyles: list of turnstyle instances
        probe: trained TurnstyleProbe with weights
        layer_index: which hidden layer to extract (e.g., 23 for SmolLM2)
        pool: "mean" or "last" — how to pool hidden states across positions
        """
        first = turnstyles[0]
        super().__init__(first.model, first.tokenizer, first.device, first.bias_strength)
        self.turnstyles = turnstyles
        self.probe = probe
        self.layer_index = layer_index
        self.pool = pool
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

    def generate(self, prompt: str, max_new_tokens: int = 50):
        """Generate with routing: regex first, probe fallback.

        Returns: (text, diagnostic_or_None)
        """
        # Step 1: Try regex parsers
        matches = self.parse(prompt)

        if matches is not None:
            # Regex matched — use first match
            t, parsed = matches[0]
            return t.generate(prompt, max_new_tokens=max_new_tokens)

        # Step 2: No regex match — use probe
        candidates, h = self._probe_route(prompt)

        if candidates:
            t, _score = candidates[0]

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

            # Step 4: Fall back to regex on routed turnstyle
            return t.generate(prompt, max_new_tokens=max_new_tokens)

        # Step 5: No match at all — free generation
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
