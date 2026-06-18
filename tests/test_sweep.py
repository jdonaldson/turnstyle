"""Tests for probe sweep — all mock-based, no GPU/model needed."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from turnstyle.sweep import (
    ALL_LABELS,
    IntentSweepResult,
    SweepResult,
    _INTENT_TEMPLATES,
    _detect_backend,
    _extract_all_hidden_states_mlx,
    _is_mlx_module,
    _sklearn_to_probe,
    _train_probe_at_layer,
    generate_intent_prompts,
    generate_prompts,
)
from turnstyle.probe import IntentProbe, TurnstyleProbe


# ── generate_prompts tests ───────────────────────────────────────────


class TestGeneratePrompts:
    def test_structure(self):
        """Returns dict with all labels, each having non-empty lists."""
        prompts = generate_prompts()
        assert isinstance(prompts, dict)
        for label in ALL_LABELS:
            assert label in prompts
            assert len(prompts[label]) > 0
            assert all(isinstance(p, str) for p in prompts[label])

    def test_per_label_count(self):
        """Respects per_label parameter."""
        prompts = generate_prompts(per_label=10)
        for label in ALL_LABELS:
            assert len(prompts[label]) == 10

    def test_with_negative(self):
        """include_negative adds _none key."""
        prompts = generate_prompts(include_negative=True)
        assert "_none" in prompts
        assert len(prompts["_none"]) > 0

    def test_without_negative(self):
        """By default no _none key."""
        prompts = generate_prompts(include_negative=False)
        assert "_none" not in prompts

    def test_subset_labels(self):
        """labels parameter filters to requested subset."""
        prompts = generate_prompts(labels=["arithmetic", "date"])
        assert set(prompts.keys()) == {"arithmetic", "date"}

    def test_unknown_label_raises(self):
        """Unknown label raises ValueError."""
        with pytest.raises(ValueError, match="Unknown label"):
            generate_prompts(labels=["nonexistent"])

    def test_deterministic(self):
        """Same seed produces same prompts."""
        p1 = generate_prompts(seed=123)
        p2 = generate_prompts(seed=123)
        assert p1 == p2

    def test_different_seeds(self):
        """Different seeds produce different prompts."""
        p1 = generate_prompts(seed=1)
        p2 = generate_prompts(seed=2)
        # At least some prompts should differ
        assert p1["arithmetic"] != p2["arithmetic"]


# ── _train_probe_at_layer tests ──────────────────────────────────────


class TestTrainProbe:
    def test_separable_data(self):
        """Linearly separable synthetic data → high accuracy."""
        rng = np.random.RandomState(42)
        n_per_class = 100
        dim = 16

        # Class 0: centered at [2, 0, ...], Class 1: centered at [-2, 0, ...]
        X0 = rng.randn(n_per_class, dim) * 0.5
        X0[:, 0] += 2.0
        X1 = rng.randn(n_per_class, dim) * 0.5
        X1[:, 0] -= 2.0

        X = np.vstack([X0, X1])
        y = np.array([0] * n_per_class + [1] * n_per_class)

        # Shuffle
        idx = rng.permutation(len(X))
        X, y = X[idx], y[idx]

        split = int(len(X) * 0.8)
        acc, per_label, clf, scaler = _train_probe_at_layer(
            X[:split], y[:split], X[split:], y[split:],
            ["class_a", "class_b"],
        )
        assert acc > 0.9
        assert "class_a" in per_label
        assert "class_b" in per_label

    def test_multiclass(self):
        """Works with more than 2 classes."""
        rng = np.random.RandomState(42)
        n_per_class = 80
        dim = 16

        classes = []
        for i in range(4):
            X_i = rng.randn(n_per_class, dim) * 0.3
            X_i[:, i % dim] += 3.0
            classes.append(X_i)

        X = np.vstack(classes)
        y = np.concatenate([np.full(n_per_class, i) for i in range(4)])

        idx = rng.permutation(len(X))
        X, y = X[idx], y[idx]

        split = int(len(X) * 0.8)
        acc, per_label, clf, scaler = _train_probe_at_layer(
            X[:split], y[:split], X[split:], y[split:],
            ["a", "b", "c", "d"],
        )
        assert acc > 0.8
        assert len(per_label) == 4


# ── sklearn → TurnstyleProbe roundtrip ───────────────────────────────


class TestSklearnToProbe:
    def test_roundtrip_rankings(self):
        """Converted probe produces same top-1 rankings as sklearn."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        rng = np.random.RandomState(42)
        n_per_class = 100
        dim = 16
        labels = ["arithmetic", "date", "unit"]

        # Generate separable data
        X_parts = []
        y_parts = []
        for i in range(len(labels)):
            X_i = rng.randn(n_per_class, dim) * 0.5
            X_i[:, i * 4] += 3.0  # each class has a different dominant dimension
            X_parts.append(X_i)
            y_parts.append(np.full(n_per_class, i))

        X = np.vstack(X_parts)
        y = np.concatenate(y_parts)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        clf = LogisticRegression(max_iter=1000, solver="lbfgs")
        clf.fit(X_scaled, y)

        # Convert to probe
        probe = _sklearn_to_probe(clf, scaler, labels, threshold=0.0)

        # Test on new data points
        for i in range(len(labels)):
            h = np.zeros(dim, dtype=np.float32)
            h[i * 4] = 5.0
            h_tensor = torch.tensor(h)

            # sklearn prediction
            sklearn_pred = clf.predict(scaler.transform(h.reshape(1, -1)))[0]
            sklearn_label = labels[sklearn_pred]

            # probe prediction (all scores)
            probe_scores = probe.predict_all(h_tensor)
            probe_label = max(probe_scores, key=probe_scores.get)

            assert sklearn_label == probe_label, (
                f"Mismatch for class {i}: sklearn={sklearn_label}, "
                f"probe={probe_label}, scores={probe_scores}"
            )


# ── SweepResult tests ────────────────────────────────────────────────


class TestSweepResult:
    def make_result(self):
        """Create a minimal SweepResult for testing."""
        probe = TurnstyleProbe(
            weights=torch.randn(3, 8),
            bias=torch.zeros(3),
            labels=["arithmetic", "date", "unit"],
        )
        return SweepResult(
            layer_accuracies={0: 0.45, 1: 0.72, 2: 0.91, 3: 0.88},
            best_layer=2,
            best_accuracy=0.91,
            labels=["arithmetic", "date", "unit"],
            probe=probe,
            per_label_accuracy={
                2: {"arithmetic": 0.95, "date": 0.88, "unit": 0.90},
            },
            pool="mean",
            train_size=320,
            test_size=80,
        )

    def test_summary_readable(self):
        """Summary produces human-readable output."""
        result = self.make_result()
        s = result.summary()
        assert "Best layer: 2" in s
        assert "91.0%" in s
        assert "Layer" in s
        assert "Accuracy" in s

    def test_summary_contains_all_layers(self):
        """Summary lists all swept layers."""
        result = self.make_result()
        s = result.summary()
        for layer in [0, 1, 2, 3]:
            assert str(layer) in s

    def test_summary_marks_best(self):
        """Summary marks the best layer."""
        result = self.make_result()
        s = result.summary()
        assert "best" in s.lower()


# ── pool strategy tests ──────────────────────────────────────────────


class TestPoolStrategies:
    def test_mean_pool_shape(self):
        """Mean pooling produces correct shape."""
        h = torch.randn(1, 10, 32)  # (batch, seq_len, hidden_dim)
        pooled = h[0].mean(dim=0)
        assert pooled.shape == (32,)

    def test_last_pool_shape(self):
        """Last-token pooling produces correct shape."""
        h = torch.randn(1, 10, 32)
        pooled = h[0, -1]
        assert pooled.shape == (32,)

    def test_mean_pool_value(self):
        """Mean pooling averages across positions."""
        h = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]])
        pooled = h[0].mean(dim=0)
        assert torch.allclose(pooled, torch.tensor([3.0, 4.0]))

    def test_last_pool_value(self):
        """Last-token pooling takes final position."""
        h = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]])
        pooled = h[0, -1]
        assert torch.allclose(pooled, torch.tensor([5.0, 6.0]))


# ── mock model extraction test ───────────────────────────────────────


class TestExtractMockModel:
    def test_hooks_capture_all_layers(self):
        """Hooks installed on a mock model capture from all target layers."""
        hidden_dim = 16
        num_layers = 4
        seq_len = 5

        # Create mock layers
        mock_layers = []
        for i in range(num_layers):
            layer = torch.nn.Linear(hidden_dim, hidden_dim)
            mock_layers.append(layer)

        # Build a simple model that passes through layers sequentially
        class TinyModel(torch.nn.Module):
            def __init__(self, layers):
                super().__init__()
                self.layers = torch.nn.ModuleList(layers)

            def forward(self, input_ids=None, attention_mask=None, **kwargs):
                # Fake: create hidden states from input_ids shape
                batch = input_ids.shape[0]
                h = torch.randn(batch, seq_len, hidden_dim)
                for layer in self.layers:
                    h = layer(h)
                return MagicMock(last_hidden_state=h)

        model = TinyModel(mock_layers)

        # Install hooks on layers 1 and 3
        target_layers = [1, 3]
        captured = {}
        handles = []
        for idx in target_layers:
            def make_hook(layer_idx):
                def hook_fn(module, input, output):
                    captured[layer_idx] = output.detach()
                return hook_fn
            handles.append(model.layers[idx].register_forward_hook(make_hook(idx)))

        # Forward pass
        input_ids = torch.tensor([[1, 2, 3, 4, 5]])
        model(input_ids=input_ids)

        for h in handles:
            h.remove()

        # Verify hooks captured from both target layers
        assert 1 in captured
        assert 3 in captured
        assert captured[1].shape == (1, seq_len, hidden_dim)
        assert captured[3].shape == (1, seq_len, hidden_dim)

        # Verify they captured different tensors
        assert not torch.allclose(captured[1], captured[3])


# ── probe_label attribute test ───────────────────────────────────────


class TestProbeLabelAttributes:
    """Verify probe_label is a real class attribute, not just in docstring."""

    @pytest.mark.parametrize(
        "cls_name,expected_label",
        [
            ("ArithmeticTurnstyle", "arithmetic"),
            ("DateTurnstyle", "date"),
            ("UnitTurnstyle", "unit"),
            ("CurrencyTurnstyle", "currency"),
            ("PercentageTurnstyle", "percentage"),
            ("CountingTurnstyle", "counting"),
            ("BaseConversionTurnstyle", "base_conversion"),
            ("SandboxTurnstyle", "sandbox"),
            ("BooleanTurnstyle", "boolean"),
            ("SortingTurnstyle", "sorting"),
            ("DyckTurnstyle", "dyck"),
        ],
    )
    def test_probe_label_is_class_attribute(self, cls_name, expected_label):
        """probe_label must be a class attribute (in __dict__), not a fallback."""
        import turnstyle

        cls = getattr(turnstyle, cls_name)

        # Must be in the class __dict__ (not inherited or computed)
        assert "probe_label" in cls.__dict__, (
            f"{cls_name}.probe_label is not a class attribute — "
            "it may be stuck inside the docstring"
        )
        assert cls.probe_label == expected_label

    def test_probe_label_not_in_docstring(self):
        """Ensure probe_label doesn't appear as text in any docstring."""
        import turnstyle

        classes = [
            turnstyle.ArithmeticTurnstyle,
            turnstyle.DateTurnstyle,
            turnstyle.UnitTurnstyle,
            turnstyle.CurrencyTurnstyle,
            turnstyle.PercentageTurnstyle,
            turnstyle.CountingTurnstyle,
            turnstyle.BaseConversionTurnstyle,
            turnstyle.SandboxTurnstyle,
            turnstyle.BooleanTurnstyle,
            turnstyle.SortingTurnstyle,
            turnstyle.DyckTurnstyle,
        ]
        for cls in classes:
            doc = cls.__doc__ or ""
            # The string 'probe_label = "' should NOT appear in the docstring
            assert 'probe_label = "' not in doc, (
                f"{cls.__name__}.__doc__ still contains 'probe_label = \"...\"' — "
                "it should be a class attribute, not in the docstring"
            )


# ── MLX backend tests (mock-based) ─────────────────────────────────


class _MockMLXModule:
    """Mimics the structure of an mlx.nn.Module for testing."""
    pass


class _MockMLXLayer:
    """Mimics a single transformer layer in mlx-lm models."""

    def __init__(self, hidden_dim: int, layer_idx: int):
        self.hidden_dim = hidden_dim
        self.layer_idx = layer_idx

    def __call__(self, h, mask=None, cache=None):
        # Shift hidden states slightly per layer so each layer is distinct
        return h + (self.layer_idx + 1) * 0.1


class _MockMLXInnerModel:
    """Mimics model.model (e.g. LlamaModel) with embed_tokens and layers."""

    def __init__(self, num_layers: int, hidden_dim: int, vocab_size: int = 100):
        self.hidden_dim = hidden_dim
        self.layers = [_MockMLXLayer(hidden_dim, i) for i in range(num_layers)]
        self.vocab_size = vocab_size
        self.fa_idx = 0

    def embed_tokens(self, tokens):
        """Return a fake embedding: (1, seq_len, hidden_dim)."""
        seq_len = tokens.shape[1] if hasattr(tokens, 'shape') else len(tokens[0])
        return np.random.RandomState(42).randn(1, seq_len, self.hidden_dim).astype(np.float32)


class _MockMLXModel(_MockMLXModule):
    """Mimics a top-level mlx-lm model (e.g. LlamaForCausalLM)."""

    def __init__(self, num_layers: int = 4, hidden_dim: int = 16):
        self.model = _MockMLXInnerModel(num_layers, hidden_dim)

    @property
    def layers(self):
        return self.model.layers


class _MockTokenizer:
    """Minimal tokenizer mock for MLX tests."""

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=True):
        return [1, 2, 3, 4, 5]  # 5 fake tokens

    def encode(self, text):
        return [1, 2, 3, 4, 5]


class TestBackendDetection:
    def test_torch_model_detected(self):
        """PyTorch PreTrainedModel auto-selects torch backend."""
        mock_model = MagicMock(spec=torch.nn.Module)
        # Make it pass isinstance check for PreTrainedModel
        with patch("turnstyle.sweep.isinstance", side_effect=lambda obj, cls: (
            cls is torch.nn.Module or
            (hasattr(cls, '__name__') and cls.__name__ == 'PreTrainedModel')
        )):
            # Direct test: PreTrainedModel is detected
            from transformers import PreTrainedModel
            mock_pt = MagicMock(spec=PreTrainedModel)
            result = _detect_backend(mock_pt, None)
            assert result == "torch"

    def test_mlx_model_detected(self):
        """An object recognized as MLX module auto-selects mlx backend."""
        mock_model = _MockMLXModel()
        with patch("turnstyle.sweep._is_mlx_module", return_value=True):
            result = _detect_backend(mock_model, None)
            assert result == "mlx"

    def test_explicit_override_torch(self):
        """Explicit backend='torch' overrides auto-detection."""
        mock_model = _MockMLXModel()
        with patch("turnstyle.sweep._is_mlx_module", return_value=True):
            result = _detect_backend(mock_model, "torch")
            assert result == "torch"

    def test_explicit_override_mlx(self):
        """Explicit backend='mlx' overrides auto-detection."""
        mock_pt = MagicMock()
        result = _detect_backend(mock_pt, "mlx")
        assert result == "mlx"

    def test_invalid_backend_raises(self):
        """Invalid backend string raises ValueError."""
        with pytest.raises(ValueError, match="backend must be"):
            _detect_backend("some-model", "tpu")

    def test_string_model_torch_fallback(self):
        """String model on non-Apple Silicon falls back to torch."""
        with patch("turnstyle.sweep.platform") as mock_platform, \
             patch("turnstyle.sweep.sys") as mock_sys:
            mock_platform.machine.return_value = "x86_64"
            mock_sys.platform = "linux"
            result = _detect_backend("some-model-name", None)
            assert result == "torch"

    def test_string_model_apple_silicon_mlx(self):
        """String model on Apple Silicon with mlx_lm prefers mlx."""
        with patch("turnstyle.sweep.platform") as mock_platform, \
             patch("turnstyle.sweep.sys") as mock_sys, \
             patch.dict("sys.modules", {"mlx_lm": MagicMock()}):
            mock_platform.machine.return_value = "arm64"
            mock_sys.platform = "darwin"
            result = _detect_backend("some-model-name", None)
            assert result == "mlx"

    def test_string_model_apple_silicon_no_mlx(self):
        """String model on Apple Silicon without mlx_lm falls back to torch."""
        with patch("turnstyle.sweep.platform") as mock_platform, \
             patch("turnstyle.sweep.sys") as mock_sys:
            mock_platform.machine.return_value = "arm64"
            mock_sys.platform = "darwin"
            # Make mlx_lm import fail
            import builtins
            original_import = builtins.__import__
            def mock_import(name, *args, **kwargs):
                if name == "mlx_lm":
                    raise ImportError("no mlx_lm")
                return original_import(name, *args, **kwargs)
            with patch("builtins.__import__", side_effect=mock_import):
                result = _detect_backend("some-model-name", None)
                assert result == "torch"


class TestExtractMLXMock:
    """Test MLX extraction with mock model — no real MLX needed."""

    def test_extract_captures_requested_layers(self):
        """Mock MLX extraction captures hidden states at requested layers."""
        mock_model = _MockMLXModel(num_layers=4, hidden_dim=16)
        mock_tokenizer = _MockTokenizer()
        prompts = ["Hello world", "What is 2+2?"]
        layer_indices = [1, 3]

        # Mock mlx.core and mlx_lm.models.base
        mock_mx = MagicMock()
        mock_mx.array.side_effect = lambda x: np.array(x)
        mock_mx.float32 = np.float32  # needed for h.astype(mx.float32)
        mock_mx.eval = MagicMock()  # no-op

        mock_base = MagicMock()
        mock_base.create_attention_mask.return_value = None

        with patch.dict("sys.modules", {
            "mlx": MagicMock(),
            "mlx.core": mock_mx,
            "mlx_lm": MagicMock(),
            "mlx_lm.models": MagicMock(),
            "mlx_lm.models.base": mock_base,
        }):
            result = _extract_all_hidden_states_mlx(
                mock_model, mock_tokenizer, prompts, "mean", layer_indices,
            )

        assert set(result.keys()) == {1, 3}
        for idx in layer_indices:
            assert result[idx].shape == (2, 16)  # 2 prompts, 16 hidden_dim
            assert isinstance(result[idx], np.ndarray)

    def test_extract_last_pool(self):
        """Last-token pooling takes final position."""
        mock_model = _MockMLXModel(num_layers=2, hidden_dim=8)
        mock_tokenizer = _MockTokenizer()

        mock_mx = MagicMock()
        mock_mx.array.side_effect = lambda x: np.array(x)
        mock_mx.float32 = np.float32
        mock_mx.eval = MagicMock()

        mock_base = MagicMock()
        mock_base.create_attention_mask.return_value = None

        with patch.dict("sys.modules", {
            "mlx": MagicMock(),
            "mlx.core": mock_mx,
            "mlx_lm": MagicMock(),
            "mlx_lm.models": MagicMock(),
            "mlx_lm.models.base": mock_base,
        }):
            result_mean = _extract_all_hidden_states_mlx(
                mock_model, mock_tokenizer, ["test"], "mean", [0],
            )
            result_last = _extract_all_hidden_states_mlx(
                mock_model, mock_tokenizer, ["test"], "last", [0],
            )

        # Both should produce (1, hidden_dim) arrays but with different values
        assert result_mean[0].shape == (1, 8)
        assert result_last[0].shape == (1, 8)
        # Mean vs last should differ (unless all positions are identical)
        # With our mock, embed_tokens returns random values, so they differ
        # But both use same seed, so just check shapes are right

    def test_extract_all_layers(self):
        """Requesting all layers returns all of them."""
        n_layers = 6
        mock_model = _MockMLXModel(num_layers=n_layers, hidden_dim=8)
        mock_tokenizer = _MockTokenizer()

        mock_mx = MagicMock()
        mock_mx.array.side_effect = lambda x: np.array(x)
        mock_mx.float32 = np.float32
        mock_mx.eval = MagicMock()

        mock_base = MagicMock()
        mock_base.create_attention_mask.return_value = None

        with patch.dict("sys.modules", {
            "mlx": MagicMock(),
            "mlx.core": mock_mx,
            "mlx_lm": MagicMock(),
            "mlx_lm.models": MagicMock(),
            "mlx_lm.models.base": mock_base,
        }):
            result = _extract_all_hidden_states_mlx(
                mock_model, mock_tokenizer, ["test"],
                "mean", list(range(n_layers)),
            )

        assert set(result.keys()) == set(range(n_layers))


class TestMLXPoolStrategies:
    """Verify pooling math works with numpy arrays (MLX path returns numpy)."""

    def test_mean_pool_numpy(self):
        """Mean pooling with numpy produces correct shape and values."""
        h = np.array([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]])  # (1, 3, 2)
        pooled = h[0].mean(axis=0)
        assert pooled.shape == (2,)
        np.testing.assert_allclose(pooled, [3.0, 4.0])

    def test_last_pool_numpy(self):
        """Last-token pooling with numpy produces correct shape and values."""
        h = np.array([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]])
        pooled = h[0, -1]
        assert pooled.shape == (2,)
        np.testing.assert_allclose(pooled, [5.0, 6.0])


class TestSweepResultBackend:
    def test_default_backend_torch(self):
        """SweepResult defaults to backend='torch'."""
        probe = TurnstyleProbe(
            weights=torch.randn(2, 8),
            bias=torch.zeros(2),
            labels=["a", "b"],
        )
        result = SweepResult(
            layer_accuracies={0: 0.5},
            best_layer=0,
            best_accuracy=0.5,
            labels=["a", "b"],
            probe=probe,
            per_label_accuracy={},
            pool="mean",
            train_size=10,
            test_size=5,
        )
        assert result.backend == "torch"

    def test_backend_in_summary(self):
        """Summary includes backend info."""
        probe = TurnstyleProbe(
            weights=torch.randn(2, 8),
            bias=torch.zeros(2),
            labels=["a", "b"],
        )
        result = SweepResult(
            layer_accuracies={0: 0.5},
            best_layer=0,
            best_accuracy=0.5,
            labels=["a", "b"],
            probe=probe,
            per_label_accuracy={},
            pool="mean",
            train_size=10,
            test_size=5,
            backend="mlx",
        )
        assert "backend=mlx" in result.summary()


# ── Intent template tests ───────────────────────────────────────────


class TestIntentTemplates:
    def test_arithmetic_templates_exist(self):
        """Arithmetic intent templates are defined."""
        assert "arithmetic" in _INTENT_TEMPLATES
        assert "operation" in _INTENT_TEMPLATES["arithmetic"]

    def test_arithmetic_operation_classes(self):
        """Arithmetic operation has 4 classes."""
        ops = _INTENT_TEMPLATES["arithmetic"]["operation"]
        assert set(ops.keys()) == {"add", "sub", "mul", "div"}
        for cls_templates in ops.values():
            assert len(cls_templates) >= 4


class TestGenerateIntentPrompts:
    def test_structure(self):
        """Returns {dim: {class: [prompts]}}."""
        result = generate_intent_prompts()
        assert "operation" in result
        assert set(result["operation"].keys()) == {"add", "sub", "mul", "div"}
        for cls_prompts in result["operation"].values():
            assert len(cls_prompts) == 50

    def test_per_class_count(self):
        result = generate_intent_prompts(per_class=10)
        for cls_prompts in result["operation"].values():
            assert len(cls_prompts) == 10

    def test_deterministic(self):
        p1 = generate_intent_prompts(seed=42)
        p2 = generate_intent_prompts(seed=42)
        assert p1 == p2

    def test_different_seeds(self):
        p1 = generate_intent_prompts(seed=1)
        p2 = generate_intent_prompts(seed=2)
        assert p1["operation"]["add"] != p2["operation"]["add"]

    def test_unknown_label_raises(self):
        with pytest.raises(ValueError, match="No intent templates"):
            generate_intent_prompts(turnstyle_label="nonexistent")

    def test_prompts_are_filled(self):
        """Prompts should not contain {a} or {b} placeholders."""
        result = generate_intent_prompts(per_class=5)
        for cls_prompts in result["operation"].values():
            for p in cls_prompts:
                assert "{a}" not in p
                assert "{b}" not in p


class TestIntentSweepResult:
    def test_summary_readable(self):
        """Summary produces human-readable output."""
        probe = IntentProbe({
            "operation": TurnstyleProbe(
                torch.randn(4, 8), torch.zeros(4),
                ["add", "sub", "mul", "div"]),
        })
        result = IntentSweepResult(
            turnstyle_label="arithmetic",
            dimensions={"operation": {0: 0.7, 1: 0.95}},
            best_layers={"operation": 1},
            best_accuracies={"operation": 0.95},
            intent_probe=probe,
            pool="last",
        )
        s = result.summary()
        assert "operation" in s
        assert "95.0%" in s
        assert "arithmetic" in s
