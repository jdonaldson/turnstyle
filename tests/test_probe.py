"""Tests for TurnstyleProbe and RoutingTurnstyle.

All tests use mock objects — no model or GPU required.
"""

import tempfile
from unittest.mock import MagicMock, patch

import torch
import pytest

from turnstyle.probe import TurnstyleProbe, IntentProbe, RoutingTurnstyle
from turnstyle.core import Turnstyle


# ── Fixtures ────────────────────────────────────────────────────────────


def make_probe(labels=None, threshold=0.5, hidden_dim=8):
    """Create a probe with known weights for deterministic testing."""
    labels = labels or ["arithmetic", "date"]
    num_types = len(labels)
    # Weights that produce predictable scores:
    # Row 0 (arithmetic): positive weights in first half
    # Row 1 (date): positive weights in second half
    weights = torch.zeros(num_types, hidden_dim)
    half = hidden_dim // 2
    weights[0, :half] = 2.0   # arithmetic responds to first half
    weights[1, half:] = 2.0   # date responds to second half
    bias = torch.zeros(num_types)
    return TurnstyleProbe(weights, bias, labels, threshold)


class MockTurnstyle(Turnstyle):
    """Turnstyle that parses only prompts containing its keyword."""

    def __init__(self, keyword, label, device="cpu"):
        model = MagicMock()
        tokenizer = MagicMock()
        super().__init__(model, tokenizer, device)
        self.keyword = keyword
        self.probe_label = label
        self._generated = False

    def parse(self, prompt: str):
        if self.keyword in prompt.lower():
            return {"matched": self.keyword}
        return None

    def make_processor(self, parsed, max_new_tokens: int):
        return MagicMock()

    def generate(self, prompt: str, max_new_tokens: int = 50):
        self._generated = True
        return f"[{self.probe_label}] answer", None


# ── TurnstyleProbe tests ───────────────────────────────────────────────


class TestTurnstyleProbe:
    def test_predict_above_threshold(self):
        probe = make_probe(threshold=0.5)
        # Hidden state with large values in first half → arithmetic scores high
        h = torch.zeros(8)
        h[:4] = 3.0
        scores = probe.predict(h)
        assert "arithmetic" in scores
        assert scores["arithmetic"] > 0.5

    def test_predict_below_threshold(self):
        probe = make_probe(threshold=0.5)
        # Hidden state near zero → sigmoid(0) ≈ 0.5, use high threshold
        probe.threshold = 0.99
        h = torch.zeros(8)
        scores = probe.predict(h)
        assert len(scores) == 0

    def test_predict_multi_label(self):
        probe = make_probe(threshold=0.5)
        # Large values everywhere → both labels active
        h = torch.ones(8) * 3.0
        scores = probe.predict(h)
        assert "arithmetic" in scores
        assert "date" in scores

    def test_predict_selective(self):
        probe = make_probe(threshold=0.5)
        # Only second half active → only date
        h = torch.zeros(8)
        h[4:] = 3.0
        scores = probe.predict(h)
        assert "date" in scores
        # arithmetic should have low score (sigmoid of 0 ≈ 0.5)
        # With threshold at 0.5, it might just barely be included
        # Use stricter threshold for clean test
        probe.threshold = 0.6
        scores = probe.predict(h)
        assert "arithmetic" not in scores
        assert "date" in scores

    def test_predict_all_ignores_threshold(self):
        probe = make_probe(threshold=0.99)
        h = torch.zeros(8)
        scores = probe.predict_all(h)
        assert "arithmetic" in scores
        assert "date" in scores
        assert len(scores) == 2

    def test_save_load_roundtrip(self):
        probe = make_probe(labels=["arithmetic", "date", "unit"], threshold=0.42)
        h = torch.ones(8) * 2.0
        original_scores = probe.predict_all(h)

        with tempfile.NamedTemporaryFile(suffix=".pt") as f:
            probe.save(f.name)
            loaded = TurnstyleProbe.load(f.name)

        assert loaded.labels == probe.labels
        assert loaded.threshold == probe.threshold
        loaded_scores = loaded.predict_all(h)
        for label in probe.labels:
            assert abs(original_scores[label] - loaded_scores[label]) < 1e-6

    def test_score_ordering(self):
        """Higher activation in a label's region → higher score."""
        probe = make_probe(threshold=0.0)
        h = torch.zeros(8)
        h[:4] = 5.0  # strong arithmetic signal
        h[4:] = 0.5  # weak date signal
        scores = probe.predict_all(h)
        assert scores["arithmetic"] > scores["date"]

    def test_predict_best_returns_top_label(self):
        """predict_best returns the highest-scoring label."""
        probe = make_probe(threshold=0.5)
        h = torch.zeros(8)
        h[:4] = 5.0  # strong arithmetic signal
        label, score = probe.predict_best(h)
        assert label == "arithmetic"
        assert score > 0.5

    def test_predict_best_date(self):
        """predict_best returns date when second half is active."""
        probe = make_probe(threshold=0.5)
        h = torch.zeros(8)
        h[4:] = 5.0  # strong date signal
        label, score = probe.predict_best(h)
        assert label == "date"
        assert score > 0.5


# ── IntentProbe tests ──────────────────────────────────────────────────


class TestIntentProbe:
    def make_intent_probe(self, hidden_dim=8):
        """Create an IntentProbe with two dimensions."""
        # Operation probe: 4 classes (add, sub, mul, div)
        op_weights = torch.zeros(4, hidden_dim)
        op_weights[0, 0] = 3.0  # add
        op_weights[1, 1] = 3.0  # sub
        op_weights[2, 2] = 3.0  # mul
        op_weights[3, 3] = 3.0  # div
        op_probe = TurnstyleProbe(
            op_weights, torch.zeros(4),
            ["add", "sub", "mul", "div"])

        # Operand_a probe: 3 classes
        a_weights = torch.zeros(3, hidden_dim)
        a_weights[0, 4] = 3.0  # "10"
        a_weights[1, 5] = 3.0  # "20"
        a_weights[2, 6] = 3.0  # "30"
        a_probe = TurnstyleProbe(
            a_weights, torch.zeros(3),
            ["10", "20", "30"])

        return IntentProbe({"operation": op_probe, "operand_a": a_probe})

    def test_predict_returns_all_dimensions(self):
        ip = self.make_intent_probe()
        h = torch.zeros(8)
        h[0] = 5.0  # add
        h[5] = 5.0  # operand_a = 20
        result = ip.predict(h)
        assert "operation" in result
        assert "operand_a" in result
        assert result["operation"][0] == "add"
        assert result["operand_a"][0] == "20"

    def test_predict_confidence(self):
        ip = self.make_intent_probe()
        h = torch.zeros(8)
        h[2] = 5.0  # mul
        result = ip.predict(h)
        label, conf = result["operation"]
        assert label == "mul"
        assert conf > 0.9

    def test_save_load_roundtrip(self):
        ip = self.make_intent_probe()
        h = torch.zeros(8)
        h[1] = 5.0  # sub
        h[6] = 5.0  # 30
        original = ip.predict(h)

        with tempfile.TemporaryDirectory() as d:
            ip.save(d)
            loaded = IntentProbe.load(d)

        loaded_result = loaded.predict(h)
        assert loaded_result["operation"][0] == original["operation"][0]
        assert loaded_result["operand_a"][0] == original["operand_a"][0]
        assert abs(loaded_result["operation"][1] - original["operation"][1]) < 1e-6

    def test_dimensions_property(self):
        ip = self.make_intent_probe()
        assert set(ip.dimensions.keys()) == {"operation", "operand_a"}


# ── RoutingTurnstyle tests ─────────────────────────────────────────────


class TestRoutingTurnstyle:
    def make_router(self, probe=None, turnstyles=None):
        """Create a RoutingTurnstyle with mock components."""
        t_arith = MockTurnstyle("what is", "arithmetic")
        t_date = MockTurnstyle("how many days", "date")
        turnstyles = turnstyles or [t_arith, t_date]
        probe = probe or make_probe()
        router = RoutingTurnstyle(
            turnstyles=turnstyles,
            probe=probe,
            layer_index=23,
        )
        return router, turnstyles

    def test_regex_first(self):
        """When regex matches, probe is never consulted."""
        router, turnstyles = self.make_router()
        matches = router.parse("What is 445 + 152?")
        assert matches is not None
        assert len(matches) == 1
        t, parsed = matches[0]
        assert t.probe_label == "arithmetic"

    def test_regex_returns_none_on_miss(self):
        """When no regex matches, parse returns None."""
        router, _ = self.make_router()
        matches = router.parse("Sum of four hundred and one fifty-two")
        assert matches is None

    def test_label_to_turnstyle_mapping(self):
        """Probe labels correctly map to turnstyle instances."""
        router, turnstyles = self.make_router()
        assert "arithmetic" in router.label_to_turnstyle
        assert "date" in router.label_to_turnstyle
        assert router.label_to_turnstyle["arithmetic"] is turnstyles[0]
        assert router.label_to_turnstyle["date"] is turnstyles[1]

    def test_label_fallback_from_classname(self):
        """Without probe_label, label is derived from class name."""
        class FooTurnstyle(Turnstyle):
            def parse(self, prompt): return None
            def make_processor(self, parsed, max_new_tokens): return None

        t = FooTurnstyle(MagicMock(), MagicMock(), "cpu")
        probe = TurnstyleProbe(
            weights=torch.ones(1, 8),
            bias=torch.zeros(1),
            labels=["foo"],
        )
        router = RoutingTurnstyle([t], probe, layer_index=0)
        assert "foo" in router.label_to_turnstyle

    def test_multiple_regex_matches(self):
        """When multiple turnstyles match, all are returned."""
        t1 = MockTurnstyle("test", "arithmetic")
        t2 = MockTurnstyle("test", "date")
        probe = make_probe()
        router = RoutingTurnstyle([t1, t2], probe, layer_index=0)
        matches = router.parse("test prompt")
        assert len(matches) == 2

    def test_generate_regex_path(self):
        """generate() uses regex match when available."""
        router, turnstyles = self.make_router()
        text, diag = router.generate("What is 3 + 4?")
        assert turnstyles[0]._generated
        assert not turnstyles[1]._generated
        assert "[arithmetic]" in text

    def test_generate_probe_fallback(self):
        """generate() falls back to probe when regex fails."""
        router, turnstyles = self.make_router()
        mock_h = torch.zeros(8)

        # Mock the probe route to return arithmetic
        with patch.object(router, '_probe_route',
                          return_value=([(turnstyles[0], 0.9)], mock_h)):
            text, diag = router.generate("Sum of four hundred and one fifty-two")

        assert turnstyles[0]._generated
        assert "[arithmetic]" in text

    def test_generate_no_match_free_generation(self):
        """When both regex and probe fail, model generates freely."""
        router, turnstyles = self.make_router()

        # Mock probe to return empty (no match)
        with patch.object(router, '_probe_route', return_value=([], None)):
            # Mock the model generation pipeline
            mock_output = torch.tensor([[1, 2, 3, 4, 5]])
            router.model.generate = MagicMock(return_value=mock_output)
            router.tokenizer.apply_chat_template = MagicMock(return_value="template")
            router.tokenizer.return_value = {"input_ids": torch.tensor([[1, 2]])}
            router.tokenizer.to = MagicMock(return_value={"input_ids": torch.tensor([[1, 2]])})
            # Make tokenizer callable return a dict-like with .to()
            mock_inputs = MagicMock()
            mock_inputs.__getitem__ = lambda self, k: torch.tensor([[1, 2]])
            mock_inputs.to = MagicMock(return_value=mock_inputs)
            router.tokenizer.return_value = mock_inputs
            router.tokenizer.decode = MagicMock(return_value="free text")

            text, diag = router.generate("Something completely unrelated")

        assert text == "free text"
        assert diag is None
        assert not turnstyles[0]._generated
        assert not turnstyles[1]._generated

    def test_probe_route_delegates_to_highest_scorer(self):
        """Probe fallback delegates to the highest-scoring turnstyle."""
        router, turnstyles = self.make_router()
        mock_h = torch.zeros(8)

        # Probe returns date as higher scorer
        with patch.object(router, '_probe_route',
                          return_value=([(turnstyles[1], 0.95),
                                         (turnstyles[0], 0.7)], mock_h)):
            text, diag = router.generate("Novel phrasing about dates")

        assert turnstyles[1]._generated
        assert not turnstyles[0]._generated
        assert "[date]" in text

    def test_generate_uses_parse_from_hidden(self):
        """When probe routes and parse_from_hidden succeeds, uses processor."""
        router, turnstyles = self.make_router()
        mock_h = torch.zeros(8)

        # Make arithmetic turnstyle's parse_from_hidden return a result
        turnstyles[0].parse_from_hidden = MagicMock(return_value={"matched": True})
        mock_proc = MagicMock()
        mock_proc.proof = None
        turnstyles[0].make_processor = MagicMock(return_value=mock_proc)

        # Mock model generation pipeline
        mock_output = torch.tensor([[1, 2, 3, 4, 5]])
        router.model.generate = MagicMock(return_value=mock_output)
        router.tokenizer.apply_chat_template = MagicMock(return_value="template")
        mock_inputs = MagicMock()
        mock_inputs.__getitem__ = lambda self, k: torch.tensor([[1, 2]])
        mock_inputs.to = MagicMock(return_value=mock_inputs)
        router.tokenizer.return_value = mock_inputs
        router.tokenizer.decode = MagicMock(return_value="probe-parsed result")

        with patch.object(router, '_probe_route',
                          return_value=([(turnstyles[0], 0.9)], mock_h)):
            text, diag = router.generate("Natural language math question")

        # parse_from_hidden was called with the hidden state
        turnstyles[0].parse_from_hidden.assert_called_once_with(mock_h)
        # make_processor was called (not t.generate)
        turnstyles[0].make_processor.assert_called_once()
        assert not turnstyles[0]._generated  # t.generate was NOT called
        assert text == "probe-parsed result"

    def test_generate_falls_through_when_parse_from_hidden_returns_none(self):
        """When parse_from_hidden returns None, falls back to t.generate."""
        router, turnstyles = self.make_router()
        mock_h = torch.zeros(8)

        # parse_from_hidden returns None (not confident enough)
        turnstyles[0].parse_from_hidden = MagicMock(return_value=None)

        with patch.object(router, '_probe_route',
                          return_value=([(turnstyles[0], 0.9)], mock_h)):
            text, diag = router.generate("Natural language math question")

        turnstyles[0].parse_from_hidden.assert_called_once_with(mock_h)
        assert turnstyles[0]._generated  # fell through to t.generate()

    def test_pool_mean(self):
        """Mean pooling averages across positions."""
        router, _ = self.make_router()
        router.pool = "mean"
        hidden = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])  # (1, 2, 2)
        pooled = router._pool_hidden(hidden)
        assert torch.allclose(pooled, torch.tensor([2.0, 3.0]))

    def test_pool_last(self):
        """Last-token pooling takes the final position."""
        router, _ = self.make_router()
        router.pool = "last"
        hidden = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])  # (1, 2, 2)
        pooled = router._pool_hidden(hidden)
        assert torch.allclose(pooled, torch.tensor([3.0, 4.0]))
