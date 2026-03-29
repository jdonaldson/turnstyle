"""Tests for TurnstyleProbe, MultiPositionProbe, MetacognitiveProbe,
StrategyRouter, and RoutingTurnstyle.

All tests use mock objects — no model or GPU required.
"""

import tempfile
from unittest.mock import MagicMock, patch

import torch
import pytest

from turnstyle.probe import (
    LAST_TOKEN,
    MEAN_POOL,
    ExtractionPoint,
    TurnstyleProbe,
    MultiPositionProbe,
    IntentProbe,
    MetacognitiveProbe,
    StrategyRouter,
    RoutingTurnstyle,
)
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


# ── MultiPositionProbe tests ──────────────────────────────────────


def make_multipos_probe(
    labels=None, threshold=0.5, hidden_dim=8,
):
    """Create a MultiPositionProbe with two extraction points.

    Point 0: pos0@L1 — first half of weights respond to this
    Point 1: last@L23 — second half of weights respond to this
    Total feature dim = hidden_dim * 2 (one per extraction point).
    """
    labels = labels or ["pattern_a", "pattern_b", "pattern_c"]
    num_labels = len(labels)
    total_dim = hidden_dim * 2  # two extraction points

    weights = torch.zeros(num_labels, total_dim)
    # pattern_a: responds to first extraction point (pos0@L1), dims 0..hidden_dim
    weights[0, :hidden_dim] = 2.0
    # pattern_b: responds to second extraction point (last@L23), dims hidden_dim..2*hidden_dim
    weights[1, hidden_dim:] = 2.0
    # pattern_c: responds to both equally
    weights[2, :] = 1.0

    bias = torch.zeros(num_labels)
    points = [
        ExtractionPoint(layer=1, position=0),       # pos0@L1
        ExtractionPoint(layer=23, position=LAST_TOKEN),  # last@L23
    ]
    return MultiPositionProbe(weights, bias, labels, points, threshold)


class TestMultiPositionProbe:
    def test_required_layers(self):
        probe = make_multipos_probe()
        assert probe.required_layers == {1, 23}

    def test_assemble_concatenates_correctly(self):
        probe = make_multipos_probe(hidden_dim=4)
        # Mock hidden states: layer 1 and layer 23, each (seq_len=3, hidden_dim=4)
        layer_hidden = {
            1: torch.tensor([[1.0, 2.0, 3.0, 4.0],
                             [5.0, 6.0, 7.0, 8.0],
                             [9.0, 10.0, 11.0, 12.0]]),
            23: torch.tensor([[0.1, 0.2, 0.3, 0.4],
                              [0.5, 0.6, 0.7, 0.8],
                              [0.9, 1.0, 1.1, 1.2]]),
        }
        assembled = probe.assemble(layer_hidden)
        # pos0@L1 = [1,2,3,4], last@L23 = [0.9,1.0,1.1,1.2]
        expected = torch.tensor([1.0, 2.0, 3.0, 4.0, 0.9, 1.0, 1.1, 1.2])
        assert torch.allclose(assembled, expected)

    def test_assemble_mean_pool(self):
        """MEAN_POOL position averages across all token positions."""
        points = [ExtractionPoint(layer=0, position=MEAN_POOL)]
        weights = torch.ones(1, 4)
        probe = MultiPositionProbe(weights, torch.zeros(1), ["x"], points)
        layer_hidden = {
            0: torch.tensor([[2.0, 4.0, 6.0, 8.0],
                             [4.0, 6.0, 8.0, 10.0]]),
        }
        assembled = probe.assemble(layer_hidden)
        expected = torch.tensor([3.0, 5.0, 7.0, 9.0])  # mean of rows
        assert torch.allclose(assembled, expected)

    def test_assemble_position_clamped(self):
        """Position index > seq_len is clamped to last token."""
        points = [ExtractionPoint(layer=0, position=99)]
        weights = torch.ones(1, 4)
        probe = MultiPositionProbe(weights, torch.zeros(1), ["x"], points)
        layer_hidden = {
            0: torch.tensor([[1.0, 2.0, 3.0, 4.0],
                             [5.0, 6.0, 7.0, 8.0]]),
        }
        assembled = probe.assemble(layer_hidden)
        # position 99 clamped to last = [5,6,7,8]
        assert torch.allclose(assembled, torch.tensor([5.0, 6.0, 7.0, 8.0]))

    def test_predict_responds_to_first_point(self):
        """Strong signal at pos0@L1 → pattern_a wins."""
        probe = make_multipos_probe(hidden_dim=4, threshold=0.5)
        layer_hidden = {
            1: torch.tensor([[5.0, 5.0, 5.0, 5.0],  # pos0: strong signal
                             [0.0, 0.0, 0.0, 0.0]]),
            23: torch.tensor([[0.0, 0.0, 0.0, 0.0],
                              [0.0, 0.0, 0.0, 0.0]]),  # last: no signal
        }
        assembled = probe.assemble(layer_hidden)
        label, score = probe.predict_best(assembled)
        assert label == "pattern_a"

    def test_predict_responds_to_second_point(self):
        """Strong signal at last@L23 → pattern_b wins."""
        probe = make_multipos_probe(hidden_dim=4, threshold=0.5)
        layer_hidden = {
            1: torch.tensor([[0.0, 0.0, 0.0, 0.0],
                             [0.0, 0.0, 0.0, 0.0]]),
            23: torch.tensor([[0.0, 0.0, 0.0, 0.0],
                              [5.0, 5.0, 5.0, 5.0]]),  # last: strong signal
        }
        assembled = probe.assemble(layer_hidden)
        label, score = probe.predict_best(assembled)
        assert label == "pattern_b"

    def test_predict_from_layers(self):
        """Full pipeline: layer dict → assemble → predict."""
        probe = make_multipos_probe(hidden_dim=4, threshold=0.5)
        layer_hidden = {
            1: torch.tensor([[5.0, 5.0, 5.0, 5.0],
                             [0.0, 0.0, 0.0, 0.0]]),
            23: torch.tensor([[0.0, 0.0, 0.0, 0.0],
                              [0.0, 0.0, 0.0, 0.0]]),
        }
        scores = probe.predict_from_layers(layer_hidden)
        assert "pattern_a" in scores

    def test_predict_threshold_filters(self):
        """High threshold filters low-confidence labels."""
        probe = make_multipos_probe(hidden_dim=4, threshold=0.99)
        layer_hidden = {
            1: torch.tensor([[0.1, 0.1, 0.1, 0.1],
                             [0.0, 0.0, 0.0, 0.0]]),
            23: torch.tensor([[0.0, 0.0, 0.0, 0.0],
                              [0.0, 0.0, 0.0, 0.0]]),
        }
        assembled = probe.assemble(layer_hidden)
        scores = probe.predict(assembled)
        assert len(scores) == 0  # nothing above 0.99

    def test_predict_all_ignores_threshold(self):
        probe = make_multipos_probe(hidden_dim=4, threshold=0.99)
        layer_hidden = {
            1: torch.zeros(2, 4),
            23: torch.zeros(2, 4),
        }
        assembled = probe.assemble(layer_hidden)
        scores = probe.predict_all(assembled)
        assert len(scores) == 3  # all labels present

    def test_save_load_roundtrip(self):
        probe = make_multipos_probe(hidden_dim=4, threshold=0.42)
        layer_hidden = {
            1: torch.tensor([[3.0, 3.0, 3.0, 3.0],
                             [0.0, 0.0, 0.0, 0.0]]),
            23: torch.tensor([[0.0, 0.0, 0.0, 0.0],
                              [1.0, 1.0, 1.0, 1.0]]),
        }
        assembled = probe.assemble(layer_hidden)
        original_scores = probe.predict_all(assembled)

        with tempfile.NamedTemporaryFile(suffix=".pt") as f:
            probe.save(f.name)
            loaded = MultiPositionProbe.load(f.name)

        assert loaded.labels == probe.labels
        assert loaded.threshold == probe.threshold
        assert len(loaded.extraction_points) == len(probe.extraction_points)
        for orig, load in zip(probe.extraction_points, loaded.extraction_points):
            assert orig.layer == load.layer
            assert orig.position == load.position

        loaded_assembled = loaded.assemble(layer_hidden)
        loaded_scores = loaded.predict_all(loaded_assembled)
        for label in probe.labels:
            assert abs(original_scores[label] - loaded_scores[label]) < 1e-6

    def test_three_extraction_points(self):
        """Probe with 3 extraction points assembles correctly."""
        points = [
            ExtractionPoint(layer=0, position=0),
            ExtractionPoint(layer=12, position=LAST_TOKEN),
            ExtractionPoint(layer=23, position=LAST_TOKEN),
        ]
        hidden_dim = 4
        total_dim = hidden_dim * 3
        weights = torch.zeros(2, total_dim)
        weights[0, :hidden_dim] = 1.0  # responds to L0 pos0
        weights[1, hidden_dim * 2:] = 1.0  # responds to L23 last
        bias = torch.zeros(2)
        probe = MultiPositionProbe(weights, bias, ["a", "b"], points)

        layer_hidden = {
            0: torch.tensor([[5.0, 5.0, 5.0, 5.0]]),
            12: torch.tensor([[0.0, 0.0, 0.0, 0.0]]),
            23: torch.tensor([[0.0, 0.0, 0.0, 0.0]]),
        }
        assembled = probe.assemble(layer_hidden)
        assert assembled.shape == (total_dim,)
        label, _ = probe.predict_best(assembled)
        assert label == "a"  # strong signal at L0 pos0


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


# ── MetacognitiveProbe tests ─────────────────────────────────────────


def make_metacognitive_probe(succeed_weight=3.0, fail_weight=0.0, hidden_dim=8, threshold=0.7):
    """Create a MetacognitiveProbe with known weights.

    succeed_weight controls how strongly the first half of hidden dims
    activates the "succeed" label.
    """
    weights = torch.zeros(2, hidden_dim)
    half = hidden_dim // 2
    weights[0, :half] = fail_weight     # "fail" responds to first half
    weights[1, half:] = succeed_weight  # "succeed" responds to second half
    bias = torch.zeros(2)
    inner = TurnstyleProbe(weights, bias, ["fail", "succeed"])
    return MetacognitiveProbe(inner, threshold=threshold)


class TestMetacognitiveProbe:
    def test_needs_intervention_high_succeed(self):
        """When succeed score > threshold, no intervention needed."""
        gate = make_metacognitive_probe(succeed_weight=3.0, threshold=0.7)
        h = torch.zeros(8)
        h[4:] = 5.0  # strong succeed signal
        needs_help, confidence = gate.needs_intervention(h)
        assert needs_help is False
        assert confidence > 0.7

    def test_needs_intervention_low_succeed(self):
        """When succeed score < threshold, intervention needed."""
        gate = make_metacognitive_probe(succeed_weight=3.0, threshold=0.7)
        h = torch.zeros(8)
        # No activation in succeed region → sigmoid(0) ≈ 0.5 < 0.7
        needs_help, confidence = gate.needs_intervention(h)
        assert needs_help is True
        assert confidence > 0.0

    def test_needs_intervention_uncertain(self):
        """Near-0.5 scores default to intervention."""
        gate = make_metacognitive_probe(succeed_weight=0.1, threshold=0.7)
        h = torch.ones(8) * 0.1  # weak signal everywhere
        needs_help, _ = gate.needs_intervention(h)
        assert needs_help is True  # uncertain → intervene

    def test_save_load_roundtrip(self):
        gate = make_metacognitive_probe(succeed_weight=2.5, threshold=0.6)
        h = torch.zeros(8)
        h[4:] = 4.0
        orig_needs, orig_conf = gate.needs_intervention(h)

        with tempfile.NamedTemporaryFile(suffix=".pt") as f:
            gate.save(f.name)
            # Threshold is now persisted — no need to pass it
            loaded = MetacognitiveProbe.load(f.name)

        assert loaded.threshold == 0.6
        loaded_needs, loaded_conf = loaded.needs_intervention(h)
        assert orig_needs == loaded_needs
        assert abs(orig_conf - loaded_conf) < 1e-6

    def test_save_load_threshold_override(self):
        """Explicit threshold on load overrides saved value."""
        gate = make_metacognitive_probe(succeed_weight=2.5, threshold=0.6)

        with tempfile.NamedTemporaryFile(suffix=".pt") as f:
            gate.save(f.name)
            loaded = MetacognitiveProbe.load(f.name, threshold=0.9)

        assert loaded.threshold == 0.9

    def test_threshold_adjustable(self):
        """Different thresholds change gate behavior on same input."""
        h = torch.zeros(8)
        h[4:] = 0.1  # weak succeed signal → sigmoid(1.0*0.1*4=0.4) ≈ 0.60

        # Low threshold → no intervention (0.60 > 0.3)
        gate_low = make_metacognitive_probe(succeed_weight=1.0, threshold=0.3)
        needs_help_low, _ = gate_low.needs_intervention(h)

        # High threshold → intervention needed (0.60 < 0.999)
        gate_high = make_metacognitive_probe(succeed_weight=1.0, threshold=0.999)
        needs_help_high, _ = gate_high.needs_intervention(h)

        assert needs_help_low is False
        assert needs_help_high is True


# ── StrategyRouter tests ─────────────────────────────────────────────


def make_strategy_probe(hidden_dim=8, succeed_dims=None, weight=3.0):
    """Create a binary succeed/fail probe for a strategy."""
    weights = torch.zeros(2, hidden_dim)
    succeed_dims = succeed_dims or [4, 5, 6, 7]
    for d in succeed_dims:
        weights[1, d] = weight  # "succeed" responds to these dims
    bias = torch.zeros(2)
    return TurnstyleProbe(weights, bias, ["fail", "succeed"])


class TestStrategyRouter:
    def test_route_picks_best_strategy(self):
        """Routes to the strategy with highest succeed score."""
        router = StrategyRouter(default_strategy="baseline")
        # Strategy A: responds to dims 0-3
        router.add_strategy("sql", make_strategy_probe(succeed_dims=[0, 1, 2, 3]))
        # Strategy B: responds to dims 4-7
        router.add_strategy("regex", make_strategy_probe(succeed_dims=[4, 5, 6, 7]))

        h = torch.zeros(8)
        h[:4] = 5.0  # strong signal in sql region
        name, conf = router.route(h)
        assert name == "sql"
        assert conf > 0.5

    def test_route_default_when_empty(self):
        """No strategies → returns default."""
        router = StrategyRouter(default_strategy="baseline")
        h = torch.ones(8)
        name, conf = router.route(h)
        assert name == "baseline"
        assert conf == 0.0

    def test_add_multiple_strategies(self):
        """Multiple probes coexist correctly."""
        router = StrategyRouter()
        router.add_strategy("a", make_strategy_probe(succeed_dims=[0, 1]))
        router.add_strategy("b", make_strategy_probe(succeed_dims=[2, 3]))
        router.add_strategy("c", make_strategy_probe(succeed_dims=[4, 5]))
        assert len(router.strategies) == 3

        # Activate dim 4-5 → strategy c wins
        h = torch.zeros(8)
        h[4:6] = 5.0
        name, _ = router.route(h)
        assert name == "c"

    def test_save_load_roundtrip(self):
        router = StrategyRouter(default_strategy="fallback")
        router.add_strategy("sql", make_strategy_probe(succeed_dims=[0, 1, 2, 3]))
        router.add_strategy("regex", make_strategy_probe(succeed_dims=[4, 5, 6, 7]))

        h = torch.zeros(8)
        h[:4] = 5.0
        orig_name, orig_conf = router.route(h)

        with tempfile.TemporaryDirectory() as d:
            router.save(d)
            loaded = StrategyRouter.load(d)

        assert loaded.default_strategy == "fallback"
        loaded_name, loaded_conf = loaded.route(h)
        assert orig_name == loaded_name
        assert abs(orig_conf - loaded_conf) < 1e-6


# ── RoutingTurnstyle + metacognitive gate tests ──────────────────────


class TestRoutingWithGate:
    def make_router(self):
        """Create a RoutingTurnstyle with mock components."""
        t_arith = MockTurnstyle("what is", "arithmetic")
        t_date = MockTurnstyle("how many days", "date")
        turnstyles = [t_arith, t_date]
        probe = make_probe()
        router = RoutingTurnstyle(
            turnstyles=turnstyles,
            probe=probe,
            layer_index=23,
        )
        return router, turnstyles

    def _mock_free_generate(self, router):
        """Set up mocks so _free_generate returns predictable output."""
        mock_output = torch.tensor([[1, 2, 3, 4, 5]])
        router.model.generate = MagicMock(return_value=mock_output)
        router.tokenizer.apply_chat_template = MagicMock(return_value="template")
        mock_inputs = MagicMock()
        mock_inputs.__getitem__ = lambda self, k: torch.tensor([[1, 2]])
        mock_inputs.to = MagicMock(return_value=mock_inputs)
        router.tokenizer.return_value = mock_inputs
        router.tokenizer.decode = MagicMock(return_value="free generation")

    def test_gate_skips_turnstyle_when_model_confident(self):
        """Metacognitive probe says succeed → skip turnstyle, free generate."""
        router, turnstyles = self.make_router()

        # Attach a gate that always says "model will succeed"
        gate = MagicMock()
        gate.needs_intervention = MagicMock(return_value=(False, 0.95))
        turnstyles[0].metacognitive_probe = gate

        self._mock_free_generate(router)

        # Mock _extract_hidden since we don't have a real model
        with patch.object(router, '_extract_hidden', return_value=torch.zeros(8)):
            text, diag = router.generate("What is 3 + 4?")

        assert not turnstyles[0]._generated  # turnstyle was skipped
        assert text == "free generation"
        assert diag is None

    def test_gate_applies_turnstyle_when_model_uncertain(self):
        """Metacognitive probe says fail → turnstyle applied as normal."""
        router, turnstyles = self.make_router()

        # Attach a gate that says "model needs help"
        gate = MagicMock()
        gate.needs_intervention = MagicMock(return_value=(True, 0.8))
        turnstyles[0].metacognitive_probe = gate

        with patch.object(router, '_extract_hidden', return_value=torch.zeros(8)):
            text, diag = router.generate("What is 3 + 4?")

        assert turnstyles[0]._generated  # turnstyle was used
        assert "[arithmetic]" in text

    def test_no_gate_applies_turnstyle_normally(self):
        """Without metacognitive probe, regex match → turnstyle as before."""
        router, turnstyles = self.make_router()
        # No gate set (default)
        text, diag = router.generate("What is 3 + 4?")
        assert turnstyles[0]._generated
        assert "[arithmetic]" in text

    def test_strategy_router_selects_strategy(self):
        """Strategy router picks between strategies on probe fallback."""
        router, turnstyles = self.make_router()
        mock_h = torch.zeros(8)

        # Set up a strategy router that prefers "date"
        strategy_router = MagicMock()
        strategy_router.route = MagicMock(return_value=("date", 0.9))
        router.strategy_router = strategy_router

        with patch.object(router, '_probe_route',
                          return_value=([(turnstyles[0], 0.8)], mock_h)):
            text, diag = router.generate("Some ambiguous prompt")

        # Strategy router rerouted from arithmetic to date
        assert turnstyles[1]._generated
        assert "[date]" in text

    def test_strategy_router_ignored_when_strategy_unknown(self):
        """Strategy router returns unknown label → falls through to probe pick."""
        router, turnstyles = self.make_router()
        mock_h = torch.zeros(8)

        strategy_router = MagicMock()
        strategy_router.route = MagicMock(return_value=("unknown_strategy", 0.9))
        router.strategy_router = strategy_router

        with patch.object(router, '_probe_route',
                          return_value=([(turnstyles[0], 0.8)], mock_h)):
            text, diag = router.generate("Some prompt")

        # Falls through to probe's pick (arithmetic)
        assert turnstyles[0]._generated

    def test_gate_on_probe_fallback_path(self):
        """Metacognitive gate checked on probe-routed turnstyle too."""
        router, turnstyles = self.make_router()
        mock_h = torch.zeros(8)

        # Attach gate to arithmetic that says "model will succeed"
        gate = MagicMock()
        gate.needs_intervention = MagicMock(return_value=(False, 0.95))
        turnstyles[0].metacognitive_probe = gate

        self._mock_free_generate(router)

        with patch.object(router, '_probe_route',
                          return_value=([(turnstyles[0], 0.9)], mock_h)):
            text, diag = router.generate("Natural language math question")

        assert not turnstyles[0]._generated  # skipped by gate
        assert text == "free generation"


# ── RoutingTurnstyle + MultiPositionProbe integration ───────────────


class TestRoutingWithMultiPosition:
    def make_multipos_router(self):
        """Create a RoutingTurnstyle with a MultiPositionProbe."""
        t_arith = MockTurnstyle("what is", "arithmetic")
        t_date = MockTurnstyle("how many days", "date")
        turnstyles = [t_arith, t_date]

        # MultiPositionProbe: pos0@L1 + last@L23
        hidden_dim = 8
        total_dim = hidden_dim * 2
        weights = torch.zeros(2, total_dim)
        weights[0, :hidden_dim] = 2.0   # arithmetic: responds to pos0@L1
        weights[1, hidden_dim:] = 2.0   # date: responds to last@L23
        bias = torch.zeros(2)
        points = [
            ExtractionPoint(layer=1, position=0),
            ExtractionPoint(layer=23, position=LAST_TOKEN),
        ]
        probe = MultiPositionProbe(
            weights, bias, ["arithmetic", "date"], points, threshold=0.5,
        )

        router = RoutingTurnstyle(
            turnstyles=turnstyles,
            probe=probe,
            layer_index=23,  # for downstream (metacognitive gate, parse_from_hidden)
        )
        return router, turnstyles

    def test_regex_still_works(self):
        """Regex path is unaffected by MultiPositionProbe."""
        router, turnstyles = self.make_multipos_router()
        text, diag = router.generate("What is 3 + 4?")
        assert turnstyles[0]._generated
        assert "[arithmetic]" in text

    def test_probe_fallback_uses_multi_position(self):
        """Probe fallback assembles features from multiple extraction points."""
        router, turnstyles = self.make_multipos_router()

        # Mock the forward pass: _probe_route calls _install_hooks, model(),
        # then assembles. We mock at the _probe_route level to verify it
        # returns the right candidate based on multi-position features.
        mock_h = torch.zeros(8)  # downstream hidden state

        # Simulate: strong signal at pos0@L1 → arithmetic wins
        with patch.object(
            router, '_probe_route',
            return_value=([(turnstyles[0], 0.9)], mock_h),
        ):
            text, diag = router.generate("Sum of four hundred and one fifty-two")

        assert turnstyles[0]._generated
        assert "[arithmetic]" in text

    def test_probe_route_installs_multi_hooks(self):
        """_probe_route installs hooks on all required layers."""
        router, turnstyles = self.make_multipos_router()

        # Track which layers get hooked
        hooked_layers = []
        original_install = router._install_hooks

        def tracking_install(layers=None):
            if layers:
                hooked_layers.extend(layers)
            original_install(layers)

        # Mock model forward pass
        mock_output = MagicMock()
        router.model.return_value = mock_output
        router.tokenizer.apply_chat_template = MagicMock(return_value="template")
        mock_inputs = MagicMock()
        mock_inputs.__getitem__ = lambda self, k: torch.tensor([[1, 2]])
        mock_inputs.to = MagicMock(return_value=mock_inputs)
        router.tokenizer.return_value = mock_inputs

        # Pre-populate captured hidden states (since model is mock)
        def fake_install(layers=None):
            hooked_layers.clear()
            if layers:
                hooked_layers.extend(layers)
            router._captured_hidden = {
                1: torch.zeros(1, 3, 8),   # layer 1: (batch, seq, dim)
                23: torch.zeros(1, 3, 8),  # layer 23
            }

        with patch.object(router, '_install_hooks', side_effect=fake_install):
            with patch.object(router, '_remove_hooks'):
                candidates, h = router._probe_route("test prompt")

        # Both layer 1 and 23 should be requested
        assert 1 in hooked_layers
        assert 23 in hooked_layers

    def test_downstream_hidden_from_layer_index(self):
        """The returned hidden state comes from layer_index, not the assembled vector."""
        router, turnstyles = self.make_multipos_router()

        # Pre-populate captured hidden states with distinct values per layer
        layer1_hidden = torch.ones(1, 3, 8) * 1.0  # layer 1
        layer23_hidden = torch.ones(1, 3, 8) * 7.0  # layer 23

        def fake_install(layers=None):
            router._captured_hidden = {
                1: layer1_hidden,
                23: layer23_hidden,
            }

        router.tokenizer.apply_chat_template = MagicMock(return_value="template")
        mock_inputs = MagicMock()
        mock_inputs.__getitem__ = lambda self, k: torch.tensor([[1, 2]])
        mock_inputs.to = MagicMock(return_value=mock_inputs)
        router.tokenizer.return_value = mock_inputs

        with patch.object(router, '_install_hooks', side_effect=fake_install):
            with patch.object(router, '_remove_hooks'):
                router.model.return_value = MagicMock()
                candidates, h = router._probe_route("test")

        # h should be pooled from layer_index=23, not assembled
        # Default pool="mean", so mean of (1, 3, 8) where all values are 7.0
        assert h is not None
        assert h.shape == (8,)
        assert torch.allclose(h, torch.tensor([7.0] * 8))
