"""ChoiceProbe — wraps a per-option autoprobe ProbeArtifact.

Selects the best option by scoring each option's last-content-token through
the artifact's classifier at the artifact's chosen layer. Reads option facts
emitted by OptionDetector (or any other primitive that emits options) and
the cached hidden states left on `state.context` by the upstream forward
pass.

Per-task instantiation: each ChoiceProbe wraps one ProbeArtifact (one task's
fitted probe). swollm/registry.py builds the right ChoiceProbe per task by
calling autoprobe(...) and feeding the resulting ProbeArtifact in.

Single-mode artifacts (the probe scores at the prompt's last token, not
per-option) need a different primitive; ChoiceProbe asserts per_option mode.
"""
from __future__ import annotations

from turnstyle.autoprobe import ProbeArtifact
from turnstyle.blackboard import Blackboard, Has, Primitive


class ChoiceProbe(Primitive):
    """Argmax-over-options probe primitive. Fires when ≥2 option facts exist;
    emits an `answer` fact with mode="choice" and per-option scores in
    payload."""

    def __init__(self, artifact: ProbeArtifact,
                 name: str = "choice_probe",
                 priority: int = 0):
        if artifact.mode != "per_option":
            raise ValueError(
                f"ChoiceProbe requires per_option-mode ProbeArtifact; "
                f"got mode={artifact.mode!r}"
            )
        super().__init__(
            name=name,
            selector=Has("option", min_count=2),
            priority=priority,
        )
        self.artifact = artifact

    def fire(self, state: Blackboard) -> None:
        hidden = state.context.get("hidden_states")
        if hidden is None:
            # No forward pass has been run yet — caller must register an
            # upstream primitive (e.g. OptionDetector) that stashes hidden
            # states on context.
            return

        layer = self.artifact.layer
        h = hidden[layer][0]

        opts = state.has("option")
        scores: dict[str, float] = {}
        for opt in opts:
            label = opt.payload["label"]
            tok_idx = opt.payload["last_content_token_idx"]
            vec = h[tok_idx].float().cpu().numpy()
            prob = self.artifact.classifier.predict_proba(
                self.artifact.scaler.transform([vec])
            )[0, 1]
            scores[label] = float(prob)

        best_label = max(scores, key=lambda k: scores[k])
        answer_str = self.artifact._format(best_label)

        state.emit(
            kind="answer",
            payload={
                "mode": "choice",
                "answer": answer_str,
                "scores": scores,
                "layer": layer,
            },
            source=self.name,
            parent_indices=[o.timestamp for o in opts],
            confidence=scores[best_label],
        )
