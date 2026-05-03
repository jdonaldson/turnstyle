"""ChoiceProbe — wraps a per-option autoprobe ProbeArtifact.

Self-contained: runs its own position-finding via `artifact.finder` (the
finder autoprobe selected at fit time, e.g. regex `per_option_last_token`),
runs its own forward pass if not already cached, and scores at the
artifact's chosen layer. Emits option facts as a side effect for downstream
primitives that want them (logit poll, knowledge decomposition, etc.).

Selector: `Not(Has("answer"))` — fires whenever no answer exists. The
artifact's finder gates internally: if the prompt isn't multi-choice (or
the finder otherwise returns < 2 positions), emit nothing and fall through.

Why not consume OptionDetector facts: the artifact was fit on hidden states
at finder-specific positions. Reading hidden states at OptionDetector's
structural-probe positions would degrade accuracy on snarks-shaped prompts
where the regex finder and structural probe disagree on token indices.
ChoiceProbe must use the same finder it was fit with. OptionDetector still
has a role for primitives that don't have a fitted finder (LogitPoll).
"""
from __future__ import annotations

import torch

from turnstyle.autoprobe import ProbeArtifact
from turnstyle.blackboard import Blackboard, Has, Not, Primitive


class ChoiceProbe(Primitive):
    """Argmax-over-options probe primitive. Self-sufficient: finds positions
    via `artifact.finder` and scores at `artifact.layer`."""

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
            selector=Not(Has("answer")),
            priority=priority,
        )
        self.artifact = artifact

    def fire(self, state: Blackboard) -> None:
        model = state.context.get("model")
        tokenizer = state.context.get("tokenizer")
        device = state.context.get("device")
        if model is None or tokenizer is None or device is None:
            return

        encoded = state.context.get("encoded")
        if encoded is None:
            encoded = tokenizer(
                state.prompt,
                return_offsets_mapping=True,
                add_special_tokens=True,
            )
            state.context["encoded"] = encoded
        assert encoded is not None

        hidden = state.context.get("hidden_states")
        if hidden is None:
            ids = {
                "input_ids": torch.tensor([encoded["input_ids"]]).to(device),
                "attention_mask": torch.tensor([encoded["attention_mask"]]).to(device),
            }
            with torch.no_grad():
                out = model(**ids, output_hidden_states=True)
            hidden = out.hidden_states
            state.context["hidden_states"] = hidden
        assert hidden is not None

        positions = self.artifact.finder(
            state.prompt, tokenizer, encoded, hidden=hidden,
        )
        if not positions:
            return

        # Emit option facts as a side effect — provenance + downstream
        # primitive composition (e.g. logit poll). These are this probe's
        # view of where options are, not necessarily the structural probe's.
        opt_facts = []
        for tok_idx, label in positions:
            opt_facts.append(state.emit(
                kind="option",
                payload={
                    "label": label,
                    "last_content_token_idx": int(tok_idx),
                    "source_finder": "probe_artifact",
                },
                source=self.name,
            ))

        layer = self.artifact.layer
        h = hidden[layer][0]

        scores: dict[str, float] = {}
        for tok_idx, label in positions:
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
            parent_indices=[f.timestamp for f in opt_facts],
            confidence=scores[best_label],
        )
