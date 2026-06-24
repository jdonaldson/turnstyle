"""Shared boilerplate for experiment scripts.

Usage:
    from common import load_hub
    tok, mdl, solvers, hub = load_hub()
"""
from __future__ import annotations

import sys
import torch

sys.path.insert(0, "/Users/jdonaldson/Projects/swollm/src")
sys.path.insert(0, "/Users/jdonaldson/Projects/turnstyle/src")

from transformers import AutoModelForCausalLM, AutoTokenizer
from turnstyle import (
    ArithmeticTurnstyle,
    BooleanTurnstyle,
    ComparisonOrderingTurnstyle,
    DateTurnstyle,
    DyckTurnstyle,
    FormalFallaciesTurnstyle,
    NavigateTurnstyle,
    ObjectTrackingTurnstyle,
    RoutingTurnstyle,
    SortingTurnstyle,
    SQLTurnstyle,
    WebOfLiesTurnstyle,
)

DEFAULT_MODEL = "HuggingFaceTB/SmolLM2-1.7B-Instruct"


def load_model(model_id: str = DEFAULT_MODEL) -> tuple:
    """Load tokenizer and model. Returns (tok, mdl, device)."""
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    mdl = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.float16
    ).to(device).eval()
    return tok, mdl, device


def make_solvers(mdl, tok, device) -> list:
    """Instantiate all 11 core solvers."""
    return [
        ArithmeticTurnstyle(mdl, tok, device),
        BooleanTurnstyle(mdl, tok, device),
        DateTurnstyle(mdl, tok, device),
        DyckTurnstyle(mdl, tok, device),
        SortingTurnstyle(mdl, tok, device),
        FormalFallaciesTurnstyle(mdl, tok, device),
        SQLTurnstyle(mdl, tok, device),
        NavigateTurnstyle(mdl, tok, device),
        WebOfLiesTurnstyle(mdl, tok, device),
        ComparisonOrderingTurnstyle(mdl, tok, device),
        ObjectTrackingTurnstyle(mdl, tok, device),
    ]


def load_hub(model_id: str = DEFAULT_MODEL) -> tuple:
    """Full pipeline: model + solvers + routing hub.

    Returns (tok, mdl, solvers, hub).
    """
    tok, mdl, device = load_model(model_id)
    solvers = make_solvers(mdl, tok, device)
    hub = RoutingTurnstyle.build(solvers, mdl, tok, device=device, verbose=False)
    return tok, mdl, solvers, hub
