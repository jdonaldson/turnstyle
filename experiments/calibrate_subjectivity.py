"""Calibrate + persist the subjectivity BipolarAxis for the hyperbaton solver.

Sweeps candidate layers, measures hyperbaton accuracy (the axis is fit only from
opinion/material seeds — not hyperbaton data — but layer selection touches the
labels, so the picked-layer accuracy is in-sample, like swollm's hyperbaton
k-sweep). Persists the best axis into the fingerprint-keyed profile.

    python -m experiments.calibrate_subjectivity
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from turnstyle.bbh import load_task
from turnstyle.dispatch_turnstyle import DispatchTurnstyle
from turnstyle.hyperbaton import fit_subjectivity_axis, solve_hyperbaton

MODEL = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
LAYERS = [6, 8, 10, 12, 14, 16, 18]


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL)
    mdl = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16).to(device)
    exs = load_task("hyperbaton")
    print(f"hyperbaton N={len(exs)}  device={device}", flush=True)

    best = (None, -1.0, 0.0)  # (layer, acc, coverage)
    for L in LAYERS:
        axis = fit_subjectivity_axis(mdl, tok, device, L)
        correct = committed = 0
        for ex in exs:
            letter = solve_hyperbaton(ex["input"], mdl, tok, device, axis)
            if letter is not None:
                committed += 1
                correct += letter == ex["target"].strip()
        acc = correct / len(exs)
        cov = committed / len(exs)
        print(f"  L{L:2d}: acc={acc*100:5.1f}%  coverage={cov*100:5.1f}%", flush=True)
        if acc > best[1]:
            best = (L, acc, cov)

    L, acc, cov = best
    print(f"\nbest: L{L}  acc={acc*100:.1f}%  coverage={cov*100:.1f}%")
    dt = DispatchTurnstyle(mdl, tok, device)
    dt.calibrate_subjectivity(layer=L, accuracy=acc, verbose=True)
    path = dt.persist()
    print(f"persisted → {path}")


if __name__ == "__main__":
    main()
