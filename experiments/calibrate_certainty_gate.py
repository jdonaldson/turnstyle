"""Calibrate + bundle a CertaintyGate for a backbone (task #2).

`certainty_gate.py` ships the class; this produces the ARTIFACT, so
`install_certainty_gate()` has something to install without a user re-deriving a
threshold.

Held-out discipline, which is the whole point of the offset plumbing added to
`bbh.evaluate`: this calibrates on examples [0:limit] of each task, and the
end-to-end eval must then run with `offset=limit`. Calibrating and evaluating on
the same examples would make any reported accuracy-at-coverage leaked.

The threshold is a coverage QUANTILE, not the value that maximises held-out
accuracy — the latter fits the calibration set and callers have a coverage budget
anyway. AUC is recorded alongside as provenance so a shipped gate can be audited
without re-deriving it.

    uv run python experiments/calibrate_certainty_gate.py --coverage 0.8
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from turnstyle.bbh import load_task  # noqa: E402
from turnstyle.certainty_gate import CertaintyGate, accuracy_at_coverage, save_gate  # noqa: E402
from turnstyle.frame_library import FrameLibrary  # noqa: E402
from turnstyle.ir import parse_scene  # noqa: E402
from turnstyle.profile import model_fingerprint  # noqa: E402

RESULTS = Path(__file__).parent / "results"
BBH = Path.home() / "Projects" / "swollm" / "data" / "bbh_cache"
MODEL = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
# the MC tasks with real correctness variance under per-option scoring; ruin_names
# is excluded on purpose (0.05 — per-option logprob is documented-broken there)
TASKS = "snarks,disambiguation_qa,movie_recommendation,date_understanding,hyperbaton"


def auc(scores, labels):
    s, l = np.asarray(scores, float), np.asarray(labels, int)
    if l.sum() == 0 or l.sum() == len(l):
        return float("nan")
    o = np.argsort(s)
    r = np.empty(len(s), float)
    r[o] = np.arange(1, len(s) + 1)
    n1, n0 = l.sum(), len(l) - l.sum()
    return float((r[l == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


@torch.no_grad()
def coord_and_correct(model, tok, dev, cert, prompt, options, target):
    """Certainty coordinate at the post-answer position, plus whether the
    per-option argmax was right — the same readout the gate uses at runtime."""
    enc = tok(prompt, return_tensors="pt", truncation=True, max_length=1024).to(dev)
    n_prompt = enc["input_ids"].shape[1]
    best, best_k, best_h = -1e9, None, None
    for k, text in options.items():
        full = tok(f"{prompt} {text}", return_tensors="pt",
                   truncation=True, max_length=1024).to(dev)
        if full["input_ids"].shape[1] <= n_prompt:
            continue
        o = model(**full, output_hidden_states=True)
        lp = torch.log_softmax(o.logits[0, :-1].float(), -1)
        tgt = full["input_ids"][0, 1:]
        s = float(lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)[n_prompt - 1:].mean())
        if s > best:
            best, best_k = s, k
            best_h = o.hidden_states[cert.layer][0, -1].float().cpu().numpy()
    if best_k is None:
        return None
    ok = int(best_k.upper() == str(target).strip().strip("()").upper())
    return float(cert.project_norm(best_h)), ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coverage", type=float, default=0.8)
    ap.add_argument("--limit", type=int, default=40, help="calibration examples/task")
    ap.add_argument("--tasks", default=TASKS)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to(dev).eval()
    fp = model_fingerprint(model)

    lib = FrameLibrary.load_npz(glob.glob("src/turnstyle/data/frames/*.npz")[0])
    cert = lib.frames["certainty"]
    print(f"{MODEL}\n  fingerprint {fp}\n  certainty L{cert.layer} cv_r {cert.cv_r:.3f}")
    print(f"  CALIBRATING on examples [0:{args.limit}] — evaluate with "
          f"offset={args.limit} or the number is leaked\n")

    coords, oks = [], []
    for task in args.tasks.split(","):
        task = task.strip()
        n = 0
        for ex in load_task(task, cache_dir=BBH)[: args.limit]:
            sc = parse_scene(ex["input"])
            if len(sc.options) < 2:
                continue
            got = coord_and_correct(model, tok, dev, cert, ex["input"],
                                    sc.options, ex["target"])
            if got is None:
                continue
            coords.append(got[0])
            oks.append(got[1])
            n += 1
        print(f"  {task:24s} n={n}", flush=True)

    coords, oks = np.array(coords), np.array(oks)
    a = auc(coords, oks)
    thr = CertaintyGate.pick_threshold(coords, coverage=args.coverage)
    gate = CertaintyGate.from_frame(cert, threshold=thr, fingerprint=fp,
                                    auc=a, n=len(coords))

    print(f"\n  n={len(coords)}  base acc {oks.mean():.3f}  AUC {a:.3f}")
    print(f"  threshold @ {args.coverage:.0%} coverage = {thr:+.3f}")
    print(f"\n  in-calibration accuracy/coverage (NOT a held-out number):")
    for cov in (1.0, 0.9, 0.8, 0.7, 0.6):
        print(f"    {cov:>5.0%}  {accuracy_at_coverage(coords, oks, cov):.3f}")

    p = save_gate(gate)
    print(f"\nwrote {p}")
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "calibrate_certainty_gate.json").write_text(json.dumps(
        {"model": MODEL, "fingerprint": fp, "coverage": args.coverage,
         "threshold": thr, "auc": a, "n": len(coords),
         "calibration_slice": [0, args.limit]}, indent=2))


if __name__ == "__main__":
    main()
