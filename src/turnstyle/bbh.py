"""Turnstyle-native BBH evaluation — end-to-end grounded generation.

Self-contained (no swollm dependency): runs `DispatchTurnstyle.generate` over the
BBH cache shipped under `turnstyle/data/bbh_cache`, extracts the emitted answer,
and scores it against the target.

This measures what DispatchTurnstyle *actually does* end-to-end (zero-shot chat +
logit-bias grounding), not a constrained solver-accuracy proxy. The report
separates two regimes per task:

  - committed: dispatch.run() returned an Answer → biased generation (grounded)
  - abstained: fell through to plain chat generation (the base Turnstyle path)

so the gap between turnstyle's native number and a fuller harness is legible:
abstained tasks are the ones missing an ADT variant / calibrated probe.

NOTE: zero-shot chat + grounding is NOT directly comparable to swollm's 3-shot
87.84% — abstained tasks here run zero-shot and will score low. That separation
is the point of this harness, not a defect.

    python -m turnstyle.bbh                       # all tasks, limit 40/task
    python -m turnstyle.bbh --tasks snarks --limit 0   # one task, all examples
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

_PAREN = re.compile(r"\(\s*([A-Za-z])\s*\)")


def default_cache_dir() -> Path:
    """Locate the BBH cache: env override, else the most-complete of the packaged
    and repo-root candidates (the packaged dir ships only a test subset)."""
    env = os.environ.get("TURNSTYLE_BBH_CACHE")
    if env:
        return Path(env)
    here = Path(__file__).resolve()
    candidates = [here.parent / "data" / "bbh_cache",
                  here.parents[2] / "data" / "bbh_cache"]
    existing = [c for c in candidates if c.is_dir()]
    if not existing:
        return candidates[0]
    return max(existing, key=lambda c: len(list(c.glob("*.json"))))


def list_tasks(cache_dir: Path | None = None) -> list[str]:
    cache_dir = cache_dir or default_cache_dir()
    return sorted(p.stem for p in cache_dir.glob("*.json"))


def load_task(name: str, cache_dir: Path | None = None) -> list[dict]:
    cache_dir = cache_dir or default_cache_dir()
    data = json.loads((cache_dir / f"{name}.json").read_text())
    return data["examples"] if isinstance(data, dict) else data


def answer_matches(generated: str, target: str) -> bool:
    """Did the (free-form) generation answer the target? Tolerant matcher tuned
    to BBH target shapes: option letters '(A)', yes/no/true/valid words, integers,
    and exact/leading text for multi-word answers (word_sorting, dyck)."""
    g, t = generated.strip(), target.strip()
    if not g:
        return False
    if g == t or g.lower() == t.lower():
        return True

    # option-letter targets like "(A)"
    m = _PAREN.fullmatch(t)
    if m:
        letter = m.group(1).lower()
        found = _PAREN.search(g)
        if found:
            return found.group(1).lower() == letter
        bare = re.match(r"[^A-Za-z]*([A-Za-z])\b", g)   # leading bare letter
        return bool(bare) and bare.group(1).lower() == letter

    # integer target → first integer in the generation
    if re.fullmatch(r"-?\d+", t):
        nums = re.findall(r"-?\d+", g)
        return bool(nums) and nums[0] == t

    # single-word target (Yes/No/True/valid/...) → first word of generation
    if re.fullmatch(r"[A-Za-z]+", t):
        first = re.match(r"[^A-Za-z]*([A-Za-z]+)", g.lower())
        return bool(first) and first.group(1) == t.lower()

    # multi-word / structured target → leading match
    return g.lower().startswith(t.lower())


def evaluate(dt, tasks=None, cache_dir=None, limit=40, max_new_tokens=50,
             verbose_first=3, progress=True):
    """Run DispatchTurnstyle end-to-end over BBH tasks.

    limit: examples per task (0 = all). Returns {task: stats} with an
    '_aggregate' entry. Inspectable: prints the first `verbose_first` examples of
    each task verbosely, then a per-task summary line."""
    cache_dir = Path(cache_dir) if cache_dir else default_cache_dir()
    names = tasks or list_tasks(cache_dir)
    report = {}
    total_correct = total_n = 0

    for name in names:
        examples = load_task(name, cache_dir)
        if limit:
            examples = examples[:limit]
        # Activate this task's calibrated MC probe from the profile, if any
        # (one choice_artifact slot; reset so a prior task's probe can't leak).
        dt.ctx.choice_artifact = None
        used_probe = dt.use_probe(name)
        t0 = time.time()
        correct = committed = committed_correct = 0
        for i, ex in enumerate(examples):
            parsed = dt.parse(ex["input"])           # Answer | None (committed?)
            gen, _ = dt.generate(ex["input"], max_new_tokens=max_new_tokens)
            ok = answer_matches(gen, ex["target"].strip())
            correct += ok
            if parsed is not None:
                committed += 1
                committed_correct += ok
            if progress and i < verbose_first:
                src = getattr(parsed, "source", None) or "abstain→gen"
                print(f"  [{name} {i}] src={src:14s} "
                      f"gen={gen[:40]!r} tgt={ex['target']!r} "
                      f"{'✓' if ok else '✗'}", flush=True)
        n = len(examples)
        acc = correct / n if n else 0.0
        report[name] = {
            "accuracy": acc, "correct": correct, "n": n,
            "committed": committed,
            "committed_acc": committed_correct / committed if committed else None,
            "abstained": n - committed,
            "used_probe": used_probe,
            "elapsed": time.time() - t0,
        }
        total_correct += correct
        total_n += n
        if progress:
            r = report[name]
            ca = "n/a" if r["committed_acc"] is None else f"{r['committed_acc']*100:.1f}%"
            print(f"{name:42s} acc={acc*100:5.1f}%  "
                  f"committed={committed}/{n} (grounded_acc={ca})  "
                  f"{r['elapsed']:.1f}s", flush=True)

    report["_aggregate"] = {
        "accuracy": total_correct / total_n if total_n else 0.0,
        "correct": total_correct, "n": total_n, "n_tasks": len(names),
    }
    return report


def _load_model(model_id, device):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if device == "auto":
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model_id)
    mdl = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float16).to(device)
    return mdl, tok, device


def main(argv=None):
    import argparse
    p = argparse.ArgumentParser(description="Turnstyle-native BBH eval (end-to-end)")
    p.add_argument("--model", default="HuggingFaceTB/SmolLM2-1.7B-Instruct")
    p.add_argument("--device", default="auto")
    p.add_argument("--tasks", help="comma-separated task filter")
    p.add_argument("--limit", type=int, default=40, help="examples/task (0=all)")
    p.add_argument("--max-new-tokens", type=int, default=50)
    p.add_argument("--output", help="write report JSON here")
    args = p.parse_args(argv)

    from turnstyle.dispatch_turnstyle import DispatchTurnstyle
    mdl, tok, device = _load_model(args.model, args.device)
    print(f"Model: {args.model}  Device: {device}", flush=True)
    dt = DispatchTurnstyle(mdl, tok, device)
    print(f"Profile loaded: {dt.profile is not None}  "
          f"polarity: {dt.ctx.polarity_probe is not None}  "
          f"probe_tasks: {dt.profile_tasks}", flush=True)

    tasks = args.tasks.split(",") if args.tasks else None
    report = evaluate(dt, tasks, limit=args.limit,
                      max_new_tokens=args.max_new_tokens)

    agg = report["_aggregate"]
    print("-" * 70)
    print(f"AGGREGATE: {agg['accuracy']*100:.2f}%  "
          f"({agg['correct']}/{agg['n']} over {agg['n_tasks']} tasks)")
    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=2))
        print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
