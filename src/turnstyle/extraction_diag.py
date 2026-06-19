"""extraction_diag — automated diagnose-fix-consumer loop for sentence-IR extraction.

Per-(model, task) extraction tuning is mandatory ([[model_dependent_calibration]]) but it
used to be a from-scratch investigation each time. This runs the extraction path with the
`diag` dict over a batch, **categorizes every failure** into the modes that actually occur
on weak backbones, and reports which one dominates — so tuning becomes "fix the top
category, re-run" instead of eyeballing single examples.

    report = diagnose_extraction(model, tok, dev, examples, COMPARISON_ORDERING_SPEC)
    print(report.summary())
    # → accuracy + a ranked failure breakdown + sample evidence per category

Failure categories (derived from the SmolLM2 logical_deduction failures, 2026-06-18):
- exception            — sentence_ir_solve raised
- empty_entities       — no entity hints extracted (entity_pattern misses the format) →
                         the model then hallucinates; THE highest-leverage fix
- hallucinated_entity  — an extracted subj/obj is not in the prompt (few-shot leakage)
- unparseable_json     — the model's per-sentence output was not valid JSON
- normalization        — extracted name matches no prompt entity even fuzzily (spelling)
- aggregation_no_answer— records extracted but the aggregator found no unique answer
- wrong_answer         — everything parsed, answer is just wrong (representation limit)
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Callable, Optional


@dataclass
class ExtractionFailure:
    index: int
    pred: Optional[str]
    gold: str
    category: str
    evidence: str
    prompt_head: str


@dataclass
class ExtractionReport:
    task: str
    n: int
    correct: int
    categories: Counter
    failures: list  # capped sample of ExtractionFailure, balanced across categories

    @property
    def accuracy(self) -> float:
        return self.correct / self.n if self.n else 0.0

    def top_category(self) -> Optional[str]:
        return self.categories.most_common(1)[0][0] if self.categories else None

    def summary(self) -> str:
        nfail = self.n - self.correct
        lines = [
            f"ExtractionReport(task={self.task}, n={self.n})",
            f"  accuracy: {self.correct}/{self.n} = {self.accuracy:.1%}",
            f"  failure categories ({nfail} failures):",
        ]
        for cat, c in self.categories.most_common():
            frac = c / nfail if nfail else 0.0
            lines.append(f"    {cat:22s} {c:4d}  ({frac:.0%})")
        if self.failures:
            lines.append("  samples:")
            for f in self.failures:
                lines.append(f"    [{f.category}] pred={f.pred} gold={f.gold} | {f.evidence}")
        if self.top_category():
            lines.append(f"  → fix first: {self.top_category()}")
        return "\n".join(lines)


# ── failure categorization ──────────────────────────────────────────────────────

_NAME_RE = re.compile(r'"(?:subj|obj|item)":\s*"([^"]+)"')


def _categorize(prompt: str, ans, gold: str, diag: dict) -> tuple[str, str]:
    """Inspect the diag dict and classify why this example failed."""
    if diag.get("error"):
        return "exception", str(diag["error"])[:100]

    if not diag.get("entities"):
        return "empty_entities", "no entity hints (entity_pattern missed the format)"

    extractions = diag.get("sentence_extractions", [])

    unparsed = [e for e in extractions if not e.get("parsed", True)]
    if unparsed:
        return "unparseable_json", repr(unparsed[0].get("response", ""))[:80]

    prompt_lo = prompt.lower()
    for e in extractions:
        for name in _NAME_RE.findall(e.get("response", "")):
            if name and not name.replace("_", " ").isdigit() and name.lower() not in prompt_lo:
                return "hallucinated_entity", f"{name!r} not in prompt"

    entities_lo = [s.lower() for s in diag.get("entities", [])]
    for e in extractions:
        for name in _NAME_RE.findall(e.get("response", "")):
            nl = name.lower()
            if nl.isdigit():
                continue
            if not any(nl in ent or ent in nl for ent in entities_lo):
                return "normalization", f"{name!r} matches no known entity {entities_lo[:4]}"

    if ans is None:
        return "aggregation_no_answer", f"{len(extractions)} records, no unique answer"

    return "wrong_answer", f"pred={ans} gold={gold}"


# ── the harness ─────────────────────────────────────────────────────────────────

def diagnose_extraction(
    model, tokenizer, device,
    examples: list,
    spec,
    target_fn: Callable = lambda ex: ex["target"].strip(),
    task: str = "",
    limit: Optional[int] = None,
    max_samples_per_cat: int = 3,
    verbose: bool = False,
) -> ExtractionReport:
    """Run sentence-IR extraction over `examples`, categorize every failure, and report.

    `spec` is a SentenceIRSpec (e.g. COMPARISON_ORDERING_SPEC). The report's top category
    is the highest-leverage thing to fix in the consumer before re-running."""
    from turnstyle.ir import sentence_ir_solve

    rows = examples[:limit] if limit else examples
    correct = 0
    cats: Counter = Counter()
    samples: list = []
    for i, ex in enumerate(rows):
        diag: dict = {}
        try:
            ans = sentence_ir_solve(model, tokenizer, device, ex["input"], spec, diag=diag)
        except Exception as e:  # noqa: BLE001 — harness must survive any extraction error
            ans, diag["error"] = None, f"{type(e).__name__}: {e}"
        gold = target_fn(ex)
        if ans == gold:
            correct += 1
            continue
        cat, evidence = _categorize(ex["input"], ans, gold, diag)
        cats[cat] += 1
        if sum(1 for s in samples if s.category == cat) < max_samples_per_cat:
            samples.append(ExtractionFailure(
                index=i, pred=ans, gold=gold, category=cat, evidence=evidence,
                prompt_head=ex["input"][:100],
            ))
        if verbose:
            print(f"  [{i}] {'ok' if ans == gold else cat}", flush=True)

    return ExtractionReport(task=task or getattr(spec, "task", "") or "?",
                            n=len(rows), correct=correct, categories=cats, failures=samples)
