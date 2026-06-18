# Composition & Activation-Based Routing Plan

**Status**: planning | **Created**: 2026-04-07 | **Source**: dyf memory `turnstyle_composition.md`

Consolidated plan covering three levels of turnstyle composability plus the activation-based routing experiment that replaces regex routing with hidden-state probes.

---

## Motivation

Turnstyle currently routes prompts to solvers via regex in each turnstyle's `parse()` method. This has three structural limits:

1. **Single-answer assumption**: one prompt → one processor → one answer. Can't handle "What is 445+152, and how many days between Jan 1 and Mar 20?"
2. **Oracle rigidity**: `DateTurnstyle.parse()` hard-codes date arithmetic patterns. "How many days until one day after thanksgiving?" is whack-a-mole because resolve→offset→diff isn't decomposed.
3. **Brittle trigger detection**: regex-based "is"/"="/"equals" matching misses novel phrasings like "the answer comes out to" or "that gives us".

The model already has the routing signal in its hidden states (SmolLM2 layer analysis: task type resolves at L23-24). The plan below attacks all three limits without abandoning the turnstyle architecture.

---

## Level 1 — Generation-Time Composition (CompositeTurnstyle)

**Problem**: Multiple independent questions in one prompt.

### Design

- `CompositeTurnstyle(turnstyles: list[Turnstyle])` wraps N independent processors
- `ChainedLogitsProcessor` sequentially hands off: when processor N reaches DONE, processor N+1 transitions to WAITING
- Each child processor retains its own state machine; the composite tracks which child is active
- Parse phase fans out: call `turnstyle.parse(prompt)` on each child, keep the ones that matched, order them by first-match position in the prompt

### Implementation Checklist

- [ ] `src/turnstyle/composite.py` — `CompositeTurnstyle`, `ChainedLogitsProcessor`
- [ ] `src/turnstyle/__init__.py` — export `CompositeTurnstyle`
- [ ] `tests/test_composite.py` — sequential handoff, mixed-domain prompt, partial-match graceful fallback
- [ ] `README.md` — worked example
- [ ] `CHANGELOG.md` — entry under Unreleased

### Acceptance Criteria

- Prompt "What is 445+152, and how many days between Jan 1 and Mar 20?" generates both answers correctly
- Single-turnstyle prompts behave identically to pre-composite behavior
- Order-independent: prompts with "date first, math second" work equally well
- No regression on existing `tests/` suite

### Status

Implementation was started and interrupted. `src/turnstyle/composite.py` does not exist yet. Resume by re-reading the approved plan (search `~/.claude/plans/` for CompositeTurnstyle).

---

## Level 2 — Oracle-Time Composition (Composable Resolvers)

**Problem**: Single dependent computation with chained stages. "How many days until one day after thanksgiving?" = resolve holiday → apply offset → count days.

### Options Considered

| Option | Approach | Trade-off |
|---|---|---|
| A | Expand regex in `parse_date_arithmetic()` | Whack-a-mole; new pattern = new regex forever |
| **B** | **Composable resolver stages** | **Preferred — factored pipeline, one-time refactor** |
| C | Cross-turnstyle DAG oracle | Speculative, no concrete use case yet |

### Option B Design (Preferred)

Factor `DateTurnstyle` oracle into pure-function stages:

```python
# src/turnstyle/resolvers/date.py
def resolve_date(token: str, today: date) -> date: ...
def apply_offset(anchor: date, offset_phrase: str) -> date: ...
def compute_diff(a: date, b: date, unit: str = "days") -> int: ...
```

`DateTurnstyle.parse()` builds a mini-pipeline from prompt contents:

```python
def parse(self, prompt: str) -> OracleResult | None:
    stages = []
    if match := HOLIDAY_RE.search(prompt):
        stages.append(("resolve", match.group(1)))
    if match := OFFSET_RE.search(prompt):
        stages.append(("offset", match.group(1)))
    if "days until" in prompt or "days between" in prompt:
        stages.append(("diff", "days"))
    return self._run_pipeline(stages) if stages else None
```

### Implementation Checklist

- [ ] `src/turnstyle/resolvers/` package with `date.py` holding pure-function stages
- [ ] Refactor `DateTurnstyle.parse()` to build + execute resolver pipelines
- [ ] Keep existing regex fast paths as a `_fast_path()` pre-check so common cases don't lose perf
- [ ] `tests/test_date_resolvers.py` — unit tests on each stage, plus integration tests on compound prompts
- [ ] Test "one day after thanksgiving", "three weeks before Easter", "days from Christmas to New Year's"

### Acceptance Criteria

- "How many days until one day after thanksgiving?" returns correct answer
- Existing date task accuracy holds (no regression on BBH `date_understanding`)
- Adding a new date pattern touches ≤1 regex and 0 new resolvers in the common case
- Turnstyle architecture unchanged outside `DateTurnstyle` internals

### Non-Goals

- Do **not** generalize to cross-turnstyle composition (that's Option C, speculative)
- Do **not** replace regex entirely — keep it as the fast path

---

## Level 3 — Activation-Based Routing (Quick Win)

**Problem**: `Turnstyle.parse()` methods use regex to detect which solver applies. Regex misses novel phrasings. The model's L23-24 hidden states already separate task types cleanly — use them instead.

### What SmolLM2 Layer Analysis Showed (dyf memory)

- **Task type resolves at L23-24**: arithmetic vs date question is linearly separable
- **Semantic function peaks L2-8**: operand binding happens early
- **Logit lens fails on L3-25**: ~50% of hidden state magnitude is a constant FFFD-direction bias. Must use RMSNorm-normalized states, not raw residual stream.

### Architecture

```
Prompt → Model forward → h[L23-24] → Probe → turnstyle subset → parse()
```

The probe **replaces routing**, not extraction. Operand extraction (e.g., pulling "445" and "152" out of an arithmetic prompt) still uses regex or attention-based decoders. Clean split:

| Stage | Mechanism |
|---|---|
| Routing ("which turnstyle applies?") | **L23-24 linear probe** |
| Extraction ("what are the operands?") | Regex / attention decoder |
| Computation ("445+152=597") | Oracle (pure Python) |
| Generation trigger ("emit it now") | Forward hook on activations (later) |

### Experiment Plan

**Phase A — Data collection**
- [ ] `experiments/routing_probe_data.py` — collect ~100-200 labeled prompts
- [ ] Categories: arithmetic, date_arithmetic, boolean, navigation, web_of_lies, object_counting, none/passthrough
- [ ] Multi-label: a prompt can fire multiple turnstyles (feeds into CompositeTurnstyle above)
- [ ] Source from BBH task prompts + synthetic variations + adversarial paraphrases
- [ ] Store: `experiments/routing_probe_data.npz` with `prompts`, `labels` (multi-hot), `splits`

**Phase B — Feature extraction**
- [ ] Run model forward on each prompt, capture `h[L23]` and `h[L24]`
- [ ] Pool: mean over last K tokens (K=4 default), also try last-token-only
- [ ] Feature variants to compare:
  - `h[L23]` pooled
  - `h[L24]` pooled
  - `h[L24] - h[L23]` (task crystallization delta, untested but medium-high confidence)
  - `concat(h[L23], h[L24])`
- [ ] Cache to `experiments/routing_probe_features.npz`

**Phase C — Probe training**
- [ ] Train multi-label logistic regression (one-vs-rest) on each feature variant
- [ ] 5-fold CV, report per-class F1 and macro-F1
- [ ] Baseline: regex `_detect_task` heuristic (current routing)
- [ ] Also try dyf tree on L23 activations (hierarchical routing, interpretable splits)

**Phase D — Integration**
- [ ] If macro-F1 ≥ regex baseline: wire probe into `StrategyRouter` or new `ActivationRouter`
- [ ] Promote to `src/turnstyle/routing_probe.py`
- [ ] Hybrid routing: probe primary, regex fallback on low-confidence predictions (< threshold)
- [ ] `tests/test_routing_probe.py`

### Acceptance Criteria

- Macro-F1 on held-out prompts ≥ regex baseline
- Novel-phrasing prompts (not in training) routed correctly at ≥70% accuracy
- Latency overhead per prompt < 20ms on SmolLM2-1.7B (single forward pass already happens)
- No regression on existing turnstyle solver accuracy

### Confidence Ranking (from operator-binding dead end)

| Approach | Confidence | Source |
|---|---|---|
| Linear probe on L23 | **High** | measured, 67% task-type accuracy baseline |
| `h[L24] - h[L23]` delta | Medium-high | task crystallization direction, untested |
| DYF tree on L23 activations | Medium | tooling exists, interpretability bonus |
| Operator binding ratio | **Dead end** | see below |

---

## Dead Ends (Do Not Repeat)

### Operator Binding as Gear-Shift Detector (2026-03-21)

**Hypothesis**: Watch `+`/`=`/digit tokens move through layers. When `+` binds toward its operands (cosine convergence L2-L8), the model is shifting into arithmetic mode. Zero-shot geometry, no trained probe.

**Result**: Rejected.

- L0 binding ratio perfectly separates arithmetic (3.4-7.4) from non-arithmetic (<0.5), **but** this is trivial vocabulary proximity — equivalent to regex "digits next to operator"
- Binding advantage evaporates by L8 and **inverts by L12**. The hyphen in "risk-reward" binds MORE in later layers because the model discovers it's a compound concept
- By L23, all cases converge to 0.83-0.88 cosine. No separation possible at the layer where task type actually lives
- Attention sink confirmed: 98-99% of L2 attention goes to sink token; direct operator→operand attention is ~0%
- Signal is in trajectory **shape** (arithmetic starts high/flattens, non-arithmetic starts low/grows) — but classifying curves needs a trained model, defeating the zero-shot appeal

**Lesson**: The model's "this is arithmetic" representation lives in directions discoverable by **probing L23**, not in pairwise token distances at any single layer.

Scripts (archived, do not re-run): `/tmp/turnstyle_binding_experiment.py`, `/tmp/turnstyle_binding_v2.py`.

---

## Sequencing

Execute in this order — each stage unblocks the next:

1. **Level 2 (Oracle-time composition)** — smallest blast radius, immediate user value, no new infra
2. **Level 1 (CompositeTurnstyle)** — already planned and partially started, finish it
3. **Level 3 Phase A-C (Probe data + training)** — experimental, results inform Phase D
4. **Level 3 Phase D (Integration)** — gated on Phase C macro-F1 clearing regex baseline

Level 1 and Level 2 can proceed in parallel if time permits — they touch different files. Level 3 should not start until Level 1 is landed (the probe needs to produce multi-label output that CompositeTurnstyle can consume).

---

## Open Questions

- **Pooling strategy for the probe**: mean-over-last-K vs last-token-only vs attention-weighted. Phase C should sweep.
- **Threshold calibration**: at what confidence should the probe defer to regex fallback? Needs a calibration set separate from training data.
- **Backbone sensitivity**: SmolLM2 L23-24 is the signal locus. Does this hold on Qwen 1.5B / Phi 3.8B? (See dyf memory `relationship_transfer_probe.md` — Phi has deepest/most stable features at L12+, not L23.) Probe training may need to be per-backbone.
- **Extraction still regex**: cleanest split for now, but long-term could an attention-based extractor replace regex for operand pulling? Hard, not on the critical path.

---

## References

- Source memory: `~/.claude/projects/-Users-jdonaldson-Projects-dyf/memory/turnstyle_composition.md`
- SmolLM2 layer analysis: `~/.claude/projects/-Users-jdonaldson-Projects-dyf/memory/smollm2_layer_analysis.md`
- FFFD logit lens failure: `~/.claude/projects/-Users-jdonaldson-Projects-dyf/memory/fffd_logit_lens.md`
- Relational transfer probe (backbone sensitivity): `relationship_transfer_probe.md` in dyf memory
- Capability probe experiment (current): `experiments/capability_probe_phase0.md`
