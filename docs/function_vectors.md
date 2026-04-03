# Function Vector Experiment Plan

Mid-computation intervention via function vectors — a follow-up to the attention sink investigation.

## Background: Why Not Sinks?

Attention sinks (position 0) absorb 65-81% of attention mass at layers 8-20 in SmolLM2, but patching their residual produces zero steering effect. Context-specific sinks (operator tokens, key nouns) are semantically meaningful, but steering via residual patching still fails because:

1. Single-position patches get diluted by downstream layers
2. We used the LM head weight direction (output vocabulary space) as the steering signal — intermediate layers don't process that

Full results: `memory/attention_sinks.md`, scripts: `/tmp/attention_sink_probe.py`, `/tmp/context_sinks.py`, `/tmp/context_sink_steering.py`

**Conclusion**: Logit biasing at the output layer remains the correct intervention. But the question remains — can we also intervene mid-computation?

## Function Vectors: The Idea

Extract activation differences between prompted (few-shot, model succeeds) and unprompted (zero-shot, model fumbles) versions of a task. The averaged difference is a "function vector" — a direction in activation space that encodes the task instruction in the model's native signal space.

Key reference: Todd et al., "Function Vectors in Large Language Models" (2023).

**Why this might work where sinks failed**:
- Patches ALL positions simultaneously (no dilution)
- Direction is from the model's own successful activations (native signal, not LM head vocabulary space)

## Experiment Design

### Phase 1: Extract Function Vectors (Arithmetic)

Generate ~50 prompt pairs across operations (add/sub/mul/div):

```
# Zero-shot (model fumbles)
"What is 123 + 456?"

# Few-shot (model succeeds)
"What is 2 + 3? 5. What is 10 + 7? 17. What is 123 + 456?"
```

Extraction:
1. Run both through SmolLM2, extract hidden states at all 24 layers (last token)
2. `fv[layer] = mean(h_fewshot - h_zeroshot)` across all pairs
3. Validate: cosine similarity between individual differences should be high (stable direction)

### Phase 2: Inject and Measure

Inject function vector at each layer into zero-shot prompts. Measure:
- Next-token probability for correct answer
- Rank change vs. baseline
- Top-5 prediction shift

Sweep: layer × magnitude grid. Use same prompts/targets as sink experiment for direct comparison.

### Phase 2b: Function Vector at Context Sinks

Test whether attention routing matters by comparing:
1. FV at all positions (standard)
2. FV at context sinks only (operator, first digit, key noun — positions with high attention)
3. FV at non-sink positions only (control)

If (2) > (3): attention channel matters, model reads FV signal more from attended positions.
If (1) ≈ (2): sink targeting adds nothing over broadcast.
If (2) ≈ (3): attention routing irrelevant, just total injection magnitude.

**This is the definitive sink test**: right signal (FV) at right positions (context sinks). If this also fails, sinks are closed as intervention channels.

### Phase 3: "Knows But Can't Say" Tasks

Test on BBH tasks where probes show internal representation but generation fails:
- **movie_recommendation**: Probe L18 64.8%, baseline 22.3%
- **disambiguation_qa**: Probe L18 98%, baseline 34.8%

Paired examples: same task, correct vs. wrong answer. FV captures what's different in the residual stream when the model succeeds.

### Phase 4: Combine with Logit Biasing

1. Function vector alone → accuracy
2. Logit biasing alone → accuracy (already measured)
3. Both together → does the combination beat either?

Hypothesis: FV handles "mode" (task understanding), logit biasing handles "answer" (token forcing). Complementary, not redundant.

### Phase 5: Virtual Token Injection

If FVs work at context sinks, try inserting NEW key-value pairs into the KV cache:
1. `K_virtual = FV @ W_K`, `V_virtual = FV @ W_V` at target layers
2. Prepend to KV cache alongside real tokens
3. Model attends naturally (no learned invariance to novel positions)

This is soft prompting via KV injection — avoids "does the model read from this position?" entirely.

## Risks

1. **Format-specific, not task-specific**: FV might encode "I'm in a few-shot context" not "do arithmetic." Test: inject arithmetic FV on non-arithmetic prompt.
2. **Unstable direction**: Individual differences might not converge. Check cosine similarity matrix.
3. **SmolLM2 too small**: Original work used larger models. 24 layers / 2048 dim might lack capacity for clean decomposition.
4. **Wrong test case**: Arithmetic failure may be capability-limited, not mode-limited. movie_recommendation (where representation exists) may be the better first test.

## Success Criteria

- **Clear win**: Rank shift >5 at magnitude ≤20 (sink patching achieved 0)
- **Interesting**: FV + logit biasing beats logit biasing alone on any BBH task
- **Negative**: FV behaves like sink patching → close this investigation

## Implementation Notes

- Standalone scripts in `/tmp/` — don't modify turnstyle code until validated
- Same model loading pattern (chat template, MPS, bfloat16, eager attention)
- Hook-based extraction at all layers (same as probe scripts)
- Save FVs as `.pt` files for reuse
- If validated: add `FunctionVector` intervention type to turnstyle alongside `LogitsProcessor` and `TurnstyleProbe`

## Connection to Existing Findings

- Operation-specific representations emerge at L20-22 (hidden state analysis)
- Operation tokens leak at L20-22 (logit lens)
- FV should peak at these layers if capturing task computation
- If FV peak layer ≠ probe peak layer → functional vs. representational geometry differ
