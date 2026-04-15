# SmolLM2 Full Capability Eval — Error Analysis Report

**Date**: 2026-04-14
**Model**: SmolLM2-1.7B-Instruct (CPU, float32)
**Script**: `experiments/smol_capability_eval.py`
**Wall time**: ~21 min (all tiers, CPU)

## Motivation

The tier taxonomy (T0–T3) lacked empirical SmolLM2 scores for T1 and error categorization for T2/T3. This eval captures *what the model produces* at each tier — not just pass/fail — to identify where the model breaks and what kind of intervention (prompt, architecture, fallback) each failure mode needs.

---

## Tier 0: Logit Steering (Sanity Check)

**Purpose**: Confirm that regex parsing + logit biasing produces correct answers. If this fails, infrastructure is broken.

| Task | N | Correct | Accuracy | Parse failures |
|------|---|---------|----------|----------------|
| boolean_expressions | 10 | 10 | **100%** | 0 |
| dyck_languages | 10 | 10 | **100%** | 0 |
| word_sorting | 10 | 10 | **100%** | 0 |
| multistep_arithmetic_two | 10 | 10 | **100%** | 0 |

**Findings**:

- **Boolean & Dyck: 100%.** Logit steering infrastructure is sound. Model corrections needed on most examples (1/1 corrected for boolean, 1-2/N for dyck) — confirming that the model would get these wrong without biasing.
- **Sorting: 100%.** ~~90% prior~~ — `it&t` failure was a regex character class bug (`[a-z']` excluded `&`). Fixed: broadened BBH regex to `\S[\S ]*\S`.
- **Arithmetic: 100%.** ~~0% prior~~ — `parse_arithmetic()` only handled binary ops. Fixed: `parse_expression()` uses `ast.parse` + recursive `_eval_node` (no `eval()`) for nested expressions. Routes through `SequenceLogitsProcessor(immediate=True)` for negative answers (e.g. `-50`).

**Verdict**: **T0 = 40/40 (100%).** Logit steering works for all four symbolic task types. Every arithmetic example required model correction (2-3 tokens each) — the model genuinely cannot do nested arithmetic; steering does all the work.

---

## Tier 1: Simple Field Extraction (Regex Disabled)

**Purpose**: Test whether SmolLM2 can extract structured fields from prompts using the ExtractionSpec LLM fallback path (the path that activates when regex fails).

| Task | N | Correct | Accuracy | Dominant failure |
|------|---|---------|----------|-----------------|
| boolean_expressions | 20 | 7 | **35%** | assemble_fail (12/20) |
| dyck_languages | 20 | 5 | **25%** | wrong_answer (9/20), assemble_fail (6/20) |
| word_sorting | 20 | 2 | **10%** | wrong_answer (18/20) |

### Error analysis

**Boolean (35% correct, 12/20 assemble_fail)**

The ExtractionSpec asks the model to extract the boolean expression from the prompt. Three failure patterns:

1. **Bare truth value** (most common): Model outputs just `"False"` or `"True"` instead of the expression. Confidence is low (0.24–0.27). The assembler tries to `eval()` this but it's not a full expression — it happens to work when the bare value matches the answer, but that's coincidence.
2. **Echo with suffix**: Model outputs the expression *plus* appended answer, e.g. `"True and not not ( not False ) is True"`. The trailing `" is True"` prevents assembly because the expression doesn't evaluate cleanly.
3. **Correct extraction** (35%): When the model reproduces just the expression, the assembler evaluates it correctly.

**Word sorting (10% correct, 18/20 wrong_answer)**

The ExtractionSpec asks for comma-separated word list. The model's dominant failure:

- **Prompt echo**: Instead of extracting just the words, the model outputs the entire prompt: `"Sort the following words alphabetically: List: thrill, splutter, panicking..."`. The assembler splits this on commas/spaces and gets tokens like `"Sort"`, `"the"`, `"following"` mixed in with actual words, producing a wrong sorted output.
- The 2/20 correct cases happen when the word list is short enough that the model's echo accidentally starts at the right boundary.

**Dyck (25% correct, 9/20 wrong + 6/20 assemble_fail)**

The ExtractionSpec asks for the bracket sequence. Two failure patterns:

- **Partial extraction** (wrong_answer): Model extracts only the first few brackets of a long sequence, e.g. input `< [ {` → model outputs just `"[ ["`. The stack computation then produces a different closing sequence than the full input would require.
- **Closing-included** (assemble_fail): Model includes both opening and closing brackets in its extraction, producing an already-balanced sequence. The assembler rejects this ("Brackets already balanced").

### Tier 1 takeaway

SmolLM2 **cannot reliably follow extraction instructions**. The dominant failure is **prompt echo** — the model regurgitates input text rather than extracting the requested field. This is consistent with the model's 1.7B scale: instruction-following for structured extraction requires more capacity than the model has. Extraction at this scale needs either (a) more constrained prompts with strong delimiters, or (b) regex.

---

## Tier 2: JSON Triple Extraction (SentenceIRSpec)

**Purpose**: Test whether SmolLM2 can produce valid JSON triples from individual sentences, and whether the extracted triples are semantically correct.

### Tasks with SentenceIRSpec (end-to-end accuracy)

| Task | N | Correct | Accuracy | All JSON valid? | Failure mode |
|------|---|---------|----------|-----------------|--------------|
| logical_deduction_three | 5 | 0 | **0%** | Yes (5/5 sentences parse) | Wrong entity names, wrong predicates |
| tracking_shuffled_three | 5 | 0 | **0%** | Yes (7/7 sentences parse) | Type misclassification → aggregator fails |

### Raw JSON production (no aggregation)

| Task | N | All-valid examples | Partial | Notes |
|------|---|--------------------|---------|-------|
| navigate | 5 | 3 (60%) | 2 | 80–100% per-sentence JSON validity |
| web_of_lies | 5 | 5 (100%) | 0 | Perfect JSON structure on all sentences |

### Error analysis

**Logical deduction (0%, 5/5 sentences produce valid JSON)**

The model produces syntactically valid JSON triples for every sentence. But the content is wrong:

- **Hallucinated entities**: Constraint sentences about "falcon" and "blue jay" produce `{"subj": "tiger", "pred": "at_pos", "obj": -1}` — the model defaults to "tiger" (from the few-shot examples in the extraction prompt) rather than extracting the actual entity.
- **Wrong predicates**: Comparative constraints ("X is to the left of Y") sometimes produce `at_pos` instead of `lt`/`gt`. The few-shot examples cover this case, but the model latches onto the most frequent pattern.
- **Aggregation consequence**: Even when individual triples parse, the wrong entity names and predicates produce an incorrect constraint set → wrong permutation → wrong answer.

**Tracking shuffled (0%, 7/7 sentences produce valid JSON)**

Every sentence produces parseable JSON arrays. But:

- **Type misclassification**: The `classify_fn` assigns wrong sentence types — fact sentences get labeled as "query" or "preamble", causing the aggregator to misinterpret init states vs. swap actions.
- **Entity hallucination in preamble**: Preamble sentences (which should produce `null`) instead produce fabricated init states: `[{"subj": "Alice", "pred": "has", "obj": "yellow ball"}]` regardless of actual content. The model pattern-matches from the few-shot examples rather than reading the sentence.
- **Consequence**: With wrong init state and misclassified swaps, the aggregator can't find the query actor in the state dict → returns `None`.

**Navigate (60% all-valid, ~90% per-sentence)**

The model produces valid JSON for movement instructions. The triples are syntactically correct but semantically imprecise:
- `"Take 1 step backward"` → `{"subj": "Take", "pred": "1 step backward", "obj": "..."}` — the step count and direction land in the predicate rather than being properly decomposed.
- Not a blocker for detection (JSON is valid), but the schema would need to be adapted if these triples were used for simulation.

**Web of lies (100% all-valid)**

Best JSON production of any task. Every sentence produces a valid triple:
- `"Sherrie tells the truth"` → `{"subj": "Sherrie", "pred": "tells", "obj": "the truth"}`
- `"Vernell says Sherrie tells the truth"` → `{"subj": "Vernell", "pred": "says", "obj": "Sherrie tells the truth"}`

The sentences are short, formulaic, and structurally close to the generic triple format. This is the sweet spot for SmolLM2 JSON extraction.

### Tier 2 takeaway

SmolLM2 **can produce valid JSON** — the mechanical formatting works, especially for short formulaic sentences (web_of_lies: 100%). The failure is **semantic**: wrong entity names (hallucinated from few-shot), wrong predicate types, wrong sentence classification. The model can format but can't extract. This confirms the "model transcribes, code simulates" architecture: SmolLM2 can produce structured output *when the mapping from text to structure is nearly 1:1* (web_of_lies), but fails when extraction requires interpretation (logical_deduction, tracking).

---

## Tier 3: SQL Generation

**Purpose**: Test whether SmolLM2 can extract table structure from text and generate correct SQL queries.

| Task | N | Correct | Accuracy | Bottleneck |
|------|---|---------|----------|-----------|
| penguins_in_a_table | 10 | 1 | **10%** | Table extraction (6/10 fail) |
| object_counting | 5 | 2 | **40%** | SQL correctness (3/5 wrong result) |

### Error analysis

**Penguins (10%, 6/10 no_tables)**

The dominant failure is at the **first step** — SmolLM2 cannot extract table structure from the BBH penguin prompts:

- **no_tables (6/10)**: `_model_extract()` fails to parse the CSV-like text into a `{table_name: (columns, rows)}` dict. The penguin prompts embed tables as inline comma-separated text without markdown table formatting, and SmolLM2's generic JSON extraction can't handle this format.
- **wrong_result (2/10)**: Of the 4 examples where tables do parse, 2 produce wrong SQL results — the model generates syntactically valid SQL that queries the wrong column or applies the wrong condition.
- **no_match (1/10)**: SQL executes correctly but the result doesn't match any option letter.
- **correct (1/10)**: One example works end-to-end.

**Object counting (40%, 3/5 wrong_result)**

Table extraction works on all 5 examples (the generic `_model_extract_table` handles the object-listing format). But SQL queries are too simple:

- Example: target=8, SQL=`SELECT COUNT(*) FROM data WHERE type = 'instrument'` → result=7. The model miscategorizes one item or uses too-narrow a WHERE clause.
- Example: target=15, SQL=`SELECT COUNT(*) FROM data` → result=13. The model counts extracted rows but the extraction missed items.
- The 2/5 correct cases are simpler prompts where a bare `SELECT COUNT(*)` happens to match.

### Tier 3 takeaway

Two distinct bottlenecks:

1. **Table extraction** (penguins): SmolLM2 can't parse inline CSV text into structured tables without the deterministic parser. The generic `_model_extract_table` path requires the model to produce JSON that maps to columns/rows — too much for 1.7B on unstructured table formats.
2. **SQL correctness** (object_counting): When tables are provided, SmolLM2 generates syntactically valid SQL (~100%) but semantically wrong queries (~60% of the time). The errors are subtle: wrong WHERE clauses, missing items in extraction, or overly broad/narrow counts.

---

## Cross-Tier Summary

| Tier | Capability tested | SmolLM2 score | Key insight |
|------|-------------------|---------------|-------------|
| T0 | Logit steering (with regex) | **100%** (40/40) | Infrastructure works; all parser gaps closed |
| T1 | Field extraction (no regex) | **23%** avg | Prompt echo dominates; model can't follow extraction instructions |
| T2 | JSON triple extraction | **0%** end-to-end; **100%** JSON validity (wol) | Can format JSON; can't extract semantics |
| T3 | SQL generation | **20%** avg | Table extraction is bottleneck; SQL syntax OK, semantics wrong |

### Implications for architecture

1. **Regex is load-bearing.** T1 shows that disabling regex drops accuracy from 100% to 23%. The ExtractionSpec LLM fallback is not a viable replacement at 1.7B — it's an emergency path, not a general one.

2. **JSON formatting ≠ extraction.** T2 shows SmolLM2 can produce valid JSON (especially for simple sentence structures), but the content is hallucinated from few-shot examples rather than extracted from the input. This means SentenceIRSpec is viable *only* for tasks where the sentence structure maps near-1:1 to the triple schema (web_of_lies). For tasks requiring interpretation (logical_deduction), deterministic solvers remain necessary.

3. **SQL is a two-step problem.** T3 shows that (a) table extraction requires deterministic parsing (SmolLM2 can't handle it), and (b) even with correct tables, SQL generation is ~60% wrong. The current `parse_tables_fn` + `repair_sql` architecture is correct — model-based extraction is not a viable alternative at this scale.

4. **The tier taxonomy holds.** The empirical scores match the predicted difficulty ordering: T0 > T1 > T2/T3. The "model transcribes, code simulates" pattern is validated — SmolLM2 is a capable transcriber for constrained formats but not an extractor or reasoner.
