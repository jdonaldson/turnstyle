# ⊢ Turnstyle

Symbolic coprocessors for LLM generation. Ground model outputs in exact computation via logit biasing.

The model handles language. Turnstyles handle facts.

```python
from turnstyle import ArithmeticTurnstyle

t = ArithmeticTurnstyle(model, tokenizer, device)
text, proof = t.generate("What is 445 + 152?")

print(text)           # "The sum of 445 and 152 is 597."
print(proof.inline()) # ⊢ 445+152=5̲97 ∎
print(proof.detail())
# ⊢ 445+152=597  1/3 corrected  Δ=0.05
#   d0: [6→5]  logit_gap=+1.0
#   trigger@step 14/19  state=DONE
```

## How it works

A turnstyle is a `LogitsProcessor` that intercepts digit generation. When the model is about to output an answer, the turnstyle biases logits toward the symbolically-computed correct digits. Every intervention is audited.

**Annotation marks:**
- `5̲` underline — digit corrected by coprocessor
- `5̂` circumflex — digit the model never emitted
- `⊢` turnstile — "this was derived"
- `∎` QED — "proof complete"

## Turnstyles

| Turnstyle | Domain | Example |
|-----------|--------|---------|
| `ArithmeticTurnstyle` | `+`, `-`, `*`, `/` | "What is 445 + 152?" |
| `DateTurnstyle` | Days/weeks between dates | "How many days between 2026-01-01 and 2026-03-20?" |
| `UnitTurnstyle` | Physical unit conversion | "How many km is 26.2 miles?" |
| `CurrencyTurnstyle` | Currency conversion | "How much is 100 USD in EUR?" |
| `PercentageTurnstyle` | Percentages, tips, discounts | "What is 15% of 230?" |
| `CountingTurnstyle` | Letters, vowels, words | "How many r's in 'strawberry'?" |
| `BaseConversionTurnstyle` | Binary, hex, octal | "What is 255 in binary?" |
| `SandboxTurnstyle` | Arbitrary Python via WASM | "What does \`sum(range(101))\` return?" |

Each turnstyle follows the same pattern: `parse()` computes an oracle answer, `make_processor()` sets up digit biasing, `generate()` runs the model with grounded outputs.

## SandboxTurnstyle

Executes arbitrary Python in a WASM sandbox (Deno + Pyodide) and biases digit logits toward the computed result. The model writes the prose; the sandbox guarantees the number.

```python
from turnstyle import SandboxTurnstyle, DenoPyodideBackend

backend = DenoPyodideBackend()
t = SandboxTurnstyle(model, tokenizer, device, backend=backend)

text, proof = t.generate("What does `sum(range(101))` return?")
# proof.answer == 5050
```

Supports fenced code blocks, inline backtick expressions, and directive prompts. See [docs/sandbox.md](docs/sandbox.md) for full details.

**Requirements:** [Deno](https://deno.land) installed on PATH. No network access or filesystem access from sandboxed code.

## Install

```bash
pip install turnstyle
```

Requires `torch` and `transformers`. Works with any HuggingFace causal LM.

For sandbox support, install [Deno](https://deno.land):
```bash
curl -fsSL https://deno.land/install.sh | sh
```
