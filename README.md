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

- **`ArithmeticTurnstyle`** — `+`, `-`, `*`, `/` (integer division)
- More coming: dates, unit conversion, lookups, ...

## Install

```bash
pip install turnstyle
```

Requires `torch` and `transformers`. Works with any HuggingFace causal LM.
