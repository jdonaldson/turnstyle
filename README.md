<img src="turnstyle.png" alt="⊢ turnstyle" />

Ground LLM generation in real computation. The model writes prose; turnstyles guarantee the facts.

A turnstyle intercepts generation at the token level, running an oracle (anything from `a + b` to arbitrary Python in a WASM sandbox) and steering the model toward the correct answer. Every intervention is audited.

```python
from turnstyle import SandboxTurnstyle

t = SandboxTurnstyle(model, tokenizer, device)

# The model writes the explanation. The sandbox computes the answer.
text, proof = t.generate("""What does this return?
```python
primes = [n for n in range(2, 100) if all(n % i != 0 for i in range(2, int(n**0.5)+1))]
len(primes)
```""")
# proof.answer == 25
```

The code runs in a WASM sandbox — no network, no filesystem, no syscalls. The model can't hallucinate a number that the sandbox actually computed.

## Turnstyles

Every turnstyle follows the same pattern: `parse()` runs an oracle, `make_processor()` wires the answer into logit biasing, `generate()` lets the model write freely while the coprocessor enforces correctness.

| Turnstyle | Oracle | Example |
|-----------|--------|---------|
| `SandboxTurnstyle` | Arbitrary Python in WASM | `` "What does `sum(range(101))` return?" `` |
| `ArithmeticTurnstyle` | `+`, `-`, `*`, `/` | "What is 445 + 152?" |
| `DateTurnstyle` | Date arithmetic | "How many days between Jan 1 and Mar 20?" |
| `UnitTurnstyle` | Physical unit conversion | "How many km is 26.2 miles?" |
| `CurrencyTurnstyle` | Currency conversion | "How much is 100 USD in EUR?" |
| `PercentageTurnstyle` | Percentages, tips, discounts | "What is 15% of 230?" |
| `CountingTurnstyle` | Letters, vowels, words | "How many r's in 'strawberry'?" |
| `BaseConversionTurnstyle` | Binary, hex, octal | "What is 255 in binary?" |

The specialized turnstyles (arithmetic, dates, etc.) are fast pattern-matched oracles. `SandboxTurnstyle` is the general case — if you can write Python for it, you can ground generation in it.

## How it works

Under the hood, a turnstyle is a `LogitsProcessor`. After the oracle computes the answer, the processor biases digit logits so the model emits the correct value. If the oracle fails (parse miss, non-numeric result, timeout), the model generates freely — no biasing, no crash.

```python
from turnstyle import ArithmeticTurnstyle

t = ArithmeticTurnstyle(model, tokenizer, device)
text, proof = t.generate("What is 445 + 152?")

print(proof.inline()) # ⊢ 445+152=5̲97 ∎
print(proof.detail())
# ⊢ 445+152=597  1/3 corrected  Δ=0.05
#   d0: [6→5]  logit_gap=+1.0
#   trigger@step 14/19  state=DONE
```

**Annotation marks:**
- `5̲` underline — digit corrected by coprocessor
- `5̂` circumflex — digit the model never emitted
- `⊢` turnstile — "this was derived"
- `∎` QED — "proof complete"

## SandboxTurnstyle

Extracts Python from prompts via fenced code blocks, inline backticks (`` `expr` ``), "what does X return" patterns, or directives (`Evaluate: expr`). Bare arithmetic falls through to `ArithmeticTurnstyle`.

```python
from turnstyle import SandboxTurnstyle

t = SandboxTurnstyle(model, tokenizer, device)

text, proof = t.generate("What does `sum(range(101))` return?")
text, proof = t.generate("Evaluate: sum(int(d) for d in str(2**100))")
```

See [docs/sandbox.md](docs/sandbox.md) for the full reference — code extraction patterns, backends, error behavior, and V1 limitations.

## Install

```bash
pip install turnstyle
```

Requires `torch` and `transformers`. Works with any HuggingFace causal LM.

For sandbox support (runs Python in a WASM sandbox):
```bash
pip install turnstyle[sandbox]
```

This installs `wasmtime` and auto-downloads CPython WASM on first use. Falls back to [Deno](https://deno.land) + Pyodide if wasmtime is unavailable.

## References

Turnstyle is an implementation of **neurosymbolic programming** — combining neural generation with symbolic computation through constrained decoding.

- Chaudhuri, S., Ellis, K., Polozov, O., Singh, R., Solar-Lezama, A., & Yue, Y. (2021). [Neurosymbolic Programming](https://www.nowpublishers.com/article/Details/PGL-049). *Foundations and Trends in Programming Languages*, 7(3), 158–243.
- [Neuro-symbolic AI](https://en.wikipedia.org/wiki/Neuro-symbolic_AI) — Wikipedia overview of the broader field.
