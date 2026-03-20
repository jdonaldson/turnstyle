"""Demo: arithmetic turnstyle on SmolLM2-1.7B."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from turnstyle import ArithmeticTurnstyle, extract_number

model_name = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
print(f"Loading {model_name}...")
device = "mps" if torch.backends.mps.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name, dtype=torch.bfloat16, attn_implementation="eager",
).to(device).eval()

t = ArithmeticTurnstyle(model, tokenizer, device)

problems = [
    "What is 445 + 152?",
    "What is 314159 + 265358?",
    "What is 7654321 / 111?",
    "What is 99999 * 99999?",
]

for prompt in problems:
    text, proof = t.generate(prompt, max_new_tokens=80)
    number = extract_number(text)
    print(f"\n  Q: {prompt}")
    print(f"  A: {text}")
    if proof:
        print(f"     {proof.inline()}")
        if proof.diagnostics:
            print(f"     {proof.diagnostic_summary()}")
