"""SmolLM2 arithmetic sanity check.

Verify the model can do 0-9 × 0-9 / + / - reliably enough to land probes.

Tries SmolLM2-1.7B-Instruct and SmolLM2-1.7B (base) on raw `a op b =` and
few-shot variants. Reports first-token argmax accuracy per format per op.

Uses MPS + float32 + local_files_only (model cache populated already).
"""
from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def first_char_expected(op: str, a: int, b: int) -> str:
    if op == "*":
        c = a * b
    elif op == "+":
        c = a + b
    else:
        c = a - b
    return "-" if c < 0 else str(c)[0]


@torch.no_grad()
def baseline_eval(model, tok, prompt_fn, device="cpu"):
    out = {}
    for op in ("*", "+", "-"):
        correct = 0
        examples = []
        for a in range(10):
            for b in range(10):
                prompt = prompt_fn(op, a, b)
                ids = tok.encode(prompt, add_special_tokens=False,
                                 return_tensors="pt").to(device)
                logits = model(ids, use_cache=False).logits[0, -1, :]
                top_id = int(logits.argmax().item())
                pred = tok.decode([top_id])
                exp = first_char_expected(op, a, b)
                if pred.lstrip().startswith(exp):
                    correct += 1
                if len(examples) < 3:
                    examples.append((prompt, pred, exp))
        out[op] = (correct, 100, examples)
    return out


def run(name):
    print(f"\n========== {name} ==========")
    tok = AutoTokenizer.from_pretrained(name, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        name, torch_dtype=torch.float32, local_files_only=True
    ).to(DEVICE).eval()
    print(f"  layers={model.config.num_hidden_layers}, "
          f"hidden={model.config.hidden_size}, "
          f"vocab={model.config.vocab_size}")

    # Tokenization check
    for p in ["5*3=", "\n5*3=", "1*2=2\n3*4=12\n5*3="]:
        ids = tok.encode(p, add_special_tokens=False)
        toks = [tok.decode([i]) for i in ids]
        print(f"  tok({p!r}) = {len(ids)} → {toks}")

    fmts = {
        "raw":     lambda op, a, b: f"{a}{op}{b}=",
        "newline": lambda op, a, b: f"\n{a}{op}{b}=",
    }
    fewshot_ex = {
        "*": [("1*2", "2"), ("3*4", "12")],
        "+": [("1+2", "3"), ("3+4", "7")],
        "-": [("3-1", "2"), ("7-4", "3")],
    }
    fmts["fewshot"] = (lambda op, a, b:
        f"{fewshot_ex[op][0][0]}={fewshot_ex[op][0][1]}\n"
        f"{fewshot_ex[op][1][0]}={fewshot_ex[op][1][1]}\n"
        f"{a}{op}{b}=")

    for fmt_name, fn in fmts.items():
        print(f"\n  Format: {fmt_name}")
        res = baseline_eval(model, tok, fn, device=DEVICE)
        for op, (c, t, ex) in res.items():
            print(f"    {op}: {c:3d}/{t} = {c/t:.0%}")
            for p, pred, exp in ex[:1]:
                print(f"      {p!r} → {pred!r} (want {exp!r})")

    del model
    if DEVICE == "mps":
        torch.mps.empty_cache()


def main():
    print(f"Device: {DEVICE}")
    run("HuggingFaceTB/SmolLM2-1.7B")        # base
    run("HuggingFaceTB/SmolLM2-1.7B-Instruct")  # instruct


if __name__ == "__main__":
    main()
