import json
import sys
from pathlib import Path
sys.path.insert(0, '/Users/jdonaldson/Projects/turnstyle/experiments')

import torch
from times_table_trace import encode, ITOS, load

ROOT = Path("/Users/jdonaldson/Projects/turnstyle/experiments/data/nanogpt_times_table")
OUT = ROOT / "predicted_chars.json"

model = load("cpu")
model.eval()


def expected_first_char(a, b, op_name):
    if op_name == "mul":
        return str(a * b)[0]
    if op_name == "add":
        return str(a + b)[0]
    return str(a - b)[0]


results = {}
for op_name, op_char in [("mul", "*"), ("add", "+"), ("sub", "-")]:
    pred_chars = {}
    exp_chars = {}
    cnt_pred = {}
    cnt_exp = {}
    mismatches = []
    for a in range(10):
        for b in range(10):
            prompt = f"\n{a}{op_char}{b}="
            ids = torch.tensor([encode(prompt)], dtype=torch.long)
            with torch.no_grad():
                logits, _, _ = model(ids)
            pred_id = int(logits[0, -1].argmax().item())
            pred_char = ITOS[pred_id]
            exp_char = expected_first_char(a, b, op_name)
            pred_chars[f"{a},{b}"] = pred_char
            exp_chars[f"{a},{b}"] = exp_char
            cnt_pred[pred_char] = cnt_pred.get(pred_char, 0) + 1
            cnt_exp[exp_char] = cnt_exp.get(exp_char, 0) + 1
            if pred_char != exp_char:
                mismatches.append([a, b, exp_char, pred_char])
    results[op_name] = {
        "predicted": pred_chars,
        "expected": exp_chars,
        "pred_counts": cnt_pred,
        "exp_counts": cnt_exp,
        "mismatches": mismatches,
    }
    print(f"{op_name}:")
    print(f"  predicted counts: {cnt_pred}")
    print(f"  expected counts:  {cnt_exp}")
    print(f"  mismatches: {len(mismatches)}")
    for m in mismatches[:5]:
        print(f"    ({m[0]},{m[1]}): expected {m[2]!r}, predicted {m[3]!r}")

OUT.write_text(json.dumps(results, indent=2))
print(f"Wrote {OUT}")
