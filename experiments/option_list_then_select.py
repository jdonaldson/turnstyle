"""Does the model enumerate options as an internal LIST, then SELECT from it?

Architectural prior: causal masking means option A's tokens cannot attend to
options that come later. So the full option set is never co-represented at the
per-option positions — only the final position (after all options) can see them
all. A literal "list-then-select" is therefore impossible until the end; the
per-option computation must be streaming / in-place.

Test 1 (the "list" test): for each option k, truncate the prompt right after
option k and recompute option k's last-token state. If the model co-represents
all options as a list, dropping later options perturbs earlier ones. Causally
it must NOT. Δ≈0 ⇒ no list; each option's rep depends only on preceding context.

Test 2 (the "select" test): append "\\nAnswer: (" and logit-lens the final
position across layers for the correct option letter. Shows WHERE selection
crystallizes (and that it's a final-position event, not a per-option one).
"""
from __future__ import annotations

import json
import re
import sys

import numpy as np
import torch

sys.path.insert(0, "/Users/jdonaldson/Projects/turnstyle/experiments")
from common import load_model  # noqa: E402

BBH_CACHE = "/Users/jdonaldson/Projects/swollm/data/bbh_cache"
TASKS = ["snarks", "date_understanding", "logical_deduction_three_objects",
         "movie_recommendation"]
N = 12
OPTION_RE = re.compile(r"^\(([A-Z])\)", re.MULTILINE)


def load_task(name):
    with open(f"{BBH_CACHE}/{name}.json") as f:
        return json.load(f)


def option_spans(text):
    marks = list(OPTION_RE.finditer(text))
    out = []
    for k, m in enumerate(marks):
        end = marks[k + 1].start() if k + 1 < len(marks) else len(text)
        out.append((m.start(), end, k, m.group(1)))
    return out


def main():
    tok, mdl, device = load_model()
    print(f"model loaded on {device}\n", flush=True)
    L = 8

    # ---- Test 1: option co-representation (TOKEN-level truncation invariance) ----
    print("=== Test 1: 'list' test — does option k's state depend on later options? ===")
    print("(max |Δ| between full-prompt and token-truncated-after-k, SAME token/position;")
    print(" ≈float16 noise ⇒ no co-represented list. Hidden-state norm shown for scale.)\n")
    for task in TASKS:
        diffs, norms = [], []
        for ex in load_task(task)[:N]:
            text = ex["input"]
            spans = option_spans(text)
            if len(spans) < 2:
                continue
            enc = tok(text, return_offsets_mapping=True, return_tensors="pt")
            offsets = enc["offset_mapping"][0].tolist()
            ids = enc["input_ids"].to(device)
            with torch.no_grad():
                full = mdl(input_ids=ids,
                           attention_mask=enc["attention_mask"].to(device),
                           output_hidden_states=True).hidden_states[L][0]
            for (s, e, k, _) in spans[:-1]:        # skip last (truncation == full)
                content_end = s + len(text[s:e].rstrip())   # last non-ws char of option k
                pos = max((ti for ti, (a, b) in enumerate(offsets)
                           if a != b and a <= content_end - 1 < b), default=None)
                if pos is None:
                    continue
                with torch.no_grad():
                    trunc = mdl(input_ids=ids[:, :pos + 1],
                                output_hidden_states=True).hidden_states[L][0, -1]
                h_full = full[pos]
                diffs.append(float(torch.max(torch.abs(h_full - trunc)).item()))
                norms.append(float(torch.linalg.norm(h_full).item()))
        d = np.nanmax(diffs) if diffs else float("nan")
        med = np.nanmedian(diffs) if diffs else float("nan")
        nrm = np.nanmedian(norms) if norms else float("nan")
        print(f"  {task:32s}  max|Δ|={d:.2e}  median|Δ|={med:.2e}  "
              f"(median ‖h‖={nrm:.1f}, n={len(diffs)})")

    # ---- Test 2: selection timing via logit lens at the answer position ----
    print("\n=== Test 2: 'select' test — correct-letter accuracy by layer at 'Answer: (' ===\n")
    n_layers = mdl.config.num_hidden_layers
    layers = list(range(0, n_layers + 1, 2))
    letter_ids = {c: tok.encode(c, add_special_tokens=False)[0]
                  for c in "ABCDEFGH"}
    norm = mdl.model.norm
    head = mdl.lm_head
    per_layer_correct = {l: 0 for l in layers}
    total = 0
    for task in TASKS:
        for ex in load_task(task)[:N]:
            text = ex["input"]
            spans = option_spans(text)
            if len(spans) < 2:
                continue
            target = re.search(r"\(([A-Z])\)", ex["target"])
            if not target:
                continue
            gold = target.group(1)
            present = [s[3] for s in spans]
            cand = torch.tensor([letter_ids[c] for c in present], device=device)
            prompt = text + "\nAnswer: ("
            enc = tok(prompt, return_tensors="pt")
            with torch.no_grad():
                out = mdl(input_ids=enc["input_ids"].to(device),
                          attention_mask=enc["attention_mask"].to(device),
                          output_hidden_states=True)
            for l in layers:
                h = out.hidden_states[l][0, -1]
                logits = head(norm(h))
                pick = present[int(torch.argmax(logits[cand]).item())]
                if pick == gold:
                    per_layer_correct[l] += 1
            total += 1
    print(f"  (n={total} prompts, {n_layers} layers)\n")
    for l in layers:
        acc = per_layer_correct[l] / total if total else 0
        bar = "█" * int(acc * 40)
        print(f"  L{l:2d}: {acc:5.1%} {bar}")


if __name__ == "__main__":
    main()
