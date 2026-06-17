"""Detect candidates being 'pulled together' for the argmax via attention.

Selection fires at the final position (option_list_then_select.py); the only
channel for the candidate scores to reach it is attention. So detect the
pulling-together as: the answer position's attention onto the option spans,
per layer/head, and whether it is COMPETITIVE (more on the eventual winner).

Signals:
  (1) total option-span attention mass from the answer position, by layer
      -> should rise where selection fires (L18+).
  (2) winner asymmetry = attn(chosen span) - mean(attn other spans), by layer
      -> positive & growing late = competitive pulling = argmax signature.
  (3) top (layer, head) by option-span attention at late layers = selection heads.
"""
from __future__ import annotations

import json
import re
import sys
from collections import defaultdict

import numpy as np
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

MODEL_ID = "HuggingFaceTB/SmolLM2-1.7B-Instruct"


def load_model():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    mdl = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.float16, attn_implementation="eager"
    ).to(device).eval()
    return tok, mdl, device


BBH_CACHE = "/Users/jdonaldson/Projects/swollm/data/bbh_cache"
TASKS = ["snarks", "date_understanding", "logical_deduction_three_objects",
         "movie_recommendation"]
N = 15
OPTION_RE = re.compile(r"^\(([A-Z])\)", re.MULTILINE)


def load_task(name):
    with open(f"{BBH_CACHE}/{name}.json") as f:
        return json.load(f)


def option_spans(text):
    marks = list(OPTION_RE.finditer(text))
    out = []
    for k, m in enumerate(marks):
        end = marks[k + 1].start() if k + 1 < len(marks) else len(text)
        out.append((m.start(), end, m.group(1)))
    return out


def span_token_idx(offsets, s, e):
    return [ti for ti, (a, b) in enumerate(offsets) if a != b and s <= (a + b) / 2 < e]


def main():
    tok, mdl, device = load_model()
    print(f"model loaded on {device}\n", flush=True)
    n_layers = mdl.config.num_hidden_layers
    norm, head = mdl.model.norm, mdl.lm_head
    letter_ids = {c: tok.encode(c, add_special_tokens=False)[0] for c in "ABCDEFGH"}

    layer_total = defaultdict(list)     # layer -> [option mass per prompt]
    layer_asym = defaultdict(list)      # layer -> [winner - mean(others)]
    head_late = defaultdict(list)       # (layer, head) -> [option mass], L>=18
    LATE = int(0.75 * n_layers)

    for task in TASKS:
        for ex in load_task(task)[:N]:
            text = ex["input"]
            spans = option_spans(text)
            if len(spans) < 2:
                continue
            present = [s[2] for s in spans]
            prompt = text + "\nAnswer: ("
            enc = tok(prompt, return_offsets_mapping=True, return_tensors="pt")
            offsets = enc["offset_mapping"][0].tolist()
            opt_idx = [span_token_idx(offsets, s, e) for (s, e, _) in spans]
            if any(len(ix) == 0 for ix in opt_idx):
                continue
            with torch.no_grad():
                out = mdl(input_ids=enc["input_ids"].to(device),
                          attention_mask=enc["attention_mask"].to(device),
                          output_hidden_states=True, output_attentions=True)
            ans = enc["input_ids"].shape[1] - 1     # answer position (last token)
            cand = torch.tensor([letter_ids[c] for c in present], device=device)
            logits = head(norm(out.hidden_states[-1][0, ans]))
            chosen = int(torch.argmax(logits[cand]).item())

            for L in range(n_layers):
                A = out.attentions[L][0, :, ans, :].float().cpu().numpy()  # [heads, seq]
                head_mass = np.array([A[:, ix].sum(axis=1) for ix in opt_idx])  # [opt, heads]
                per_opt = head_mass.mean(axis=1)        # mean over heads -> [opt]
                layer_total[L].append(float(per_opt.sum()))
                others = np.delete(per_opt, chosen)
                layer_asym[L].append(float(per_opt[chosen] - others.mean()))
                if L >= LATE:
                    tot_by_head = head_mass.sum(axis=0)  # [heads] option mass this layer
                    for h, m in enumerate(tot_by_head):
                        head_late[(L, h)].append(float(m))

    print("Layer | option-span attn mass (answer pos) | winner asymmetry")
    print("-" * 60)
    for L in range(0, n_layers, 2):
        print(f"  L{L:2d}  |  {np.mean(layer_total[L]):.3f}"
              f"  {'█' * int(np.mean(layer_total[L]) * 40):<20s}"
              f"|  {np.mean(layer_asym[L]):+.4f}")

    print("\nTop 'selection heads' (layer, head) by option-span attention at L>=%d:" % LATE)
    ranked = sorted(head_late.items(), key=lambda kv: -np.mean(kv[1]))[:8]
    for (L, h), v in ranked:
        print(f"  L{L} H{h:2d}:  mean option mass = {np.mean(v):.3f}  (n={len(v)})")


if __name__ == "__main__":
    main()
