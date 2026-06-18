"""Causal confirmation: are the late selection heads pulling candidates together
for the argmax, or just reporting a decision already made?

Intervention: mask the ANSWER position's attention to a chosen span at chosen
layers (attention knockout via per-layer additive mask), then see if the
model's pick flips.

Conditions:
  KO_winner_late  : mask answer->WINNER span at L18-21   (predict: many flips)
  KO_winner_early : mask answer->WINNER span at L8-11     (control: few flips)
  KO_runnerup_late: mask answer->RUNNER-UP span at L18-21 (control: few flips)

If only KO_winner_late flips picks, the late answer->winner attention edges are
causally load-bearing for selection.
"""
from __future__ import annotations

import json
import re

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
BBH_CACHE = "/Users/jdonaldson/Projects/swollm/data/bbh_cache"
TASKS = ["snarks", "date_understanding", "logical_deduction_three_objects",
         "movie_recommendation"]
N = 15
OPTION_RE = re.compile(r"^\(([A-Z])\)", re.MULTILINE)
LATE = [18, 19, 20, 21]
EARLY = [8, 9, 10, 11]


def load_task(name):
    with open(f"{BBH_CACHE}/{name}.json") as f:
        return json.load(f)


def option_spans(text):
    marks = list(OPTION_RE.finditer(text))
    return [(m.start(),
             marks[k + 1].start() if k + 1 < len(marks) else len(text),
             m.group(1)) for k, m in enumerate(marks)]


def span_tokens(offsets, s, e):
    return [ti for ti, (a, b) in enumerate(offsets) if a != b and s <= (a + b) / 2 < e]


_edits = [0]


def make_hook(ans_pos, ko_tokens):
    def hook(module, args, kwargs):
        am = kwargs.get("attention_mask", None)
        if isinstance(am, torch.Tensor):
            am = am.clone()
            am[..., ans_pos, ko_tokens] = torch.finfo(am.dtype).min
            kwargs["attention_mask"] = am
            _edits[0] += 1
            return (args, kwargs)
        return None
    return hook


def run(mdl, ids, cand, present, layers=None, ans_pos=None, ko_tokens=None):
    handles = []
    if layers is not None:
        for L in layers:
            h = mdl.model.layers[L].self_attn.register_forward_pre_hook(
                make_hook(ans_pos, ko_tokens), with_kwargs=True)
            handles.append(h)
    try:
        with torch.no_grad():
            out = mdl(input_ids=ids, output_hidden_states=True)
            logits = mdl.lm_head(mdl.model.norm(out.hidden_states[-1][0, -1]))
            probs = torch.softmax(logits[cand], dim=-1)
        pick = int(torch.argmax(probs).item())
        return pick, probs.float().cpu().numpy()
    finally:
        for h in handles:
            h.remove()


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    mdl = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.float16, attn_implementation="eager").to(device).eval()
    print(f"model loaded on {device}\n", flush=True)
    letter_ids = {c: tok.encode(c, add_special_tokens=False)[0] for c in "ABCDEFGH"}

    flips = {"winner_late": 0, "winner_early": 0, "runnerup_late": 0}
    wprob_drop = {"winner_late": [], "winner_early": [], "runnerup_late": []}
    total = 0
    for task in TASKS:
        for ex in load_task(task)[:N]:
            text = ex["input"]
            spans = option_spans(text)
            if len(spans) < 2:
                continue
            present = [s[2] for s in spans]
            enc = tok(text + "\nAnswer: (", return_offsets_mapping=True, return_tensors="pt")
            offsets = enc["offset_mapping"][0].tolist()
            ids = enc["input_ids"].to(device)
            opt_tokens = [span_tokens(offsets, s, e) for (s, e, _) in spans]
            if any(len(t) == 0 for t in opt_tokens):
                continue
            ans = ids.shape[1] - 1
            cand = torch.tensor([letter_ids[c] for c in present], device=device)

            base_pick, base_probs = run(mdl, ids, cand, present)
            winner = base_pick
            order = np.argsort(-base_probs)
            runnerup = int(order[1]) if order[0] == winner else int(order[0])

            for name, layers, ko in [
                ("winner_late", LATE, winner),
                ("winner_early", EARLY, winner),
                ("runnerup_late", LATE, runnerup),
            ]:
                pick, probs = run(mdl, ids, cand, present, layers=layers,
                                  ans_pos=ans, ko_tokens=opt_tokens[ko])
                if pick != base_pick:
                    flips[name] += 1
                wprob_drop[name].append(float(base_probs[winner] - probs[winner]))
            total += 1
        print(f"  done {task}", flush=True)

    print(f"\nn={total} prompts, mask edits applied={_edits[0]} "
          f"(should be ~{total*3*4})\n")
    print("Condition         | pick-flip rate | mean winner-prob drop")
    print("-" * 58)
    for name in ["winner_late", "winner_early", "runnerup_late"]:
        fr = flips[name] / total if total else 0
        wd = np.mean(wprob_drop[name]) if wprob_drop[name] else float("nan")
        print(f"  {name:16s}|   {fr:6.1%}      |  {wd:+.4f}")
    print("\nPredict: winner_late >> winner_early and >> runnerup_late "
          "if late answer->winner attention is causal.")


if __name__ == "__main__":
    main()
