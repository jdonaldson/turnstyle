"""Find demo examples that TRIP vanilla SmolLM2 but turnstyle gets right.
For each candidate: vanilla greedy answer, turnstyle (source/text/proof), target,
and a verdict (good trip-up = vanilla wrong AND turnstyle right). Also surfaces the
current proof string per solver so we know what to enrich.
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from turnstyle.bbh import load_task, answer_matches
from turnstyle.dispatch_turnstyle import DispatchTurnstyle

MID = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
dev = "mps" if torch.backends.mps.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained(MID)
mdl = AutoModelForCausalLM.from_pretrained(MID, dtype=torch.float16).to(dev)
dt = DispatchTurnstyle(mdl, tok, dev)

# (task, indices) — hard, deterministic tasks where SmolLM2 should struggle.
CANDIDATES = [
    ("multistep_arithmetic_two", [0, 3, 7]),
    ("date_understanding", [5, 8, 11, 22]),
    ("tracking_shuffled_objects_three_objects", [0, 2]),
    ("tracking_shuffled_objects_five_objects", [0, 2]),
    ("dyck_languages", [3, 6, 9]),
    ("logical_deduction_five_objects", [0, 2]),
    ("word_sorting", [4, 9]),
    ("boolean_expressions", [4, 9]),
]


def vanilla(prompt):
    t = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
    inp = tok(t, return_tensors="pt").to(dev)
    with torch.no_grad():
        o = mdl.generate(**inp, max_new_tokens=80, do_sample=False)
    return tok.decode(o[0][inp["input_ids"].shape[1]:], skip_special_tokens=True).strip()


for task, idxs in CANDIDATES:
    exs = load_task(task)
    for i in idxs:
        if i >= len(exs):
            continue
        ex = exs[i]; tgt = ex["target"].strip()
        v = vanilla(ex["input"])
        ans = dt.parse(ex["input"])
        v_ok = answer_matches(v, tgt)
        t_ok = ans is not None and answer_matches(ans.text, tgt)
        verdict = "★ TRIP-UP" if (not v_ok and t_ok) else ("both-ok" if (v_ok and t_ok) else "check")
        print(f"\n[{task} #{i}] target={tgt!r}  {verdict}")
        print(f"  vanilla({v_ok}): {v[:75]!r}")
        print(f"  turnstyle({t_ok}): src={ans.source if ans else None} text={ans.text if ans else None}")
        print(f"  proof: {ans.proof if ans else None}")
