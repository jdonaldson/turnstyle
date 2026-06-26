"""DateExtract viability diagnostic: can SmolLM2 do the EXTRACTION half of a
date_understanding problem, leaving arithmetic to symbolic?

Asks the model for JSON {today: ISO, offset: {count,unit,dir}} over the full problem
(NOT the final answer — that stays symbolic). Then deterministically applies the offset
via relativedelta, formats MM/DD/YYYY, matches an option. Categorizes every failure so
we see SmolLM2's actual failure modes (the diagnose-first step before designing a schema):
  json_fail | bad_today | bad_offset | no_match(resolved but wrong) | OK

  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python experiments/dateextract_diag.py
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import json, re
from datetime import date
from dateutil.relativedelta import relativedelta

from turnstyle.bbh import load_task, answer_matches
from turnstyle.ir import _parse_json

PROMPT = """You extract date information from a word problem. Output ONLY a JSON object with two keys:
- "today": the reference date the problem establishes, in YYYY-MM-DD format. Resolve holidays and described dates to an actual calendar date.
- "offset": an object {{"count": integer, "unit": one of "day","week","month","year", "dir": "before" or "after"}} for the shift the QUESTION asks for. Use count 0 if it asks for today itself.
Do NOT compute the final answer.

Problem: It is 03/15/2020 today. What is the date one week from today?
JSON: {{"today": "2020-03-15", "offset": {{"count": 1, "unit": "week", "dir": "after"}}}}

Problem: Yesterday was New Year's Day of 1990. What is the date 3 days ago?
JSON: {{"today": "1990-01-02", "offset": {{"count": 3, "unit": "day", "dir": "before"}}}}

Problem: {problem}
JSON:"""

UNIT = {"day": "days", "days": "days", "week": "weeks", "weeks": "weeks",
        "month": "months", "months": "months", "year": "years", "years": "years"}


def options_of(text):
    sec = text.split("Options:")[-1]
    return dict(re.findall(r'\(([A-Z])\)\s+([^\n(]+)', sec))


def resolve(data):
    """data -> (MM/DD/YYYY string) or raise. Symbolic; model only supplied fields."""
    today = date.fromisoformat(data["today"].strip())
    off = data.get("offset") or {}
    cnt = int(off.get("count", 0))
    unit = UNIT.get(str(off.get("unit", "day")).lower(), "days")
    if str(off.get("dir", "after")).lower().startswith("bef"):
        cnt = -cnt
    result = today + relativedelta(**{unit: cnt})
    return result.strftime("%m/%d/%Y")


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    mid = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(mid)
    mdl = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.float16).to(dev)
    print(f"model on {dev}", flush=True)

    exs = load_task("date_understanding")[:40]
    cats = {"json_fail": 0, "bad_today": 0, "bad_offset": 0, "no_match": 0, "OK": 0}
    for i, ex in enumerate(exs):
        problem = ex["input"].split("Options:")[0].strip()
        tgt = ex["target"].strip()
        opts = options_of(ex["input"])
        msg = [{"role": "user", "content": PROMPT.format(problem=problem)}]
        ct = tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        enc = tok(ct, return_tensors="pt").to(dev)
        with torch.no_grad():
            out = mdl.generate(**enc, max_new_tokens=80, do_sample=False)
        resp = tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        data = _parse_json(resp)

        cat, got = None, None
        if not isinstance(data, dict) or "today" not in data:
            cat = "json_fail"
        else:
            try:
                got = resolve(data)
            except Exception:
                cat = "bad_today"
            if cat is None:
                # find which option matches the resolved date -> letter
                letter = next((f"({k})" for k, v in opts.items() if v.strip() == got), None)
                if letter is None:
                    cat = "no_match"
                else:
                    cat = "OK" if answer_matches(letter, tgt) else "no_match"
                    got = f"{got}->{letter}"
        cats[cat] += 1
        flag = "OK " if cat == "OK" else "XX "
        print(f"[{i:2d}] {flag}{cat:9s} resp={resp[:70]!r} got={got} tgt={tgt}", flush=True)

    n = len(exs)
    print(f"\n=== {cats} | OK {cats['OK']}/{n} = {cats['OK']/n*100:.1f}% ===")


if __name__ == "__main__":
    main()
