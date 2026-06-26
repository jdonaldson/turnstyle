"""Do the 4 'knowledge wall' BBH tasks carry EXTRACTABLE signal in SmolLM2's hidden
states (recognition) even though generation is flat at baseline? Run autoprobe (layer ×
finder × mode sweep, 5-fold CV, cheap-baseline lift, ship rule) on each. If a probe
ships (CV >= 60% AND >= +10pp over the best cheap baseline), the wall is a GENERATION
limit, not a representation limit -- and a probe moves it, pure SmolLM2.

  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python experiments/knowledge_extract.py
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
from turnstyle.bbh import load_task
from turnstyle.autoprobe import autoprobe

# (task, generation baseline on this model: 3-shot swollm / zero-shot native)
TASKS = [
    ("causal_judgement", "58.7 / 50.0", 0.50),          # binary, chance .50
    ("sports_understanding", "54.7 / 35.0", 0.50),      # binary, chance .50
    ("movie_recommendation", "22.3 / 17.5", 0.25),      # MC ~4-5 opt
    ("salient_translation_error_detection", "13.8 / 7.5", 0.167),  # 6-way MC
]


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    mid = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(mid)
    mdl = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.float16).to(dev)
    print(f"== {mid} on {dev} ==", flush=True)

    summary = []
    for task, gen_base, chance in TASKS:
        exs = load_task(task)
        print(f"\n{'='*70}\n{task}  (gen baseline {gen_base}, chance {chance:.2f}, N={len(exs)})\n{'='*70}", flush=True)
        res = autoprobe(exs, lambda ex: ex["target"].strip(), mdl, tok, dev, verbose=True)
        chosen = res.chosen
        cv = chosen[3] if chosen else None
        best_cheap = max(res.cheap_baselines.values()) if res.cheap_baselines else None
        summary.append((task, gen_base, chance, cv, best_cheap, res.ship, res.reason))

    print(f"\n\n{'#'*72}\nSUMMARY: extractable knowledge in SmolLM2?\n{'#'*72}")
    print(f"{'task':42s} {'gen':>11s} {'chance':>6s} {'probeCV':>8s} {'cheap':>6s} {'SHIP':>5s}")
    for task, gen_base, chance, cv, cheap, ship, reason in summary:
        cvs = f"{cv:.1%}" if cv is not None else "  -  "
        chs = f"{cheap:.1%}" if cheap is not None else "  -  "
        print(f"{task:42s} {gen_base:>11s} {chance:>6.2f} {cvs:>8s} {chs:>6s} {str(ship):>5s}  {reason}")


if __name__ == "__main__":
    main()
