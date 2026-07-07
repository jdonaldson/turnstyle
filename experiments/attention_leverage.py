"""
attention_leverage.py — WHERE does Mistral leverage attention? (the SmolLM2
comparison, and the transport-timing clue for the torus question)

SmolLM2's map (attention_sinks + hidden_state_analysis, 2026-03/04): from L8
on, 65-81% of attention parks on the pos-0 sink; leverage is a few LATE
targeted heads (L20.H0 operand-copy 45-57%, L22-23 assembly) feeding the L23
MLP which does ~70% of the compute. Attention = late transport, MLP = compute.
GPT-2 is the opposite (L11 attention-dominated) — a trained strategy, not an
architectural constant.

Why it matters here: at the month token of "a {wd} in {mo}", weekday identity
can only arrive VIA attention. torus_hunt finds weekday readable (cv~.96) but
never ring-shaped at that anchor. If Mistral's cross-token transport is LATE
(SmolLM2-style), the weekday arrives after the mid-stack shape-formation
window — a mechanistic explanation for order-without-topology at the
composite level. If transport is mid-stack and the torus still doesn't form,
the failure is representational, not timing.

Conditions:
  calendar  84 wd x mo pairs x 2 templates; anchors = month-token & final.
            Spans: weekday, pos0 (sink), month-local.
  arith     word arithmetic ("What is four plus seven? The answer is"),
            anchor = final; spans: operands, pos0. Apples-to-apples with the
            SmolLM2 answer-flow map.

Per layer: attention mass anchor->span (head-mean and head-max), and
||attn_out|| / ||mlp_out|| at the anchor (the leverage curve).

Run:  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python \
        experiments/attention_leverage.py [model_id]
Writes experiments/results/attention_leverage.json.
"""
from __future__ import annotations
import json, os, sys

import numpy as np

WD = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday",
      "Sunday"]
MO = ["January", "February", "March", "April", "May", "June", "July",
      "August", "September", "October", "November", "December"]
T_CAL = ["They met on a {a} in {b}.", "It happened on a {a} in {b}."]
NUMW = ["two", "three", "four", "five", "six", "seven", "eight", "nine"]


def span_token_idxs(offs, cs, ce):
    return [k for k, (s, e) in enumerate(offs) if e > cs and s < ce]


def run_prompts(mdl, tok, dev, prompts, n_layers):
    """prompts: list of (sentence, spans_dict, anchors_dict) where spans/
    anchors give char ranges. Returns aggregated per-layer metrics."""
    import torch
    attn_outs, mlp_outs = {}, {}
    hooks = []
    for i, lyr in enumerate(mdl.model.layers):
        hooks.append(lyr.self_attn.register_forward_hook(
            lambda m, inp, out, i=i: attn_outs.__setitem__(
                i, out[0].detach())))
        hooks.append(lyr.mlp.register_forward_hook(
            lambda m, inp, out, i=i: mlp_outs.__setitem__(i, out.detach())))

    agg = {}  # (anchor, span or 'ratio') -> [n_layers] running sums
    def add(key, L, v):
        arr = agg.setdefault(key, np.zeros((n_layers, 2)))
        arr[L, 0] += v; arr[L, 1] += 1

    try:
        for sent, spans, anchors in prompts:
            enc = tok(sent, return_offsets_mapping=True, return_tensors="pt")
            offs = enc.pop("offset_mapping")[0].tolist()
            enc = {k: v.to(dev) for k, v in enc.items()}
            attn_outs.clear(); mlp_outs.clear()
            with torch.no_grad():
                out = mdl(**enc, output_attentions=True)
            A = [a[0].float().cpu().numpy() for a in out.attentions]
            span_idx = {nm: span_token_idxs(offs, cs, ce)
                        for nm, (cs, ce) in spans.items()}
            span_idx["pos0"] = [0]
            for aname, (cs, ce) in anchors.items():
                idxs = span_token_idxs(offs, cs, ce)
                q = idxs[-1] if idxs else len(offs) - 1
                for L in range(n_layers):
                    for nm, sidx in span_idx.items():
                        if not sidx or min(sidx) > q:
                            continue
                        m = A[L][:, q, sidx].sum(-1)   # per head
                        add((aname, nm, "mean"), L, float(m.mean()))
                        add((aname, nm, "max"), L, float(m.max()))
                    ao = float(attn_outs[L][0, q].norm())
                    mo_ = float(mlp_outs[L][0, q].norm())
                    add((aname, "ratio", "attn/mlp"), L, ao / (mo_ + 1e-9))
    finally:
        for h in hooks:
            h.remove()
    return {k: (v[:, 0] / np.maximum(v[:, 1], 1)).round(4).tolist()
            for k, v in agg.items()}


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    mid = sys.argv[1] if len(sys.argv) > 1 else "mistralai/Mistral-7B-Instruct-v0.3"
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(mid)
    mdl = AutoModelForCausalLM.from_pretrained(
        mid, dtype=torch.float16, attn_implementation="eager").to(dev)
    n_layers = mdl.config.num_hidden_layers
    results: dict = {"model": mid}

    # --- calendar ---------------------------------------------------------
    prompts = []
    for a in WD:
        for b in MO:
            for tmpl in T_CAL:
                sent = tmpl.format(a=a, b=b)
                ca = sent.rfind(a); cb = sent.rfind(b)
                prompts.append((sent,
                                {"wd": (ca, ca + len(a)),
                                 "mo": (cb, cb + len(b))},
                                {"mo_tok": (cb, cb + len(b)),
                                 "end": (len(sent) - 1, len(sent))}))
    print(f"calendar: {len(prompts)} prompts", flush=True)
    cal = run_prompts(mdl, tok, dev, prompts, n_layers)
    results["calendar"] = cal

    # --- arithmetic -------------------------------------------------------
    prompts = []
    for a in NUMW:
        for b in NUMW:
            if a == b:
                continue
            sent = f"What is {a} plus {b}? The answer is"
            ca = sent.find(a); cb = sent.rfind(b)
            prompts.append((sent,
                            {"op1": (ca, ca + len(a)),
                             "op2": (cb, cb + len(b))},
                            {"end": (len(sent) - 1, len(sent))}))
    print(f"arith: {len(prompts)} prompts", flush=True)
    ar = run_prompts(mdl, tok, dev, prompts, n_layers)
    results["arith"] = ar

    # --- print the two maps -----------------------------------------------
    print("\n=== calendar: attention leverage per layer ===")
    print("  L | mo_tok->wd mean/max | ->pos0 | end->wd mean/max | "
          "attn/mlp(mo_tok)")
    for L in range(n_layers):
        print(f" {L:2d} | {cal[('mo_tok','wd','mean')][L]:.3f} "
              f"{cal[('mo_tok','wd','max')][L]:.3f} | "
              f"{cal[('mo_tok','pos0','mean')][L]:.3f} | "
              f"{cal[('end','wd','mean')][L]:.3f} "
              f"{cal[('end','wd','max')][L]:.3f} | "
              f"{cal[('mo_tok','ratio','attn/mlp')][L]:.2f}")
    print("\n=== arith: attention leverage per layer ===")
    print("  L | end->op1 mean/max | end->op2 mean/max | ->pos0 | attn/mlp")
    for L in range(n_layers):
        print(f" {L:2d} | {ar[('end','op1','mean')][L]:.3f} "
              f"{ar[('end','op1','max')][L]:.3f} | "
              f"{ar[('end','op2','mean')][L]:.3f} "
              f"{ar[('end','op2','max')][L]:.3f} | "
              f"{ar[('end','pos0','mean')][L]:.3f} | "
              f"{ar[('end','ratio','attn/mlp')][L]:.2f}")

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "results", "attention_leverage.json")
    payload: dict = {"model": mid}
    for cond in ("calendar", "arith"):
        payload[cond] = {"|".join(k): v for k, v in results[cond].items()}
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
