"""
calendar_events.py — name the heads that deliver the calendar to the
prediction site (Mistral-7B).

torus_hunt: the wd x mo torus assembles at the sentence-END token at L31,
after the month base-ring (L18+) and a late weekday delivery.
attention_leverage: end->wd max-head pulls 0.42@L19 and 0.50@L30. This
script identifies WHICH heads, using the validated per-head event detector
(attention_events v2, which blindly rediscovered SmolLM2's L20.H0).

Corpus: all 84 "They met on a {wd} in {mo}." cells + 16 diverse filler
prompts (fillers keep per-head baselines honest — calendar-specific
transport must stand out against generic behavior).

Reports:
  1. DELIVERY TABLE: per (layer, head), mean transported value
     weekday-span -> final token across the 84 calendar prompts (and
     month-span -> final token; and weekday-span -> month token). The
     ranked table names the delivery heads.
  2. Top head-events overall (v2 z) — discovery pass.
  3. Decomposition of the top delivery heads on a sample sentence
     (sources + OV direct-logit-attribution).

Run:  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python \
        experiments/calendar_events.py
Writes experiments/results/calendar_events.json.
"""
from __future__ import annotations
import json, os

import numpy as np

MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
WD = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday",
      "Sunday"]
MO = ["January", "February", "March", "April", "May", "June", "July",
      "August", "September", "October", "November", "December"]
TEMPLATE = "They met on a {wd} in {mo}."
FILLER = [
    "The rain fell softly on the quiet village all night.",
    "He opened the door and stepped into the cold morning air.",
    "What is four plus seven? The answer is",
    "The capital of France is",
    "Sarah gave the book to Emma because she had finished it.",
    "The basket held three apples, two pears, and one lemon.",
    "The oak is taller than the pine, but the redwood is tallest.",
    "for i in range(10): total = total + i",
    "El gato duerme en la casa.",
    "All cats are animals. Whiskers is a cat, so Whiskers is",
    "The trophy did not fit in the suitcase because it was too big.",
    "Water is made of hydrogen and",
    "She poured the tea and sat down by the window.",
    "SELECT name FROM users WHERE age > 30",
    "If it rains, the ground gets wet. It rained, therefore",
    "Gold is heavier than silver, and lead is heavier still.",
]
TOP_EVENTS = 15
TOP_DELIVERY = 10


def span_idxs(offs, cs, ce):
    return [k for k, (s, e) in enumerate(offs) if e > cs and s < ce]


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL)
    mdl = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.float16, attn_implementation="eager").to(dev)
    cfg = mdl.config
    n_layers, n_heads = cfg.num_hidden_layers, cfg.num_attention_heads
    n_kv = getattr(cfg, "num_key_value_heads", n_heads)
    d_head = cfg.hidden_size // n_heads
    rep = n_heads // n_kv
    W_U = mdl.lm_head.weight.detach().float().cpu().numpy()
    W_O = [lyr.self_attn.o_proj.weight.detach().float().cpu().numpy()
           for lyr in mdl.model.layers]

    v_store, o_store = {}, {}
    hooks = []
    for i, lyr in enumerate(mdl.model.layers):
        hooks.append(lyr.self_attn.v_proj.register_forward_hook(
            lambda m, inp, out, i=i: v_store.__setitem__(
                i, out.detach().float())))
        hooks.append(lyr.self_attn.o_proj.register_forward_pre_hook(
            lambda m, inp, i=i: o_store.__setitem__(
                i, inp[0].detach().float())))

    prompts = [("cal", TEMPLATE.format(wd=a, mo=b), a, b)
               for a in WD for b in MO]
    prompts += [("filler", t, None, None) for t in FILLER]
    print(f"{MODEL}: {n_layers}L x {n_heads}H (kv {n_kv}), "
          f"{len(prompts)} prompts", flush=True)

    samples = []
    sample_ix = {}          # (pi, L, t) -> sample, for delivery decomposition
    spans = {}              # pi -> (wd_span, mo_span, T)
    # delivery accumulators: (n_layers, n_heads) sums over calendar prompts
    dl_wd_end = np.zeros((n_layers, n_heads))
    dl_mo_end = np.zeros((n_layers, n_heads))
    dl_wd_mo = np.zeros((n_layers, n_heads))
    n_cal = 0

    for pi, (cat, text, a, b) in enumerate(prompts):
        enc = tok(text, return_offsets_mapping=True, return_tensors="pt")
        offs = enc.pop("offset_mapping")[0].tolist()
        ids = enc["input_ids"][0]
        T = len(ids)
        toks = [tok.decode([i]) for i in ids]
        wd_span = span_idxs(offs, text.rfind(a),
                            text.rfind(a) + len(a)) if a else []
        mo_span = span_idxs(offs, text.rfind(b),
                            text.rfind(b) + len(b)) if b else []
        v_store.clear(); o_store.clear()
        with torch.no_grad():
            out = mdl(**{k: v.to(dev) for k, v in enc.items()},
                      output_attentions=True)
        A = [x[0].float().cpu().numpy() for x in out.attentions]
        if cat == "cal":
            n_cal += 1
        for L in range(n_layers):
            v = v_store[L][0].cpu().numpy().reshape(T, n_kv, d_head)
            vn = np.repeat(np.linalg.norm(v, axis=-1), rep, axis=1).T  # [H,T]
            oin = o_store[L][0].cpu().numpy().reshape(
                T, n_heads, d_head).astype(np.float16)
            # A[L] is [H, tq, tk]; weight each key tk by its value norm
            AV = A[L] * vn[:, None, :]                      # [H, tq, tk]
            for t in range(1, T):
                w = AV[:, t, 1:t + 1]                        # [H, t]
                s = {"pi": pi, "cat": cat, "L": L, "t": t,
                     "_hs": w.sum(1), "_w": w,
                     "_oin": oin[t], "_toks": toks}
                samples.append(s)
                sample_ix[(pi, L, t)] = s
            spans[pi] = (wd_span, mo_span, T)
            if cat == "cal":
                tf = T - 1
                if wd_span:
                    dl_wd_end[L] += AV[:, tf, wd_span].sum(-1)
                    if mo_span:
                        tm = mo_span[-1]
                        dl_wd_mo[L] += AV[:, tm, wd_span].sum(-1)
                if mo_span:
                    dl_mo_end[L] += AV[:, tf, mo_span].sum(-1)

    for h in hooks:
        h.remove()

    # per-head z baselines
    hs_sum = np.zeros((n_layers, n_heads)); hs_sq = np.zeros((n_layers, n_heads))
    hs_n = np.zeros((n_layers, n_heads))
    for s in samples:
        hs_sum[s["L"]] += s["_hs"]; hs_sq[s["L"]] += s["_hs"] ** 2
        hs_n[s["L"]] += 1
    mu_h = hs_sum / hs_n
    sd_h = np.sqrt(np.maximum(hs_sq / hs_n - mu_h ** 2, 0)) + 1e-9

    def decompose(e, h):
        toks = e["_toks"]; w = e["_w"]; L = e["L"]
        srcs = np.argsort(-w[h])[:3]
        src_s = ", ".join(f"'{toks[s + 1]}'({w[h, s]:.1f})"
                          for s in srcs if w[h, s] > 0)
        contrib = e["_oin"][h].astype(np.float32) @ \
            W_O[L][:, h * d_head:(h + 1) * d_head].T
        logits = contrib @ W_U.T
        ov = ", ".join(tok.decode([int(i)]).strip() or "␣"
                       for i in np.argsort(-logits)[:5])
        return src_s, ov

    # ---- 1. delivery tables ---------------------------------------------
    results: dict = {"model": MODEL, "delivery": {}}
    for name, tbl in (("wd->end", dl_wd_end), ("mo->end", dl_mo_end),
                      ("wd->mo_tok", dl_wd_mo)):
        m = tbl / max(n_cal, 1)
        flat = [(float(m[L, h]), L, h) for L in range(n_layers)
                for h in range(n_heads)]
        flat.sort(key=lambda x: -x[0])
        print(f"\n=== delivery: {name} (mean transported value, "
              f"{n_cal} calendar prompts) ===")
        rows = []
        pi0 = 0                              # first calendar prompt as example
        wd0, mo0, T0 = spans[pi0]
        for val, L, h in flat[:TOP_DELIVERY]:
            t_ex = mo0[-1] if name == "wd->mo_tok" else T0 - 1
            ex = sample_ix[(pi0, L, t_ex)]
            src_s, ov = decompose(ex, h)
            rows.append({"L": L, "head": h, "mean_delivery": round(val, 2),
                         "example_sources": src_s, "ov_top": ov})
            print(f"  L{L:2d} H{h:2d}  {val:6.2f}   e.g. <- {src_s}")
            print(f"           OV writes: {ov}")
        results["delivery"][name] = rows

    # ---- 2. global head-events (discovery) ------------------------------
    head_events = []
    for s in samples:
        zh = (s["_hs"] - mu_h[s["L"]]) / sd_h[s["L"]]
        for h in np.where(zh >= 5.0)[0]:
            head_events.append((float(zh[h]), s, int(h)))
    head_events.sort(key=lambda e: -e[0])
    print(f"\n=== top {TOP_EVENTS} head-events (z>=5) ===")
    ev_out = []
    for z, e, h in head_events[:TOP_EVENTS]:
        src_s, ov = decompose(e, h)
        rec = {"z": round(z, 1), "L": e["L"], "head": h,
               "cat": e["cat"], "text": prompts[e["pi"]][1],
               "dest": e["_toks"][e["t"]],
               "sources": src_s, "ov_top": ov}
        ev_out.append(rec)
        print(f"\nz={z:5.1f} L{e['L']:2d} H{h:2d} [{rec['cat']:6s}] "
              f"dest='{rec['dest']}'  \"{rec['text'][:55]}\"")
        print(f"    <- {src_s}")
        print(f"    OV writes: {ov}")
    results["head_events"] = ev_out

    outp = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "results", "calendar_events.json")
    with open(outp, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {outp}")


if __name__ == "__main__":
    main()
