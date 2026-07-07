"""
attention_events.py — unsupervised attention-event detection on SmolLM2.

Premise (from attention_leverage + the SmolLM2 answer-flow work): attention's
resting state is PARKED on sinks; computation announces itself when heads
un-park and transport real content. Raw attention mass is a bad importance
signal (sinks get the most); the honest event score weights attention by the
VALUE NORM of the source — sinks/punctuation have near-zero values, so they
self-discount.

Event score, per (layer, dest position t>0), summed over heads:
    score(l, t) = sum_h sum_{s>0} A[l,h,t,s] * ||v[l,h,s]||
z-scored WITHIN layer across every (prompt, position) in the corpus.
No topics, no probes, no labels — events are wherever transport spikes.

Decomposition per event (working backwards to WHAT was computed):
  - top heads by transported value
  - each head's top source tokens (what was fetched, from where)
  - OV direct-logit-attribution: the head's output slice through W_O then
    the unembedding — what the head WROTE, in vocabulary space.

Planted validation: the corpus includes arithmetic — the known SmolLM2
signature (answer-position event ~L20, sources = operands, OV = operand
word forms) must surface in the top events unprompted, or the detector is
not trustworthy.

Scope caveat: "all events" means all events THIS CORPUS elicits. The
detector is topic-free; coverage is corpus-relative.

Run:  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python \
        experiments/attention_events.py
Writes experiments/results/attention_events.json.
"""
from __future__ import annotations
import json, os

import numpy as np

MODEL = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
TOP_EVENTS = 30
Z_EVENT = 3.0

CORPUS = [
    ("arith", "What is four plus seven? The answer is"),
    ("arith", "What is six times three? The answer is"),
    ("arith", "Nine minus two equals"),
    ("arith", "If you have five apples and eat two, you are left with"),
    ("arith", "Twelve divided by four is"),
    ("arith", "The sum of eight and eight is"),
    ("dates", "Today is Tuesday. Tomorrow will be"),
    ("dates", "The meeting was moved from Friday to the following Monday."),
    ("dates", "She was born in March and her brother in November."),
    ("dates", "Christmas falls on December 25th every year."),
    ("dates", "The deadline is the last day of February."),
    ("dates", "They met on a Monday in January."),
    ("recall", "The capital of France is"),
    ("recall", "Water is made of hydrogen and"),
    ("recall", "The author of Romeo and Juliet is"),
    ("recall", "The largest planet in the solar system is"),
    ("recall", "The chemical symbol for gold is"),
    ("recall", "Mount Everest is located in"),
    ("coref", "Sarah gave the book to Emma because she had finished it."),
    ("coref", "The trophy did not fit in the suitcase because it was too big."),
    ("coref", "Tom called his father, and he was happy to hear from him."),
    ("coref", "The dog chased the cat until it climbed a tree."),
    ("coref", "Anna told Maria that her car was blocking the driveway."),
    ("coref", "The lawyer questioned the witness, and she answered calmly."),
    ("count", "The basket held three apples, two pears, and one lemon."),
    ("count", "I have a piano, a violin, a drum, and a flute."),
    ("count", "The list contains red, blue, green, and yellow items."),
    ("count", "There were four cars, two trucks, and a motorcycle outside."),
    ("compare", "The oak is taller than the pine, but the redwood is tallest."),
    ("compare", "Gold is heavier than silver, and lead is heavier still."),
    ("compare", "A whale is bigger than a dolphin, which is bigger than a fish."),
    ("compare", "The red house is older than the blue one."),
    ("narrative", "The rain fell softly on the quiet village all night."),
    ("narrative", "He opened the door and stepped into the cold morning air."),
    ("narrative", "The children laughed as the kite rose above the trees."),
    ("narrative", "She poured the tea and sat down by the window."),
    ("code", "for i in range(10): total = total + i"),
    ("code", "if x > 5 and y < 3: return True"),
    ("code", "The function takes a list and returns its length."),
    ("code", "SELECT name FROM users WHERE age > 30"),
    ("multiling", "El gato duerme en la casa."),
    ("multiling", "Le train arrive à la gare."),
    ("multiling", "Der Hund läuft im Park."),
    ("multiling", "The word for bread in Spanish is"),
    ("logic", "All cats are animals. Whiskers is a cat, so Whiskers is"),
    ("logic", "If it rains, the ground gets wet. It rained, therefore"),
    ("logic", "Either the key is in the drawer or on the hook. It is not in the drawer."),
    ("logic", "No fish can fly. A salmon is a fish, so a salmon cannot"),
]


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
    W_U = mdl.lm_head.weight.detach().float().cpu().numpy()   # [V, d_model]
    W_O = [lyr.self_attn.o_proj.weight.detach().float().cpu().numpy()
           for lyr in mdl.model.layers]                # [d_model, d_model]

    v_store, o_store = {}, {}
    hooks = []
    for i, lyr in enumerate(mdl.model.layers):
        hooks.append(lyr.self_attn.v_proj.register_forward_hook(
            lambda m, inp, out, i=i: v_store.__setitem__(
                i, out.detach().float())))
        hooks.append(lyr.self_attn.o_proj.register_forward_pre_hook(
            lambda m, inp, i=i: o_store.__setitem__(
                i, inp[0].detach().float())))

    samples = []      # per (prompt, layer, t): score + context for decomp
    per_layer_scores = [[] for _ in range(n_layers)]
    print(f"{MODEL}: {n_layers}L x {n_heads}H (kv {n_kv}), "
          f"{len(CORPUS)} prompts", flush=True)

    for pi, (cat, text) in enumerate(CORPUS):
        enc = tok(text, return_tensors="pt")
        ids = enc["input_ids"][0]
        T = len(ids)
        toks = [tok.decode([i]) for i in ids]
        v_store.clear(); o_store.clear()
        import torch as _t
        with _t.no_grad():
            out = mdl(**{k: v.to(dev) for k, v in enc.items()},
                      output_attentions=True)
        A = [a[0].float().cpu().numpy() for a in out.attentions]  # [H,T,T]
        for L in range(n_layers):
            v = v_store[L][0].cpu().numpy().reshape(T, n_kv, d_head)
            vn = np.linalg.norm(v, axis=-1)            # [T, n_kv]
            vn_h = np.repeat(vn, rep, axis=1).T        # [H, T]
            oin = o_store[L][0].cpu().numpy().reshape(T, n_heads, d_head)
            for t in range(1, T):
                w = A[L][:, t, 1:t + 1] * vn_h[:, 1:t + 1]   # [H, t] non-sink
                sc = float(w.sum())
                per_layer_scores[L].append(sc)
                samples.append({"pi": pi, "cat": cat, "L": L, "t": t,
                                "score": sc,
                                "sink": float(A[L][:, t, 0].mean()),
                                "_w": w, "_oin": oin[t], "_toks": toks})

    for h in hooks:
        h.remove()

    def decompose_head(e, h):
        toks = e["_toks"]; w = e["_w"]; L = e["L"]
        srcs = np.argsort(-w[h])[:3]
        src_s = ", ".join(f"'{toks[s + 1]}'({w[h, s]:.1f})"
                          for s in srcs if w[h, s] > 0)
        contrib = e["_oin"][h] @ W_O[L][:, h * d_head:(h + 1) * d_head].T
        logits = contrib @ W_U.T
        top_v = np.argsort(-logits)[:5]
        ov = ", ".join(tok.decode([int(i)]).strip() or "␣" for i in top_v)
        return {"head": int(h), "sources": src_s, "ov_top": ov}

    # ---- v1: layer-sum events -------------------------------------------
    mu = [np.mean(s) for s in per_layer_scores]
    sd = [np.std(s) + 1e-9 for s in per_layer_scores]
    for s in samples:
        s["z"] = (s["score"] - mu[s["L"]]) / sd[s["L"]]

    counts = {}
    for s in samples:
        if s["z"] >= Z_EVENT:
            counts[s["L"]] = counts.get(s["L"], 0) + 1
    print("\n=== v1 per-layer event counts (layer-sum z>=3) ===")
    print(" ".join(f"L{L}:{counts.get(L, 0)}" for L in range(n_layers)))

    # ---- v2: per-HEAD events (an event is a head-level outlier) ---------
    hs_sum = np.zeros((n_layers, n_heads))
    hs_sq = np.zeros((n_layers, n_heads))
    hs_n = np.zeros((n_layers, n_heads))
    for s in samples:
        hs = s["_w"].sum(1)                      # [H]
        s["_hs"] = hs
        hs_sum[s["L"]] += hs
        hs_sq[s["L"]] += hs ** 2
        hs_n[s["L"]] += 1
    mu_h = hs_sum / hs_n
    sd_h = np.sqrt(np.maximum(hs_sq / hs_n - mu_h ** 2, 0)) + 1e-9

    head_events = []
    for s in samples:
        zh = (s["_hs"] - mu_h[s["L"]]) / sd_h[s["L"]]
        for h in np.where(zh >= Z_EVENT)[0]:
            head_events.append((float(zh[h]), s, int(h)))
    head_events.sort(key=lambda e: -e[0])

    h_counts = {}
    for z, s, h in head_events:
        h_counts[s["L"]] = h_counts.get(s["L"], 0) + 1
    print("\n=== v2 per-layer head-event counts (per-head z>=3) ===")
    print(" ".join(f"L{L}:{h_counts.get(L, 0)}" for L in range(n_layers)))

    print(f"\n=== v2 top {TOP_EVENTS} head-events ===", flush=True)
    out_events = []
    for z, e, h in head_events[:TOP_EVENTS]:
        hi = decompose_head(e, h)
        rec = {"cat": e["cat"], "text": CORPUS[e["pi"]][1], "L": e["L"],
               "dest": e["_toks"][e["t"]], "z": round(z, 1),
               "sink": round(e["sink"], 2), "head": hi}
        out_events.append(rec)
        print(f"\nz={z:5.1f} L{e['L']:2d} H{h:2d} [{rec['cat']:9s}] "
              f"dest='{rec['dest']}'  \"{rec['text'][:58]}\"")
        print(f"    <- {hi['sources']}")
        print(f"    OV writes: {hi['ov_top']}")

    # ---- planted validation: best late-layer arith head-event -----------
    arith = [(z, s, h) for z, s, h in head_events
             if s["cat"] == "arith" and s["L"] >= 12]
    print("\n=== planted arithmetic validation (per-head, L>=12) ===")
    if arith:
        for z, s, h in arith[:5]:
            hi = decompose_head(s, h)
            print(f"  z={z:.1f} L{s['L']} H{h} dest='{s['_toks'][s['t']]}' "
                  f"\"{CORPUS[s['pi']][1][:45]}\" <- {hi['sources']} | "
                  f"OV: {hi['ov_top']}")
    else:
        print("  STILL MISSING — per-head z did not recover it either")

    outp = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "results", "attention_events.json")
    with open(outp, "w") as f:
        json.dump({"model": MODEL, "z_event": Z_EVENT,
                   "v1_layer_event_counts": {str(k): v for k, v in
                                             sorted(counts.items())},
                   "v2_head_event_counts": {str(k): v for k, v in
                                            sorted(h_counts.items())},
                   "head_events": out_events,
                   "arith_validation": [
                       {"z": round(z, 1), "L": s["L"], "head": h,
                        "dest": s["_toks"][s["t"]],
                        "text": CORPUS[s["pi"]][1]}
                       for z, s, h in arith[:5]]}, f, indent=2)
    print(f"\nwrote {outp}")


if __name__ == "__main__":
    main()
