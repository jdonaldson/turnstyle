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

    mu = [np.mean(s) for s in per_layer_scores]
    sd = [np.std(s) + 1e-9 for s in per_layer_scores]
    for s in samples:
        s["z"] = (s["score"] - mu[s["L"]]) / sd[s["L"]]

    events = sorted(samples, key=lambda s: -s["z"])[:TOP_EVENTS]
    counts = {}
    for s in samples:
        if s["z"] >= Z_EVENT:
            counts[s["L"]] = counts.get(s["L"], 0) + 1

    print("\n=== per-layer event counts (z>=3) ===")
    print(" ".join(f"L{L}:{counts.get(L, 0)}" for L in range(n_layers)))

    print(f"\n=== top {TOP_EVENTS} events ===", flush=True)
    out_events = []
    for e in events:
        toks = e["_toks"]; w = e["_w"]; L = e["L"]; t = e["t"]
        head_tot = w.sum(1)
        tops = np.argsort(-head_tot)[:2]
        head_info = []
        for h in tops:
            srcs = np.argsort(-w[h])[:3]
            src_s = ", ".join(f"'{toks[s + 1]}'({w[h, s]:.1f})"
                              for s in srcs if w[h, s] > 0)
            contrib = e["_oin"][h] @ W_O[L][:, h * d_head:(h + 1) * d_head].T
            logits = contrib @ W_U.T
            top_v = np.argsort(-logits)[:5]
            ov = ", ".join(tok.decode([int(i)]).strip() or "␣"
                           for i in top_v)
            head_info.append({"head": int(h), "sources": src_s,
                              "ov_top": ov})
        rec = {"cat": e["cat"], "text": CORPUS[e["pi"]][1], "L": L,
               "dest": toks[t], "z": round(float(e["z"]), 1),
               "sink": round(e["sink"], 2), "heads": head_info}
        out_events.append(rec)
        print(f"\nz={rec['z']:5.1f} L{L:2d} [{rec['cat']:9s}] "
              f"dest='{rec['dest']}' sink={rec['sink']:.2f}  "
              f"\"{rec['text'][:60]}\"")
        for hi in head_info:
            print(f"    H{hi['head']:2d} <- {hi['sources']}")
            print(f"        OV writes: {hi['ov_top']}")

    outp = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "results", "attention_events.json")
    with open(outp, "w") as f:
        json.dump({"model": MODEL, "z_event": Z_EVENT,
                   "layer_event_counts": {str(k): v for k, v in
                                          sorted(counts.items())},
                   "events": out_events}, f, indent=2)
    print(f"\nwrote {outp}")


if __name__ == "__main__":
    main()
