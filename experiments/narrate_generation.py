"""
narrate_generation.py — narrate a model's ANSWER via its attention events.

The product of the event-detector arc: as the model generates each token,
score every head at the generating position against per-head corpus
baselines (attention_events v2, the metric that blindly rediscovered
L20.H0); for heads that spike, decompose the event — what was pulled, from
where, what the head wrote (OV) — and emit a human-readable trace:

  > "What is four plus seven? The answer is"
    + ' eleven'    L20 H0  z=9   <- 'four','seven'   writes: eleven, four
                   L17 H26 z=5   <- 'seven'          writes: ...
    provenance: the answer drew on {'four', 'seven'}

Baselines are computed fresh on the 48-prompt topic-free corpus (~1 min),
then several demo answers are narrated (arith, coref, next-day, recall).

Run:  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python \
        experiments/narrate_generation.py
Writes experiments/results/narrate_generation.json.
"""
from __future__ import annotations
import json, os, sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from attention_events import CORPUS, MODEL

DEMOS = [
    "What is four plus seven? The answer is",
    "Today is Tuesday. Tomorrow will be",
    "Sarah gave the book to Emma because she had finished it. The book was returned by",
    "The capital of France is",
    "What is nine minus three? The answer is",
    "Anna told Maria that her car was blocking the driveway. The driveway was blocked by",
    "All cats are animals. Whiskers is a cat, so Whiskers is",
    "Gold is heavier than silver, and lead is heavier than gold. The heaviest metal of the three is",
    "Monday, Tuesday, Wednesday,",
    "The word for bread in Spanish is",
    "Christmas falls in the month of",
    "The basket held three apples and two pears. The number of apples was",
]
GEN_TOKENS = 5
Z_SHOW = 4.0
TOP_SHOW = 3

# presentation-layer word classes for verbalization (NOT solver logic)
FUNCTION_WORDS = {"the", "a", "an", "is", "was", "were", "be", "will",
                  "would", "to", "of", "in", "on", "by", "for", "and", "or",
                  "so", "that", "it", "she", "he", "her", "his", "had",
                  "has", "?", ".", ",", "what", "answer", "word", "number",
                  "month", "total", "next", ""}


def verbalize(gen, events):
    """Mechanical event -> English clause rules."""
    clauses = []
    gl = gen.strip().lower()
    for ev in events:
        srcs = [s for s in ev["sources"] if s]
        content = [s for s in srcs if s.lower() not in FUNCTION_WORDS]
        ovl = [o.lower() for o in ev["ov"][:2]]
        tag = f"(L{ev['L']} H{ev['head']})"
        if gl and any(gl == o or gl in o or o in gl for o in ovl if o):
            src_s = " and ".join(f"'{s}'" for s in (content or srcs)[:2])
            clauses.append(f"wrote '{gen.strip()}' directly while reading "
                           f"{src_s} {tag}")
        elif len(content) >= 2:
            clauses.append("gathered " + " and ".join(
                f"'{s}'" for s in content[:3]) + f" {tag}")
        elif len(content) == 1:
            clauses.append(f"retrieved '{content[0]}' {tag}")
        elif srcs:
            clauses.append(f"followed the cue '"
                           + " ".join(srcs[:3]) + f"' {tag}")
        else:
            clauses.append(f"moved diffuse context {tag}")
    if not clauses:
        return f"To produce '{gen.strip()}': no head stood out (routine)."
    return (f"To produce '{gen.strip()}', the model "
            + "; ".join(clauses) + ".")


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
            lambda m, inp, i=i: o_store.__setitem__(i, inp[0].detach().float())))

    def head_scores(ids):
        """Forward; per-head transported-value scores at the LAST position.
        Returns (next_id, hs [L,H], ctx) where ctx holds decomposition."""
        T = len(ids)
        v_store.clear(); o_store.clear()
        with torch.no_grad():
            out = mdl(input_ids=torch.tensor([ids]).to(dev),
                      output_attentions=True)
        A = [a[0].float().cpu().numpy() for a in out.attentions]
        hs = np.zeros((n_layers, n_heads))
        ws, oins = [], []
        for L in range(n_layers):
            v = v_store[L][0].cpu().numpy().reshape(T, n_kv, d_head)
            vn = np.repeat(np.linalg.norm(v, axis=-1), rep, axis=1).T
            w = A[L][:, T - 1, 1:] * vn[:, 1:]       # [H, T-1] non-sink
            hs[L] = w.sum(1)
            ws.append(w)
            oins.append(o_store[L][0, T - 1].cpu().numpy()
                        .reshape(n_heads, d_head))
        nxt = int(out.logits[0, -1].argmax())
        return nxt, hs, (ws, oins)

    # ---- baselines over the topic-free corpus (every position, one
    # forward per prompt — causal attention makes position-t scores in the
    # full pass identical to prefix-t last-position scores) ---------------
    print("computing per-head baselines on the corpus...", flush=True)
    bl_sum = np.zeros((n_layers, n_heads)); bl_sq = np.zeros((n_layers, n_heads))
    bl_n = 0
    for _, text in CORPUS:
        ids = tok(text)["input_ids"]
        T = len(ids)
        v_store.clear(); o_store.clear()
        with torch.no_grad():
            out = mdl(input_ids=torch.tensor([ids]).to(dev),
                      output_attentions=True)
        A = [a[0].float().cpu().numpy() for a in out.attentions]
        for L in range(n_layers):
            v = v_store[L][0].cpu().numpy().reshape(T, n_kv, d_head)
            vn = np.repeat(np.linalg.norm(v, axis=-1), rep, axis=1).T
            for t in range(1, T):
                hs_t = (A[L][:, t, 1:t + 1] * vn[:, 1:t + 1]).sum(1)
                bl_sum[L] += hs_t; bl_sq[L] += hs_t ** 2
        bl_n += T - 1
    mu = bl_sum / bl_n
    sd = np.sqrt(np.maximum(bl_sq / bl_n - mu ** 2, 0)) + 1e-9
    print(f"baselines from {bl_n} positions", flush=True)

    # ---- narrated generation --------------------------------------------
    results: dict = {"model": MODEL, "demos": []}
    for prompt in DEMOS:
        ids = tok(prompt)["input_ids"]
        print(f"\n▶ \"{prompt}\"", flush=True)
        demo = {"prompt": prompt, "steps": []}
        provenance = set()
        for _ in range(GEN_TOKENS):
            nxt, hs, (ws, oins) = head_scores(ids)
            toks = [tok.decode([i]) for i in ids]
            z = (hs - mu) / sd
            flat = [(float(z[L, h]), L, h) for L in range(n_layers)
                    for h in range(n_heads) if z[L, h] >= Z_SHOW]
            flat.sort(key=lambda x: -x[0])
            gen = tok.decode([nxt])
            step = {"token": gen, "events": []}
            print(f"  +{gen!r}")
            for zv, L, h in flat[:TOP_SHOW]:
                w = ws[L][h]
                srcs = np.argsort(-w)[:3]
                src_toks = [toks[s + 1].strip() for s in srcs if w[s] > 0.5]
                provenance.update(src_toks)
                contrib = oins[L][h] @ W_O[L][:, h * d_head:(h + 1) * d_head].T
                ov = [tok.decode([int(i)]).strip() or "␣"
                      for i in np.argsort(-(contrib @ W_U.T))[:4]]
                step["events"].append(
                    {"L": L, "head": h, "z": round(zv, 1),
                     "sources": src_toks, "ov": ov})
                print(f"      L{L:2d} H{h:2d} z={zv:4.1f}  <- "
                      f"{', '.join(repr(s) for s in src_toks) or '(diffuse)'}"
                      f"   writes: {', '.join(ov)}", flush=True)
            step["sentence"] = verbalize(gen, step["events"])
            print(f"      » {step['sentence']}")
            ids = ids + [nxt]
            demo["steps"].append(step)
            if gen.strip() in (".", "</s>", ""):
                break
        answer = tok.decode(ids[len(tok(prompt)['input_ids']):])
        demo["answer"] = answer
        demo["provenance"] = sorted(provenance)
        content_prov = sorted(s for s in provenance
                              if s.lower() not in FUNCTION_WORDS)
        demo["summary"] = (f"It answered {answer.strip()!r} drawing on "
                           + (", ".join(f"'{s}'" for s in content_prov)
                              or "routine context") + ".")
        print(f"  answer: {answer!r}")
        print(f"  ∴ {demo['summary']}")
        results["demos"].append(demo)

    for h in hooks:
        h.remove()
    outp = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "results", "narrate_generation.json")
    with open(outp, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {outp}")


if __name__ == "__main__":
    main()
