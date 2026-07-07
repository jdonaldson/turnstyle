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
    "The capital of Wakanda is",
    "Christmas falls in the month of",
    "Gold is heavier than silver, and lead is heavier than gold. The heaviest metal of the three is",
    "Monday, Tuesday, Wednesday,",
    "The chemical symbol for gold is",
    "The opposite of generous is",
]
GEN_TOKENS = 4
Z_SHOW = 4.0
TOP_SHOW = 3

# presentation-layer word classes for verbalization (NOT solver logic)
FUNCTION_WORDS = {"the", "a", "an", "is", "was", "were", "be", "will",
                  "would", "to", "of", "in", "on", "by", "for", "and", "or",
                  "so", "that", "it", "she", "he", "her", "his", "had",
                  "has", "?", ".", ",", "what", "answer", "word", "number",
                  "month", "total", "next", "capital", "symbol", "opposite",
                  "chemical", "falls", ""}

# heads identified earlier in this arc (event detector + attribution runs)
HEAD_NOTES = {
    (19, 27): "fact-recall", (19, 28): "fact-recall",
    (19, 23): "operand-gather", (17, 26): "arithmetic-gather",
    (20, 0): "operand-copy", (21, 13): "name-mover",
    (21, 2): "name-writer", (21, 18): "name-writer",
    (20, 27): "name-gather", (20, 17): "claim-tracker",
    (20, 1): "day-sequence", (16, 3): "list-continuation",
    (4, 14): "instruction-reader", (7, 29): "answer-cue",
    (6, 31): "question-format", (20, 16): "premise-copy",
    (21, 27): "premise-copy", (19, 13): "antonym",
    (20, 8): "attribute-lookup", (21, 4): "comparative",
}
# SmolLM2 direct-logit artifact tokens (massive-activation junk)
OV_ARTIFACTS = {"neurotransmit", "chimpan", "caterpill", "dprinting",
                "opio", "cryptocur", "recipesteps", "controlplane",
                "keynot", "<filename>", "earthqu", "meticul", "soriasis"}


def _readable_ov(ov):
    keep = [o for o in ov
            if o and o.lower() not in OV_ARTIFACTS and o.isalpha()
            and len(o) >= 2]
    return keep[:3] if len(keep) >= 2 else []


def verbalize(gen, events):
    """Event -> explanatory English. Distinguishes recall (whole-word
    lookup), assembly (subword fragments of an unfamiliar name), gathering
    (multiple operands), copying (OV writes the token), and format cues."""
    clauses = []
    gl = gen.strip().lower()
    for ev in events:
        raw = ev.get("sources_raw", ev["sources"])
        srcs = [s.strip() for s in raw if s.strip()]
        frags = [s.strip() for s in raw
                 if s and not s.startswith(" ") and s.strip()]
        content = [s for s in srcs if s.lower() not in FUNCTION_WORDS]
        ovk = _readable_ov(ev["ov"])
        role = HEAD_NOTES.get((ev["L"], ev["head"]))
        who = (f"the {role} head (L{ev['L']} H{ev['head']})" if role
               else f"head L{ev['L']} H{ev['head']}")
        wrote = f", writing {'/'.join(ovk)}" if ovk else ""
        if gl and any(gl == o.lower() or gl in o.lower() or o.lower() in gl
                      for o in ev["ov"][:2] if o):
            src_s = " and ".join(f"'{s}'" for s in (content or srcs)[:2])
            clauses.append(f"{who} inspected {src_s} and wrote "
                           f"'{gen.strip()}' into the answer itself")
        elif frags and len(frags) >= max(1, len(content)):
            clauses.append(f"{who} was still piecing together the "
                           f"unfamiliar name ({' + '.join(repr(f) for f in frags[:3])})"
                           f" — assembly, not recall")
        elif len(content) >= 2:
            clauses.append(f"{who} collected "
                           + " and ".join(f"'{s}'" for s in content[:3])
                           + f" together{wrote}")
        elif len(content) == 1:
            clauses.append(f"{who} looked up '{content[0]}' from the "
                           f"prompt{wrote}")
        elif srcs:
            clauses.append(f"{who} checked the format cue "
                           f"'{' '.join(srcs[:3])}' (what kind of token "
                           f"comes next), not any content")
        else:
            clauses.append(f"{who} moved diffuse context")
    if not clauses:
        return f"To produce '{gen.strip()}': no head stood out (routine)."
    return (f"To produce '{gen.strip()}': "
            + "; ".join(clauses) + ".")


def answer_verdict(steps):
    """Grounding verdict over the whole answer, from event types."""
    lookups, assemblies, cues = 0, 0, 0
    for st in steps:
        for ev in st["events"]:
            raw = ev.get("sources_raw", ev["sources"])
            srcs = [s.strip() for s in raw if s.strip()]
            frags = [s for s in raw if s and not s.startswith(" ")]
            content = [s for s in srcs if s.lower() not in FUNCTION_WORDS]
            if frags and len(frags) >= max(1, len(content)):
                assemblies += 1
            elif content:
                lookups += 1
            else:
                cues += 1
    if assemblies > lookups:
        return ("UNGROUNDED-ASSEMBLY: the dominant activity was piecing an "
                "unfamiliar name together, not recalling anything about it")
    if lookups == 0:
        return ("FORMAT-DRIVEN: no content was looked up — the answer "
                "came from the sentence template, not from a fact")
    return (f"GROUNDED: {lookups} content lookup(s) fed the answer "
            f"({cues} format checks, {assemblies} assembly events)")


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
                src_raw = [toks[s + 1] for s in srcs if w[s] > 0.5]
                src_toks = [s.strip() for s in src_raw]
                provenance.update(src_toks)
                contrib = oins[L][h] @ W_O[L][:, h * d_head:(h + 1) * d_head].T
                ov = [tok.decode([int(i)]).strip() or "␣"
                      for i in np.argsort(-(contrib @ W_U.T))[:4]]
                step["events"].append(
                    {"L": L, "head": h, "z": round(zv, 1),
                     "sources": src_toks, "sources_raw": src_raw, "ov": ov})
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
        demo["verdict"] = answer_verdict(demo["steps"])
        print(f"  answer: {answer!r}")
        print(f"  ∴ {demo['summary']}")
        print(f"  ⚖ {demo['verdict']}")
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
