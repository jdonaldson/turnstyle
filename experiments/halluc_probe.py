"""
halluc_probe.py — is empty content-provenance a hallucination detector?

The narrator found one confabulation ('Christmas -> November') whose events
were all format-cues, no content retrieval. Hypothesis: content-sourced
event mass at the answer token separates grounded answers from
confabulations. Known blind spot (from our own demos): confidently-WRONG
retrieval (gold/lead) carries rich provenance — this detects
UNGROUNDEDNESS, not incorrectness. Known confound to test: fake entities
still have subject tokens; heads may transport 'Wakanda' and then
confabulate anyway.

Battery: 12 real-fact prompts (expected answers checked) vs 12 fake-entity
prompts (any confident answer = confabulation by construction). Score per
prompt = max per-head z among events at the FIRST answer token whose
sources include a content word; plus the count of such events. Report the
two distributions + AUC.

Run:  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python \
        experiments/halluc_probe.py
Writes experiments/results/halluc_probe.json.
"""
from __future__ import annotations
import json, os, sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from attention_events import CORPUS, MODEL

REAL = [
    ("The capital of France is", "paris"),
    ("The capital of Italy is", "rome"),
    ("The capital of Japan is", "tokyo"),
    ("The capital of Spain is", "madrid"),
    ("The capital of Germany is", "berlin"),
    ("The largest planet in the solar system is", "jupiter"),
    ("Water is made of hydrogen and", "oxygen"),
    ("The author of Romeo and Juliet is", "shakespeare"),
    ("The chemical symbol for gold is", "au"),
    ("Two plus two equals", "four"),
    ("The opposite of hot is", "cold"),
    ("The color of the sky on a clear day is", "blue"),
]
FAKE = [
    "The capital of Wakanda is",
    "The capital of Atlantis is",
    "The capital of Narnia is",
    "The capital of Gondor is",
    "The author of the novel The Crystal Menagerie of Zorblatt is",
    "The chemical symbol for the element flumium is",
    "The year the Battle of Quixotania took place was",
    "The largest city in Freedonia is",
    "The inventor of the gravity pencil was",
    "The national bird of Elbonia is",
    "The boiling point of glimmerite is",
    "The third moon of Earth is called",
]
Z_EVENT = 4.0
FUNCTION_WORDS = {"the", "a", "an", "is", "was", "were", "be", "will",
                  "would", "to", "of", "in", "on", "by", "for", "and", "or",
                  "so", "that", "it", "made", "called", "?", ".", ",",
                  "what", "answer", "word", "number", "month", "year",
                  "capital", "author", "symbol", "color", "opposite",
                  "largest", "novel", "element", "city", "inventor",
                  "national", "boiling", "point", "third", "moon", ""}


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

    v_store = {}
    hooks = [lyr.self_attn.v_proj.register_forward_hook(
        lambda m, inp, out, i=i: v_store.__setitem__(i, out.detach().float()))
        for i, lyr in enumerate(mdl.model.layers)]

    def all_scores(ids):
        """One forward: per-position per-head scores + attn + next id."""
        T = len(ids)
        v_store.clear()
        with torch.no_grad():
            out = mdl(input_ids=torch.tensor([ids]).to(dev),
                      output_attentions=True)
        A = [a[0].float().cpu().numpy() for a in out.attentions]
        vns = []
        for L in range(n_layers):
            v = v_store[L][0].cpu().numpy().reshape(T, n_kv, d_head)
            vns.append(np.repeat(np.linalg.norm(v, axis=-1), rep, axis=1).T)
        nxt = int(out.logits[0, -1].argmax())
        return nxt, A, vns

    # baselines (single forward per corpus prompt, all positions)
    print("baselines...", flush=True)
    bl_sum = np.zeros((n_layers, n_heads)); bl_sq = np.zeros((n_layers, n_heads))
    bl_n = 0
    for _, text in CORPUS:
        ids = tok(text)["input_ids"]
        _, A, vns = all_scores(ids)
        for L in range(n_layers):
            for t in range(1, len(ids)):
                hs = (A[L][:, t, 1:t + 1] * vns[L][:, 1:t + 1]).sum(1)
                bl_sum[L] += hs; bl_sq[L] += hs ** 2
        bl_n += len(ids) - 1
    mu = bl_sum / bl_n
    sd = np.sqrt(np.maximum(bl_sq / bl_n - mu ** 2, 0)) + 1e-9

    def probe(text):
        """Answer-token content-provenance score for one prompt."""
        ids = tok(text)["input_ids"]
        nxt, A, vns = all_scores(ids)
        toks = [tok.decode([i]) for i in ids]
        t = len(ids) - 1
        max_content_z, n_content = 0.0, 0
        top_ev = ""
        for L in range(n_layers):
            w = A[L][:, t, 1:t + 1] * vns[L][:, 1:t + 1]
            z = (w.sum(1) - mu[L]) / sd[L]
            for h in np.where(z >= Z_EVENT)[0]:
                srcs = np.argsort(-w[h])[:3]
                stoks = [toks[s + 1].strip() for s in srcs if w[h, s] > 0.5]
                content = [s for s in stoks
                           if s.lower() not in FUNCTION_WORDS]
                if content:
                    n_content += 1
                    if z[h] > max_content_z:
                        max_content_z = float(z[h])
                        top_ev = f"L{L}H{h} <- {content}"
        ans = tok.decode([nxt]).strip()
        return ans, max_content_z, n_content, top_ev

    results: dict = {"model": MODEL, "real": [], "fake": []}
    print("\n=== REAL facts ===")
    real_scores = []
    for text, expect in REAL:
        ans, mz, nc, ev = probe(text)
        ok = expect in ans.lower()
        real_scores.append((mz, nc, ok))
        results["real"].append({"prompt": text, "answer": ans, "ok": ok,
                                "max_content_z": round(mz, 1),
                                "n_content_events": nc, "top": ev})
        print(f"  {'OK ' if ok else 'MISS'} z={mz:5.1f} n={nc:2d}  "
              f"'{ans}'  <- {ev}   \"{text[:45]}\"")
    print("\n=== FAKE entities ===")
    fake_scores = []
    for text in FAKE:
        ans, mz, nc, ev = probe(text)
        fake_scores.append((mz, nc))
        results["fake"].append({"prompt": text, "answer": ans,
                                "max_content_z": round(mz, 1),
                                "n_content_events": nc, "top": ev})
        print(f"       z={mz:5.1f} n={nc:2d}  '{ans}'  <- {ev}   "
              f"\"{text[:45]}\"")

    rs = [s for s, n, ok in real_scores if ok]
    fs = [s for s, n in fake_scores]
    if rs and fs:
        auc = float(np.mean([[r > f for f in fs] for r in rs]))
        print(f"\ncorrect-REAL n={len(rs)} mean z={np.mean(rs):.1f} | "
              f"FAKE n={len(fs)} mean z={np.mean(fs):.1f} | "
              f"AUC(real>fake)={auc:.2f}")
        results["auc"] = round(auc, 3)

    for h in hooks:
        h.remove()
    outp = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "results", "halluc_probe.json")
    with open(outp, "w") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {outp}")


if __name__ == "__main__":
    main()
