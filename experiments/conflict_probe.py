"""
conflict_probe.py — is an OVERRULED SPECIALIST a wrong-answer signal?

The gold/lead narration showed the failure legibly: the comparative head
(L21 H4) fetched 'lead' (correct) and OV-wrote it, but the emitted answer
was 'gold' (the name-gather head's candidate won). Hypothesis: at the
answer token, when strong events recommend DIFFERENT entities — or a
specialist's OV recommendation disagrees with the emitted token — the
answer is likely wrong. Forward-only, no labels at inference.

Battery: transitive chains "{A} is {cmp} than {B}, and {C} is {cmp} than
{A}. The {sup} of the three is" (correct = C), 5 domains x 3 rotations.
Per prompt, at the answer position:
  - events z>=4 -> readable OV tokens -> mapped to prompt entities
  - rec_set   distinct entities recommended by head OVs
  - overruled 1 if some recommendation != the emitted entity
Report per-prompt table + AUC(overruled/conflict predicts wrongness).

Run:  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python \
        experiments/conflict_probe.py
Writes experiments/results/conflict_probe.json.
"""
from __future__ import annotations
import json, os, sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from attention_events import CORPUS, MODEL

DOMAINS = [
    (("gold", "silver", "lead"), "heavier", "heaviest"),
    (("oak", "pine", "cedar"), "taller", "tallest"),
    (("whale", "dolphin", "shark"), "bigger", "biggest"),
    (("Tom", "Sam", "Ben"), "older", "oldest"),
    (("granite", "marble", "chalk"), "harder", "hardest"),
]
Z_EVENT = 4.0


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

    def forward(ids):
        T = len(ids)
        v_store.clear(); o_store.clear()
        with torch.no_grad():
            out = mdl(input_ids=torch.tensor([ids]).to(dev),
                      output_attentions=True)
        A = [a[0].float().cpu().numpy() for a in out.attentions]
        vns = []
        for L in range(n_layers):
            v = v_store[L][0].cpu().numpy().reshape(T, n_kv, d_head)
            vns.append(np.repeat(np.linalg.norm(v, axis=-1), rep, axis=1).T)
        return out, A, vns

    print("baselines...", flush=True)
    bl_sum = np.zeros((n_layers, n_heads)); bl_sq = np.zeros((n_layers, n_heads))
    bl_n = 0
    for _, text in CORPUS:
        ids = tok(text)["input_ids"]
        _, A, vns = forward(ids)
        for L in range(n_layers):
            for t in range(1, len(ids)):
                hs = (A[L][:, t, 1:t + 1] * vns[L][:, 1:t + 1]).sum(1)
                bl_sum[L] += hs; bl_sq[L] += hs ** 2
        bl_n += len(ids) - 1
    mu = bl_sum / bl_n
    sd = np.sqrt(np.maximum(bl_sq / bl_n - mu ** 2, 0)) + 1e-9

    def entity_of(token_str, ents):
        t = token_str.strip().lower()
        if len(t) < 3:
            return None
        for e in ents:
            el = e.lower()
            if t == el or el.startswith(t) or t.startswith(el):
                return e
        return None

    rows = []
    print("\nprompt | correct? emitted | recs | overruled conflict")
    for ents, cmp_w, sup_w in DOMAINS:
        for rot in range(3):
            A_, B_, C_ = (ents[rot % 3], ents[(rot + 1) % 3],
                          ents[(rot + 2) % 3])
            text = (f"{A_.capitalize()} is {cmp_w} than {B_}, and {C_} is "
                    f"{cmp_w} than {A_}. The {sup_w} of the three is")
            ids = tok(text)["input_ids"]
            out, A, vns = forward(ids)
            t = len(ids) - 1
            toks = [tok.decode([i]) for i in ids]
            emitted = tok.decode([int(out.logits[0, -1].argmax())])
            em_ent = entity_of(emitted, ents)
            recs = set()
            for L in range(n_layers):
                w = A[L][:, t, 1:t + 1] * vns[L][:, 1:t + 1]
                z = (w.sum(1) - mu[L]) / sd[L]
                for h in np.where(z >= Z_EVENT)[0]:
                    oin = o_store[L][0, t].cpu().numpy().reshape(
                        n_heads, d_head)
                    contrib = oin[h] @ W_O[L][:, h * d_head:(h + 1) * d_head].T
                    for i in np.argsort(-(contrib @ W_U.T))[:3]:
                        e = entity_of(tok.decode([int(i)]), ents)
                        if e:
                            recs.add(e)
            correct = em_ent is not None and em_ent.lower() == C_.lower()
            overruled = int(any(r.lower() != (em_ent or "").lower()
                                for r in recs))
            conflict = int(len(recs) >= 2)
            rows.append({"prompt": text, "correct": bool(correct),
                         "emitted": emitted.strip(), "target": C_,
                         "recs": sorted(recs), "overruled": overruled,
                         "conflict": conflict})
            print(f"  {'OK ' if correct else 'X  '} '{emitted.strip():8s}'"
                  f" (want {C_:8s}) recs={sorted(recs)} "
                  f"ovr={overruled} conf={conflict}  \"{text[:52]}\"")

    wrong = [r for r in rows if not r["correct"]]
    right = [r for r in rows if r["correct"]]
    for feat in ("overruled", "conflict"):
        w = [r[feat] for r in wrong]; g = [r[feat] for r in right]
        if w and g:
            auc = float(np.mean([[wi > gi for gi in g] for wi in w]) +
                        0.5 * np.mean([[wi == gi for gi in g] for wi in w]))
            print(f"\n{feat}: wrong mean {np.mean(w):.2f} vs right mean "
                  f"{np.mean(g):.2f} | AUC(wrongness)={auc:.2f}")
    print(f"accuracy: {len(right)}/{len(rows)}")

    for h in hooks:
        h.remove()
    outp = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "results", "conflict_probe.json")
    with open(outp, "w") as f:
        json.dump({"model": MODEL, "rows": rows}, f, indent=2)
    print(f"wrote {outp}")


if __name__ == "__main__":
    main()
