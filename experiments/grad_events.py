"""
grad_events.py — attribution-weighted attention events (grad x activation).

The event detector measures TRAFFIC (attention x value norm). This measures
CONSEQUENCE: one backward pass gives d(target)/d(attention edge) for every
edge; attribution = edge weight x its gradient. Crossing the two splits
events into moved-and-mattered vs moved-and-ignored — the cheap first-order
stand-in for ablation.

Validation task: next-day prediction, where the output MUST use the weekday.
  prompt  "They met on a {wd} in {mo}. The next day was a"
  target  logit(successor day) - mean logit(other 6 days)
The weekday-CONTRAST target cancels generic say-a-weekday credit; only
weekday-dependent computation earns attribution. Question: is L30 H18 (the
angle-copy delivery head) load-bearing, and which heads matter that the
transport tables missed?

Mechanics: params frozen (no param grads); the embedding output is re-leafed
with requires_grad so the graph exists; eager attention so out.attentions
are graph tensors; retain_grad + backward; fp32 (MPS fp16 backward NaNs,
cf frame_steering).

Run:  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python \
        experiments/grad_events.py
Writes experiments/results/grad_events.json.
"""
from __future__ import annotations
import json, os

import numpy as np

MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
WD = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday",
      "Sunday"]
MO = ["January", "April", "July", "October"]      # 7x4 = 28 prompts
TEMPLATE = "They met on a {wd} in {mo}. The next day was a"
TOP_HEADS = 15


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL)
    mdl = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.float32, attn_implementation="eager").to(dev)
    for p in mdl.parameters():
        p.requires_grad_(False)
    n_layers = mdl.config.num_hidden_layers
    n_heads = mdl.config.num_attention_heads

    def embed_leaf(m, inp, out):
        out = out.detach()
        out.requires_grad_(True)
        return out
    h_embed = mdl.model.embed_tokens.register_forward_hook(embed_leaf)

    day_ids = [tok.encode(" " + w, add_special_tokens=False)[0] for w in WD]

    attr_sum = np.zeros((n_layers, n_heads))
    attr_wd_src = np.zeros((n_layers, n_heads))   # edges sourced in wd span
    top_edges = {}                                 # (L,h) -> best |edge| info
    n_ok = 0
    prompts = [(a, b) for a in WD for b in MO]
    print(f"{MODEL}: {len(prompts)} prompts, fp32 backward", flush=True)

    for a, b in prompts:
        text = TEMPLATE.format(wd=a, mo=b)
        enc = tok(text, return_offsets_mapping=True, return_tensors="pt")
        offs = enc.pop("offset_mapping")[0].tolist()
        cs = text.find(a)
        wd_span = [k for k, (s, e) in enumerate(offs)
                   if e > cs and s < cs + len(a)]
        toks = [tok.decode([i]) for i in enc["input_ids"][0]]
        out = mdl(**{k: v.to(dev) for k, v in enc.items()},
                  output_attentions=True)
        for att in out.attentions:
            att.retain_grad()
        logits = out.logits[0, -1]
        succ = day_ids[(WD.index(a) + 1) % 7]
        others = [d for d in day_ids if d != succ]
        target = logits[succ] - torch.stack([logits[o] for o in others]).mean()
        pred = int(torch.tensor([logits[d] for d in day_ids]).argmax())
        n_ok += int(day_ids[pred] == succ)
        mdl.zero_grad(set_to_none=True)
        target.backward()
        for L, att in enumerate(out.attentions):
            g = att.grad
            if g is None:
                continue
            ax = (att[0] * g[0]).detach().float().cpu().numpy()  # [H, T, T]
            attr_sum[L] += ax.sum((1, 2))
            attr_wd_src[L] += ax[:, :, wd_span].sum((1, 2))
            for h in range(n_heads):
                m = np.abs(ax[h])
                q, k = np.unravel_index(np.argmax(m), m.shape)
                v = float(ax[h, q, k])
                cur = top_edges.get((L, h))
                if cur is None or abs(v) > abs(cur[0]):
                    top_edges[(L, h)] = (v, toks[q], toks[k])
        del out
        if dev == "mps":
            torch.mps.empty_cache()
    h_embed.remove()

    attr_sum /= len(prompts)
    attr_wd_src /= len(prompts)
    print(f"\nnext-day accuracy (argmax over 7 day tokens): "
          f"{n_ok}/{len(prompts)}", flush=True)

    results: dict = {"model": MODEL, "accuracy": f"{n_ok}/{len(prompts)}",
                     "tables": {}}
    for name, tbl in (("total attribution", attr_sum),
                      ("attribution via weekday-span sources", attr_wd_src)):
        flat = [(float(tbl[L, h]), L, h) for L in range(n_layers)
                for h in range(n_heads)]
        flat.sort(key=lambda x: -abs(x[0]))
        print(f"\n=== top {TOP_HEADS} heads by |{name}| ===")
        rows = []
        for v, L, h in flat[:TOP_HEADS]:
            ev, qt, kt = top_edges[(L, h)]
            rows.append({"L": L, "head": h, "attr": round(v, 3),
                         "top_edge": f"'{qt}' <- '{kt}' ({ev:+.3f})"})
            print(f"  L{L:2d} H{h:2d}  {v:+8.3f}   top edge: "
                  f"'{qt}' <- '{kt}' ({ev:+.3f})")
        results["tables"][name] = rows
        r18 = 1 + sum(1 for v, L, h in flat
                      if abs(v) > abs(tbl[30, 18]))
        print(f"  [L30 H18 rank in this table: #{r18}, "
              f"attr {tbl[30, 18]:+.3f}]")
        results["tables"][name + "_L30H18_rank"] = r18

    outp = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "results", "grad_events.json")
    with open(outp, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {outp}")


if __name__ == "__main__":
    main()
