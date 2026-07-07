"""
weight_mine_head.py — ONE EXAMPLE of weight-informed stimulus mining.

Can the model's weights alone tell us what a head wants, well enough to
write its stimuli? Target: SmolLM2 L20 H0 — the operand-copy head that was
hand-mapped (2026-04) and blindly rediscovered by the event detector — so
the mined table has a known right answer: keys ~ number words, OV ~ copies
them. Second row for free: L21 H13 (the Emma name-mover; expect name-ish
keys, copy OV).

Method (first-order, no forward passes):
  QK   A(D) = W_Qh^T R(-D) W_Kh over RMSNorm'd embeddings (R = the RoPE
       relative rotation at key-query offset D — RoPE makes preference
       DISTANCE-dependent, which we mine rather than fight). SVD(A) ->
       top singular pairs -> top vocab tokens on the query and key sides.
  OV   for mined key tokens: W_U(W_Oh W_Vh e_k) top outputs + a copy score
       (argmax == key?) over a common-token sample.
LIVE CHECK: assemble token-level stimuli (key + filler x(D-1) + query) from
the top mined pairs, forward once, read the head's actual attention
query->key vs its mean elsewhere. Weights predicted it; does it fire?

Run:  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python \
        experiments/weight_mine_head.py
Writes experiments/results/weight_mine_head.json.
"""
from __future__ import annotations
import json, os

import numpy as np

MODEL = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
HEADS = [(20, 0, "operand copier"), (21, 13, "Emma name-mover")]
OFFSETS = [1, 2, 4, 8]
R_SV = 3          # singular pairs to mine
TOPK = 8


def rope_rel(d_head, base, delta):
    """Relative RoPE rotation matrix R(delta), HF rotate-half convention."""
    half = d_head // 2
    inv = 1.0 / (base ** (np.arange(half) / half))
    ang = delta * inv
    c, s = np.cos(ang), np.sin(ang)
    R = np.zeros((d_head, d_head))
    for i in range(half):
        R[i, i] = c[i]; R[i + half, i + half] = c[i]
        R[i, i + half] = -s[i]; R[i + half, i] = s[i]
    return R


def top_tokens(vec, En, tok, k=TOPK, sign=1):
    sc = sign * (En @ vec)
    idx = np.argsort(-sc)[:k]
    return [tok.decode([int(i)]).strip() or "␣" for i in idx], idx


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL)
    mdl = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.float16, attn_implementation="eager").to(dev)
    cfg = mdl.config
    n_heads = cfg.num_attention_heads
    n_kv = getattr(cfg, "num_key_value_heads", n_heads)
    d_model = cfg.hidden_size
    d_head = d_model // n_heads
    base = float(getattr(cfg, "rope_theta", 10000.0))

    E = mdl.model.embed_tokens.weight.detach().float().cpu().numpy()
    W_U = mdl.lm_head.weight.detach().float().cpu().numpy()
    rms = np.sqrt((E ** 2).mean(1, keepdims=True)) + 1e-6

    results: dict = {"model": MODEL, "heads": {}}
    stimuli = []

    for L, h, label in HEADS:
        lyr = mdl.model.layers[L]
        gam = lyr.input_layernorm.weight.detach().float().cpu().numpy()
        En = (E / rms) * gam                      # RMSNorm'd embeddings
        sl = slice(h * d_head, (h + 1) * d_head)
        kv_h = h // (n_heads // n_kv)
        slk = slice(kv_h * d_head, (kv_h + 1) * d_head)
        Wq = lyr.self_attn.q_proj.weight.detach().float().cpu().numpy()[sl]
        Wk = lyr.self_attn.k_proj.weight.detach().float().cpu().numpy()[slk]
        Wv = lyr.self_attn.v_proj.weight.detach().float().cpu().numpy()[slk]
        Wo = lyr.self_attn.o_proj.weight.detach().float().cpu().numpy()[:, sl]

        print(f"\n================ L{L} H{h} ({label}) ================")
        hrec: dict = {"label": label, "offsets": {}}
        for D in OFFSETS:
            A = Wq.T @ rope_rel(d_head, base, -D) @ Wk      # [dm, dm]
            U, S, Vt = np.linalg.svd(A)
            print(f"--- offset {D} (top {R_SV} singular pairs, "
                  f"s={S[:R_SV].round(0)}) ---")
            rows = []
            for i in range(R_SV):
                for sign in (1, -1):
                    qt, _ = top_tokens(U[:, i], En, tok, sign=sign)
                    kt, kidx = top_tokens(Vt[i], En, tok, sign=sign)
                    rows.append({"sv": i, "sign": sign, "q": qt, "k": kt})
                    print(f"  sv{i}{'+' if sign > 0 else '-'} "
                          f"Q: {', '.join(qt[:6])}")
                    print(f"        K: {', '.join(kt[:6])}")
            hrec["offsets"][D] = rows
        # OV on the offset-1 sv0 key tokens + copy score on common tokens
        _, kidx = top_tokens(np.linalg.svd(
            Wq.T @ rope_rel(d_head, base, -1) @ Wk)[2][0], En, tok)
        print("--- OV (what it writes for those keys) ---")
        ovs = []
        for ki in kidx[:5]:
            out = W_U @ (Wo @ (Wv @ En[ki]))
            tops = [tok.decode([int(i)]).strip() or "␣"
                    for i in np.argsort(-out)[:5]]
            ovs.append({"key": tok.decode([int(ki)]).strip(), "ov": tops})
            print(f"  '{tok.decode([int(ki)]).strip()}' -> {tops}")
        common = np.argsort(-np.abs(En).sum(1))[:2000]
        hits = 0
        for t in common[:400]:
            out = W_U @ (Wo @ (Wv @ En[t]))
            hits += int(np.argmax(out) == t)
        print(f"copy score (argmax==key over 400 tokens): {hits / 400:.2f}")
        hrec["ov"] = ovs
        hrec["copy_score"] = hits / 400
        results["heads"][f"L{L}H{h}"] = hrec

        # stimuli from top mined pair at each offset
        for D in (2, 4):
            r0 = hrec["offsets"][D][0]
            q_tok, k_tok = r0["q"][0], r0["k"][0]
            stimuli.append((L, h, D, k_tok, q_tok))

    # ---- live check ------------------------------------------------------
    print("\n================ live check ================")
    live = []
    filler_id = tok.encode(" the", add_special_tokens=False)[0]
    for L, h, D, k_tok, q_tok in stimuli:
        k_ids = tok.encode(" " + k_tok, add_special_tokens=False)
        q_ids = tok.encode(" " + q_tok, add_special_tokens=False)
        ids = [tok.bos_token_id] if tok.bos_token_id is not None else []
        kpos = len(ids)
        ids += k_ids[:1] + [filler_id] * (D - 1) + q_ids[:1]
        qpos = len(ids) - 1
        import torch as _t
        with _t.no_grad():
            out = mdl(input_ids=_t.tensor([ids]).to(dev),
                      output_attentions=True)
        A = out.attentions[L][0, h].float().cpu().numpy()
        got = float(A[qpos, kpos])
        rest = float(A[qpos].mean())
        live.append({"L": L, "h": h, "D": D, "key": k_tok, "query": q_tok,
                     "attn_q_to_k": round(got, 3),
                     "mean_attn": round(rest, 3),
                     "ratio": round(got / (rest + 1e-9), 1)})
        print(f"L{L} H{h} D={D}: '{k_tok}' ... '{q_tok}'  "
              f"attn={got:.3f} (mean {rest:.3f}, x{got / (rest + 1e-9):.1f})")
    results["live"] = live

    outp = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "results", "weight_mine_head.json")
    with open(outp, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {outp}")


if __name__ == "__main__":
    main()
