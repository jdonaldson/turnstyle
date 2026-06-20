"""Qualitative write-side comparison: print Evaluation vs Potency steered generations.

The quantitative steer_substrate.py was mis-instrumented (distinct-2 misses word-salad).
Eyeball it instead: does the clean axis (Evaluation) and the noisier axis (Potency) each
steer COHERENTLY on its own theme, or does one break down?
"""
import sys
sys.path.insert(0, "experiments")
sys.path.insert(0, "src")
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import pole_probe as PP
from osgood_epa import EPA

L = 9
dev = "mps" if torch.backends.mps.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained(PP.MODEL_ID)
mdl = AutoModelForCausalLM.from_pretrained(PP.MODEL_ID, dtype=torch.float16).to(dev).eval()


def adj_vec(words):
    out = []
    for w in words:
        sent = f"It is very {w}."
        cs = sent.rfind(w); ce = cs + len(w)
        enc = tok(sent, return_offsets_mapping=True, return_tensors="pt")
        offs = enc.pop("offset_mapping")[0].tolist()
        enc = {k: v.to(dev) for k, v in enc.items()}
        with torch.no_grad():
            o = mdl(**enc, output_hidden_states=True)
        tk = next((k for k, (s, e) in enumerate(offs) if e > cs and s < ce), -1)
        out.append(o.hidden_states[L][0, tk])
    return torch.stack(out)


vecs = {}
for name in ("evaluation", "potency"):
    hi = adj_vec(EPA[name]["en"]["hi"]); lo = adj_vec(EPA[name]["en"]["lo"])
    v = hi.mean(0) - lo.mean(0); vecs[name] = v / v.norm()

state = {"v": None, "alpha": 0.0}


def hook(m, i, out):
    if state["alpha"] == 0.0:
        return out
    h = out[0] if isinstance(out, tuple) else out
    h = h + state["alpha"] * h.norm(dim=-1, keepdim=True) * state["v"].to(h.dtype)
    return (h,) + out[1:] if isinstance(out, tuple) else h


mdl.model.layers[L].register_forward_hook(hook)
prompts = ["The neighborhood I live in is", "When I think about the project, it seems"]
for name, v in vecs.items():
    print(f"\n===== steering on {name.upper()} axis =====")
    for pr in prompts:
        print(f"PROMPT: {pr}")
        for a in (-0.3, 0.0, 0.3):
            state["v"] = v; state["alpha"] = a
            enc = {k: x.to(dev) for k, x in tok(pr, return_tensors="pt").items()}
            with torch.no_grad():
                g = mdl.generate(**enc, max_new_tokens=30, do_sample=False,
                                 repetition_penalty=1.3, pad_token_id=tok.eos_token_id)
            txt = tok.decode(g[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
            print(f"  a={a:+.1f}  {txt.strip()[:100]}")
        print()
