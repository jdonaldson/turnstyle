"""
freq_audit_family.py — the frequency audit applied to the REST of the shipped
frame family: number, time, material(-naturalness), with size as a known-clean
control. Audits CANONICAL_FRAMES directly (the artifact, not experiment copies),
each frame with its own template.

Motivation: space died under this audit (0.886 -> 0.162 residualized — it was
word rarity). Number is the highest-risk survivor: "one" is a far commoner WORD
than "billion", so the family's strongest frame (r~0.95) could be partially
frequency. The blog post publicly promises this audit; the ATOM discussion
leans on the number direction.

Per frame: (a) label<->zipf correlation, (b) raw best CV r over layers,
(c) zipf-residualized best CV r. Same method as freq_residual.py (label
residualization; the opinion no-op there validated it as non-destructive).

Writes experiments/results/freq_audit_family.json.
Run:  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python experiments/freq_audit_family.py
"""
from __future__ import annotations
import json, os, sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from ordering_frames import cv_r
from turnstyle.frame_library import CANONICAL_FRAMES

from wordfreq import zipf_frequency

AUDIT = ["number", "time", "material", "size"]   # size = known-clean control
DEFAULT_TEMPLATE = "It is a {w} object."


def collect_last_t(model, tok, device, words, template):
    """Last-subword hidden states, per-frame template (ordering_frames.collect_last
    generalized)."""
    import torch
    out = {}
    for w in words:
        sent = template.format(w=w)
        cs = sent.rfind(w); ce = cs + len(w)
        enc = tok(sent, return_offsets_mapping=True, return_tensors="pt")
        offs = enc.pop("offset_mapping")[0].tolist()
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            hs = model(**enc, output_hidden_states=True).hidden_states
        stk = torch.stack(hs, 0)[:, 0, :, :]
        idxs = [k for k, (s, e) in enumerate(offs) if e > cs and s < ce] or [-1]
        out[w] = stk[:, idxs[-1], :].float().cpu().numpy()
    return out


def residualize(y, z):
    z1 = np.stack([z, np.ones_like(z)], axis=1)
    beta, *_ = np.linalg.lstsq(z1, y, rcond=None)
    return y - z1 @ beta


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    mid = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(mid)
    mdl = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.float16).to(dev)

    results = {"model": mid, "frames": {}}
    print("frame       n    label|zipf r    raw r@L        residualized r@L")
    for name in AUDIT:
        spec = CANONICAL_FRAMES[name]
        template = spec.get("template", DEFAULT_TEMPLATE)
        words = list(spec["data"])
        y = np.array([spec["data"][w] for w in words], dtype=float)
        z = np.array([zipf_frequency(w, "en") for w in words])
        lab_r = float(np.corrcoef(y, z)[0, 1])
        acts = collect_last_t(mdl, tok, dev, words, template)
        n_layers = acts[words[0]].shape[0]
        y_res = residualize(y, z)
        best_raw = best_res = (-9, -1)
        for L in range(n_layers):
            X = np.array([acts[w][L] for w in words])
            r_raw, r_res = cv_r(X, y), cv_r(X, y_res)
            if r_raw > best_raw[0]: best_raw = (r_raw, L)
            if r_res > best_res[0]: best_res = (r_res, L)
        results["frames"][name] = {
            "n": len(words), "template": template,
            "label_zipf_r": round(lab_r, 3),
            "raw": {"r": round(best_raw[0], 3), "layer": best_raw[1]},
            "residualized": {"r": round(best_res[0], 3), "layer": best_res[1]}}
        print(f"{name:10s} {len(words):3d}     {lab_r:+.3f}       "
              f"{best_raw[0]:+.3f}@L{best_raw[1]:<3d}   {best_res[0]:+.3f}@L{best_res[1]}",
              flush=True)

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "results", "freq_audit_family.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
