"""Is there a GERMANIC construction that clarifies the directional sense of links/rechts,
pulling 'rechts' away from its recht=law / richtig=correct fusion?

The bare adverb 'rechts' never reaches the lateral cluster (cos with 'links' ~0 every layer).
But German has spatial-ONLY constructions the bare word lacks. We pool the directional word's
subspan inside each construction and score, per layer:
  lateral  = cos(left_form, right_form)        (global-centered) — want HIGH (they co-cluster)
  lawpull  = cos(right_form, 'Recht'/law)                        — want LOW
  margin   = lateral - lawpull                                   — the disambiguation gain
The construction with the biggest margin is what clarifies direction in Germanic.

  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python experiments/germanic_directional.py
"""
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import numpy as np

# (label, left_sentence, left_target, right_sentence, right_target)
DE_FORMS = [
    ("bare",    "links",              "links", "rechts",              "rechts"),
    ("nach",    "nach links",         "links", "nach rechts",         "rechts"),
    ("Seite",   "die linke Seite",    "linke", "die rechte Seite",    "rechte"),
    ("Hand",    "die linke Hand",     "linke", "die rechte Hand",     "rechte"),
    ("herum",   "linksherum",         "links", "rechtsherum",         "rechts"),
    ("abbiegen","Biege links ab.",    "links", "Biege rechts ab.",    "rechts"),
    ("Pfeil",   "der Pfeil zeigt nach links", "links",
                "der Pfeil zeigt nach rechts", "rechts"),   # "arrow points to the {dir}"
]
NL_FORMS = [
    ("bare",  "links",           "links", "rechts",           "rechts"),
    ("naar",  "naar links",      "links", "naar rechts",      "rechts"),
    ("kant",  "de linkerkant",   "linker", "de rechterkant",  "rechter"),
    ("hand",  "de linkerhand",   "linker", "de rechterhand",  "rechter"),
    ("af",    "Sla linksaf.",    "links", "Sla rechtsaf.",    "rechts"),
]
# law / correct anchors (the distractor cluster) — German is the reference for both (shared cognate)
DE_LAW = [("das Recht", "Recht"), ("das Gesetz", "Gesetz"), ("Das ist richtig.", "richtig")]


def pool(model, tok, dev, sentence, target):
    import torch
    cs = sentence.rfind(target); ce = cs + len(target)
    enc = tok(sentence, return_offsets_mapping=True, return_tensors="pt")
    offs = enc.pop("offset_mapping")[0].tolist()
    enc = {k: v.to(dev) for k, v in enc.items()}
    with torch.no_grad():
        hs = model(**enc, output_hidden_states=True).hidden_states
    stk = torch.stack(hs, 0)[:, 0, :, :]
    idxs = [k for k, (s, e) in enumerate(offs) if e > cs and s < ce] or [-1]
    return stk[:, idxs[-1], :].float().cpu().numpy()


def run(model, tok, dev, forms, law_anchor, title):
    L_acts = {lab: pool(model, tok, dev, ls, lt) for lab, ls, lt, _, _ in forms}
    R_acts = {lab: pool(model, tok, dev, rs, rt) for lab, _, _, rs, rt in forms}
    law = pool(model, tok, dev, *law_anchor)
    nL = law.shape[0]
    labels = [lab for lab, *_ in forms]
    print(f"\n=== {title} (law anchor: '{law_anchor[0]}') ===")
    print("margin = cos(left,right) - cos(right,law), per-dim Z-SCORED (neutralizes massive "
          "activations). higher = clearer direction; cos(L,R)>0 = the pair co-clusters laterally.")
    hdr = "  ".join(f"{lab:>8}" for lab in labels)
    print(f"{'L':>3} | {hdr}")
    best = {lab: (-9.0, -1) for lab in labels}
    for Lyr in range(nL):
        # z-score per dimension across ALL pooled vectors at this layer (kills rogue dims)
        stack = np.array([L_acts[lab][Lyr] for lab in labels]
                         + [R_acts[lab][Lyr] for lab in labels] + [law[Lyr]])
        mu, sd = stack.mean(0), stack.std(0) + 1e-6
        Z = (stack - mu) / sd
        n = len(labels)
        Lz = {lab: Z[i] for i, lab in enumerate(labels)}
        Rz = {lab: Z[n + i] for i, lab in enumerate(labels)}
        wz = Z[2 * n]

        def cos(a, b):
            return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
        cells = []
        for lab in labels:
            m = cos(Lz[lab], Rz[lab]) - cos(Rz[lab], wz)
            cells.append(f"{m:+.2f}")
            if m > best[lab][0]:
                best[lab] = (m, Lyr)
        if Lyr % 3 == 0:
            print(f"{Lyr:>3} | " + "  ".join(f"{c:>8}" for c in cells), flush=True)
    print("best margin/layer: " + "  ".join(
        f"{lab}={m:+.2f}@L{L}" for lab, (m, L) in best.items()))


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    mid = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(mid)
    mdl = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.float16).to(dev).eval()
    run(mdl, tok, dev, DE_FORMS, DE_LAW[0], "GERMAN constructions")
    run(mdl, tok, dev, NL_FORMS, DE_LAW[0], "DUTCH constructions (vs German law cognate)")


if __name__ == "__main__":
    main()
