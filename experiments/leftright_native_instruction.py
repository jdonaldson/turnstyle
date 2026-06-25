"""Does each word's NATIVE navigation instruction gate the lateral sense cross-lingually?

leftright_xlingual_spatial.py was confounded: it put foreign words in an ENGLISH frame
("Turn to the rechts" — code-mixed, ungrammatical), so the instruction couldn't gate the
foreign polyseme. Here each left/right word sits in its OWN language's "turn {w}" instruction
(_collect pools just the word's subword span, isolating its gated representation). Fit the
lateral direction on ENGLISH (Turn left/right), project the native-framed foreign pairs
(language-centered) per layer. Also report cos(leftword,rightword) WITHIN each native frame —
if the native instruction gates the sense, this should be high (like EN spatial 0.65-0.75).

  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python experiments/leftright_native_instruction.py
"""
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import numpy as np
from turnstyle.frame_library import _ridge_dir

# (lang, left_word, right_word, native_instruction_template with {w})
NATIVE = [
    ("en", "left", "right", "Turn {w}."),
    ("es", "izquierda", "derecha", "Gira a la {w}."),
    ("fr", "gauche", "droite", "Tourne à {w}."),
    ("de", "links", "rechts", "Biege {w} ab."),
    ("it", "sinistra", "destra", "Gira a {w}."),
    ("pt", "esquerda", "direita", "Vire à {w}."),
    ("nl", "links", "rechts", "Sla {w}af."),
]


def collect_word_in(model, tok, dev, word, sentence):
    """Pool the last subword of `word` within its native `sentence` (offset-mapped)."""
    import torch
    cs = sentence.rfind(word); ce = cs + len(word)
    enc = tok(sentence, return_offsets_mapping=True, return_tensors="pt")
    offs = enc.pop("offset_mapping")[0].tolist()
    enc = {k: v.to(dev) for k, v in enc.items()}
    with torch.no_grad():
        hs = model(**enc, output_hidden_states=True).hidden_states
    stk = torch.stack(hs, 0)[:, 0, :, :]
    idxs = [k for k, (s, e) in enumerate(offs) if e > cs and s < ce] or [-1]
    return stk[:, idxs[-1], :].float().cpu().numpy()        # (L+1, H)


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    mid = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(mid)
    mdl = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.float16).to(dev).eval()

    # acts[lang] = {"L": (n_layers,H), "R": (n_layers,H)} from the NATIVE instruction
    acts = {}
    for lang, lw, rw, tmpl in NATIVE:
        acts[lang] = {"L": collect_word_in(mdl, tok, dev, lw, tmpl.format(w=lw)),
                      "R": collect_word_in(mdl, tok, dev, rw, tmpl.format(w=rw))}
    nL = acts["en"]["L"].shape[0]

    langs = [lang for lang, *_ in NATIVE]

    # fit lateral direction on EN (native "Turn left/right")
    print("native-instruction cross-lingual left/right (fit on EN, language-centered)")
    print(f"{'L':>3} | xling sign /12 | per-lang cos(L,R) in native frame, GLOBAL-mean-"
          "centered (lateral-cluster gating check)")
    best = (0, -1)
    for L in range(nL):
        # global mean over all 14 native-framed words at this layer
        G = np.mean([acts[lg][s][L] for lg in langs for s in ("L", "R")], 0)
        Xf = np.array([acts["en"]["L"][L], acts["en"]["R"][L]])
        mu, sd = Xf.mean(0), Xf.std(0) + 1e-6
        d = _ridge_dir((Xf - mu) / sd, np.array([-1.0, 1.0]))

        hit, cos_cells = 0, []
        for lang in langs:
            if lang == "en":
                continue
            l, r = acts[lang]["L"][L], acts[lang]["R"][L]
            off = (l + r) / 2.0                                # per-language centering (sign)
            hit += (float(((l - off) / sd) @ d) < 0) + (float(((r - off) / sd) @ d) > 0)
            lc, rc = l - G, r - G                              # global-centered (gating cos)
            c = float(lc @ rc / (np.linalg.norm(lc) * np.linalg.norm(rc) + 1e-9))
            cos_cells.append(f"{lang}:{c:+.2f}")
        if hit > best[0]:
            best = (hit, L)
        if L % 2 == 0 or hit >= 10:
            print(f"{L:>3} | {hit:>11}/12 | " + " ".join(cos_cells), flush=True)
    print(f"\nbest cross-lingual sign acc: {best[0]}/12 @ L{best[1]} (chance 6). "
          "SIGN-ACC is the transfer metric; cos(L,R) high+ = both land in one lateral cluster.")


if __name__ == "__main__":
    main()
