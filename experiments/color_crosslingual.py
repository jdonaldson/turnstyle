"""Cross-lingual color: does an ENGLISH-fit color axis pole es/fr/de color words,
and AT WHICH LAYER? This disambiguates the within-EN L0/L1 peak: if color is pure
lexical/embedding lookup, EN-fit axes won't transfer at L0 (different-language tokens
have different embeddings) and cross-lingual alignment should peak DEEPER, where the
model maps color words to a shared meaning. If L0 transfers cross-lingually too, the
early peak is genuinely semantic, not orthographic.

Method: fit ridge color directions (L*,a*,b*) on ~90 EN colors at each layer (EN
standardization); project the 12 basic-color-term translations onto those EN axes;
correlate projection with the (language-independent) true Lab, per layer per language.
"""
from __future__ import annotations

import sys
import numpy as np

sys.path.insert(0, "experiments")
from color_affect_frame import COLORS, _hex_to_lab

# 12 Berlin-Kay basic color terms, translated; Lab is language-independent (the hex).
BASIC = [
    ("000000", {"en": "black", "es": "negro", "fr": "noir", "de": "schwarz"}),
    ("ffffff", {"en": "white", "es": "blanco", "fr": "blanc", "de": "weiß"}),
    ("ff0000", {"en": "red", "es": "rojo", "fr": "rouge", "de": "rot"}),
    ("008000", {"en": "green", "es": "verde", "fr": "vert", "de": "grün"}),
    ("ffff00", {"en": "yellow", "es": "amarillo", "fr": "jaune", "de": "gelb"}),
    ("0000ff", {"en": "blue", "es": "azul", "fr": "bleu", "de": "blau"}),
    ("a52a2a", {"en": "brown", "es": "marrón", "fr": "marron", "de": "braun"}),
    ("ffa500", {"en": "orange", "es": "naranja", "fr": "orange", "de": "orange"}),
    ("ffc0cb", {"en": "pink", "es": "rosa", "fr": "rose", "de": "rosa"}),
    ("800080", {"en": "purple", "es": "morado", "fr": "violet", "de": "lila"}),
    ("808080", {"en": "grey", "es": "gris", "fr": "gris", "de": "grau"}),
    ("40e0d0", {"en": "turquoise", "es": "turquesa", "fr": "turquoise", "de": "türkis"}),
]

TEMPLATES = {"en": "It is {w}.", "es": "Es {w}.",
             "fr": "C'est {w}.", "de": "Es ist {w}."}


def collect(model, tok, device, words, template):
    """Return {w: {'first','last','mean': (L+1,H)}} — pool the word's subword span
    three ways in one forward pass, so we can compare token-readout choices."""
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
        stk = torch.stack(hs, 0)[:, 0, :, :]            # (L+1, T, H)
        idxs = [k for k, (s, e) in enumerate(offs) if e > cs and s < ce] or [-1]
        out[w] = {
            "first": stk[:, idxs[0], :].float().cpu().numpy(),
            "last": stk[:, idxs[-1], :].float().cpu().numpy(),
            "mean": stk[:, idxs, :].mean(1).float().cpu().numpy(),
        }
    return out


def ridge_dir(Xs, y, alpha=10.0):
    w = np.linalg.solve(Xs.T @ Xs + alpha * np.eye(Xs.shape[1]), Xs.T @ (y - y.mean()))
    return w / (np.linalg.norm(w) + 1e-9)


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    mid = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(mid)
    mdl = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.float16).to(dev)

    basic_en = {c[1]["en"] for c in BASIC}
    train = [c for c in COLORS if c not in basic_en and c not in ("fuchsia", "aqua")]
    LAB_tr = np.array([_hex_to_lab(COLORS[w]) for w in train])
    LAB_te = np.array([_hex_to_lab(h) for h, _ in BASIC])
    print(f"train EN colors={len(train)}  test basic colors={len(BASIC)} x4 langs", flush=True)

    tr = collect(mdl, tok, dev, train, TEMPLATES["en"])
    te = {lang: collect(mdl, tok, dev, [c[1][lang] for c in BASIC], TEMPLATES[lang])
          for lang in TEMPLATES}
    n_layers = tr[train[0]]["first"].shape[0]
    chans = ["L*", "a*", "b*"]

    def analyze(pool):
        prof = {lang: [] for lang in TEMPLATES}
        perchan = {lang: {c: [] for c in chans} for lang in TEMPLATES}
        for layer in range(n_layers):
            Xtr = np.array([tr[w][pool][layer] for w in train])
            mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
            Xtrs = (Xtr - mu) / sd
            dirs = [ridge_dir(Xtrs, LAB_tr[:, j]) for j in range(3)]
            for lang in TEMPLATES:
                Xte = np.array([te[lang][c[1][lang]][pool][layer] for c in BASIC])
                Xtes = (Xte - mu) / sd
                cs = []
                for j in range(3):
                    proj = Xtes @ dirs[j]
                    r = np.corrcoef(proj, LAB_te[:, j])[0, 1]
                    r = r if np.isfinite(r) else 0.0
                    cs.append(r); perchan[lang][chans[j]].append(r)
                prof[lang].append(float(np.mean(cs)))
        return prof, perchan

    for pool in ("first", "last", "mean"):
        prof, perchan = analyze(pool)
        print(f"\n========== POOL = {pool.upper()} ==========")
        print("  best layer per language (mean-channel alignment):")
        for lang in TEMPLATES:
            a = np.array(prof[lang]); b = int(a.argmax())
            print(f"    {lang}: best {a[b]:+.3f} @L{b}  (L0={a[0]:+.3f}, L1={a[1]:+.3f})")
        print("  per-channel best (cross-lingual avg es/fr/de):")
        for c in chans:
            s = np.mean([perchan[l][c] for l in ("es", "fr", "de")], 0)
            b = int(s.argmax())
            print(f"    {c}: best {s[b]:+.3f} @L{b}  (L0={s[0]:+.3f}, L1={s[1]:+.3f})")
        xl = np.mean([prof[l] for l in ("es", "fr", "de")], 0)
        b = int(xl.argmax())
        print(f"  ALL-cross-lingual mean: best {xl[b]:+.3f} @L{b}  "
              f"(L0={xl[0]:+.3f}, L1={xl[1]:+.3f}, peak-vs-L0 Δ={xl[b]-xl[0]:+.3f})")


if __name__ == "__main__":
    main()
