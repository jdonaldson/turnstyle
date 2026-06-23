"""AUDIT (audit-the-generator): the first-token readout that distorted cross-lingual
COLOR may also have understated cross-lingual AFFECT (the EPA/polarity transfer numbers
that underpin the affect frame). Same test as color_crosslingual but for valence:
fit an Evaluation axis on English ACT words, project translated valence-signed affect
adjectives (es/fr/de) onto it, FIRST vs LAST vs MEAN subword readout, by layer.

If last-token lifts affect cross-lingual like it did color, the project's affect
transfer numbers (0.74–0.9) are understated and the same lesson applies.
"""
from __future__ import annotations

import sys
import numpy as np

sys.path.insert(0, "experiments")
from epa_external_validation import load_act

# valence-signed affect adjectives, translated; valence is language-independent.
AFFECT = [
    (+3, {"en": "good", "es": "bueno", "fr": "bon", "de": "gut"}),
    (-3, {"en": "bad", "es": "malo", "fr": "mauvais", "de": "schlecht"}),
    (+3, {"en": "happy", "es": "feliz", "fr": "heureux", "de": "glücklich"}),
    (-3, {"en": "sad", "es": "triste", "fr": "triste", "de": "traurig"}),
    (+2, {"en": "beautiful", "es": "hermoso", "fr": "beau", "de": "schön"}),
    (-2, {"en": "ugly", "es": "feo", "fr": "laid", "de": "hässlich"}),
    (+2, {"en": "kind", "es": "amable", "fr": "gentil", "de": "nett"}),
    (-3, {"en": "cruel", "es": "cruel", "fr": "cruel", "de": "grausam"}),
    (+3, {"en": "wonderful", "es": "maravilloso", "fr": "merveilleux", "de": "wunderbar"}),
    (-3, {"en": "terrible", "es": "terrible", "fr": "terrible", "de": "schrecklich"}),
    (+1, {"en": "calm", "es": "tranquilo", "fr": "calme", "de": "ruhig"}),
    (-2, {"en": "angry", "es": "enfadado", "fr": "fâché", "de": "wütend"}),
    (+2, {"en": "brave", "es": "valiente", "fr": "courageux", "de": "mutig"}),
    (-2, {"en": "lonely", "es": "solitario", "fr": "seul", "de": "einsam"}),
]
TEMPLATES = {"en": "It is {w}.", "es": "Es {w}.",
             "fr": "C'est {w}.", "de": "Es ist {w}."}


def collect(model, tok, device, words, template):
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
        out[w] = {"first": stk[:, idxs[0], :].float().cpu().numpy(),
                  "last": stk[:, idxs[-1], :].float().cpu().numpy(),
                  "mean": stk[:, idxs, :].mean(1).float().cpu().numpy()}
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

    act = load_act()
    test_en = {a[1]["en"] for a in AFFECT}
    train = [w for w in sorted(act) if w not in test_en][:500]
    E_tr = np.array([act[w][0] for w in train])             # Evaluation rating
    VAL = np.array([v for v, _ in AFFECT])
    # report token counts (the suspected mechanism)
    for lang in TEMPLATES:
        n = [len(tok.encode(a[1][lang], add_special_tokens=False)) for a in AFFECT]
        print(f"{lang}: avg_tokens={np.mean(n):.2f} single={sum(x==1 for x in n)}/{len(AFFECT)}", flush=True)

    tr = collect(mdl, tok, dev, train, TEMPLATES["en"])
    te = {lang: collect(mdl, tok, dev, [a[1][lang] for a in AFFECT], TEMPLATES[lang])
          for lang in TEMPLATES}
    n_layers = tr[train[0]]["first"].shape[0]

    def analyze(pool):
        prof = {lang: [] for lang in TEMPLATES}
        for layer in range(n_layers):
            Xtr = np.array([tr[w][pool][layer] for w in train])
            mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
            d = ridge_dir((Xtr - mu) / sd, E_tr)
            for lang in TEMPLATES:
                Xte = np.array([te[lang][a[1][lang]][pool][layer] for a in AFFECT])
                proj = ((Xte - mu) / sd) @ d
                r = np.corrcoef(proj, VAL)[0, 1]
                prof[lang].append(float(r) if np.isfinite(r) else 0.0)
        return prof

    for pool in ("first", "last", "mean"):
        prof = analyze(pool)
        print(f"\n===== POOL={pool.upper()} (corr EN-Eval-axis proj vs valence) =====")
        for lang in TEMPLATES:
            a = np.array(prof[lang]); b = int(a.argmax())
            print(f"  {lang}: best {a[b]:+.3f} @L{b}  (L0={a[0]:+.3f})")
        xl = np.mean([prof[l] for l in ("es", "fr", "de")], 0); b = int(xl.argmax())
        print(f"  cross-ling mean: best {xl[b]:+.3f} @L{b}  (L0={xl[0]:+.3f})")


if __name__ == "__main__":
    main()
