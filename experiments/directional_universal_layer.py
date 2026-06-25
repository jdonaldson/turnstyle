"""Is there a UNIVERSAL layer for directional sense (compass + left/right) across languages,
setting aside the Germanic rechts=recht fusion?

Compass (monosemous) peaked early-mid (L6-7); Romance left/right (polysemous, context-gated)
resolved later. Do they share a good layer? Per layer, fit NS/EW on EN compass and LR on EN
left/right, then measure cross-lingual sign accuracy (language-centered) for:
  compass  = primary-axis sign of n/s/e/w in each language ("Move {w}." frame)
  lateral  = left/right sign in each language's NATIVE turn-instruction
Romance set {es,fr,it,pt}; Germanic {de,nl} shown as a contrast at the chosen layer.

  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python experiments/directional_universal_layer.py
"""
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import math
import numpy as np
from turnstyle.frame_library import _collect, _ridge_dir

S = 1 / math.sqrt(2)
CMP_TMPL = "Move {w}."
NS_FIT = {"north": 1, "south": -1, "east": 0, "west": 0,
          "northeast": S, "northwest": S, "southeast": -S, "southwest": -S}
EW_FIT = {"east": 1, "west": -1, "north": 0, "south": 0,
          "northeast": S, "northwest": -S, "southeast": S, "southwest": -S}
CMP_FIT = list(NS_FIT)

# compass primary-axis tests per language (word, axis, sign)
COMPASS = {
    "es": [("norte","ns",1),("sur","ns",-1),("este","ew",1),("oeste","ew",-1)],
    "fr": [("nord","ns",1),("sud","ns",-1),("est","ew",1),("ouest","ew",-1)],
    "it": [("nord","ns",1),("sud","ns",-1),("est","ew",1),("ovest","ew",-1)],
    "pt": [("norte","ns",1),("sul","ns",-1),("leste","ew",1),("oeste","ew",-1)],
    "de": [("norden","ns",1),("süden","ns",-1),("osten","ew",1),("westen","ew",-1)],
    "nl": [("noorden","ns",1),("zuiden","ns",-1),("oosten","ew",1),("westen","ew",-1)],
}
# lateral: native turn-instruction (sentence, target_word, sign)
LAT_FIT = [("Turn left.", "left", -1), ("Turn right.", "right", 1)]
LATERAL = {
    "es": [("Gira a la izquierda.","izquierda",-1),("Gira a la derecha.","derecha",1)],
    "fr": [("Tourne à gauche.","gauche",-1),("Tourne à droite.","droite",1)],
    "it": [("Gira a sinistra.","sinistra",-1),("Gira a destra.","destra",1)],
    "pt": [("Vire à esquerda.","esquerda",-1),("Vire à direita.","direita",1)],
    "de": [("Biege links ab.","links",-1),("Biege rechts ab.","rechts",1)],
    "nl": [("Sla linksaf.","links",-1),("Sla rechtsaf.","rechts",1)],
}
ROMANCE = ["es", "fr", "it", "pt"]
GERMANIC = ["de", "nl"]


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


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    mid = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(mid)
    mdl = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.float16).to(dev).eval()

    # compass acts (Move-frame, isolated words)
    cmp_words = CMP_FIT + [w for v in COMPASS.values() for (w, _, _) in v]
    cmp_acts = _collect(mdl, tok, dev, list(dict.fromkeys(cmp_words)), CMP_TMPL, pool="last")
    # lateral acts (native sentences)
    lat_fit_acts = {w: pool(mdl, tok, dev, s, w) for (s, w, _) in LAT_FIT}
    lat_acts = {lg: {w: pool(mdl, tok, dev, s, w) for (s, w, _) in items}
                for lg, items in LATERAL.items()}
    nL = cmp_acts[CMP_FIT[0]].shape[0]
    ns_y = np.array([NS_FIT[w] for w in CMP_FIT], float)
    ew_y = np.array([EW_FIT[w] for w in CMP_FIT], float)
    lr_y = np.array([s for (_, _, s) in LAT_FIT], float)

    def compass_acc(L, langs):
        Xf = np.array([cmp_acts[w][L] for w in CMP_FIT])
        mu, sd = Xf.mean(0), Xf.std(0) + 1e-6
        nsd, ewd = _ridge_dir((Xf - mu) / sd, ns_y), _ridge_dir((Xf - mu) / sd, ew_y)
        hit = tot = 0
        for lg in langs:
            off = np.mean([cmp_acts[w][L] for (w, _, _) in COMPASS[lg]], 0)
            for (w, ax, sgn) in COMPASS[lg]:
                z = (cmp_acts[w][L] - off) / sd
                proj = z @ (nsd if ax == "ns" else ewd)
                hit += (proj > 0) == (sgn > 0); tot += 1
        return hit, tot

    def lateral_acc(L, langs):
        Xf = np.array([lat_fit_acts["left"][L], lat_fit_acts["right"][L]])
        mu, sd = Xf.mean(0), Xf.std(0) + 1e-6
        d = _ridge_dir((Xf - mu) / sd, lr_y)
        hit = tot = 0
        for lg in langs:
            ws = [w for (_, w, _) in LATERAL[lg]]
            off = np.mean([lat_acts[lg][w][L] for w in ws], 0)
            for (_, w, sgn) in LATERAL[lg]:
                proj = ((lat_acts[lg][w][L] - off) / sd) @ d
                hit += (proj > 0) == (sgn > 0); tot += 1
        return hit, tot

    print("ROMANCE {es,fr,it,pt}: cross-lingual directional sign accuracy per layer")
    print(f"{'L':>3} | {'compass/16':>11} {'lateral/8':>10} {'combined/24':>12}  (chance 8 / 4 / 12)")
    rows = []
    for L in range(nL):
        ch, ct = compass_acc(L, ROMANCE)
        lh, lt = lateral_acc(L, ROMANCE)
        rows.append((L, ch, lh, ch + lh))
        print(f"{L:>3} | {ch:>7}/16   {lh:>6}/8   {ch + lh:>8}/24", flush=True)

    bestC = max(rows, key=lambda r: r[1])
    bestL = max(rows, key=lambda r: r[2])
    bestComb = max(rows, key=lambda r: r[3])
    print(f"\nbest compass: L{bestC[0]} ({bestC[1]}/16)   "
          f"best lateral: L{bestL[0]} ({bestL[2]}/8)   "
          f"best COMBINED: L{bestComb[0]} ({bestComb[3]}/24)")
    # top-3 combined layers
    top = sorted(rows, key=lambda r: -r[3])[:5]
    print("top-5 combined layers: " + ", ".join(f"L{L}={c}/24(cmp{ch},lat{lh})"
          for (L, ch, lh, c) in top))

    print("\nGERMANIC {de,nl} contrast at those layers (expect compass OK, lateral poor):")
    for L in sorted({bestC[0], bestComb[0], bestL[0]}):
        gch, _ = compass_acc(L, GERMANIC)
        glh, _ = lateral_acc(L, GERMANIC)
        print(f"  L{L}: compass {gch}/8  lateral {glh}/4")


if __name__ == "__main__":
    main()
