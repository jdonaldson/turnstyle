"""Anchor-framing transfer: is the STATED date the reference (today), or displaced
(tomorrow/yesterday)? This is the other half of the date 'glue' (direction was the offset
glue). 3-class probe (today/tomorrow/yesterday), train EN, test ES/FR, per layer. Chance
0.333.

Isolation: the framing word lives ONLY in the anchor sentence; the question is CONSTANT
('3 days from now') so the probe can't read framing off the question. 6 surface forms per
class so it learns the concept, not a phrase. If framing transfers like direction (~0.85),
direction+framing = a complete probe-readable, cross-lingual anchor+offset reader.

  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python experiments/date_crosslingual_framing.py
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import numpy as np

CLASSES = ["today", "tomorrow", "yesterday"]

# (en, es, fr-with-article) — 18 distinct dates; classes take 0-5 / 6-11 / 12-17
DATES = [
    ("Sep 9, 1909", "9 de septiembre de 1909", "le 9 septembre 1909"),
    ("Mar 3, 2001", "3 de marzo de 2001", "le 3 mars 2001"),
    ("Apr 10, 1985", "10 de abril de 1985", "le 10 avril 1985"),
    ("Jun 7, 1990", "7 de junio de 1990", "le 7 juin 1990"),
    ("Jan 2, 1958", "2 de enero de 1958", "le 2 janvier 1958"),
    ("Oct 16, 1924", "16 de octubre de 1924", "le 16 octobre 1924"),
    ("Jul 29, 2002", "29 de julio de 2002", "le 29 juillet 2002"),
    ("Nov 12, 2019", "12 de noviembre de 2019", "le 12 novembre 2019"),
    ("Feb 1, 1987", "1 de febrero de 1987", "le 1 février 1987"),
    ("May 9, 2017", "9 de mayo de 2017", "le 9 mai 2017"),
    ("Dec 31, 1929", "31 de diciembre de 1929", "le 31 décembre 1929"),
    ("Aug 16, 2009", "16 de agosto de 2009", "le 16 août 2009"),
    ("Apr 19, 1969", "19 de abril de 1969", "le 19 avril 1969"),
    ("Mar 20, 2020", "20 de marzo de 2020", "le 20 mars 2020"),
    ("May 6, 1992", "6 de mayo de 1992", "le 6 mai 1992"),
    ("Jun 18, 2019", "18 de junio de 2019", "le 18 juin 2019"),
    ("Jul 4, 1976", "4 de julio de 1976", "le 4 juillet 1976"),
    ("Oct 31, 1888", "31 de octubre de 1888", "le 31 octobre 1888"),
]

# 6 surface forms per class per language, {d} = localized date
FRAMES = {
    "en": {
        "today": ["Today is {d}.", "It is {d} today.", "The current date is {d}.",
                  "As of today, it is {d}.", "Today's date is {d}.", "Right now it is {d}."],
        "tomorrow": ["Tomorrow is {d}.", "Tomorrow will be {d}.", "The day after today is {d}.",
                     "Tomorrow's date is {d}.", "It will be {d} tomorrow.", "Tomorrow is going to be {d}."],
        "yesterday": ["Yesterday was {d}.", "Yesterday's date was {d}.", "The day before today was {d}.",
                      "It was {d} yesterday.", "Yesterday it was {d}.", "The previous day was {d}."],
    },
    "es": {
        "today": ["Hoy es {d}.", "Es {d} hoy.", "La fecha actual es {d}.",
                  "A día de hoy, es {d}.", "La fecha de hoy es {d}.", "Ahora mismo es {d}."],
        "tomorrow": ["Mañana es {d}.", "Mañana será {d}.", "El día después de hoy es {d}.",
                     "La fecha de mañana es {d}.", "Será {d} mañana.", "Mañana va a ser {d}."],
        "yesterday": ["Ayer fue {d}.", "La fecha de ayer fue {d}.", "El día antes de hoy fue {d}.",
                      "Fue {d} ayer.", "Ayer era {d}.", "El día anterior fue {d}."],
    },
    "fr": {
        "today": ["Aujourd'hui, c'est {d}.", "C'est {d} aujourd'hui.", "La date actuelle est {d}.",
                  "À ce jour, on est {d}.", "La date d'aujourd'hui est {d}.", "En ce moment, on est {d}."],
        "tomorrow": ["Demain, c'est {d}.", "Demain sera {d}.", "Le jour après aujourd'hui est {d}.",
                     "La date de demain est {d}.", "Ce sera {d} demain.", "Demain, ça va être {d}."],
        "yesterday": ["Hier, c'était {d}.", "La date d'hier était {d}.", "Le jour avant aujourd'hui était {d}.",
                      "C'était {d} hier.", "Hier, on était {d}.", "Le jour précédent était {d}."],
    },
}
QUESTION = {
    "en": " What is the date 3 days from now in MM/DD/YYYY?",
    "es": " ¿Cuál es la fecha dentro de 3 días en formato MM/DD/YYYY?",
    "fr": " Quelle est la date dans 3 jours au format MM/DD/YYYY ?",
}
LANGS = ["en", "es", "fr"]
DIDX = {"en": 0, "es": 1, "fr": 2}


def build(lang):
    """-> (list[str], labels). class c uses dates[6c:6c+6], variant i with date 6c+i."""
    prompts, y = [], []
    for c, cls in enumerate(CLASSES):
        for i in range(6):
            d = DATES[6 * c + i][DIDX[lang]]
            prompts.append(FRAMES[lang][cls][i].format(d=d) + QUESTION[lang])
            y.append(c)
    return prompts, np.array(y)


def last_states(model, tok, device, prompts):
    import torch
    out = []
    for p in prompts:
        ct = tok.apply_chat_template([{"role": "user", "content": p}],
                                     tokenize=False, add_generation_prompt=True)
        enc = tok(ct, return_tensors="pt").to(device)
        with torch.no_grad():
            hs = model(**enc, output_hidden_states=True).hidden_states
        out.append(torch.stack(hs, 0)[:, 0, -1, :].float().cpu().numpy())
    return np.stack(out, 1)


def transfer(Xen, Xes, Xfr, y, C=0.3):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_predict, LeaveOneOut
    sc = StandardScaler().fit(Xen)
    en = sc.transform(Xen)
    loo = (cross_val_predict(LogisticRegression(C=C, max_iter=3000), en, y,
                             cv=LeaveOneOut()) == y).mean()
    clf = LogisticRegression(C=C, max_iter=3000).fit(en, y)
    ep, fp = clf.predict(sc.transform(Xes)), clf.predict(sc.transform(Xfr))
    return loo, (ep == y).mean(), (fp == y).mean(), ep, fp


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    mid = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(mid)
    mdl = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.float16).to(dev)
    y = build("en")[1]
    print(f"model on {dev}  N={len(y)} (framing 3-class, chance 0.333)", flush=True)

    acts = {lg: last_states(mdl, tok, dev, build(lg)[0]) for lg in LANGS}
    nL = acts["en"].shape[0]

    print("\n=== anchor-framing (today/tomorrow/yesterday) transfer: train EN, test ES/FR ===")
    print(f"  {'L':>3s} | {'EN-loo':>7s} {'ES':>5s} {'FR':>5s} | {'ES+FR':>6s}")
    best = (-9, -1)
    perL = {}
    for L in range(nL):
        loo, es, fr, ep, fp = transfer(acts["en"][L], acts["es"][L], acts["fr"][L], y)
        m = (es + fr) / 2
        perL[L] = (ep, fp)
        if m > best[0]:
            best = (m, L)
        if L % 2 == 0 or L == nL - 1:
            print(f"  {L:>3d} | {loo:>7.2f} {es:>5.2f} {fr:>5.2f} | {m:>6.2f}", flush=True)
    L = best[1]
    print(f"  >>> best mean ES/FR framing transfer = {best[0]:.2f} @L{L}")

    ep, fp = perL[L]
    print(f"\n=== framing confusion @L{L} (true -> pred) ===")
    for lang, pred in (("ES", ep), ("FR", fp)):
        rows = [f"{CLASSES[y[i]][:3]}->{CLASSES[pred[i]][:3]}" + ("" if pred[i] == y[i] else "X")
                for i in range(len(y))]
        print(f"  {lang}: " + " ".join(rows), flush=True)


if __name__ == "__main__":
    main()
