"""Follow-up: is the offset UNIT (day/week/month/year) cross-lingually readable, like
direction was? Train a 4-class EN probe on the last-token state, test ES/FR, per layer.
Chance = 0.25. If unit transfers like direction (88%), the WHOLE offset (direction x unit
x magnitude-from-digits) is recoverable cross-lingually from a mid-stack probe.

Balanced set: uniform "Today is {date}" anchor (so only the offset phrase varies -> the
probe must read the offset, not the anchor), 4 per unit, 8 forward / 8 backward.
Also re-checks DIRECTION on this fresh set as a replication.

  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python experiments/date_crosslingual_unit.py
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import numpy as np

# unit: 0=day 1=week 2=month 3=year ; dir: 0=back 1=fwd
UNITS = ["day", "week", "month", "year"]
P = [
    dict(unit=0, dir=1, en="Today is Sep 9, 1909. What is the date tomorrow in MM/DD/YYYY?",
         es="Hoy es 9 de septiembre de 1909. ¿Cuál es la fecha de mañana en formato MM/DD/YYYY?",
         fr="Aujourd'hui, c'est le 9 septembre 1909. Quelle est la date de demain au format MM/DD/YYYY ?"),
    dict(unit=0, dir=1, en="Today is Mar 3, 2001. What is the date 5 days later in MM/DD/YYYY?",
         es="Hoy es 3 de marzo de 2001. ¿Cuál es la fecha 5 días después en formato MM/DD/YYYY?",
         fr="Aujourd'hui, c'est le 3 mars 2001. Quelle est la date 5 jours plus tard au format MM/DD/YYYY ?"),
    dict(unit=0, dir=0, en="Today is Apr 10, 1985. What is the date 10 days ago in MM/DD/YYYY?",
         es="Hoy es 10 de abril de 1985. ¿Cuál es la fecha de hace 10 días en formato MM/DD/YYYY?",
         fr="Aujourd'hui, c'est le 10 avril 1985. Quelle est la date d'il y a 10 jours au format MM/DD/YYYY ?"),
    dict(unit=0, dir=0, en="Today is Jun 7, 1990. What is the date yesterday in MM/DD/YYYY?",
         es="Hoy es 7 de junio de 1990. ¿Cuál es la fecha de ayer en formato MM/DD/YYYY?",
         fr="Aujourd'hui, c'est le 7 juin 1990. Quelle est la date d'hier au format MM/DD/YYYY ?"),
    dict(unit=1, dir=1, en="Today is Jan 2, 1958. What is the date one week from today in MM/DD/YYYY?",
         es="Hoy es 2 de enero de 1958. ¿Cuál es la fecha dentro de una semana en formato MM/DD/YYYY?",
         fr="Aujourd'hui, c'est le 2 janvier 1958. Quelle est la date dans une semaine au format MM/DD/YYYY ?"),
    dict(unit=1, dir=1, en="Today is Oct 16, 1924. What is the date two weeks from today in MM/DD/YYYY?",
         es="Hoy es 16 de octubre de 1924. ¿Cuál es la fecha dentro de dos semanas en formato MM/DD/YYYY?",
         fr="Aujourd'hui, c'est le 16 octobre 1924. Quelle est la date dans deux semaines au format MM/DD/YYYY ?"),
    dict(unit=1, dir=0, en="Today is Jul 29, 2002. What is the date one week ago in MM/DD/YYYY?",
         es="Hoy es 29 de julio de 2002. ¿Cuál es la fecha de hace una semana en formato MM/DD/YYYY?",
         fr="Aujourd'hui, c'est le 29 juillet 2002. Quelle est la date d'il y a une semaine au format MM/DD/YYYY ?"),
    dict(unit=1, dir=0, en="Today is Nov 12, 2019. What is the date two weeks ago in MM/DD/YYYY?",
         es="Hoy es 12 de noviembre de 2019. ¿Cuál es la fecha de hace dos semanas en formato MM/DD/YYYY?",
         fr="Aujourd'hui, c'est le 12 novembre 2019. Quelle est la date d'il y a deux semaines au format MM/DD/YYYY ?"),
    dict(unit=2, dir=1, en="Today is Feb 1, 1987. What is the date one month from today in MM/DD/YYYY?",
         es="Hoy es 1 de febrero de 1987. ¿Cuál es la fecha dentro de un mes en formato MM/DD/YYYY?",
         fr="Aujourd'hui, c'est le 1 février 1987. Quelle est la date dans un mois au format MM/DD/YYYY ?"),
    dict(unit=2, dir=1, en="Today is May 9, 2017. What is the date three months from today in MM/DD/YYYY?",
         es="Hoy es 9 de mayo de 2017. ¿Cuál es la fecha dentro de tres meses en formato MM/DD/YYYY?",
         fr="Aujourd'hui, c'est le 9 mai 2017. Quelle est la date dans trois mois au format MM/DD/YYYY ?"),
    dict(unit=2, dir=0, en="Today is Dec 31, 1929. What is the date one month ago in MM/DD/YYYY?",
         es="Hoy es 31 de diciembre de 1929. ¿Cuál es la fecha de hace un mes en formato MM/DD/YYYY?",
         fr="Aujourd'hui, c'est le 31 décembre 1929. Quelle est la date d'il y a un mois au format MM/DD/YYYY ?"),
    dict(unit=2, dir=0, en="Today is Aug 16, 2009. What is the date two months ago in MM/DD/YYYY?",
         es="Hoy es 16 de agosto de 2009. ¿Cuál es la fecha de hace dos meses en formato MM/DD/YYYY?",
         fr="Aujourd'hui, c'est le 16 août 2009. Quelle est la date d'il y a deux mois au format MM/DD/YYYY ?"),
    dict(unit=3, dir=1, en="Today is Apr 19, 1969. What is the date one year from today in MM/DD/YYYY?",
         es="Hoy es 19 de abril de 1969. ¿Cuál es la fecha dentro de un año en formato MM/DD/YYYY?",
         fr="Aujourd'hui, c'est le 19 avril 1969. Quelle est la date dans un an au format MM/DD/YYYY ?"),
    dict(unit=3, dir=1, en="Today is Sep 9, 1909. What is the date two years from today in MM/DD/YYYY?",
         es="Hoy es 9 de septiembre de 1909. ¿Cuál es la fecha dentro de dos años en formato MM/DD/YYYY?",
         fr="Aujourd'hui, c'est le 9 septembre 1909. Quelle est la date dans deux ans au format MM/DD/YYYY ?"),
    dict(unit=3, dir=0, en="Today is May 6, 1992. What is the date one year ago in MM/DD/YYYY?",
         es="Hoy es 6 de mayo de 1992. ¿Cuál es la fecha de hace un año en formato MM/DD/YYYY?",
         fr="Aujourd'hui, c'est le 6 mai 1992. Quelle est la date d'il y a un an au format MM/DD/YYYY ?"),
    dict(unit=3, dir=0, en="Today is Mar 20, 2020. What is the date ten years ago in MM/DD/YYYY?",
         es="Hoy es 20 de marzo de 2020. ¿Cuál es la fecha de hace diez años en formato MM/DD/YYYY?",
         fr="Aujourd'hui, c'est le 20 mars 2020. Quelle est la date d'il y a diez años... ".replace("diez años...", "diez años en formato MM/DD/YYYY?"),
         ),
]
# fix Y4 fr (typo guard above only touched es); set explicitly
P[15]["fr"] = "Aujourd'hui, c'est le 20 mars 2020. Quelle est la date d'il y a dix ans au format MM/DD/YYYY ?"

LANGS = ["en", "es", "fr"]


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
    es_pred = clf.predict(sc.transform(Xes))
    fr_pred = clf.predict(sc.transform(Xfr))
    return loo, (es_pred == y).mean(), (fr_pred == y).mean(), es_pred, fr_pred


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    mid = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(mid)
    mdl = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.float16).to(dev)
    print(f"model on {dev}  N={len(P)} (unit chance 0.25, dir chance 0.50)", flush=True)

    acts = {lg: last_states(mdl, tok, dev, [p[lg] for p in P]) for lg in LANGS}
    yu = np.array([p["unit"] for p in P])
    yd = np.array([p["dir"] for p in P])
    nL = acts["en"].shape[0]

    print("\n=== offset UNIT (4-class) transfer: train EN, test ES/FR ===")
    print(f"  {'L':>3s} | {'EN-loo':>7s} {'ES':>5s} {'FR':>5s} | {'ES+FR':>6s}   ||  dir ES+FR")
    bestU = (-9, -1)
    perL = {}
    for L in range(nL):
        loo, es, fr, ep, fp = transfer(acts["en"][L], acts["es"][L], acts["fr"][L], yu)
        dloo, des, dfr, _, _ = transfer(acts["en"][L], acts["es"][L], acts["fr"][L], yd, C=0.5)
        m = (es + fr) / 2
        perL[L] = (ep, fp)
        if m > bestU[0]:
            bestU = (m, L)
        if L % 2 == 0 or L == nL - 1:
            print(f"  {L:>3d} | {loo:>7.2f} {es:>5.2f} {fr:>5.2f} | {m:>6.2f}   ||  {(des+dfr)/2:.2f}", flush=True)
    L = bestU[1]
    print(f"  >>> best mean ES/FR UNIT transfer = {bestU[0]:.2f} @L{L}")

    ep, fp = perL[L]
    print(f"\n=== unit confusion @L{L} (true -> pred), ES then FR ===")
    for lang, pred in (("ES", ep), ("FR", fp)):
        rows = [f"{UNITS[yu[i]]}->{UNITS[pred[i]]}" + ("" if pred[i] == yu[i] else " X")
                for i in range(len(P))]
        print(f"  {lang}: " + " | ".join(rows), flush=True)


if __name__ == "__main__":
    main()
