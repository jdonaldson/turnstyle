"""Do BBH date problems share a cross-lingual layer? Two tests, both per-layer:

(1) RETRIEVAL: does EN problem i's last-token state retrieve its ES/FR translation
    (top-1, cosine, z-scored)? Chance = 1/16. The L0 embedding row is the LEXICAL
    control — dates (03/20/2020) + 'MM/DD/YYYY' are shared TOKENS across languages, so
    L0 alignment is digit-driven. A MID-STACK PEAK ABOVE L0 = semantic alignment beyond
    shared tokens.
(2) DIRECTION PROBE: train EN logistic 'is the question offset BACKWARD (ago/yesterday)
    or FORWARD (tomorrow/from today)?', test on ES & FR. Direction is NOT in the shared
    digits — it needs ago/hace/il-y-a — so a consistent transfer layer here is semantic.
    Several problems are minimal pairs (same anchor, opposite direction) to isolate it.

  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python experiments/date_crosslingual.py
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import numpy as np

# dir: 0 = backward (ago/yesterday/before), 1 = forward (tomorrow/from today/later)
PROBLEMS = [
    # ---- forward ----
    dict(dir=1,
         en="Today is Christmas Eve of 1937. What is the date tomorrow in MM/DD/YYYY?",
         es="Hoy es Nochebuena de 1937. ¿Cuál es la fecha de mañana en formato MM/DD/YYYY?",
         fr="Aujourd'hui, c'est le réveillon de Noël de 1937. Quelle est la date de demain au format MM/DD/YYYY ?"),
    dict(dir=1,
         en="Tomorrow is 11/12/2019. What is the date one week from today in MM/DD/YYYY?",
         es="Mañana es 11/12/2019. ¿Cuál es la fecha dentro de una semana en formato MM/DD/YYYY?",
         fr="Demain, c'est le 11/12/2019. Quelle est la date dans une semaine au format MM/DD/YYYY ?"),
    dict(dir=1,
         en="Jane scheduled 3 appointments with 5 people for tomorrow (Tue, 7/9/1972). What is the date tomorrow in MM/DD/YYYY?",
         es="Jane programó 3 citas con 5 personas para mañana (mar, 7/9/1972). ¿Cuál es la fecha de mañana en formato MM/DD/YYYY?",
         fr="Jane a programmé 3 rendez-vous avec 5 personnes pour demain (mar, 7/9/1972). Quelle est la date de demain au format MM/DD/YYYY ?"),
    dict(dir=1,
         en="Jane booked a flight for tomorrow, Jul 29, 2002. What is the date one week from today in MM/DD/YYYY?",
         es="Jane reservó un vuelo para mañana, 29 de julio de 2002. ¿Cuál es la fecha dentro de una semana en formato MM/DD/YYYY?",
         fr="Jane a réservé un vol pour demain, le 29 juillet 2002. Quelle est la date dans une semaine au format MM/DD/YYYY ?"),
    dict(dir=1,
         en="Today's meeting is rescheduled to 11 am tomorrow, 10/16/1924. What is the date one week from today in MM/DD/YYYY?",
         es="La reunión de hoy se reprograma para mañana a las 11 am, 10/16/1924. ¿Cuál es la fecha dentro de una semana en formato MM/DD/YYYY?",
         fr="La réunion d'aujourd'hui est reportée à demain 11h, le 10/16/1924. Quelle est la date dans une semaine au format MM/DD/YYYY ?"),
    dict(dir=1,
         en="Jane and John married on Jan 2, 1958. It is their 5-year anniversary today. What is the date one week from today in MM/DD/YYYY?",
         es="Jane y John se casaron el 2 de enero de 1958. Hoy es su quinto aniversario. ¿Cuál es la fecha dentro de una semana en formato MM/DD/YYYY?",
         fr="Jane et John se sont mariés le 2 janvier 1958. Aujourd'hui, c'est leur cinquième anniversaire de mariage. Quelle est la date dans une semaine au format MM/DD/YYYY ?"),
    dict(dir=1,
         en="On May 9th, 2017 Jane bought 40 eggs. She ate one per day. Today she ran out of eggs. What is the date 24 hours later in MM/DD/YYYY?",
         es="El 9 de mayo de 2017 Jane compró 40 huevos. Comió uno por día. Hoy se quedó sin huevos. ¿Cuál es la fecha 24 horas después en formato MM/DD/YYYY?",
         fr="Le 9 mai 2017, Jane a acheté 40 œufs. Elle en a mangé un par jour. Aujourd'hui, elle n'a plus d'œufs. Quelle est la date 24 heures plus tard au format MM/DD/YYYY ?"),
    dict(dir=1,
         en="Jane quit her job on Mar 20, 2020. 176 days have passed since then. What is the date tomorrow in MM/DD/YYYY?",
         es="Jane renunció a su trabajo el 20 de marzo de 2020. Han pasado 176 días desde entonces. ¿Cuál es la fecha de mañana en formato MM/DD/YYYY?",
         fr="Jane a quitté son emploi le 20 mars 2020. 176 jours se sont écoulés depuis. Quelle est la date de demain au format MM/DD/YYYY ?"),
    # ---- backward ----
    dict(dir=0,
         en="Today is Sep 9, 1909. What is the date 10 days ago in MM/DD/YYYY?",
         es="Hoy es 9 de septiembre de 1909. ¿Cuál es la fecha de hace 10 días en formato MM/DD/YYYY?",
         fr="Aujourd'hui, c'est le 9 septembre 1909. Quelle est la date d'il y a 10 jours au format MM/DD/YYYY ?"),
    dict(dir=0,
         en="Today is Apr 10, 1985. Jane's appointment will be 3 days later. What is the date one week ago from today in MM/DD/YYYY?",
         es="Hoy es 10 de abril de 1985. La cita de Jane será 3 días después. ¿Cuál es la fecha de hace una semana en formato MM/DD/YYYY?",
         fr="Aujourd'hui, c'est le 10 avril 1985. Le rendez-vous de Jane aura lieu 3 jours plus tard. Quelle est la date d'il y a une semaine au format MM/DD/YYYY ?"),
    dict(dir=0,
         en="Yesterday, Jan 21, 2011, Jane ate 2 pizzas and 5 wings. What is the date one week ago from today in MM/DD/YYYY?",
         es="Ayer, 21 de enero de 2011, Jane comió 2 pizzas y 5 alitas. ¿Cuál es la fecha de hace una semana en formato MM/DD/YYYY?",
         fr="Hier, le 21 janvier 2011, Jane a mangé 2 pizzas et 5 ailes de poulet. Quelle est la date d'il y a une semaine au format MM/DD/YYYY ?"),
    dict(dir=0,
         en="Yesterday was April 30, 2021. What is the date one week ago from today in MM/DD/YYYY?",
         es="Ayer fue 30 de abril de 2021. ¿Cuál es la fecha de hace una semana en formato MM/DD/YYYY?",
         fr="Hier, c'était le 30 avril 2021. Quelle est la date d'il y a une semaine au format MM/DD/YYYY ?"),
    dict(dir=0,
         en="Tomorrow is 11/12/2019. What is the date yesterday in MM/DD/YYYY?",
         es="Mañana es 11/12/2019. ¿Cuál es la fecha de ayer en formato MM/DD/YYYY?",
         fr="Demain, c'est le 11/12/2019. Quelle est la date d'hier au format MM/DD/YYYY ?"),
    dict(dir=0,
         en="Yesterday, Jan 21, 2011, Jane ate 2 pizzas and 5 wings. What is the date 10 days ago in MM/DD/YYYY?",
         es="Ayer, 21 de enero de 2011, Jane comió 2 pizzas y 5 alitas. ¿Cuál es la fecha de hace 10 días en formato MM/DD/YYYY?",
         fr="Hier, le 21 janvier 2011, Jane a mangé 2 pizzas et 5 ailes de poulet. Quelle est la date d'il y a 10 jours au format MM/DD/YYYY ?"),
    dict(dir=0,
         en="Jane thinks today is 6/18/2019, but John thinks today is 6/19/2019. Jane is correct. What is the date yesterday in MM/DD/YYYY?",
         es="Jane cree que hoy es 6/18/2019, pero John cree que hoy es 6/19/2019. Jane tiene razón. ¿Cuál es la fecha de ayer en formato MM/DD/YYYY?",
         fr="Jane pense qu'aujourd'hui c'est le 6/18/2019, mais John pense qu'aujourd'hui c'est le 6/19/2019. Jane a raison. Quelle est la date d'hier au format MM/DD/YYYY ?"),
    dict(dir=0,
         en="Jane quit her job on Mar 20, 2020. 176 days have passed since then. What is the date one year ago from today in MM/DD/YYYY?",
         es="Jane renunció a su trabajo el 20 de marzo de 2020. Han pasado 176 días desde entonces. ¿Cuál es la fecha de hace un año en formato MM/DD/YYYY?",
         fr="Jane a quitté son emploi le 20 mars 2020. 176 jours se sont écoulés depuis. Quelle est la date d'il y a un an au format MM/DD/YYYY ?"),
]

LANGS = ["en", "es", "fr"]


def last_states(model, tok, device, prompts):
    """Last-token hidden state, all layers, for each chat-templated prompt. -> [L, N, H]."""
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


def zscore(X):
    return (X - X.mean(0)) / (X.std(0) + 1e-6)


def retrieval(A, B):
    """top-1 retrieval acc both directions, z-scored cosine. A,B = [N,H]."""
    a, b = zscore(A), zscore(B)
    a /= (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)
    b /= (np.linalg.norm(b, axis=1, keepdims=True) + 1e-9)
    S = a @ b.T
    n = S.shape[0]
    fwd = (S.argmax(1) == np.arange(n)).mean()
    bwd = (S.argmax(0) == np.arange(n)).mean()
    return (fwd + bwd) / 2


def dir_transfer(Xen, Xes, Xfr, y):
    """Train logistic on EN, test ES/FR. Also EN LOO. Returns (loo, es, fr)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_predict, LeaveOneOut
    sc = StandardScaler().fit(Xen)
    en = sc.transform(Xen)
    clf = LogisticRegression(C=0.5, max_iter=2000)
    loo = (cross_val_predict(LogisticRegression(C=0.5, max_iter=2000), en, y,
                             cv=LeaveOneOut()) == y).mean()
    clf.fit(en, y)
    es = (clf.predict(sc.transform(Xes)) == y).mean()
    fr = (clf.predict(sc.transform(Xfr)) == y).mean()
    return loo, es, fr


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    mid = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(mid)
    mdl = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.float16).to(dev)
    print(f"model on {dev}  N={len(PROBLEMS)} (chance retrieval={1/len(PROBLEMS):.3f})", flush=True)

    acts = {lg: last_states(mdl, tok, dev, [p[lg] for p in PROBLEMS]) for lg in LANGS}
    y = np.array([p["dir"] for p in PROBLEMS])
    nL = acts["en"].shape[0]

    print("\n=== (1) cross-lingual retrieval (top-1, z-cosine) — L0 = lexical baseline ===")
    print(f"  {'L':>3s} | {'en-es':>6s} {'en-fr':>6s} {'es-fr':>6s} | {'mean':>6s}")
    for L in range(nL):
        ee = retrieval(acts["en"][L], acts["es"][L])
        ef = retrieval(acts["en"][L], acts["fr"][L])
        sf = retrieval(acts["es"][L], acts["fr"][L])
        if L % 2 == 0 or L == nL - 1:
            print(f"  {L:>3d} | {ee:>6.2f} {ef:>6.2f} {sf:>6.2f} | {(ee+ef+sf)/3:>6.2f}", flush=True)

    print("\n=== (2) offset-direction transfer: train EN, test ES/FR (chance 0.50) ===")
    print(f"  {'L':>3s} | {'EN-loo':>7s} {'ES':>5s} {'FR':>5s} | {'ES+FR':>6s}")
    best = (-9, -1)
    for L in range(nL):
        loo, es, fr = dir_transfer(acts["en"][L], acts["es"][L], acts["fr"][L], y)
        m = (es + fr) / 2
        if m > best[0]:
            best = (m, L)
        if L % 2 == 0 or L == nL - 1:
            print(f"  {L:>3d} | {loo:>7.2f} {es:>5.2f} {fr:>5.2f} | {m:>6.2f}", flush=True)
    print(f"  >>> best mean ES/FR direction transfer = {best[0]:.2f} @L{best[1]}")


if __name__ == "__main__":
    main()
