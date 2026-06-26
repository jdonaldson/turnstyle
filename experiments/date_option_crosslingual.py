"""Cross-lingual transfer of the per-option date-SELECTION probe + closest-date control.

(1) Train the binary 'is this the correct option?' probe on EN examples [30:150] (disjoint
    from the test set), test SELECTION on 18 problems whose STEMS are translated to ES/FR
    but whose option-dates are identical (language-invariant MM/DD/YYYY). If selection holds
    near the EN ~63%, it's a genuine multilingual date selector — the payoff of the thread.
(2) Closest-date control: pick the option whose date is nearest a date in the stem. If the
    probe beats this, it's recognition, not a 'pick the closest date' heuristic (BBH
    distractors are deliberate near-misses, so this should be weak).

  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python experiments/date_option_crosslingual.py
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import re
import numpy as np
from datetime import date
from turnstyle.bbh import load_task
from turnstyle.dates import _parse_date_str

# translated STEM+question (idx -> (es, fr)); option block is reused verbatim from EN
TR = {
 0: ("Hoy es Nochebuena de 1937. ¿Cuál es la fecha de mañana en formato MM/DD/YYYY?",
     "Aujourd'hui, c'est le réveillon de Noël de 1937. Quelle est la date de demain au format MM/DD/YYYY ?"),
 2: ("Jane y John se casaron el 2 de enero de 1958. Hoy es su quinto aniversario. ¿Cuál es la fecha dentro de una semana en formato MM/DD/YYYY?",
     "Jane et John se sont mariés le 2 janvier 1958. Aujourd'hui, c'est leur cinquième anniversaire de mariage. Quelle est la date dans une semaine au format MM/DD/YYYY ?"),
 3: ("Jane programó 3 citas con 5 personas para mañana (mar, 7/9/1972). ¿Cuál es la fecha de hace una semana en formato MM/DD/YYYY?",
     "Jane a programmé 3 rendez-vous avec 5 personnes pour demain (mar, 7/9/1972). Quelle est la date d'il y a une semaine au format MM/DD/YYYY ?"),
 4: ("La reunión de hoy se reprograma para mañana a las 11 am, 10/16/1924. ¿Cuál es la fecha dentro de una semana en formato MM/DD/YYYY?",
     "La réunion d'aujourd'hui est reportée à demain 11h, le 10/16/1924. Quelle est la date dans une semaine au format MM/DD/YYYY ?"),
 5: ("Jane visita la librería el 16 de cada mes desde octubre de 2009. Hoy es su quinta visita a la librería. ¿Cuál es la fecha de hace un año en formato MM/DD/YYYY?",
     "Jane visite la librairie le 16 de chaque mois depuis octobre 2009. Aujourd'hui, c'est sa cinquième visite à la librairie. Quelle est la date d'il y a un an au format MM/DD/YYYY ?"),
 6: ("El 9 de mayo de 2017 Jane compró 40 huevos. Comió uno por día. Hoy se quedó sin huevos. ¿Cuál es la fecha 24 horas después en formato MM/DD/YYYY?",
     "Le 9 mai 2017, Jane a acheté 40 œufs. Elle en a mangé un par jour. Aujourd'hui, elle n'a plus d'œufs. Quelle est la date 24 heures plus tard au format MM/DD/YYYY ?"),
 7: ("Mañana es 11/12/2019. ¿Cuál es la fecha de ayer en formato MM/DD/YYYY?",
     "Demain, c'est le 11/12/2019. Quelle est la date d'hier au format MM/DD/YYYY ?"),
 8: ("Jane cree que hoy es 6/18/2019, pero John cree que hoy es 6/19/2019. Jane tiene razón. ¿Cuál es la fecha de ayer en formato MM/DD/YYYY?",
     "Jane pense qu'aujourd'hui c'est le 6/18/2019, mais John pense qu'aujourd'hui c'est le 6/19/2019. Jane a raison. Quelle est la date d'hier au format MM/DD/YYYY ?"),
 11: ("Hoy es 10 de abril de 1985. La cita de Jane será 3 días después. ¿Cuál es la fecha de hace una semana en formato MM/DD/YYYY?",
      "Aujourd'hui, c'est le 10 avril 1985. Le rendez-vous de Jane aura lieu 3 jours plus tard. Quelle est la date d'il y a une semaine au format MM/DD/YYYY ?"),
 14: ("Mañana es 11/12/2019. ¿Cuál es la fecha dentro de una semana en formato MM/DD/YYYY?",
      "Demain, c'est le 11/12/2019. Quelle est la date dans une semaine au format MM/DD/YYYY ?"),
 16: ("El 6 de mayo de 1992 le parece a Jane como ayer, pero en realidad fue hace diez años. ¿Cuál es la fecha de hoy en formato MM/DD/YYYY?",
      "Le 6 mai 1992 semble hier pour Jane, mais c'était en fait il y a dix ans. Quelle est la date d'aujourd'hui au format MM/DD/YYYY ?"),
 18: ("Hoy es 9 de septiembre de 1909. ¿Cuál es la fecha de hace 10 días en formato MM/DD/YYYY?",
      "Aujourd'hui, c'est le 9 septembre 1909. Quelle est la date d'il y a 10 jours au format MM/DD/YYYY ?"),
 19: ("La hora local actual es las 3:02 pm del 5/4/2004. ¿Cuál es la fecha de hace un año en formato MM/DD/YYYY?",
      "L'heure locale actuelle est 15h02 le 5/4/2004. Quelle est la date d'il y a un an au format MM/DD/YYYY ?"),
 20: ("Ayer, 21 de enero de 2011, Jane comió 2 pizzas y 5 alitas. ¿Cuál es la fecha de hace una semana en formato MM/DD/YYYY?",
      "Hier, le 21 janvier 2011, Jane a mangé 2 pizzas et 5 ailes de poulet. Quelle est la date d'il y a une semaine au format MM/DD/YYYY ?"),
 22: ("Jane está celebrando el último día de enero de 2012. ¿Cuál es la fecha de ayer en formato MM/DD/YYYY?",
      "Jane fête le dernier jour de janvier 2012. Quelle est la date d'hier au format MM/DD/YYYY ?"),
 25: ("Jane reservó un vuelo para mañana, 29 de julio de 2002. ¿Cuál es la fecha dentro de una semana en formato MM/DD/YYYY?",
      "Jane a réservé un vol pour demain, le 29 juillet 2002. Quelle est la date dans une semaine au format MM/DD/YYYY ?"),
 26: ("La fecha límite es el 1 de junio de 2021, que está a 2 días de ahora. ¿Cuál es la fecha de hace una semana en formato MM/DD/YYYY?",
      "La date limite est le 1 juin 2021, soit dans 2 jours. Quelle est la date d'il y a une semaine au format MM/DD/YYYY ?"),
 28: ("Ayer fue 30 de abril de 2021. ¿Cuál es la fecha de hace una semana en formato MM/DD/YYYY?",
      "Hier, c'était le 30 avril 2021. Quelle est la date d'il y a une semaine au format MM/DD/YYYY ?"),
}


def parse_options(text):
    sec = text.split("Options:")[-1]
    return [(l, v.strip()) for l, v in re.findall(r'\(([A-Z])\)\s+([^\n(]+)', sec)]


def opt_idxs(chat, offs, opts):
    start = chat.find("Options:"); cur = start if start >= 0 else 0
    out = []
    for letter, dt in opts:
        pos = chat.find(dt, cur)
        if pos < 0:
            out.append((letter, None)); continue
        end = pos + len(dt); cur = end
        t = [k for k, (s, e) in enumerate(offs) if e > pos and s < end]
        out.append((letter, t[-1] if t else None))
    return out


def stem_dates(pre):
    out = []
    for c in re.findall(r'\d{1,2}/\d{1,2}/\d{2,4}', pre):
        d = _parse_date_str(c)
        if d:
            out.append(d)
    for m in re.finditer(r'[A-Z][a-z]+\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4}', pre):
        d = _parse_date_str(m.group(0))
        if d:
            out.append(d)
    return out


def closest_acc(examples):
    ok = n = 0
    for ex in examples:
        pre = ex["input"].split("Options:")[0]
        sd = stem_dates(pre)
        opts = parse_options(ex["input"])
        if not sd or not opts:
            continue
        best, bestdist = None, 1e18
        for letter, dv in opts:
            od = _parse_date_str(dv)
            if od is None:
                continue
            dist = min(abs((od - s).days) for s in sd)
            if dist < bestdist:
                bestdist, best = dist, letter
        if best is None:
            continue
        n += 1
        ok += int(f"({best})" == ex["target"].strip())
    return ok, n


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    mid = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(mid)
    mdl = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.float16).to(dev)
    print(f"model on {dev}", flush=True)

    examples = load_task("date_understanding")[:150]

    def collect(full_text, target):
        opts = parse_options(full_text)
        ct = tok.apply_chat_template([{"role": "user", "content": full_text}],
                                     tokenize=False, add_generation_prompt=True)
        enc = tok(ct, return_offsets_mapping=True, return_tensors="pt")
        offs = enc.pop("offset_mapping")[0].tolist()
        enc = {k: v.to(dev) for k, v in enc.items()}
        with torch.no_grad():
            hs = mdl(**enc, output_hidden_states=True).hidden_states
        stk = torch.stack(hs, 0)[:, 0, :, :].float().cpu().numpy()
        vecs, ys = [], []
        for letter, ti in opt_idxs(ct, offs, opts):
            if ti is None:
                continue
            vecs.append(stk[:, ti, :].astype(np.float16))
            ys.append(1 if f"({letter})" == target else 0)
        return np.stack(vecs) if vecs else None, np.array(ys)  # [n_opt, L, H], [n_opt]

    # ---- training rows from EN [30:150] ----
    tr_v, tr_y, tr_g = [], [], []
    for gid, ex in enumerate(examples[30:150]):
        V, y = collect(ex["input"], ex["target"].strip())
        if V is None:
            continue
        for j in range(len(y)):
            tr_v.append(V[j]); tr_y.append(y[j]); tr_g.append(gid)
        if gid % 40 == 0:
            print(f"  train {gid+1}/120", flush=True)
    TRV = np.stack(tr_v).astype(np.float32); TRY = np.array(tr_y)
    nL = TRV.shape[1]

    # ---- test problems (EN/ES/FR) ----
    test = {"en": [], "es": [], "fr": []}
    for idx, (es, fr) in TR.items():
        ex = examples[idx]
        _, _, post = ex["input"].partition("Options:")
        tgt = ex["target"].strip()
        test["en"].append(collect(ex["input"], tgt) + (tgt,))
        test["es"].append(collect(es + "\nOptions:" + post, tgt) + (tgt,))
        test["fr"].append(collect(fr + "\nOptions:" + post, tgt) + (tgt,))
    print(f"  test problems: {len(TR)} x3 langs", flush=True)

    ck, cn = closest_acc(examples[:150])
    print(f"\nclosest-date control (n={cn}): {ck}/{cn} = {ck/cn:.3f}", flush=True)

    def sel_acc(clf, sc, items, L):
        ok = n = 0
        for V, y, tgt in items:
            if V is None or y.sum() == 0:
                continue
            prob = clf.predict_proba(sc.transform(V[:, L, :].astype(np.float32)))[:, 1]
            ok += int(y[int(np.argmax(prob))] == 1); n += 1
        return ok, n

    print("\n=== cross-lingual selection (train EN[30:150], test 18 translated) ===")
    print(f"  {'L':>3s} | {'EN':>5s} {'ES':>5s} {'FR':>5s}")
    best = (-9, -1)
    for L in range(nL):
        sc = StandardScaler().fit(TRV[:, L, :])
        clf = LogisticRegression(C=0.3, max_iter=2000, class_weight="balanced")
        clf.fit(sc.transform(TRV[:, L, :]), TRY)
        eo, en_ = sel_acc(clf, sc, test["en"], L)
        so, sn = sel_acc(clf, sc, test["es"], L)
        fo, fn = sel_acc(clf, sc, test["fr"], L)
        m = (so / sn + fo / fn) / 2
        if m > best[0]:
            best = (m, L)
        if L % 2 == 0 or L == nL - 1:
            print(f"  {L:>3d} | {eo/en_:>5.2f} {so/sn:>5.2f} {fo/fn:>5.2f}", flush=True)
    print(f"  >>> best mean ES/FR selection = {best[0]:.2f} @L{best[1]}  (chance ~0.17)")


if __name__ == "__main__":
    main()
