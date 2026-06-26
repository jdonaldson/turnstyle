"""Head-to-head: a CROSS-LINGUAL SYMBOLIC solver vs the option-probe (72%) on the 18
translated date problems. Uses only closed-class tiny lexicons (months/units/directions/
framing/numbers in EN/ES/FR) + language-free numeric dates + relativedelta arithmetic.
If symbolic >= probe multilingually, the multilingual date capability is symbolic-first.

No model. Decides: where the anchor parses, symbolic should win exactly; the probe's niche
is the anchors symbolic CAN'T parse (holidays/recurring/anniversary) but might RECOGNIZE.

  .venv/bin/python experiments/date_xlingual_symbolic.py
"""
from __future__ import annotations
import sys, re
from datetime import date, timedelta
from dateutil.relativedelta import relativedelta as rd
sys.path.insert(0, "experiments")
from date_option_crosslingual import TR
from turnstyle.bbh import load_task

MONTHS = {
    'january': 1, 'february': 2, 'march': 3, 'april': 4, 'may': 5, 'june': 6, 'july': 7,
    'august': 8, 'september': 9, 'october': 10, 'november': 11, 'december': 12,
    'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'jun': 6, 'jul': 7, 'aug': 8, 'sep': 9,
    'oct': 10, 'nov': 11, 'dec': 12,
    'enero': 1, 'febrero': 2, 'marzo': 3, 'abril': 4, 'mayo': 5, 'junio': 6, 'julio': 7,
    'agosto': 8, 'septiembre': 9, 'setiembre': 9, 'octubre': 10, 'noviembre': 11, 'diciembre': 12,
    'janvier': 1, 'février': 2, 'fevrier': 2, 'mars': 3, 'avril': 4, 'mai': 5, 'juin': 6,
    'juillet': 7, 'août': 8, 'aout': 8, 'septembre': 9, 'octobre': 10, 'novembre': 11,
    'décembre': 12, 'decembre': 12,
}
NUM = {'a': 1, 'an': 1, 'one': 1, 'un': 1, 'uno': 1, 'una': 1, 'une': 1,
       'two': 2, 'dos': 2, 'deux': 2, 'three': 3, 'tres': 3, 'trois': 3,
       'four': 4, 'cuatro': 4, 'quatre': 4, 'five': 5, 'cinco': 5, 'cinq': 5,
       'ten': 10, 'diez': 10, 'dix': 10}
UNIT = {  # token -> relativedelta kwarg
    'day': 'days', 'days': 'days', 'día': 'days', 'dia': 'days', 'días': 'days', 'dias': 'days',
    'jour': 'days', 'jours': 'days',
    'week': 'weeks', 'weeks': 'weeks', 'semana': 'weeks', 'semanas': 'weeks',
    'semaine': 'weeks', 'semaines': 'weeks',
    'month': 'months', 'months': 'months', 'mes': 'months', 'meses': 'months', 'mois': 'months',
    'year': 'years', 'years': 'years', 'año': 'years', 'ano': 'years', 'años': 'years',
    'anos': 'years', 'an': 'years', 'ans': 'years', 'année': 'years', 'annee': 'years',
}
BACK = ['ago', 'before', 'hace', 'antes', 'il y a', 'avant']
FWD = ['from today', 'from now', 'later', 'after', 'dentro', 'dans', 'plus tard', 'después',
       'despues', 'après', 'apres']
TODAY_W = ['today', 'hoy', "aujourd'hui", 'aujourd']
TMRW_W = ['tomorrow', 'mañana', 'manana', 'demain']
YDAY_W = ['yesterday', 'ayer', 'hier']
QSPLIT = re.compile(r'(what is the date|cuál es la fecha|cual es la fecha|quelle est la date)',
                    re.IGNORECASE)


def parse_dates(text):
    out = []
    for c in re.findall(r'\d{1,2}/\d{1,2}/\d{2,4}', text):
        p = c.split('/')
        try:
            y = int(p[2]); y += 1900 if y < 100 else 0
            out.append(date(y, int(p[0]), int(p[1])))
        except ValueError:
            pass
    low = text.lower()
    for m in re.finditer(r'(\d{1,2})\s+de\s+(\w+)\s+de\s+(\d{4})', low):      # es: 6 de mayo de 1992
        if m.group(2) in MONTHS:
            out.append(date(int(m.group(3)), MONTHS[m.group(2)], int(m.group(1))))
    for m in re.finditer(r'(\d{1,2})\s+(\w+)\s+(\d{4})', low):                # fr: 6 mai 1992
        if m.group(2) in MONTHS:
            out.append(date(int(m.group(3)), MONTHS[m.group(2)], int(m.group(1))))
    for m in re.finditer(r'([a-zéûôà]+)\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})', low):  # en: May 6, 1992
        if m.group(1) in MONTHS:
            out.append(date(int(m.group(3)), MONTHS[m.group(1)], int(m.group(2))))
    return out


def framing_disp(stem):
    s = stem.lower()
    if any(w in s for w in TMRW_W):
        return -1
    if any(w in s for w in YDAY_W):
        return +1
    return 0


def parse_offset(q):
    s = q.lower()
    if any(w in s for w in TMRW_W):
        return rd(days=1)
    if any(w in s for w in YDAY_W):
        return rd(days=-1)
    if re.search(r'24\s+(hours|horas|heures)', s):
        return rd(days=1)
    # generic: count + unit + direction
    sign = -1 if any(w in s for w in BACK) else (1 if any(w in s for w in FWD) else None)
    unit = next((UNIT[t] for t in re.findall(r"[a-zéûôà]+", s) if t in UNIT), None)
    cnt = next((NUM[t] for t in re.findall(r"[a-z]+", s) if t in NUM), None)
    cd = re.search(r'\b(\d{1,2})\b', s)
    if cd:
        cnt = int(cd.group(1))
    if unit is None or sign is None:
        if any(w in s for w in TODAY_W):
            return rd(days=0)
        return None
    return rd(**{unit: sign * (cnt or 1)})


def parse_options(text):
    sec = text.split("Options:")[-1]
    return [(l, v.strip()) for l, v in re.findall(r'\(([A-Z])\)\s+([^\n(]+)', sec)]


def opt_date(s):
    m = re.match(r'(\d{1,2})/(\d{1,2})/(\d{4})', s.strip())
    return date(int(m.group(3)), int(m.group(1)), int(m.group(2))) if m else None


def solve(full):
    pre = full.split("Options:")[0]
    parts = QSPLIT.split(pre)
    stem = parts[0] if len(parts) > 1 else pre
    q = parts[-1] if len(parts) > 1 else pre
    dates = parse_dates(stem)
    off = parse_offset(q)
    if not dates or off is None:
        return None
    today = dates[0] + timedelta(days=framing_disp(stem))
    target = today + off
    for letter, dv in parse_options(full):
        if opt_date(dv) == target:
            return f"({letter})"
    return None


def main():
    exs = load_task("date_understanding")[:150]
    accs = {"en": [0, 0], "es": [0, 0], "fr": [0, 0]}
    print(f"{'idx':>4s} {'lang':>4s}  {'pred':>5s} {'tgt':>5s}  ok")
    for idx, (es, fr) in TR.items():
        ex = exs[idx]; tgt = ex["target"].strip()
        _, _, post = ex["input"].partition("Options:")
        full = {"en": ex["input"], "es": es + "\nOptions:" + post, "fr": fr + "\nOptions:" + post}
        for lg in ("en", "es", "fr"):
            pred = solve(full[lg])
            ok = int(pred == tgt)
            accs[lg][0] += ok; accs[lg][1] += 1
            if lg == "en":
                print(f"{idx:>4d} {lg:>4s}  {str(pred):>5s} {tgt:>5s}  {'ok' if ok else 'XX'}"
                      + ("" if pred else "  (abstain)"))
    print("\n=== cross-lingual symbolic vs option-probe (0.72) ===")
    for lg in ("en", "es", "fr"):
        ok, n = accs[lg]
        print(f"  {lg}: {ok}/{n} = {ok/n:.2f}")


if __name__ == "__main__":
    main()
