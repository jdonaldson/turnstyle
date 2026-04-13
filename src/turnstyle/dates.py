"""Date/time turnstyle — grounds date arithmetic in datetime computation.

Handles:
    "How many days between March 20 and June 15?"
    "How many days from 2026-01-01 to 2026-12-31?"
    "How many weeks between April 1 and July 1?"
    "How many days until Christmas?"
"""

from __future__ import annotations

import re
from datetime import date, datetime, timedelta

from turnstyle.arithmetic import ArithmeticLogitsProcessor
from turnstyle.core import Turnstyle
from turnstyle.extract import ExtractionSpec, FieldSpec

# ── date parsing ─────────────────────────────────────────────────────

MONTHS = {
    'january': 1, 'february': 2, 'march': 3, 'april': 4,
    'may': 5, 'june': 6, 'july': 7, 'august': 8,
    'september': 9, 'october': 10, 'november': 11, 'december': 12,
    'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4,
    'jun': 6, 'jul': 7, 'aug': 8, 'sep': 9, 'sept': 9,
    'oct': 10, 'nov': 11, 'dec': 12,
}

# TODO(no-keyword): HOLIDAYS is a hardcoded knowledge dict — cultural knowledge
# that should come from the model, not be enumerated here. Replace with LLM
# extraction: model resolves holiday names to (month, day); deterministic date
# arithmetic runs downstream.
HOLIDAYS = {
    'christmas': (12, 25), 'christmas day': (12, 25),
    'new year': (1, 1), "new year's": (1, 1), "new year's day": (1, 1),
    'valentine': (2, 14), "valentine's": (2, 14), "valentine's day": (2, 14),
    'halloween': (10, 31),
    'independence day': (7, 4), 'july 4th': (7, 4), '4th of july': (7, 4),
}


def _parse_date(text: str, reference_year: int | None = None) -> date | None:
    """Try to parse a date from text. Returns None on failure."""
    text = text.strip().strip('.,?!')

    # Check holidays
    lower = text.lower()
    for name, (m, d) in HOLIDAYS.items():
        if name in lower:
            year = reference_year or date.today().year
            return date(year, m, d)

    # ISO: 2026-03-20
    m = re.match(r'(\d{4})-(\d{1,2})-(\d{1,2})', text)
    if m:
        return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))

    # US numeric: 3/20/2026
    m = re.match(r'(\d{1,2})/(\d{1,2})/(\d{4})', text)
    if m:
        return date(int(m.group(3)), int(m.group(1)), int(m.group(2)))

    # "March 20, 2026" or "March 20 2026"
    m = re.match(r'([A-Za-z]+)\s+(\d{1,2}),?\s*(\d{4})', text)
    if m:
        month = MONTHS.get(m.group(1).lower())
        if month:
            return date(int(m.group(3)), month, int(m.group(2)))

    # "20 March 2026"
    m = re.match(r'(\d{1,2})\s+([A-Za-z]+)\s+(\d{4})', text)
    if m:
        month = MONTHS.get(m.group(2).lower())
        if month:
            return date(int(m.group(3)), month, int(m.group(1)))

    # "March 20" (no year — use reference year or current)
    m = re.match(r'([A-Za-z]+)\s+(\d{1,2})', text)
    if m:
        month = MONTHS.get(m.group(1).lower())
        if month:
            year = reference_year or date.today().year
            return date(year, month, int(m.group(2)))

    return None


# ── expression parsing ───────────────────────────────────────────────

def parse_date_arithmetic(text: str):
    """Extract date arithmetic from text. Returns (expression, answer_int, unit) or None.

    Supported patterns:
        "How many days between {date} and {date}?"
        "How many days from {date} to {date}?"
        "How many weeks between {date} and {date}?"
        "How many days until {date}?"
    """
    lower = text.lower()

    # "How many {unit} between/from {date} and/to {date}"
    m = re.search(
        r'how many\s+(days?|weeks?|months?)\s+(?:between|from)\s+(.+?)\s+(?:and|to)\s+(.+?)[\?\.]?\s*$',
        lower)
    if m:
        unit = m.group(1).rstrip('s') or 'day'
        d1 = _parse_date(m.group(2))
        d2 = _parse_date(m.group(3))
        if d1 and d2:
            delta = abs((d2 - d1).days)
            if unit == 'week':
                answer = delta // 7
            else:
                answer = delta
            expr = f"days({d1.isoformat()},{d2.isoformat()})"
            if unit == 'week':
                expr = f"weeks({d1.isoformat()},{d2.isoformat()})"
            return expr, answer, unit

    # "How many days until {date}"
    m = re.search(r'how many\s+(days?|weeks?)\s+until\s+(.+?)[\?\.]?\s*$', lower)
    if m:
        unit = m.group(1).rstrip('s') or 'day'
        d = _parse_date(m.group(2))
        if d:
            today = date.today()
            delta = (d - today).days
            if delta < 0:
                # Next year
                d = d.replace(year=d.year + 1)
                delta = (d - today).days
            if unit == 'week':
                answer = delta // 7
            else:
                answer = delta
            expr = f"until({d.isoformat()})"
            return expr, answer, unit

    return None


_DATE_UNITS = ["days", "weeks"]


def _assemble_date(fields: dict) -> tuple[str, int, str]:
    """Assemble date extraction fields into parse() tuple format."""
    date1_str = fields["date1"].strip().strip('.,?!')
    date2_str = fields["date2"].strip().strip('.,?!')
    unit = fields["unit"].rstrip("s") or "day"  # normalize: "days" → "day"

    d1 = _parse_date(date1_str)
    d2 = _parse_date(date2_str)
    if d1 is None or d2 is None:
        raise ValueError(f"Could not parse dates: {date1_str!r}, {date2_str!r}")

    delta = abs((d2 - d1).days)
    if unit == "week":
        answer = delta // 7
        expr = f"weeks({d1.isoformat()},{d2.isoformat()})"
    else:
        answer = delta
        expr = f"days({d1.isoformat()},{d2.isoformat()})"
    return expr, answer, unit


DATE_EXTRACTION_SPEC = ExtractionSpec(
    fields=[
        FieldSpec(
            name="date1",
            prompt_template=(
                "What is the first/start date mentioned in this text? "
                "Return just the date.\nText: {input}\nDate:"
            ),
        ),
        FieldSpec(
            name="date2",
            prompt_template=(
                "What is the second/end date mentioned in this text? "
                "Return just the date.\nText: {input}\nDate:"
            ),
        ),
        FieldSpec(
            name="unit",
            prompt_template=(
                "What unit of time is being asked about?\n"
                "Text: {input}\nUnit:"
            ),
            options=_DATE_UNITS,
        ),
    ],
    assemble=_assemble_date,
)


class DateTurnstyle(Turnstyle):
    """Grounds date arithmetic in datetime computation.

        t = DateTurnstyle(model, tokenizer, device)
        text, proof = t.generate("How many days between March 20 and June 15?")
    """

    probe_label = "date"
    extraction_spec = DATE_EXTRACTION_SPEC
    examples = [
        'Today is Christmas Eve of 1937. What is the date tomorrow in MM/DD/YYYY?\nOptions:\n(A) 12/11/1937\n(B) 12/25/1937\n(C) 01/04/1938\n(D) 12/04/1937\n(E) 12/25/2006\n(F) 07/25/1937',
        'In the UK, people usually put the day before the month when formatting the date. Therefore, today is 02/01/1987 to them. What is the date a month ago in MM/DD/YYYY?\nOptions:\n(A) 12/02/1986\n(B) 12/01/1986\n(C) 03/02/1986\n(D) 12/02/2032\n(E) 12/02/2062\n(F) 02/06/1987',
        'Jane and John married on Jan 2, 1958. It is their 5-year anniversary today. What is the date one week from today in MM/DD/YYYY?\nOptions:\n(A) 01/16/1961\n(B) 01/09/1961\n(C) 01/09/1963\n(D) 03/09/1960\n(E) 07/09/1960\n(F) 02/28/1961',
        'Jane scheduled 3 apointments with 5 poeple for tomorrow (Tue, 7/9/1972). What is the date one week ago from today in MM/DD/YYYY?\nOptions:\n(A) 07/09/1972\n(B) 12/01/1972\n(C) 07/01/2008\n(D) 05/13/1972\n(E) 07/01/1972\n(F) 08/12/1972',
        "Today's meeting is rescheduled to 11 am tomorrow, 10/16/1924. What is the date one week from today in MM/DD/YYYY?\nOptions:\n(A) 10/22/1944\n(B) 10/22/1924\n(C) 10/07/1924\n(D) 10/21/1924\n(E) 08/22/1925\n(F) 11/21/1924",
        'Jane visits the bookstore on the 16th of each month starting from the October of 2009. It is her 5th visit to the bookstore today. What is the date one year ago from today in MM/DD/YYYY?\nOptions:\n(A) 08/16/2009\n(B) 02/16/2009\n(C) 02/16/2004\n(D) 03/13/2009\n(E) 01/26/2009\n(F) 02/15/2009',
        'On May 9th, 2017 Jane bought 40 eggs. She ate one per day. Today she ran out of eggs. What is the date 24 hours later in MM/DD/YYYY?\nOptions:\n(A) 06/19/2017\n(B) 07/17/2017\n(C) 06/20/2017\n(D) 06/18/2017\n(E) 06/15/2017\n(F) 07/10/2017',
        'Tomorrow is 11/12/2019. What is the date yesterday in MM/DD/YYYY?\nOptions:\n(A) 11/10/2076\n(B) 11/10/2019\n(C) 09/10/2019\n(D) 11/06/2019\n(E) 11/11/2019\n(F) 11/17/2019',
        'Jane thinks today is 6/18/2019, but John thinks today is 6/19/2019. Jane is correct. What is the date yesterday in MM/DD/YYYY?\nOptions:\n(A) 06/17/2063\n(B) 05/18/2019\n(C) 05/20/2019\n(D) 06/17/2019\n(E) 05/13/2019\n(F) 06/08/2019',
        'The first day of 2019 is a Tuesday, and today is the first Monday of 2019. What is the date one year ago from today in MM/DD/YYYY?\nOptions:\n(A) 01/08/2018\n(B) 01/07/2018\n(C) 01/02/2018\n(D) 01/12/2018\n(E) 05/07/2017\n(F) 10/29/2017',
        'Yesterday was 12/31/1929. Today could not be 12/32/1929 because December has only 31 days. What is the date today in MM/DD/YYYY?\nOptions:\n(A) 01/08/1930\n(B) 01/06/1930\n(C) 01/01/1930\n(D) 12/31/1929\n(E) 10/01/1930\n(F) 02/27/1930',
        "Today is Apr 10, 1985. Jane's appointment will be 3 days later. What is the date one week ago from today in MM/DD/YYYY?\nOptions:\n(A) 05/13/1985\n(B) 04/03/1985\n(C) 03/10/1986\n(D) 04/01/1985\n(E) 04/13/1985\n(F) 04/12/1986",
        'Jane quited her job on Mar 20, 2020. 176 days have passed since then. What is the date one year ago from today in MM/DD/YYYY?\nOptions:\n(A) 09/20/2019\n(B) 10/01/2019\n(C) 10/24/2019\n(D) 09/12/1970\n(E) 09/12/2019',
        'Today is the palindrome day of 2020, because the MMDDYYYY format of the date is the same backwards as forwards. What is the date a month ago in MM/DD/YYYY?\nOptions:\n(A) 01/01/2020\n(B) 01/02/2020\n(C) 10/02/2019\n(D) 11/05/2019\n(E) 01/02/1931\n(F) 01/03/2020',
        'Tomorrow is 11/12/2019. What is the date one week from today in MM/DD/YYYY?\nOptions:\n(A) 08/21/2019\n(B) 11/18/2019\n(C) 11/18/2025\n(D) 11/18/2011\n(E) 02/18/2020\n(F) 11/04/2019',
        'Jane scheduled 3 apointments with 5 poeple for tomorrow (Tue, 7/9/1972). What is the date tomorrow in MM/DD/YYYY?\nOptions:\n(A) 07/09/1923\n(B) 08/05/1972\n(C) 09/01/1972\n(D) 07/09/1972\n(E) 07/09/2007\n(F) 09/04/1972',
        'May 6, 1992 is like yesterday to Jane, but that is actually ten years ago. What is the date today in MM/DD/YYYY?\nOptions:\n(A) 05/06/2095\n(B) 05/06/1935\n(C) 05/06/1912\n(D) 08/06/2002\n(E) 05/06/2002\n(F) 05/06/1964',
        'Jane quited her job on Mar 20, 2020. 176 days have passed since then. What is the date tomorrow in MM/DD/YYYY?\nOptions:\n(A) 09/13/2020\n(B) 08/14/2020\n(C) 09/13/2057\n(D) 07/26/2020\n(E) 10/04/2020\n(F) 09/13/2084',
        'Today is Sep 9, 1909. What is the date 10 days ago in MM/DD/YYYY?\nOptions:\n(A) 08/29/1909\n(B) 08/30/1967\n(C) 08/30/1909\n(D) 08/30/1977\n(E) 08/02/1909\n(F) 08/16/1909',
        'The current local time is 3:02 pm of 5/4/2004. What is the date one year ago from today in MM/DD/YYYY?\nOptions:\n(A) 06/14/2003\n(B) 05/03/2003\n(C) 12/04/2002\n(D) 06/02/2003\n(E) 05/04/2003',
        'Yesterday, Jan 21, 2011, Jane ate 2 pizzas and 5 wings. What is the date one week ago from today in MM/DD/YYYY?\nOptions:\n(A) 12/06/2010\n(B) 01/15/2011\n(C) 01/17/2011\n(D) 03/15/2011\n(E) 12/24/2010',
        'Yesterday, Jan 21, 2011, Jane ate 2 pizzas and 5 wings. What is the date 10 days ago in MM/DD/YYYY?\nOptions:\n(A) 01/18/2011\n(B) 01/12/2011\n(C) 01/12/2069\n(D) 01/13/2011\n(E) 05/12/2010\n(F) 08/12/2010',
        'Jane is celebrating the last day of Jan 2012. What is the date yesterday in MM/DD/YYYY?\nOptions:\n(A) 01/29/2012\n(B) 09/30/2011\n(C) 02/06/2012\n(D) 01/30/2012\n(E) 01/30/1925',
        '2015 is coming in 36 hours. What is the date one week ago from today in MM/DD/YYYY?\nOptions:\n(A) 12/22/2040\n(B) 12/23/2014\n(C) 12/22/2014\n(D) 01/05/2015\n(E) 12/22/2086\n(F) 02/06/2015',
        'In the UK, people usually put the day before the month when formatting the date. Therefore, today is 02/01/1987 to them. What is the date yesterday in MM/DD/YYYY?\nOptions:\n(A) 01/01/1987\n(B) 10/19/1986\n(C) 01/08/1987\n(D) 01/01/2015\n(E) 01/01/1934\n(F) 12/29/1986',
        'Jane booked a flight for tomorrow, Jul 29, 2002. What is the date one week from today in MM/DD/YYYY?\nOptions:\n(A) 08/03/2002\n(B) 10/22/2002\n(C) 06/03/2002\n(D) 08/04/2002\n(E) 08/04/1930\n(F) 07/04/2003',
        'The deadline is Jun 1, 2021, which is 2 days away from now. What is the date one week ago from today in MM/DD/YYYY?\nOptions:\n(A) 05/23/2020\n(B) 04/13/2021\n(C) 05/24/2021\n(D) 05/23/2054\n(E) 05/23/1940\n(F) 05/23/2021',
        'Jane quited her job on Mar 20, 2020. 176 days have passed since then. What is the date one week ago from today in MM/DD/YYYY?\nOptions:\n(A) 02/05/2021\n(B) 09/05/2020\n(C) 08/15/2020\n(D) 09/06/2020\n(E) 09/04/2020',
        'Yesterday was April 30, 2021. What is the date one week ago from today in MM/DD/YYYY?\nOptions:\n(A) 05/15/2021\n(B) 04/23/2021\n(C) 04/15/2021\n(D) 06/05/2021\n(E) 04/19/2021\n(F) 04/24/2021',
        'Jane visits the bookstore on the 16th of each month starting from the October of 2009. It is her 5th visit to the bookstore today. What is the date today in MM/DD/YYYY?\nOptions:\n(A) 02/06/2010\n(B) 12/21/2009\n(C) 02/16/1933\n(D) 02/16/2010\n(E) 11/30/2009\n(F) 05/24/2010',
    ]

    def parse(self, prompt: str):
        return None  # routing via probe, fields via extraction

    def make_processor(self, parsed, max_new_tokens: int):
        expr, answer, unit = parsed
        answer_digits = [int(d) for d in str(abs(answer))]
        return ArithmeticLogitsProcessor(
            self.tokenizer, answer_digits, expr, answer,
            self.bias_strength, max_new_tokens=max_new_tokens)
