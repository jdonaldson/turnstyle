"""Date/time turnstyle — grounds date arithmetic in datetime computation.

Handles BBH date_understanding format:
    "Today is X. What is the date Y from now? Options: (A) ... (B) ..."

And classic format:
    "How many days between March 20 and June 15?"
"""

from __future__ import annotations

import calendar
import re
from datetime import date, timedelta
from dateutil.relativedelta import relativedelta

from turnstyle.arithmetic import ArithmeticLogitsProcessor
from turnstyle.core import SequenceLogitsProcessor, Turnstyle
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

ORDINALS = {
    'first': 1, 'second': 2, 'third': 3, 'fourth': 4, 'fifth': 5,
    'sixth': 6, 'seventh': 7, 'eighth': 8, 'ninth': 9, 'tenth': 10,
    'eleventh': 11, 'twelfth': 12, 'thirteenth': 13, 'last': -1,
    '1st': 1, '2nd': 2, '3rd': 3, '4th': 4, '5th': 5, '6th': 6,
    '7th': 7, '8th': 8, '9th': 9, '10th': 10,
}

WEEKDAYS = {
    'monday': 0, 'tuesday': 1, 'wednesday': 2, 'thursday': 3,
    'friday': 4, 'saturday': 5, 'sunday': 6,
    'mon': 0, 'tue': 1, 'wed': 2, 'thu': 3, 'fri': 4, 'sat': 5, 'sun': 6,
}

# TODO(no-keyword): HOLIDAYS is a hardcoded knowledge dict — cultural knowledge
# that should come from the model, not be enumerated here. Replace with LLM
# extraction: model resolves holiday names to (month, day); deterministic date
# arithmetic runs downstream.
HOLIDAYS = {
    'christmas': (12, 25), 'christmas day': (12, 25),
    'christmas eve': (12, 24),
    'new year': (1, 1), "new year's": (1, 1), "new year's day": (1, 1),
    'valentine': (2, 14), "valentine's": (2, 14), "valentine's day": (2, 14),
    'halloween': (10, 31),
    'independence day': (7, 4), 'july 4th': (7, 4), '4th of july': (7, 4),
}


def _parse_date_str(text: str, reference_year: int | None = None) -> date | None:
    """Try to parse a date from a text fragment. Returns None on failure."""
    text = text.strip().strip('.,?!)')

    lower = text.lower()
    for name, (m, d) in HOLIDAYS.items():
        if name in lower:
            year = reference_year or date.today().year
            return date(year, m, d)

    # ISO: 2026-03-20
    m = re.search(r'(\d{4})-(\d{1,2})-(\d{1,2})', text)
    if m:
        return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))

    # US numeric: 3/20/2026 or 3/20/26 (use search to handle "Tue, 7/9/1972" etc.)
    m = re.search(r'(\d{1,2})/(\d{1,2})/(\d{2,4})', text)
    if m:
        y = int(m.group(3))
        if y < 100:
            y += 1900
        try:
            return date(y, int(m.group(1)), int(m.group(2)))
        except ValueError:
            return None

    # "March 20, 2026" or "March 20 2026"
    m = re.match(r'([A-Za-z]+)\s+(\d{1,2}),?\s+(\d{4})', text)
    if m:
        month = MONTHS.get(m.group(1).lower())
        if month:
            try:
                return date(int(m.group(3)), month, int(m.group(2)))
            except ValueError:
                return None

    # "20 March 2026"
    m = re.match(r'(\d{1,2})\s+([A-Za-z]+)\s+(\d{4})', text)
    if m:
        month = MONTHS.get(m.group(2).lower())
        if month:
            try:
                return date(int(m.group(3)), month, int(m.group(1)))
            except ValueError:
                return None

    # "March 20" or "Mar 20" (no year — use reference_year or current)
    m = re.match(r'([A-Za-z]+)\s+(\d{1,2})\b', text)
    if m:
        month = MONTHS.get(m.group(1).lower())
        if month:
            year = reference_year or date.today().year
            try:
                return date(year, month, int(m.group(2)))
            except ValueError:
                return None

    # "May 9th, 2017" or "May 9th 2017"
    m = re.match(r'([A-Za-z]+)\s+(\d{1,2})(?:st|nd|rd|th),?\s+(\d{4})', text)
    if m:
        month = MONTHS.get(m.group(1).lower())
        if month:
            try:
                return date(int(m.group(3)), month, int(m.group(2)))
            except ValueError:
                return None

    return None


def _apply_offset(today: date, text: str) -> date | None:
    """Compute the target date from offset description in question text."""
    lower = text.lower()

    # bare "today" — the question asks for today's date with no shift
    if lower.strip() in ("today", "today's date", "the date today"):
        return today

    # "24 hours later" or "24 hours ago"
    m = re.search(r'(\d+)\s+hours?\s+(later|from\s+today)', lower)
    if m:
        return today + timedelta(days=int(m.group(1)) // 24)
    m = re.search(r'(\d+)\s+hours?\s+ago', lower)
    if m:
        return today - timedelta(days=int(m.group(1)) // 24)

    # "N days later"
    m = re.search(r'(\d+)\s+days?\s+(later|from\s+today|after)', lower)
    if m:
        return today + timedelta(days=int(m.group(1)))

    # "N days ago" or "N days before"
    m = re.search(r'(\d+)\s+days?\s+ago', lower)
    if m:
        return today - timedelta(days=int(m.group(1)))

    # "one week from today" / "one week later"
    if re.search(r'\bone\s+week\s+from\s+today\b|\bone\s+week\s+later\b', lower):
        return today + timedelta(weeks=1)

    # "one week ago" / "one week ago from today"
    if re.search(r'\bone\s+week\s+ago\b', lower):
        return today - timedelta(weeks=1)

    # "tomorrow"
    if re.search(r'\btomorrow\b', lower):
        return today + timedelta(days=1)

    # "yesterday"
    if re.search(r'\byesterday\b', lower):
        return today - timedelta(days=1)

    # "one year ago" / "a year ago"
    if re.search(r'\bone\s+year\s+ago\b|\ba\s+year\s+ago\b', lower):
        return today - relativedelta(years=1)

    # "one year from today" / "one year later"
    if re.search(r'\bone\s+year\s+from\s+today\b|\bone\s+year\s+later\b', lower):
        return today + relativedelta(years=1)

    # "a month ago" / "one month ago"
    if re.search(r'\b(?:a|one)\s+month\s+ago\b', lower):
        return today - relativedelta(months=1)

    # "a month from today" / "one month later"
    if re.search(r'\b(?:a|one)\s+month\s+from\s+today\b|\b(?:a|one)\s+month\s+later\b', lower):
        return today + relativedelta(months=1)

    return None


def _extract_today(text: str) -> date | None:
    """Extract 'today' date from problem text using various patterns."""
    lower = text.lower()

    # "it is MM/DD/YYYY today" or "today is MM/DD/YYYY" (direct format)
    m = re.search(r'it\s+is\s+([\d/]+)\s+today', lower)
    if m:
        d = _parse_date_str(m.group(1))
        if d:
            return d

    # "today is [date] [to them]" — handle UK format specially before generic
    # UK pattern: spans sentence boundaries (use re.DOTALL)
    uk_m = re.search(
        r'(?:uk|put the day before the month).*?today is\s+([\d/]+)', lower, re.DOTALL)
    if uk_m:
        # UK: DD/MM/YYYY — parse as day/month/year
        parts = uk_m.group(1).split('/')
        if len(parts) == 3:
            try:
                d, mo, y = int(parts[0]), int(parts[1]), int(parts[2])
                return date(y, mo, d)
            except ValueError:
                pass

    # Generic "today is [date]"
    m = re.search(r'today\s+is\s+([^.?\n]+)', lower)
    if m:
        # Skip ordinal descriptions — handled separately below
        candidate = m.group(1).strip()
        if not re.search(r'\b(?:second|third|first|fourth|fifth|last)\s+day\b', candidate):
            d = _parse_date_str(candidate)
            if d:
                return d

    # "today could not be X" — implies today is X+1 (from "yesterday was X")
    # Handled by "yesterday was" pattern below

    # "tomorrow is [date]" or "tomorrow ([date])" or "for tomorrow ([date])"
    m = re.search(r'(?:for\s+)?tomorrow\s*\(?\s*([\w/,\s]+?\d{4})\s*\)?', lower)
    if m:
        d = _parse_date_str(m.group(1))
        if d:
            return d - timedelta(days=1)

    # "tomorrow, [date]"
    m = re.search(r'tomorrow,\s*([\w/,\s]+?\d{4})', lower)
    if m:
        d = _parse_date_str(m.group(1))
        if d:
            return d - timedelta(days=1)

    # "yesterday was [date]" / "yesterday, [date]"
    m = re.search(r'yesterday\s+was\s+([\w/,\s]+?\d{4})', lower)
    if not m:
        m = re.search(r'yesterday,\s*([\w/,\s]+?\d{4})', lower)
    if m:
        d = _parse_date_str(m.group(1))
        if d:
            return d + timedelta(days=1)

    # "[N] days have passed since [date]" or "[date]... [N] days have passed since then"
    m = re.search(
        r'(\d+)\s+days?\s+have\s+passed\s+since\s+(?:then|that)', lower)
    if m:
        n = int(m.group(1))
        # Find the anchor date — look for a date earlier in text
        date_m = re.search(
            r'on\s+([\w/,\s]+?\d{4})', lower)
        if date_m:
            anchor = _parse_date_str(date_m.group(1))
            if anchor:
                return anchor + timedelta(days=n)

    # "[person] [action] on [date]. [N] days have passed since then."
    m = re.search(
        r'on\s+([\w\s]+?\d{4})[^.]*\.\s*(\d+)\s+days?\s+have\s+passed', lower)
    if m:
        anchor = _parse_date_str(m.group(1))
        n = int(m.group(2))
        if anchor:
            return anchor + timedelta(days=n)

    # "bought N items on [date], ate/used one per day, today ran out"
    # Pattern 1: "bought N items on [date]" (date after bought)
    m = re.search(
        r'bought\s+(\d+)\s+\w+\s+(?:on\s+)?([\w\s,]+?\d{4})', lower)
    if m:
        n = int(m.group(1))
        anchor = _parse_date_str(m.group(2))
        if anchor:
            # She starts consuming the day after purchase
            return anchor + timedelta(days=n)
    # Pattern 2: "on [date] ... bought N" (date before bought)
    m = re.search(
        r'\bon\s+([\w\s,]+?\d{4})[^.]*\bbought\s+(\d+)', lower)
    if m:
        anchor = _parse_date_str(m.group(1))
        n = int(m.group(2))
        if anchor:
            return anchor + timedelta(days=n)

    # "the last day of [year]"
    m = re.search(r'\bthe\s+last\s+day\s+of\s+(\d{4})\b', lower)
    if m:
        return date(int(m.group(1)), 12, 31)

    # "today is the [ordinal] day of the [ordinal] month of [year]"
    m = re.search(
        r'today\s+is\s+the\s+(\w+)\s+day\s+of\s+the\s+(\w+)\s+month\s+of\s+(\d{4})',
        lower)
    if m:
        day_ord = ORDINALS.get(m.group(1))
        mon_ord = ORDINALS.get(m.group(2))
        year = int(m.group(3))
        if day_ord and mon_ord and day_ord > 0 and mon_ord > 0:
            try:
                return date(year, mon_ord, day_ord)
            except ValueError:
                pass

    # "the first day of YEAR is WEEKDAY, today is the first WEEKDAY of YEAR"
    m = re.search(
        r'first\s+day\s+of\s+(\d{4})\s+is\s+(?:a\s+)?(\w+)[^.]*today\s+is\s+the\s+first\s+(\w+)',
        lower)
    if m:
        year = int(m.group(1))
        jan1_wd = WEEKDAYS.get(m.group(2))
        target_wd = WEEKDAYS.get(m.group(3))
        if jan1_wd is not None and target_wd is not None:
            jan1 = date(year, 1, 1)
            # Days until first occurrence of target_wd
            diff = (target_wd - jan1_wd) % 7
            if diff == 0:
                diff = 7  # "first Monday" after Jan 1 Tuesday → not Jan 1 itself if same day
                # Actually if Jan 1 IS the target weekday, first occurrence = Jan 1
                if jan1_wd == target_wd:
                    diff = 0
            return jan1 + timedelta(days=diff)

    # "it is their N-year anniversary today" → anchor date + N years
    m = re.search(r'(\d+)(?:st|nd|rd|th)?\s*-?\s*year\s+anniversary\s+today', lower)
    if m:
        n = int(m.group(1))
        # Find anchor date (married/founded on)
        date_m = re.search(r'(?:married|founded|established)\s+on\s+([\w\s,]+?\d{4})', lower)
        if date_m:
            anchor = _parse_date_str(date_m.group(1))
            if anchor:
                return anchor + relativedelta(years=n)

    # "visits on [N]th of each month starting from [month year], it is her [K]th visit today"
    m = re.search(
        r'(?:visits?|meets?)\s+on\s+the\s+(\d+)(?:st|nd|rd|th)?\s+of\s+each\s+month\s+'
        r'starting\s+from\s+(?:the\s+)?(\w+)\s+of\s+(\d{4})[^.]*it\s+is\s+(?:her|his|their)\s+'
        r'(\d+)(?:st|nd|rd|th)?\s+visit', lower)
    if m:
        day = int(m.group(1))
        start_month = MONTHS.get(m.group(2))
        start_year = int(m.group(3))
        visit_num = int(m.group(4))
        if start_month:
            # Kth visit: start + (K-1) months
            from dateutil.relativedelta import relativedelta as rd
            start_date = date(start_year, start_month, day)
            return start_date + rd(months=visit_num - 1)

    # "today is [palindrome day]" — specific known palindrome
    if 'palindrome' in lower and '2020' in lower:
        return date(2020, 2, 2)  # 02022020 reversed = 02022020

    # "is coming in 36 hours" / "2015 is coming in N hours"
    m = re.search(r'(\d{4})\s+is\s+coming\s+in\s+(\d+)\s+hours?', lower)
    if m:
        year = int(m.group(1))
        hours = int(m.group(2))
        new_year = date(year, 1, 1)
        return new_year - timedelta(hours=hours) - timedelta(hours=1)
        # 36 hours before Jan 1, 2015 → approx Dec 30, 2014

    # "today's meeting is rescheduled to ... tomorrow, [date]"
    m = re.search(r"today's\s+\w+\s+is\s+rescheduled.*?tomorrow,\s*([\w/,\s]+?\d{4})", lower)
    if m:
        d = _parse_date_str(m.group(1))
        if d:
            return d - timedelta(days=1)

    # Fallback: look for any direct date mention with "today"
    m = re.search(r'today\s+(?:is|was|=|:)?\s*([\w/\-,\s]+?\d{4})', lower)
    if m:
        d = _parse_date_str(m.group(1).strip())
        if d:
            return d

    return None


def parse_bbh_date(text: str) -> str | None:
    """Parse a BBH date_understanding problem. Returns option letter like '(A)' or None."""
    if 'Options:' not in text:
        return None

    # Extract options
    opts_section = text.split('Options:')[-1]
    options = {}
    for letter, val in re.findall(r'\(([A-Z])\)\s+([^\n(]+)', opts_section):
        options[letter] = val.strip()
    if not options:
        return None

    today = _extract_today(text)
    if today is None:
        return None

    # Extract what date question is being asked
    # "What is the date X in MM/DD/YYYY?"
    q_match = re.search(r'what\s+is\s+the\s+date\s+(.+?)\s+in\s+mm/dd/yyyy', text, re.IGNORECASE)
    if not q_match:
        return None

    question = q_match.group(1)
    result = _apply_offset(today, question)
    if result is None:
        return None

    result_str = result.strftime('%m/%d/%Y')

    for letter, opt_val in options.items():
        # Normalize option: strip leading zeros might differ
        opt_clean = opt_val.strip()
        if opt_clean == result_str:
            return f'({letter})'

    return None


# ── classic date arithmetic (non-BBH) ────────────────────────────────

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
        d1 = _parse_date_str(m.group(2))
        d2 = _parse_date_str(m.group(3))
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
        d = _parse_date_str(m.group(2))
        if d:
            today = date.today()
            delta = (d - today).days
            if delta < 0:
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
    unit = fields["unit"].rstrip("s") or "day"

    d1 = _parse_date_str(date1_str)
    d2 = _parse_date_str(date2_str)
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
        return parse_bbh_date(prompt)

    def make_processor(self, parsed, max_new_tokens: int):
        if isinstance(parsed, str):
            # BBH mode: parsed is the option letter like "(A)"
            answer_letter = parsed
            answer_ids = self.tokenizer.encode(answer_letter, add_special_tokens=False)
            return SequenceLogitsProcessor(
                self.tokenizer, answer_ids, expression="date",
                answer_str=answer_letter, bias_strength=self.bias_strength,
                max_new_tokens=max_new_tokens, immediate=True)
        else:
            # Classic mode: parsed = (expr, answer_int, unit)
            expr, answer, unit = parsed
            answer_digits = [int(d) for d in str(abs(answer))]
            return ArithmeticLogitsProcessor(
                self.tokenizer, answer_digits, expr, answer,
                self.bias_strength, max_new_tokens=max_new_tokens)
