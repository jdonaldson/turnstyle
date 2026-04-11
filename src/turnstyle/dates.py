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
        "How many days between March 20 and June 15?",
        "How many days from January 1 to March 15?",
        "How many weeks between April 1 and July 1?",
        "How many days until Christmas?",
        "How many days from 2026-01-01 to 2026-12-31?",
        "How many days between February 14 and May 5?",
        "Count the days between October 31 and December 25",
        "How far apart are June 21 and September 22?",
        "Days remaining until New Year?",
        "How long from March 1 to August 31 in days?",
        "How many days between July 4 and November 11?",
        "How many weeks from May 1 to July 15?",
        "How many days from September 1 to December 1?",
        "How many days between Easter and Christmas?",
        "How many days until Valentine's Day?",
        "How many days between January 20 and March 20?",
        "How many days from April 15 to June 30?",
        "How many weeks between August 1 and October 15?",
        "How many days between the first and last day of February?",
        "How many days between November 1 and December 31?",
        "How many days from 2026-03-01 to 2026-06-01?",
        "How many days between January 1 and July 4?",
        "How far in days from March 15 to June 21?",
        "How many days until Halloween?",
        "How many days between May 25 and September 5?",
        "How many days from December 1 to February 28?",
        "How many weeks between January 1 and April 1?",
        "How many days between the summer solstice and autumn equinox?",
        "How many days from August 15 to December 25?",
        "How many days between February 1 and April 30?",
    ]

    def parse(self, prompt: str):
        return None  # routing via probe, fields via extraction

    def make_processor(self, parsed, max_new_tokens: int):
        expr, answer, unit = parsed
        answer_digits = [int(d) for d in str(abs(answer))]
        return ArithmeticLogitsProcessor(
            self.tokenizer, answer_digits, expr, answer,
            self.bias_strength, max_new_tokens=max_new_tokens)
