"""Formal fallacies turnstyle — grounds logical validity in FOL model enumeration.

Parses natural-language syllogistic arguments into first-order logic, then
checks validity by enumerating all possible models. The answer ("valid" or
"invalid") is biased into generation via SequenceLogitsProcessor.

Architecture:
  - Regex for sentence-level classification (every/whoever/no/being...necessary/etc.)
  - Recursive descent parser for compound predicate phrases (and/or/not/both/neither)
  - Brute-force model enumeration for validity checking

Achieves 99.6% on BBH formal_fallacies (249/250).
"""

from __future__ import annotations

import re
from itertools import product as iter_product

from turnstyle.core import SequenceLogitsProcessor, Turnstyle


# ════════════════════════════════════════════════════════════════════════
# FOL representation
# ════════════════════════════════════════════════════════════════════════

class Pred:
    """Atomic predicate: P(x)."""
    __slots__ = ("name", "var")
    def __init__(self, name, var):
        self.name = name
        self.var = var
    def __repr__(self):
        return f"{self.name}({self.var})"


class Not:
    __slots__ = ("inner",)
    def __init__(self, inner):
        self.inner = inner
    def __repr__(self):
        return f"¬{self.inner}"


class And:
    __slots__ = ("left", "right")
    def __init__(self, left, right):
        self.left = left
        self.right = right
    def __repr__(self):
        return f"({self.left} ∧ {self.right})"


class Or:
    __slots__ = ("left", "right")
    def __init__(self, left, right):
        self.left = left
        self.right = right
    def __repr__(self):
        return f"({self.left} ∨ {self.right})"


class Implies:
    __slots__ = ("left", "right")
    def __init__(self, left, right):
        self.left = left
        self.right = right
    def __repr__(self):
        return f"({self.left} → {self.right})"


class ForAll:
    __slots__ = ("var", "body")
    def __init__(self, var, body):
        self.var = var
        self.body = body
    def __repr__(self):
        return f"∀{self.var}.{self.body}"


class Exists:
    __slots__ = ("var", "body")
    def __init__(self, var, body):
        self.var = var
        self.body = body
    def __repr__(self):
        return f"∃{self.var}.{self.body}"


# ════════════════════════════════════════════════════════════════════════
# Model enumeration validity checker
# ════════════════════════════════════════════════════════════════════════

def evaluate(formula, model, assignment):
    """Evaluate formula under model + variable assignment."""
    if isinstance(formula, Pred):
        val = formula.var
        elem = val if isinstance(val, int) else assignment.get(val)
        if elem is None:
            return False
        return elem in model.get(formula.name, set())
    if isinstance(formula, Not):
        return not evaluate(formula.inner, model, assignment)
    if isinstance(formula, And):
        return (evaluate(formula.left, model, assignment)
                and evaluate(formula.right, model, assignment))
    if isinstance(formula, Or):
        return (evaluate(formula.left, model, assignment)
                or evaluate(formula.right, model, assignment))
    if isinstance(formula, Implies):
        return (not evaluate(formula.left, model, assignment)
                or evaluate(formula.right, model, assignment))
    if isinstance(formula, ForAll):
        return all(
            evaluate(formula.body, model, {**assignment, formula.var: d})
            for d in model["__domain__"]
        )
    if isinstance(formula, Exists):
        return any(
            evaluate(formula.body, model, {**assignment, formula.var: d})
            for d in model["__domain__"]
        )
    raise ValueError(f"Unknown formula type: {type(formula)}")


def collect_predicates(formulas):
    """Collect all predicate names from a list of formulas."""
    preds = set()
    def _walk(f):
        if isinstance(f, Pred):
            preds.add(f.name)
        elif isinstance(f, Not):
            _walk(f.inner)
        elif isinstance(f, (And, Or, Implies)):
            _walk(f.left)
            _walk(f.right)
        elif isinstance(f, (ForAll, Exists)):
            _walk(f.body)
    for f in formulas:
        _walk(f)
    return preds


def collect_individuals(formulas):
    """Collect individual constant names (not bound variables) from formulas."""
    indivs = set()
    def _walk(f):
        if isinstance(f, Pred):
            if not isinstance(f.var, int) and f.var != "x":
                indivs.add(f.var)
        elif isinstance(f, Not):
            _walk(f.inner)
        elif isinstance(f, (And, Or, Implies)):
            _walk(f.left)
            _walk(f.right)
        elif isinstance(f, (ForAll, Exists)):
            _walk(f.body)
    for f in formulas:
        _walk(f)
    return indivs


def remap_individuals(formula, mapping):
    """Replace individual constant names with integer domain elements."""
    if isinstance(formula, Pred):
        v = mapping.get(formula.var, formula.var)
        return Pred(formula.name, v)
    if isinstance(formula, Not):
        return Not(remap_individuals(formula.inner, mapping))
    if isinstance(formula, And):
        return And(remap_individuals(formula.left, mapping),
                   remap_individuals(formula.right, mapping))
    if isinstance(formula, Or):
        return Or(remap_individuals(formula.left, mapping),
                  remap_individuals(formula.right, mapping))
    if isinstance(formula, Implies):
        return Implies(remap_individuals(formula.left, mapping),
                       remap_individuals(formula.right, mapping))
    if isinstance(formula, ForAll):
        return ForAll(formula.var, remap_individuals(formula.body, mapping))
    if isinstance(formula, Exists):
        return Exists(formula.var, remap_individuals(formula.body, mapping))
    return formula


def check_validity(premises, conclusion, domain_size=3):
    """Check if conclusion follows from premises via model enumeration.

    Returns True if valid (no countermodel exists).
    """
    all_formulas = premises + [conclusion]
    predicates = sorted(collect_predicates(all_formulas))
    individuals = sorted(collect_individuals(all_formulas))

    # Map individuals to domain elements
    indiv_map = {name: i for i, name in enumerate(individuals)}
    domain_size = max(domain_size, len(individuals) + 1)
    domain_size = min(domain_size, 4)  # cap for performance

    premises = [remap_individuals(p, indiv_map) for p in premises]
    conclusion = remap_individuals(conclusion, indiv_map)
    predicates = sorted(collect_predicates(premises + [conclusion]))

    domain = list(range(domain_size))
    n_preds = len(predicates)

    # Enumerate all possible models (each predicate = subset of domain)
    pred_assignments = list(iter_product(range(2), repeat=domain_size))

    for model_tuple in iter_product(pred_assignments, repeat=n_preds):
        model = {"__domain__": domain}
        for i, pred_name in enumerate(predicates):
            model[pred_name] = {
                d for d, val in zip(domain, model_tuple[i]) if val
            }

        if all(evaluate(p, model, {}) for p in premises):
            if not evaluate(conclusion, model, {}):
                return False  # countermodel found
    return True


# ════════════════════════════════════════════════════════════════════════
# Recursive descent predicate parser
# ════════════════════════════════════════════════════════════════════════

def _split_at_connective(text, connective):
    """Split text at a logical connective, requiring both sides have prepositions."""
    pattern = r'\s+' + re.escape(connective) + r'\s+'
    for m in re.finditer(pattern, text, re.IGNORECASE):
        left = text[:m.start()].strip()
        right = text[m.end():].strip()
        has_prep = lambda s: (' of ' in s or ' to ' in s or ' for ' in s)
        if has_prep(left) and has_prep(right):
            return left, right
    return None


def _strip_article(text):
    return re.sub(r'^(?:a|an|the)\s+', '', text.strip(), flags=re.IGNORECASE)


def parse_pred_atom(text, var="x"):
    """Parse an atomic predicate: [article] relation of Entity."""
    text = _strip_article(text).strip().rstrip(',')
    text = re.sub(r'^either\s+(?:a\s+|an\s+)?', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\s+(?:who|that)\s*$', '', text, flags=re.IGNORECASE)
    name = text.lower()
    if name:
        return Pred(name, var), {name}
    return None, set()


def _normalize_connectives(text):
    """Normalize filler words in logical connectives to standard forms."""
    text = re.sub(r'\s+yet\s+not\s+', ' and not ', text, flags=re.IGNORECASE)
    text = re.sub(r'\s+while\s+not\s+', ' and not ', text, flags=re.IGNORECASE)
    text = re.sub(r'\bor,\s*otherwise,?\s*', 'or ', text, flags=re.IGNORECASE)
    text = re.sub(r'\band,?\s*in\s+addition,?\s*', 'and ', text, flags=re.IGNORECASE)
    text = re.sub(
        r'\band,?\s*in\s+the\s+same\s+time,?\s*', 'and ',
        text, flags=re.IGNORECASE,
    )
    text = re.sub(r',?\s+or\s+both\s*$', '', text, flags=re.IGNORECASE)
    text = re.sub(r',?\s+too\s*$', '', text, flags=re.IGNORECASE)
    text = re.sub(r',?\s+as\s+well\s*$', '', text, flags=re.IGNORECASE)
    return text


def parse_pred_expr(text, var="x"):
    """Recursive descent parser for compound predicate expressions.

    Handles: not, both...and, neither...nor, none of this:, and, or,
    and combinations like "not A or not B".
    """
    text = _normalize_connectives(text.strip())
    if not text:
        return None, set()
    lower = text.lower()

    # "at least one of these: A, B[,] or C"
    m = re.match(r'at\s+least\s+one\s+of\s+these:\s*(.+)', text, re.IGNORECASE)
    if m:
        items = re.split(r',?\s+or\s+|,\s+', m.group(1))
        formulas, all_preds = [], set()
        for item in items:
            f, p = parse_pred_atom(item.strip(), var)
            if f:
                formulas.append(f)
                all_preds |= p
        if formulas:
            result = formulas[-1]
            for f in reversed(formulas[:-1]):
                result = Or(f, result)
            return result, all_preds

    # "not both A and B" → Or(Not(A), Not(B))
    if lower.startswith('not both:'):
        text = 'not both ' + text[9:].strip()
        lower = text.lower()
    if lower.startswith('not both '):
        rest = text[9:].strip()
        split = _split_at_connective(rest, 'and')
        if split:
            lf, lp = parse_pred_expr(split[0], var)
            rf, rp = parse_pred_expr(split[1], var)
            if lf and rf:
                return Or(Not(lf), Not(rf)), lp | rp

    # "both A and B" → And(A, B)
    if lower.startswith('both:'):
        text = 'both ' + text[5:].strip()
        lower = text.lower()
    if lower.startswith('both '):
        rest = text[5:].strip()
        split = _split_at_connective(rest, 'and')
        if split:
            lf, lp = parse_pred_expr(split[0], var)
            rf, rp = parse_pred_expr(split[1], var)
            if lf and rf:
                return And(lf, rf), lp | rp

    # "neither A nor B" → And(Not(A), Not(B))
    if lower.startswith('neither '):
        rest = text[8:].strip()
        split = _split_at_connective(rest, 'nor')
        if split:
            lf, lp = parse_pred_expr(split[0], var)
            rf, rp = parse_pred_expr(split[1], var)
            if lf and rf:
                return And(Not(lf), Not(rf)), lp | rp

    # "none of this: A or B [or C]"
    m = re.match(r'none\s+of\s+this:\s*(.+)', text, re.IGNORECASE)
    if m:
        parts = re.split(r'\s+or\s+', m.group(1))
        formulas, all_preds = [], set()
        for part in parts:
            f, p = parse_pred_atom(part.strip(), var)
            if f:
                formulas.append(Not(f))
                all_preds |= p
        if formulas:
            result = formulas[0]
            for f in formulas[1:]:
                result = And(result, f)
            return result, all_preds

    # "not A or not B" → Or(Not(A), Not(B))
    split = _split_at_connective(text, 'or')
    if split:
        ll, rl = split[0].lower(), split[1].lower()
        if ll.startswith('not ') and rl.startswith('not '):
            lf, lp = parse_pred_expr(split[0][4:].strip(), var)
            rf, rp = parse_pred_expr(split[1][4:].strip(), var)
            if lf and rf:
                return Or(Not(lf), Not(rf)), lp | rp

    # "not A and not B" → And(Not(A), Not(B))
    split_and = _split_at_connective(text, 'and')
    if split_and:
        ll, rl = split_and[0].lower(), split_and[1].lower()
        if ll.startswith('not ') and rl.startswith('not '):
            lf, lp = parse_pred_expr(split_and[0][4:].strip(), var)
            rf, rp = parse_pred_expr(split_and[1][4:].strip(), var)
            if lf and rf:
                return And(Not(lf), Not(rf)), lp | rp

    # "A or B" → Or(A, B)
    if split:
        lf, lp = parse_pred_expr(split[0], var)
        rf, rp = parse_pred_expr(split[1], var)
        if lf and rf:
            return Or(lf, rf), lp | rp

    # "A and B" → And(A, B)
    if split_and:
        lf, lp = parse_pred_expr(split_and[0], var)
        rf, rp = parse_pred_expr(split_and[1], var)
        if lf and rf:
            return And(lf, rf), lp | rp

    # "not A" → Not(A)
    if lower.startswith('not '):
        inner, ip = parse_pred_expr(text[4:].strip(), var)
        if inner:
            return Not(inner), ip

    # Atomic predicate
    return parse_pred_atom(text, var)


# ════════════════════════════════════════════════════════════════════════
# Pattern classifier + probe-dispatched parser
# ════════════════════════════════════════════════════════════════════════

# Words that start quantified/connective sentences, never individual names.
# Used as a guard: reject name_is/name_is_not when sentence starts with these.
QUANTIFIER_WORDS = frozenset([
    'every', 'whoever', 'whatever', 'being', 'not', 'no', 'nobody',
    'nothing', 'some', 'there', 'it', 'everyone', 'everything', 'if',
    'somebody', 'to', 'someone', 'something',
])


def classify_pattern(sentence):
    """Classify sentence pattern by keyword prefix.

    Returns a label like 'every_is', 'name_is', 'being_necessary', etc.
    This is the regex-based default classifier; a trained probe can
    replace this for higher accuracy or speed.
    """
    text = sentence.strip().rstrip(".")
    text = re.sub(r'\s+is\s+also\s+', ' is ', text, flags=re.IGNORECASE)
    text = re.sub(r',?\s+too\.?\s*$', '', text, flags=re.IGNORECASE)
    lower = text.lower()

    m = re.match(r'^every\s+(.+)\s+is\s+(?:also\s+|however\s+)?(.+)$', lower)
    if m:
        subj = m.group(1)
        if re.match(r'^(.+?)\s+(?:who|that)\s+is\s+', subj):
            return "every_who_is"
        if re.match(r'^(.+?)\s+and\s+every\s+', subj):
            return "every_and_every"
        return "every_is"

    if re.match(r'^(?:whoever|whatever)\s+is\s+none\s+of\s+this:', lower):
        return "whoever_none"
    if re.match(r'^(?:whoever|whatever)\s+is\s+', lower):
        return "whoever_is"
    if re.match(r'^every(?:one|body)\s+who\s+is\s+', lower):
        return "everyone_who_is"
    if re.match(r'^everything\s+that\s+is\s+', lower):
        return "everything_that_is"

    if re.match(r'^no\s+(.+?)\s+and\s+no\s+(.+?)\s+is\s+', lower):
        return "no_and_no_is"
    m = re.match(r'^no\s+(.+)\s+is\s+(.+)$', lower)
    if m:
        subj = m.group(1)
        if re.match(r'^(.+?)\s+(?:who|that)\s+is\s+', subj):
            return "no_who_is"
        return "no_is"

    if re.match(r'^nobody\s+is\s+neither\s+', lower):
        return "nobody_neither"
    if re.match(r'^nothing\s+is\s+neither\s+', lower):
        return "nothing_neither"

    if re.match(
        r'^being\s+(.+?)\s+is\s+necessary\s+for\s+(not\s+)?being\s+', lower,
    ):
        return "being_necessary"
    if re.match(
        r'^being\s+(.+?)\s+is\s+sufficient\s+for\s+(not\s+)?being\s+', lower,
    ):
        return "being_sufficient"
    if re.match(r'^not\s+being\s+(.+?)\s+is\s+sufficient\s+for\s+', lower):
        return "not_being_sufficient"
    if re.match(r'^not\s+being\s+(.+?)\s+is\s+necessary\s+for\s+', lower):
        return "not_being_necessary"
    if re.match(r'^to\s+be\s+(.+?)\s+is\s+necessary\s+for\s+', lower):
        return "to_be_necessary"

    if re.match(r'^it\s+is\s+(?:not\s+the\s+case|false)\s+that\s+', lower):
        return "it_is_not"
    if re.match(r'^not\s+every\s+', lower):
        return "not_every"

    if re.match(r'^some\s+(.+)\s+is\s+', lower):
        return "some_is"
    if re.match(r'^some(?:one|body)\s+who\s+is\s+', lower):
        return "someone_who_is"
    if re.match(r'^there\s+is\s+(?:somebody|someone)\s+who\s+is\s+', lower):
        return "there_is_somebody"
    if re.match(r'^there\s+is\s+something\s+that\s+is\s+', lower):
        return "there_is_something"
    if re.match(r'^there\s+is\s+no\s+', lower):
        return "there_is_no"
    if re.match(r'^there\s+(?:exists|is)\s+(?:a\s+|an\s+)?', lower):
        return "there_exists"
    if re.match(r'^(?:somebody|someone)\s+is\s+', lower):
        return "somebody_is"
    if re.match(r'^something\s+is\s+', lower):
        return "something_is"

    if re.match(r'^if\s+(?:someone|something)\s+is\s+', lower):
        return "if_someone_is"

    # Individual assertions (guarded against quantifier words)
    m = re.match(r'^(.+?)\s+is\s+none\s+of\s+this:', text, re.IGNORECASE)
    if m:
        first = m.group(1).split()[0].lower()
        if first not in QUANTIFIER_WORDS:
            return "name_none_of"
    m = re.match(r'^(.+?)\s+is\s+not\s+', text, re.IGNORECASE)
    if m:
        first = m.group(1).split()[0].lower()
        if first not in QUANTIFIER_WORDS:
            return "name_is_not"
    m = re.match(r'^(.+?)\s+is\s+', text, re.IGNORECASE)
    if m:
        first = m.group(1).split()[0].lower()
        if first not in QUANTIFIER_WORDS:
            return "name_is"

    return "unknown"


def _prep_sentence(text):
    """Normalize sentence for extraction."""
    text = text.strip().rstrip('.')
    text = re.sub(r'\s+is\s+also\s+', ' is ', text, flags=re.IGNORECASE)
    text = re.sub(r',?\s+too\.?\s*$', '', text, flags=re.IGNORECASE)
    return text


def probe_parse_sentence(text, label=None):
    """Parse a sentence using targeted extraction based on a pattern label.

    If label is None, classifies using the regex classifier (oracle mode).
    Returns (formula, predicates) or (None, set()).

    The label tells us which extraction path to use, avoiding the regex
    cascade in parse_sentence(). A guard rejects 'name_is'/'name_is_not'
    when the sentence starts with a quantifier word, eliminating 94% of
    potential silent corruptions from catch-all patterns.
    """
    text = _prep_sentence(text)
    lower = text.lower()
    if label is None:
        label = classify_pattern(text)

    # ── Guard: reject catch-all labels for quantified sentences ──
    first_word = text.split()[0].lower() if text.split() else ""
    if label in ("name_is", "name_is_not") and first_word in QUANTIFIER_WORDS:
        return None, set()

    # ── Universal: every_is ──
    if label == "every_is":
        m = re.match(
            r'^every\s+(.+)\s+is\s+(?:also\s+|however\s+)?(.+)$', lower,
        )
        if m:
            ant_f, ap = parse_pred_atom(m.group(1), "x")
            cons_f, cp = parse_pred_expr(m.group(2), "x")
            if ant_f and cons_f:
                return ForAll("x", Implies(ant_f, cons_f)), ap | cp

    # ── Universal: every_who_is ──
    if label == "every_who_is":
        m = re.match(
            r'^every\s+(.+)\s+is\s+(?:also\s+|however\s+)?(.+)$', lower,
        )
        if m:
            subj, obj = m.group(1), m.group(2)
            who_m = re.match(
                r'^(.+?)\s+(?:who|that)\s+is\s+(?:also\s+)?(.+)$', subj,
            )
            if who_m:
                base_f, bp = parse_pred_atom(who_m.group(1), "x")
                qual_f, qp = parse_pred_expr(who_m.group(2), "x")
                cons_f, cp = parse_pred_expr(obj, "x")
                if base_f and qual_f and cons_f:
                    return (
                        ForAll("x", Implies(And(base_f, qual_f), cons_f)),
                        bp | qp | cp,
                    )

    # ── Universal: every_and_every ──
    if label == "every_and_every":
        m = re.match(
            r'^every\s+(.+)\s+is\s+(?:also\s+|however\s+)?(.+)$', lower,
        )
        if m:
            subj, obj = m.group(1), m.group(2)
            every_m = re.match(r'^(.+?)\s+and\s+every\s+(.+)$', subj)
            if every_m:
                f1, p1 = parse_pred_atom(every_m.group(1), "x")
                f2, p2 = parse_pred_atom(every_m.group(2), "x")
                cons_f, cp = parse_pred_expr(obj, "x")
                if f1 and f2 and cons_f:
                    return (
                        And(
                            ForAll("x", Implies(f1, cons_f)),
                            ForAll("x", Implies(f2, cons_f)),
                        ),
                        p1 | p2 | cp,
                    )

    # ── Universal: whoever_is / whatever_is ──
    if label == "whoever_is":
        m = re.match(
            r'^(?:whoever|whatever)\s+is\s+(.+)\s+is\s+'
            r'(?:also\s+|however\s+)?(.+?)(?:,\s*too)?$', lower,
        )
        if m:
            ant_f, ap = parse_pred_expr(m.group(1), "x")
            cons_f, cp = parse_pred_expr(m.group(2), "x")
            if ant_f and cons_f:
                return ForAll("x", Implies(ant_f, cons_f)), ap | cp

    # ── Universal: everyone_who_is / everybody_who_is ──
    if label == "everyone_who_is":
        m = re.match(
            r'^every(?:one|body)\s+who\s+is\s+(.+)\s+is\s+'
            r'(?:also\s+|however\s+)?(.+?)(?:,\s*too)?$', lower,
        )
        if m:
            ant_f, ap = parse_pred_expr(m.group(1), "x")
            cons_f, cp = parse_pred_expr(m.group(2), "x")
            if ant_f and cons_f:
                return ForAll("x", Implies(ant_f, cons_f)), ap | cp

    # ── Universal: everything_that_is ──
    if label == "everything_that_is":
        m = re.match(
            r'^everything\s+that\s+is\s+(?:a\s+|an\s+)?(.+)\s+is\s+'
            r'(?:also\s+|however\s+)?(.+?)(?:,\s*too)?$', lower,
        )
        if m:
            ant_f, ap = parse_pred_expr(m.group(1), "x")
            cons_f, cp = parse_pred_expr(m.group(2), "x")
            if ant_f and cons_f:
                return ForAll("x", Implies(ant_f, cons_f)), ap | cp

    # ── Negated: no_is ──
    if label == "no_is":
        m = re.match(r'^no\s+(.+)\s+is\s+(.+)$', lower)
        if m:
            ant_f, ap = parse_pred_atom(m.group(1), "x")
            obj_f, op = parse_pred_expr(m.group(2), "x")
            if ant_f and obj_f:
                return ForAll("x", Implies(ant_f, Not(obj_f))), ap | op

    # ── Negated: no_who_is ──
    if label == "no_who_is":
        m = re.match(r'^no\s+(.+)\s+is\s+(.+)$', lower)
        if m:
            subj, obj = m.group(1), m.group(2)
            who_m = re.match(
                r'^(.+?)\s+(?:who|that)\s+is\s+(?:also\s+)?(.+)$', subj,
            )
            if who_m:
                base_f, bp = parse_pred_atom(who_m.group(1), "x")
                qual_f, qp = parse_pred_expr(who_m.group(2), "x")
                obj_f, op = parse_pred_expr(obj, "x")
                if base_f and qual_f and obj_f:
                    return (
                        ForAll(
                            "x", Implies(And(base_f, qual_f), Not(obj_f)),
                        ),
                        bp | qp | op,
                    )

    # ── Negated: no_and_no_is ──
    if label == "no_and_no_is":
        m = re.match(r'^no\s+(.+?)\s+and\s+no\s+(.+?)\s+is\s+(.+)$', lower)
        if m:
            s1f, s1p = parse_pred_atom(m.group(1), "x")
            s2f, s2p = parse_pred_atom(m.group(2), "x")
            obj_f, op = parse_pred_expr(m.group(3), "x")
            if s1f and s2f and obj_f:
                return (
                    And(
                        ForAll("x", Implies(s1f, Not(obj_f))),
                        ForAll("x", Implies(s2f, Not(obj_f))),
                    ),
                    s1p | s2p | op,
                )

    # ── Nobody/nothing neither nor ──
    if label in ("nobody_neither", "nothing_neither"):
        starter = "nobody" if label == "nobody_neither" else "nothing"
        m = re.match(
            rf'^{starter}\s+is\s+neither\s+(.+?)\s+nor\s+(.+)$', lower,
        )
        if m:
            lf, lp = parse_pred_atom(m.group(1), "x")
            rf, rp = parse_pred_atom(m.group(2), "x")
            if lf and rf:
                return ForAll("x", Or(lf, rf)), lp | rp

    # ── Necessary conditions ──
    if label == "being_necessary":
        m = re.match(
            r'^being\s+(.+?)\s+is\s+necessary\s+for\s+'
            r'(not\s+)?being\s+(.+)$', lower,
        )
        if m:
            nec_f, np_ = parse_pred_expr(m.group(1), "x")
            neg = m.group(2) is not None
            cond_f, cp = parse_pred_expr(m.group(3), "x")
            if nec_f and cond_f:
                ant = Not(cond_f) if neg else cond_f
                return ForAll("x", Implies(ant, nec_f)), np_ | cp

    if label == "to_be_necessary":
        m = re.match(
            r'^to\s+be\s+(.+?)\s+is\s+necessary\s+for\s+'
            r'(not\s+)?being\s+(.+)$', lower,
        )
        if m:
            nec_f, np_ = parse_pred_expr(m.group(1), "x")
            neg = m.group(2) is not None
            cond_f, cp = parse_pred_expr(m.group(3), "x")
            if nec_f and cond_f:
                ant = Not(cond_f) if neg else cond_f
                return ForAll("x", Implies(ant, nec_f)), np_ | cp

    # ── Sufficient conditions ──
    if label == "being_sufficient":
        m = re.match(
            r'^being\s+(.+?)\s+is\s+sufficient\s+for\s+'
            r'(not\s+)?being\s+(.+)$', lower,
        )
        if m:
            suf_f, sp = parse_pred_expr(m.group(1), "x")
            neg = m.group(2) is not None
            res_f, rp = parse_pred_expr(m.group(3), "x")
            if suf_f and res_f:
                cons = Not(res_f) if neg else res_f
                return ForAll("x", Implies(suf_f, cons)), sp | rp

    if label == "not_being_sufficient":
        m = re.match(
            r'^not\s+being\s+(.+?)\s+is\s+sufficient\s+for\s+'
            r'(not\s+)?being\s+(.+)$', lower,
        )
        if m:
            ant_f, ap = parse_pred_expr(m.group(1), "x")
            neg = m.group(2) is not None
            cons_f, cp = parse_pred_expr(m.group(3), "x")
            if ant_f and cons_f:
                cons = Not(cons_f) if neg else cons_f
                return ForAll("x", Implies(Not(ant_f), cons)), ap | cp

    if label == "not_being_necessary":
        m = re.match(
            r'^not\s+being\s+(.+?)\s+is\s+necessary\s+for\s+'
            r'(not\s+)?being\s+(.+)$', lower,
        )
        if m:
            nec_f, np_ = parse_pred_expr(m.group(1), "x")
            neg = m.group(2) is not None
            cond_f, cp = parse_pred_expr(m.group(3), "x")
            if nec_f and cond_f:
                ant = Not(cond_f) if neg else cond_f
                return ForAll("x", Implies(ant, Not(nec_f))), np_ | cp

    # ── Negation ──
    if label == "it_is_not":
        m = re.match(
            r'^it\s+is\s+(?:not\s+the\s+case|false)\s+that\s+'
            r'(.+)\s+is\s+(.+)$', lower,
        )
        if m:
            name = m.group(1).strip()
            pred_f, pp = parse_pred_expr(m.group(2), name)
            if pred_f:
                return Not(pred_f), pp

    if label == "not_every":
        m = re.match(r'^not\s+every\s+(.+)\s+is\s+(.+)$', lower)
        if m:
            subj_f, sp = parse_pred_atom(m.group(1), "x")
            obj_f, op = parse_pred_expr(m.group(2), "x")
            if subj_f and obj_f:
                return Not(ForAll("x", Implies(subj_f, obj_f))), sp | op

    # ── Existential ──
    if label == "some_is":
        m = re.match(r'^some\s+(.+)\s+is\s+(.+)$', lower)
        if m:
            subj_f, sp = parse_pred_atom(m.group(1), "x")
            obj_f, op = parse_pred_expr(m.group(2), "x")
            if subj_f and obj_f:
                return Exists("x", And(subj_f, obj_f)), sp | op

    if label == "someone_who_is":
        m = re.match(
            r'^some(?:one|body)\s+who\s+is\s+(not\s+)?(.+)\s+is\s+(.+)$',
            lower,
        )
        if m:
            neg = m.group(1) is not None
            ant_f, ap = parse_pred_expr(m.group(2), "x")
            cons_f, cp = parse_pred_expr(m.group(3), "x")
            if ant_f and cons_f:
                ant = Not(ant_f) if neg else ant_f
                return ForAll("x", Implies(ant, cons_f)), ap | cp

    if label == "there_is_somebody":
        m = re.match(
            r'^there\s+is\s+(?:somebody|someone)\s+who\s+is\s+(.+)$', lower,
        )
        if m:
            pred_f, pp = parse_pred_expr(m.group(1), "x")
            if pred_f:
                return Exists("x", pred_f), pp

    if label == "there_is_something":
        m = re.match(
            r'^there\s+is\s+something\s+that\s+is\s+(.+)$', lower,
        )
        if m:
            pred_f, pp = parse_pred_expr(m.group(1), "x")
            if pred_f:
                return Exists("x", pred_f), pp

    if label == "there_is_no":
        m = re.match(
            r'^there\s+is\s+no\s+(.+?)\s+(?:who|that)\s+is\s+(.+)$', lower,
        )
        if m:
            subj_f, sp = parse_pred_atom(m.group(1), "x")
            obj_f, op = parse_pred_expr(m.group(2), "x")
            if subj_f and obj_f:
                return Not(Exists("x", And(subj_f, obj_f))), sp | op

    if label == "there_exists":
        m = re.match(
            r'^there\s+(?:exists|is)\s+(?:a\s+|an\s+)?(.+?)\s+'
            r'(?:who|that)\s+is\s+(.+)$', lower,
        )
        if m:
            subj_f, sp = parse_pred_atom(m.group(1), "x")
            obj_f, op = parse_pred_expr(m.group(2), "x")
            if subj_f and obj_f:
                return Exists("x", And(subj_f, obj_f)), sp | op

    if label == "somebody_is":
        m = re.match(r'^(?:somebody|someone)\s+is\s+(.+)$', lower)
        if m:
            pred_f, pp = parse_pred_expr(m.group(1), "x")
            if pred_f:
                return Exists("x", pred_f), pp

    if label == "something_is":
        m = re.match(r'^something\s+is\s+(not\s+)?(.+)$', lower)
        if m:
            neg = m.group(1) is not None
            pred_f, pp = parse_pred_expr(m.group(2), "x")
            if pred_f:
                inner = Not(pred_f) if neg else pred_f
                return Exists("x", inner), pp

    # ── Conditional ──
    if label == "if_someone_is":
        m = re.match(
            r'^if\s+(?:someone|something)\s+is\s+(not\s+)?(?:a\s+|an\s+)?'
            r'(.+?),?\s+then\s+that\s+(?:person|thing)\s+is\s+(.+)$', lower,
        )
        if m:
            neg = m.group(1) is not None
            ant_f, ap = parse_pred_expr(m.group(2), "x")
            cons_f, cp = parse_pred_expr(m.group(3), "x")
            if ant_f and cons_f:
                ant = Not(ant_f) if neg else ant_f
                return ForAll("x", Implies(ant, cons_f)), ap | cp

    # ── Individual (guarded catch-all) ──
    if label == "name_is_not":
        m = re.match(r'^(.+?)\s+is\s+not\s+(.+)$', text, re.IGNORECASE)
        if m:
            name = m.group(1).strip().lower()
            pred_f, pp = parse_pred_expr(m.group(2), name)
            if pred_f:
                return Not(pred_f), pp

    if label == "name_is":
        m = re.match(
            r'^(.+?)\s+is\s+(?:a\s+|an\s+)?(.+)$', text, re.IGNORECASE,
        )
        if m:
            name = m.group(1).strip().lower()
            pred_f, pp = parse_pred_expr(m.group(2), name)
            if pred_f:
                return pred_f, pp

    if label == "name_none_of":
        m = re.match(
            r'^(.+?)\s+is\s+none\s+of\s+this:\s*(.+)$',
            text, re.IGNORECASE,
        )
        if m:
            name = m.group(1).strip().lower()
            parts = re.split(r'\s+or\s+', m.group(2))
            formulas, all_preds = [], set()
            for part in parts:
                f, p = parse_pred_atom(part.strip(), name)
                if f:
                    formulas.append(Not(f))
                    all_preds |= p
            if formulas:
                result = formulas[0]
                for f in formulas[1:]:
                    result = And(result, f)
                return result, all_preds

    if label == "whoever_none":
        m = re.match(
            r'^(?:whoever|whatever)\s+is\s+none\s+of\s+this:\s*(.+?)'
            r',?\s+is\s+(?:also\s+|however\s+)?(.+?)(?:,\s*too)?$', lower,
        )
        if m:
            parts = re.split(r'\s+or\s+', m.group(1))
            formulas, all_preds = [], set()
            for part in parts:
                f, p = parse_pred_atom(part.strip(), "x")
                if f:
                    formulas.append(Not(f))
                    all_preds |= p
            cons_f, cp = parse_pred_expr(m.group(2), "x")
            if formulas and cons_f:
                ant = formulas[0]
                for f in formulas[1:]:
                    ant = And(ant, f)
                return ForAll("x", Implies(ant, cons_f)), all_preds | cp

    return None, set()


# ════════════════════════════════════════════════════════════════════════
# Sentence-level parser (regex cascade fallback)
# ════════════════════════════════════════════════════════════════════════

def parse_sentence(text):
    """Parse one NL sentence into a FOL formula.

    Returns (formula, predicates) or (None, set()).
    """
    text = text.strip().rstrip('.')
    text = re.sub(r'\s+is\s+also\s+', ' is ', text, flags=re.IGNORECASE)
    text = re.sub(r',?\s+too\.?\s*$', '', text, flags=re.IGNORECASE)
    lower = text.lower()

    # ── Universal patterns ──

    # "every X [who/that is Y] is Z"
    m = re.match(r'^every\s+(.+)\s+is\s+(?:also\s+|however\s+)?(.+)$', lower)
    if m:
        subj, obj = m.group(1), m.group(2)
        who_m = re.match(
            r'^(.+?)\s+(?:who|that)\s+is\s+(?:also\s+)?(.+)$', subj,
        )
        if who_m:
            base_f, bp = parse_pred_atom(who_m.group(1), "x")
            qual_f, qp = parse_pred_expr(who_m.group(2), "x")
            cons_f, cp = parse_pred_expr(obj, "x")
            if base_f and qual_f and cons_f:
                return (
                    ForAll("x", Implies(And(base_f, qual_f), cons_f)),
                    bp | qp | cp,
                )
        else:
            every_m = re.match(r'^(.+?)\s+and\s+every\s+(.+)$', subj)
            if every_m:
                f1, p1 = parse_pred_atom(every_m.group(1), "x")
                f2, p2 = parse_pred_atom(every_m.group(2), "x")
                cons_f, cp = parse_pred_expr(obj, "x")
                if f1 and f2 and cons_f:
                    return (
                        And(
                            ForAll("x", Implies(f1, cons_f)),
                            ForAll("x", Implies(f2, cons_f)),
                        ),
                        p1 | p2 | cp,
                    )
            ant_f, ap = parse_pred_atom(subj, "x")
            cons_f, cp = parse_pred_expr(obj, "x")
            if ant_f and cons_f:
                return ForAll("x", Implies(ant_f, cons_f)), ap | cp

    # "whoever/whatever is X is [also|however] Y"
    m = re.match(
        r'^(?:whoever|whatever)\s+is\s+(.+)\s+is\s+'
        r'(?:also\s+|however\s+)?(.+?)(?:,\s*too)?$',
        lower,
    )
    if m:
        ant_f, ap = parse_pred_expr(m.group(1), "x")
        cons_f, cp = parse_pred_expr(m.group(2), "x")
        if ant_f and cons_f:
            return ForAll("x", Implies(ant_f, cons_f)), ap | cp

    # "everyone/everybody who is X is [also|however] Y"
    m = re.match(
        r'^every(?:one|body)\s+who\s+is\s+(.+)\s+is\s+'
        r'(?:also\s+|however\s+)?(.+?)(?:,\s*too)?$',
        lower,
    )
    if m:
        ant_f, ap = parse_pred_expr(m.group(1), "x")
        cons_f, cp = parse_pred_expr(m.group(2), "x")
        if ant_f and cons_f:
            return ForAll("x", Implies(ant_f, cons_f)), ap | cp

    # "everything that is [a|an] X is [also|however] Y"
    m = re.match(
        r'^everything\s+that\s+is\s+(?:a\s+|an\s+)?(.+)\s+is\s+'
        r'(?:also\s+|however\s+)?(.+?)(?:,\s*too)?$',
        lower,
    )
    if m:
        ant_f, ap = parse_pred_expr(m.group(1), "x")
        cons_f, cp = parse_pred_expr(m.group(2), "x")
        if ant_f and cons_f:
            return ForAll("x", Implies(ant_f, cons_f)), ap | cp

    # "no X and no Y is Z"
    m = re.match(r'^no\s+(.+?)\s+and\s+no\s+(.+?)\s+is\s+(.+)$', lower)
    if m:
        subj1, subj2, obj = m.group(1), m.group(2), m.group(3)
        if ' of ' in subj1 and ' of ' in subj2:
            s1f, s1p = parse_pred_atom(subj1, "x")
            s2f, s2p = parse_pred_atom(subj2, "x")
            obj_f, op = parse_pred_expr(obj, "x")
            if s1f and s2f and obj_f:
                f1 = ForAll("x", Implies(s1f, Not(obj_f)))
                f2 = ForAll("x", Implies(s2f, Not(obj_f)))
                return And(f1, f2), s1p | s2p | op

    # "no X [who is Y] is Z"
    m = re.match(r'^no\s+(.+)\s+is\s+(.+)$', lower)
    if m:
        subj, obj = m.group(1), m.group(2)
        who_m = re.match(
            r'^(.+?)\s+(?:who|that)\s+is\s+(?:also\s+)?(.+)$', subj,
        )
        if who_m:
            base_f, bp = parse_pred_atom(who_m.group(1), "x")
            qual_f, qp = parse_pred_expr(who_m.group(2), "x")
            obj_f, op = parse_pred_expr(obj, "x")
            if base_f and qual_f and obj_f:
                return (
                    ForAll("x", Implies(And(base_f, qual_f), Not(obj_f))),
                    bp | qp | op,
                )
        else:
            ant_f, ap = parse_pred_atom(subj, "x")
            obj_f, op = parse_pred_expr(obj, "x")
            if ant_f and obj_f:
                return ForAll("x", Implies(ant_f, Not(obj_f))), ap | op

    # "nobody is neither X nor Y" → ∀x: X(x) ∨ Y(x)
    m = re.match(r'^nobody\s+is\s+neither\s+(.+?)\s+nor\s+(.+)$', lower)
    if m:
        lf, lp = parse_pred_atom(m.group(1), "x")
        rf, rp = parse_pred_atom(m.group(2), "x")
        if lf and rf:
            return ForAll("x", Or(lf, rf)), lp | rp

    # "nothing is neither X nor Y" → ∀x: X(x) ∨ Y(x)
    m = re.match(r'^nothing\s+is\s+neither\s+(.+?)\s+nor\s+(.+)$', lower)
    if m:
        lf, lp = parse_pred_atom(m.group(1), "x")
        rf, rp = parse_pred_atom(m.group(2), "x")
        if lf and rf:
            return ForAll("x", Or(lf, rf)), lp | rp

    # ── Necessary/sufficient conditions ──

    # "being X is necessary for [not] being Y"
    m = re.match(
        r'^being\s+(.+?)\s+is\s+necessary\s+for\s+(not\s+)?being\s+(.+)$',
        lower,
    )
    if m:
        nec_f, np_ = parse_pred_expr(m.group(1), "x")
        neg = m.group(2) is not None
        cond_f, cp = parse_pred_expr(m.group(3), "x")
        if nec_f and cond_f:
            ant = Not(cond_f) if neg else cond_f
            return ForAll("x", Implies(ant, nec_f)), np_ | cp

    # "being X is sufficient for [not] being Y"
    m = re.match(
        r'^being\s+(.+?)\s+is\s+sufficient\s+for\s+(not\s+)?being\s+(.+)$',
        lower,
    )
    if m:
        suf_f, sp = parse_pred_expr(m.group(1), "x")
        neg = m.group(2) is not None
        res_f, rp = parse_pred_expr(m.group(3), "x")
        if suf_f and res_f:
            cons = Not(res_f) if neg else res_f
            return ForAll("x", Implies(suf_f, cons)), sp | rp

    # "not being X is sufficient for [not] being Y"
    m = re.match(
        r'^not\s+being\s+(.+?)\s+is\s+sufficient\s+for\s+'
        r'(not\s+)?being\s+(.+)$',
        lower,
    )
    if m:
        ant_f, ap = parse_pred_expr(m.group(1), "x")
        neg = m.group(2) is not None
        cons_f, cp = parse_pred_expr(m.group(3), "x")
        if ant_f and cons_f:
            cons = Not(cons_f) if neg else cons_f
            return ForAll("x", Implies(Not(ant_f), cons)), ap | cp

    # "not being X is necessary for [not] being Y"
    m = re.match(
        r'^not\s+being\s+(.+?)\s+is\s+necessary\s+for\s+'
        r'(not\s+)?being\s+(.+)$',
        lower,
    )
    if m:
        nec_f, np_ = parse_pred_expr(m.group(1), "x")
        neg = m.group(2) is not None
        cond_f, cp = parse_pred_expr(m.group(3), "x")
        if nec_f and cond_f:
            ant = Not(cond_f) if neg else cond_f
            return ForAll("x", Implies(ant, Not(nec_f))), np_ | cp

    # "to be X is necessary for [not] being Y"
    m = re.match(
        r'^to\s+be\s+(.+?)\s+is\s+necessary\s+for\s+'
        r'(not\s+)?being\s+(.+)$',
        lower,
    )
    if m:
        nec_f, np_ = parse_pred_expr(m.group(1), "x")
        neg = m.group(2) is not None
        cond_f, cp = parse_pred_expr(m.group(3), "x")
        if nec_f and cond_f:
            ant = Not(cond_f) if neg else cond_f
            return ForAll("x", Implies(ant, nec_f)), np_ | cp

    # ── Negation ──

    # "it is not the case / false that NAME is [not] Y"
    m = re.match(
        r'^it\s+is\s+(?:not\s+the\s+case|false)\s+that\s+(.+)\s+is\s+(.+)$',
        lower,
    )
    if m:
        name = m.group(1).strip()
        pred_text = m.group(2)
        pred_f, pp = parse_pred_expr(pred_text, name)
        if pred_f:
            return Not(pred_f), pp

    # "not every X is Y"
    m = re.match(r'^not\s+every\s+(.+)\s+is\s+(.+)$', lower)
    if m:
        subj_f, sp = parse_pred_atom(m.group(1), "x")
        obj_f, op = parse_pred_expr(m.group(2), "x")
        if subj_f and obj_f:
            return Not(ForAll("x", Implies(subj_f, obj_f))), sp | op

    # ── Existential ──

    # "some X is Y"
    m = re.match(r'^some\s+(.+)\s+is\s+(.+)$', lower)
    if m:
        subj_f, sp = parse_pred_atom(m.group(1), "x")
        obj_f, op = parse_pred_expr(m.group(2), "x")
        if subj_f and obj_f:
            return Exists("x", And(subj_f, obj_f)), sp | op

    # "someone/somebody who is [not] X is Y"
    m = re.match(
        r'^some(?:one|body)\s+who\s+is\s+(not\s+)?(.+)\s+is\s+(.+)$', lower,
    )
    if m:
        neg = m.group(1) is not None
        ant_f, ap = parse_pred_expr(m.group(2), "x")
        cons_f, cp = parse_pred_expr(m.group(3), "x")
        if ant_f and cons_f:
            ant = Not(ant_f) if neg else ant_f
            return ForAll("x", Implies(ant, cons_f)), ap | cp

    # "there is somebody/someone who is X"
    m = re.match(
        r'^there\s+is\s+(?:somebody|someone)\s+who\s+is\s+(.+)$', lower,
    )
    if m:
        pred_f, pp = parse_pred_expr(m.group(1), "x")
        if pred_f:
            return Exists("x", pred_f), pp
    m = re.match(
        r'^there\s+is\s+something\s+that\s+is\s+(.+)$', lower,
    )
    if m:
        pred_f, pp = parse_pred_expr(m.group(1), "x")
        if pred_f:
            return Exists("x", pred_f), pp

    # "there is no X who is Y"
    m = re.match(
        r'^there\s+is\s+no\s+(.+?)\s+(?:who|that)\s+is\s+(.+)$', lower,
    )
    if m:
        subj_f, sp = parse_pred_atom(m.group(1), "x")
        obj_f, op = parse_pred_expr(m.group(2), "x")
        if subj_f and obj_f:
            return Not(Exists("x", And(subj_f, obj_f))), sp | op

    # "there exists/is [a|an] X who is Y"
    m = re.match(
        r'^there\s+(?:exists|is)\s+(?:a\s+|an\s+)?(.+?)\s+'
        r'(?:who|that)\s+is\s+(.+)$',
        lower,
    )
    if m:
        subj_f, sp = parse_pred_atom(m.group(1), "x")
        obj_f, op = parse_pred_expr(m.group(2), "x")
        if subj_f and obj_f:
            return Exists("x", And(subj_f, obj_f)), sp | op

    # "somebody/someone is X"
    m = re.match(r'^(?:somebody|someone)\s+is\s+(.+)$', lower)
    if m:
        pred_f, pp = parse_pred_expr(m.group(1), "x")
        if pred_f:
            return Exists("x", pred_f), pp

    # "something is [not] X"
    m = re.match(r'^something\s+is\s+(not\s+)?(.+)$', lower)
    if m:
        neg = m.group(1) is not None
        pred_f, pp = parse_pred_expr(m.group(2), "x")
        if pred_f:
            inner = Not(pred_f) if neg else pred_f
            return Exists("x", inner), pp

    # ── Conditional ──

    # "if someone/something is [not] X, then that person/thing is Y"
    m = re.match(
        r'^if\s+(?:someone|something)\s+is\s+(not\s+)?(?:a\s+|an\s+)?(.+?)'
        r',?\s+then\s+that\s+(?:person|thing)\s+is\s+(.+)$',
        lower,
    )
    if m:
        neg = m.group(1) is not None
        ant_f, ap = parse_pred_expr(m.group(2), "x")
        cons_f, cp = parse_pred_expr(m.group(3), "x")
        if ant_f and cons_f:
            ant = Not(ant_f) if neg else ant_f
            return ForAll("x", Implies(ant, cons_f)), ap | cp

    # ── Individual assertions ──

    QUANTIFIER_WORDS = frozenset([
        'every', 'whoever', 'whatever', 'being', 'not', 'no', 'nobody',
        'nothing', 'some', 'there', 'it', 'everyone', 'everything', 'if',
        'somebody', 'to',
    ])

    # "NAME is not Y"
    m = re.match(r'^(.+?)\s+is\s+not\s+(.+)$', text, re.IGNORECASE)
    if m:
        name = m.group(1).strip()
        first_word = name.split()[0].lower() if name.split() else ""
        if first_word not in QUANTIFIER_WORDS:
            pred_f, pp = parse_pred_expr(m.group(2), name.lower())
            if pred_f:
                return Not(pred_f), pp

    # "NAME is [a|an] Y"
    m = re.match(r'^(.+?)\s+is\s+(?:a\s+|an\s+)?(.+)$', text, re.IGNORECASE)
    if m:
        name = m.group(1).strip()
        first_word = name.split()[0].lower() if name.split() else ""
        if first_word not in QUANTIFIER_WORDS:
            pred_f, pp = parse_pred_expr(m.group(2), name.lower())
            if pred_f:
                return pred_f, pp

    # "X is none of this: A or B"
    m = re.match(
        r'^(.+?)\s+is\s+none\s+of\s+this:\s*(.+)$', text, re.IGNORECASE,
    )
    if m:
        name = m.group(1).strip().lower()
        parts = re.split(r'\s+or\s+', m.group(2))
        formulas, all_preds = [], set()
        for part in parts:
            f, p = parse_pred_atom(part.strip(), name)
            if f:
                formulas.append(Not(f))
                all_preds |= p
        if formulas:
            result = formulas[0]
            for f in formulas[1:]:
                result = And(result, f)
            return result, all_preds

    # "whoever/whatever is none of this: A or B, is Y"
    m = re.match(
        r'^(?:whoever|whatever)\s+is\s+none\s+of\s+this:\s*(.+?)'
        r',?\s+is\s+(?:also\s+|however\s+)?(.+?)(?:,\s*too)?$',
        lower,
    )
    if m:
        parts = re.split(r'\s+or\s+', m.group(1))
        formulas, all_preds = [], set()
        for part in parts:
            f, p = parse_pred_atom(part.strip(), "x")
            if f:
                formulas.append(Not(f))
                all_preds |= p
        cons_f, cp = parse_pred_expr(m.group(2), "x")
        if formulas and cons_f:
            ant = formulas[0]
            for f in formulas[1:]:
                ant = And(ant, f)
            return ForAll("x", Implies(ant, cons_f)), all_preds | cp

    return None, set()


# ════════════════════════════════════════════════════════════════════════
# Argument extraction & splitting
# ════════════════════════════════════════════════════════════════════════

PREAMBLE_PATS = [
    r"Here comes a perfectly valid argument:\s*",
    r"The following argument pertains to this question:\s*",
    r"The following argument seeks to clarify some such relations:\s*",
    r"Consider the following argument:\s*",
]

PREMISE_MARKERS = [
    "First of all,", "First,", "First premise:", "To start with,",
    "To begin with,", "Second,", "Second premise:", "Third,",
    "Third premise:", "Fourth,", "Fourth premise:",
    "Moreover,", "Next,", "Plus,", "Now,", "Finally,", "Additionally,",
]

CONCLUSION_MARKERS = [
    "We may conclude:", "We may conclude that",
    "So, necessarily,", "Therefore,", "It follows that",
    "Hence,", "In consequence,", "All this entails that",
    "From this follows:", "It must be true that",
    "This means that", "Drawing on these propositions, we get that",
]

# Keyword whitelist for sentence splitting
_SENT_STARTERS = (
    r'(?:Every|No|Nobody|Nothing|Whoever|Whatever|Being|Not\s+being|'
    r'There|It\s+is|Some|Everyone|Everybody|Everything|If\s+someone|'
    r'If\s+something|To\s+be|Somebody|Someone|Whatever)'
)
_SENT_SPLIT_PAT = rf'\.\s+(?={_SENT_STARTERS}\b)'


def extract_and_split(text):
    """Extract argument, split into (premise_sentences, conclusion_sentence).

    Returns (list[str], str) or (None, None) on failure.
    """
    m = re.search(r'"(.+?)"', text, re.DOTALL)
    if not m:
        return None, None
    arg = m.group(1)

    # Fix missing-space issues (e.g., "Barton.therefore" → "Barton. Therefore")
    arg = re.sub(
        r'\.([a-z])', lambda m: '. ' + m.group(1).upper(), arg,
    )

    # Strip preamble
    preamble_end = 0
    for pat in PREAMBLE_PATS:
        m2 = re.search(pat, arg)
        if m2 and m2.end() > preamble_end:
            preamble_end = m2.end()
    if preamble_end > 0:
        arg = arg[preamble_end:].strip()

    # Split off conclusion
    conclusion = None
    premises_text = arg
    for marker in CONCLUSION_MARKERS:
        pat = re.escape(marker)
        m3 = re.search(pat, arg, re.IGNORECASE)
        if m3:
            premises_text = arg[:m3.start()].strip()
            conclusion = arg[m3.end():].strip().rstrip('.')
            break

    if conclusion is None:
        return None, None

    # Split premises on markers
    ptext = premises_text
    for marker in PREMISE_MARKERS:
        ptext = ptext.replace(marker, "|||")
    parts = [p.strip().rstrip('.') for p in ptext.split("|||") if p.strip()]

    has_markers = any(marker in premises_text for marker in PREMISE_MARKERS)
    if not has_markers:
        parts = [
            p.strip().rstrip('.')
            for p in re.split(_SENT_SPLIT_PAT, premises_text)
            if p.strip()
        ]
    else:
        expanded = []
        for part in parts:
            sents = re.split(_SENT_SPLIT_PAT, part)
            expanded.extend(
                [s.strip().rstrip('.') for s in sents if s.strip()],
            )
        parts = expanded

    return parts, conclusion


# ════════════════════════════════════════════════════════════════════════
# Main solver
# ════════════════════════════════════════════════════════════════════════

def _parse_one(text, classifier=None):
    """Parse a single sentence, trying probe-dispatch first then regex fallback.

    classifier: callable(sentence) → label, or None for regex classify_pattern.
    """
    label = classifier(text) if classifier else classify_pattern(text)
    f, preds = probe_parse_sentence(text, label)
    if f is not None:
        return f, preds
    # Fallback to full regex cascade
    return parse_sentence(text)


def solve_formal_fallacy(text, classifier=None):
    """Solve a formal_fallacies example.

    Args:
        text: The full BBH example input text.
        classifier: Optional callable(sentence) → label for sentence type
            classification. Defaults to regex-based classify_pattern().
            A trained probe can be passed here for probe-dispatched parsing.

    Returns 'valid', 'invalid', or None (parse failure).
    """
    premise_texts, conclusion_text = extract_and_split(text)
    if premise_texts is None or conclusion_text is None:
        return None

    # Parse premises (with aggressive split retry on failure)
    premise_formulas = []
    for pt in premise_texts:
        f, preds = _parse_one(pt, classifier)

        # Check for suspicious parse: predicate names containing ". "
        suspicious = f is not None and any('. ' in p for p in preds)

        if f is None or suspicious:
            subs = re.split(r'\.\s+(?=[A-Z])', pt)
            if len(subs) > 1:
                retry_ok = True
                retry_formulas = []
                for sub in subs:
                    sub = sub.strip().rstrip('.')
                    if not sub:
                        continue
                    sf, _ = _parse_one(sub, classifier)
                    if sf is None:
                        retry_ok = False
                        break
                    retry_formulas.append(sf)
                if retry_ok and retry_formulas:
                    premise_formulas.extend(retry_formulas)
                    continue
            if f is not None:
                premise_formulas.append(f)
                continue
            return None
        premise_formulas.append(f)

    # Parse conclusion
    conclusion_formula, _ = _parse_one(conclusion_text, classifier)
    if conclusion_formula is None:
        return None

    # Check validity via model enumeration
    valid = check_validity(premise_formulas, conclusion_formula)
    return "valid" if valid else "invalid"


# ════════════════════════════════════════════════════════════════════════
# Turnstyle
# ════════════════════════════════════════════════════════════════════════

class FormalFallaciesTurnstyle(Turnstyle):
    """Grounds logical validity in FOL model enumeration.

    Parses NL syllogistic arguments into first-order logic, checks
    validity by enumerating all interpretations. Biases generation
    toward "valid" or "invalid".

        t = FormalFallaciesTurnstyle(model, tokenizer, device)
        text, proof = t.generate(prompt)
    """

    probe_label = "formal_fallacies"
    examples = [
        "Consider the following argument: Every elephant is a mammal. Some elephants are gray. Therefore, some mammals are gray. Is the argument valid or invalid?\nOptions:\n(A) valid\n(B) invalid",
        "Consider the following argument: All birds have wings. No penguins can fly. Therefore, some birds cannot fly. Is the argument valid or invalid?\nOptions:\n(A) valid\n(B) invalid",
        "Consider the following argument: All roses are flowers. All flowers need water. Therefore, all roses need water. Is the argument valid or invalid?\nOptions:\n(A) valid\n(B) invalid",
        "Consider the following argument: No cats are dogs. Some pets are cats. Therefore, some pets are not dogs. Is the argument valid or invalid?\nOptions:\n(A) valid\n(B) invalid",
        "Consider the following argument: All A are B. All B are C. Therefore, all A are C. Is the argument valid or invalid?\nOptions:\n(A) valid\n(B) invalid",
        "Consider the following argument: Some students are athletes. All athletes are healthy. Therefore, all students are healthy. Is the argument valid or invalid?\nOptions:\n(A) valid\n(B) invalid",
        "Consider the following argument: All mammals are warm-blooded. Whales are mammals. Therefore, whales are warm-blooded. Is the argument valid or invalid?\nOptions:\n(A) valid\n(B) invalid",
        "Consider the following argument: No fish are mammals. All sharks are fish. Therefore, no sharks are mammals. Is the argument valid or invalid?\nOptions:\n(A) valid\n(B) invalid",
        "Consider the following argument: All teachers are educated. Some educated people are poor. Therefore, some teachers are poor. Is the argument valid or invalid?\nOptions:\n(A) valid\n(B) invalid",
        "Consider the following argument: Every square is a rectangle. Every rectangle has four sides. Therefore, every square has four sides. Is the argument valid or invalid?\nOptions:\n(A) valid\n(B) invalid",
        "Consider the following argument: All politicians are liars. Some people are politicians. Therefore, some people are liars. Is the argument valid or invalid?\nOptions:\n(A) valid\n(B) invalid",
        "Consider the following argument: No reptile is warm-blooded. All snakes are reptiles. Therefore, no snake is warm-blooded. Is the argument valid or invalid?\nOptions:\n(A) valid\n(B) invalid",
        "Consider the following argument: Some cars are electric. All electric vehicles are eco-friendly. Therefore, some cars are eco-friendly. Is the argument valid or invalid?\nOptions:\n(A) valid\n(B) invalid",
        "Consider the following argument: All programmers use computers. Some teachers use computers. Therefore, some teachers are programmers. Is the argument valid or invalid?\nOptions:\n(A) valid\n(B) invalid",
        "Consider the following argument: All metals conduct electricity. Gold is a metal. Therefore, gold conducts electricity. Is the argument valid or invalid?\nOptions:\n(A) valid\n(B) invalid",
        "Consider the following argument: Some doctors are surgeons. All surgeons wear gloves. Therefore, some doctors wear gloves. Is the argument valid or invalid?\nOptions:\n(A) valid\n(B) invalid",
        "Consider the following argument: No vegetarian eats meat. All vegans are vegetarians. Therefore, no vegan eats meat. Is the argument valid or invalid?\nOptions:\n(A) valid\n(B) invalid",
        "Consider the following argument: All diamonds are hard. Some jewels are diamonds. Therefore, all jewels are hard. Is the argument valid or invalid?\nOptions:\n(A) valid\n(B) invalid",
        "Consider the following argument: Every prime number greater than 2 is odd. 7 is a prime number greater than 2. Therefore, 7 is odd. Is the argument valid or invalid?\nOptions:\n(A) valid\n(B) invalid",
        "Consider the following argument: Some athletes are tall. All tall people play basketball. Therefore, some athletes play basketball. Is the argument valid or invalid?\nOptions:\n(A) valid\n(B) invalid",
        "Consider the following argument: All humans are mortal. Socrates is a human. Therefore, Socrates is mortal. Is the argument valid or invalid?\nOptions:\n(A) valid\n(B) invalid",
        "Consider the following argument: No insect has a backbone. Spiders are insects. Therefore, spiders have no backbone. Is the argument valid or invalid?\nOptions:\n(A) valid\n(B) invalid",
        "Consider the following argument: All nurses are compassionate. Some compassionate people are good listeners. Therefore, some nurses are good listeners. Is the argument valid or invalid?\nOptions:\n(A) valid\n(B) invalid",
        "Consider the following argument: Some books are bestsellers. All bestsellers are popular. Therefore, some books are popular. Is the argument valid or invalid?\nOptions:\n(A) valid\n(B) invalid",
        "Consider the following argument: All computers need power. All phones are computers. Therefore, all phones need power. Is the argument valid or invalid?\nOptions:\n(A) valid\n(B) invalid",
        "Consider the following argument: No herbivore eats meat. All deer are herbivores. Therefore, no deer eat meat. Is the argument valid or invalid?\nOptions:\n(A) valid\n(B) invalid",
        "Consider the following argument: Some plants are poisonous. All poisonous plants are dangerous. Therefore, all plants are dangerous. Is the argument valid or invalid?\nOptions:\n(A) valid\n(B) invalid",
        "Consider the following argument: All triangles have three sides. All equilateral shapes are triangles. Therefore, all equilateral shapes have three sides. Is the argument valid or invalid?\nOptions:\n(A) valid\n(B) invalid",
        "Consider the following argument: Some leaders are corrupt. All corrupt people abuse power. Therefore, some leaders abuse power. Is the argument valid or invalid?\nOptions:\n(A) valid\n(B) invalid",
        "Consider the following argument: Every whale is a mammal. Every mammal breathes air. Therefore, every whale breathes air. Is the argument valid or invalid?\nOptions:\n(A) valid\n(B) invalid",
    ]

    def parse(self, prompt: str):
        """Parse argument and compute validity.

        Returns (summary, answer_str) or None.
        """
        answer = solve_formal_fallacy(prompt)
        if answer is None:
            return None
        # Build a compact summary for diagnostics
        premise_texts, _ = extract_and_split(prompt)
        n_premises = len(premise_texts) if premise_texts else 0
        summary = f"{n_premises} premises → {answer}"
        return summary, answer

    def make_processor(self, parsed, max_new_tokens: int):
        summary, answer_str = parsed
        answer_ids = self.tokenizer.encode(
            answer_str, add_special_tokens=False,
        )
        return SequenceLogitsProcessor(
            self.tokenizer,
            answer_ids,
            expression=summary,
            answer_str=answer_str,
            bias_strength=self.bias_strength,
            max_new_tokens=max_new_tokens,
        )
