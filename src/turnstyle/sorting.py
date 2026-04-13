"""Word sorting turnstyle — grounds alphabetical sorting in exact computation.

Handles:
    "Sort the following words: banana apple cherry"
    "Sort [cherry, banana, apple] alphabetically"
"""

from __future__ import annotations

import re

from turnstyle.core import SequenceLogitsProcessor, Turnstyle
from turnstyle.extract import ExtractionSpec, FieldSpec


def parse_sorting(text: str) -> tuple[list[str], list[str], str] | None:
    """Extract a word list from text and return sorted version.

    Returns (original_words, sorted_words, sorted_str) or None.
    """
    lower = text.lower()

    # "Sort the following words: banana apple cherry"
    # "Sort [cherry, banana, apple]"
    # "Sort: banana, apple, cherry"
    m = re.search(
        r'sort(?:ed)?(?:\s+(?:the\s+)?(?:following\s+)?(?:words|list))?'
        r'[\s:]*\[?([a-z][a-z, ]+[a-z])\]?',
        lower,
    )
    if not m:
        return None

    raw = m.group(1)
    # Split on commas and/or spaces
    words = [w.strip() for w in re.split(r'[,\s]+', raw) if w.strip()]

    if len(words) < 2:
        return None

    sorted_words = sorted(words)
    sorted_str = " ".join(sorted_words)
    return words, sorted_words, sorted_str


def _assemble_sorting(fields: dict) -> tuple[list[str], list[str], str]:
    """Assemble sorting extraction fields into parse() tuple format."""
    raw = fields["words"]
    words = [w.strip() for w in re.split(r'[,\s]+', raw) if w.strip()]
    if len(words) < 2:
        raise ValueError("Need at least 2 words to sort")
    sorted_words = sorted(words)
    return words, sorted_words, " ".join(sorted_words)


SORTING_EXTRACTION_SPEC = ExtractionSpec(
    fields=[
        FieldSpec(
            name="words",
            prompt_template=(
                "Extract the list of words to sort from this text. "
                "Return only the words, comma-separated.\n"
                "Text: {input}\nWords:"
            ),
        ),
    ],
    assemble=_assemble_sorting,
)


class SortingTurnstyle(Turnstyle):
    """Grounds alphabetical sorting in exact computation.

        t = SortingTurnstyle(model, tokenizer, device)
        text, proof = t.generate("Sort the following words: banana apple cherry")
    """

    probe_label = "sorting"
    extraction_spec = SORTING_EXTRACTION_SPEC
    examples = [
        'Sort the following words alphabetically: List: syndrome therefrom',
        'Sort the following words alphabetically: List: thrill splutter panicking scorch same dot prod obstetric malton onus drumhead delmarva barn embezzle it&t damp guru subsist entirety greene',
        'Sort the following words alphabetically: List: vegetate artillery harm fda doris prosody bainite incongruous monkey vivian',
        'Sort the following words alphabetically: List: sioux fortescue purloin percept helmsman',
        'Sort the following words alphabetically: List: indifferent trainman bootlegging',
        'Sort the following words alphabetically: List: conference apparition ignore dutton layperson coupe superstitious westward turnoff messenger copra floruit primitive implement',
        'Sort the following words alphabetically: List: covalent spiderwort horowitz divisive spiritual cheshire affluent gideon quadrature julio peanut epsilon diagnostician grover folklore gothic salient',
        'Sort the following words alphabetically: List: euclidean stonehenge hobby cloudy winsome invite thrifty fight majestic citrus surge scene',
        'Sort the following words alphabetically: List: thunderclap swab built poland',
        'Sort the following words alphabetically: List: regret starlight wallboard cotyledon more pepperoni',
        'Sort the following words alphabetically: List: burley bela arapaho bacteria bock',
        'Sort the following words alphabetically: List: scrumptious sidereal thermal yakima siena gorky saxon scottish figural hydroxyl seventeen neapolitan rampage nerve grapple fate plainfield stooge knives allotted',
        'Sort the following words alphabetically: List: lucrative you\'ve tunnel archery bride coquette polytypy barbudo radix arlen lockwood teem officious',
        'Sort the following words alphabetically: List: gentle boletus galveston aniline eddy fontainebleau wile scandalous skat sportsmen',
        'Sort the following words alphabetically: List: crowfoot scrupulous campfire contrast purgatory',
        'Sort the following words alphabetically: List: bare census intrinsic torch timeout infirm humility snagging exaltation patristic paregoric gnomon moth sorrowful manatee oblique stressful',
        'Sort the following words alphabetically: List: marlborough pyrotechnic filly grocer treadle transitive platelet deliver landau hotbox uncle siemens anger hessian gneiss convoy ninebark advent plat stapleton',
        'Sort the following words alphabetically: List: filamentous semaphore bulrush audacious xylophone sensate municipal harris intervenor battleground rubicund',
        'Sort the following words alphabetically: List: county quantify nail o\'connell phony bauer poole venice accelerate nominee raisin putnam',
        'Sort the following words alphabetically: List: bituminous ami decadent knickerbocker exeter',
        'Sort the following words alphabetically: List: slurp raytheon gloucester',
        'Sort the following words alphabetically: List: chlorate glidden incentive manatee spurt lavoisier judicatory',
        'Sort the following words alphabetically: List: shouldn\'t lorenz runneth skintight plastisol swept coven etruscan disturb',
        'Sort the following words alphabetically: List: shreveport gamut berg multiplexor bluish puerto subliminal',
        'Sort the following words alphabetically: List: daffy hypothesis croupier dockyard household peccary triode minstrelsy nepotism sawtimber mantic info confess serenade summate silver duty loam mandate',
        'Sort the following words alphabetically: List: champ jigsaw acclaim pipeline exempt gadwall hypothalamus clothbound sensory lozenge hayes conclusion delirious dyestuff hood seashell commodity plentiful sarcastic teen',
        'Sort the following words alphabetically: List: dynastic inflammable prick tristan vitiate tackle stagnate conglomerate nebulae phosphide',
        'Sort the following words alphabetically: List: dateline jill langmuir pipette household',
        'Sort the following words alphabetically: List: tip abo frond indistinguishable stockholder gasify passenger bonaventure armful oscillatory referential guiana pancreatic through embryology herman dictatorial cremate',
        'Sort the following words alphabetically: List: heterostructure bertrand careful wherewith libra eyelid feign southeastern paste snip',
    ]

    def parse(self, prompt: str):
        return None  # routing via probe, fields via extraction

    def make_processor(self, parsed, max_new_tokens: int):
        original_words, sorted_words, sorted_str = parsed
        answer_ids = self.tokenizer.encode(sorted_str, add_special_tokens=False)
        expression = " ".join(original_words)
        return SequenceLogitsProcessor(
            self.tokenizer, answer_ids, expression=expression,
            answer_str=sorted_str, bias_strength=self.bias_strength,
            max_new_tokens=max_new_tokens)
