"""Object tracking turnstyle — grounds shuffled-object tracking in state simulation.

Extracts initial object assignments and swap actions via schema extraction, then
simulates the state transitions deterministically.

Extraction schema
-----------------
init  → {"ActorName": "item", ...}   — one entry per actor
swap  → {"actor1": "Name", "actor2": "Name"}
query → {"ask": "Name"}

Handles BBH tracking_shuffled_objects_three/five/seven_objects.
"""

from __future__ import annotations

import re

from turnstyle.core import SequenceLogitsProcessor, Turnstyle
from turnstyle.ir import SentenceIRSpec, SentenceRecord


# ── closed actor set (structural) ────────────────────────────────────────────

ALL_ACTORS = ["Alice", "Bob", "Claire", "Dave", "Eve", "Fred", "Gertrude"]


# ── few-shot extraction prompt ────────────────────────────────────────────────

_EXTRACT_PROMPT = """\
Extract object tracking information as JSON.

init  → object mapping each actor to their item: {{"ActorName": "item", ...}}
swap  → {{"actor1": "Name", "actor2": "Name"}}
query → {{"ask": "Name"}}
preamble → null

Examples:
sentence: At the start of the day, Alice has a yellow ball, Bob has a white ball, and Claire has a green ball.
type: init
{{"Alice": "yellow ball", "Bob": "white ball", "Claire": "green ball"}}

sentence: At the start of the day, Alice has a piano, Bob has a guitar, Claire has a violin, Dave has a flute, and Eve has a drum.
type: init
{{"Alice": "piano", "Bob": "guitar", "Claire": "violin", "Dave": "flute", "Eve": "drum"}}

sentence: At the start of the day, Alice is playing tennis, Bob is playing soccer, and Claire is playing basketball.
type: init
{{"Alice": "tennis", "Bob": "soccer", "Claire": "basketball"}}

sentence: Then Alice and Bob swap balls.
type: swap
{{"actor1": "Alice", "actor2": "Bob"}}

sentence: Then Bob and Claire trade presents.
type: swap
{{"actor1": "Bob", "actor2": "Claire"}}

sentence: Then Dave and Eve switch hats.
type: swap
{{"actor1": "Dave", "actor2": "Eve"}}

sentence: At the end of the day, what color ball does Alice have?
type: query
{{"ask": "Alice"}}

sentence: At the end of the day, what does Bob have?
type: query
{{"ask": "Bob"}}

sentence: Alice, Bob, and Claire each have a ball.
type: preamble
null

sentence: {sentence}
type: {type}
"""


# ── sentence classifier (structural) ─────────────────────────────────────────


def _classify_tracking(sentence: str) -> str:
    if "?" in sentence:
        return "query"
    if re.search(r"\bstart\b", sentence, re.I):
        return "init"
    actor_count = sum(1 for a in ALL_ACTORS if re.search(rf"\b{a}\b", sentence))
    return "swap" if actor_count == 2 else "preamble"


# ── aggregator ────────────────────────────────────────────────────────────────


def _aggregate_tracking(
    records: list[SentenceRecord],
    options: dict[str, str],
) -> str | None:
    """Simulate object swaps from extracted records, match final state to options."""
    state: dict[str, str] = {}
    swaps: list[tuple[str, str]] = []
    query_actor: str | None = None

    for rec in records:
        d = rec.data
        if not isinstance(d, dict):
            continue
        if rec.record_type == "init":
            for actor, item in d.items():
                if isinstance(item, str):
                    state[actor] = item.strip().rstrip(".,")
        elif rec.record_type == "swap":
            if "actor1" in d and "actor2" in d:
                swaps.append((str(d["actor1"]).strip(), str(d["actor2"]).strip()))
        elif rec.record_type == "query":
            if "ask" in d:
                query_actor = str(d["ask"]).strip()

    if not state or query_actor is None:
        return None

    for a1, a2 in swaps:
        if a1 in state and a2 in state:
            state[a1], state[a2] = state[a2], state[a1]

    if query_actor not in state:
        return None

    answer_val = state[query_actor].lower()
    for letter, opt in options.items():
        if opt.lower() in answer_val or answer_val in opt.lower():
            return f"({letter})"

    return None


# ── SentenceIRSpec ────────────────────────────────────────────────────────────

OBJECT_TRACKING_SPEC = SentenceIRSpec(
    sentence_types=["init", "swap", "query", "preamble"],
    extract_prompt=_EXTRACT_PROMPT,
    aggregate=_aggregate_tracking,
    classify_fn=_classify_tracking,
    max_tokens=100,
)


# ── Turnstyle subclass ────────────────────────────────────────────────────────


class ObjectTrackingTurnstyle(Turnstyle):
    """Grounds object-tracking tasks in deterministic state simulation.

    LLM extracts actor→item assignments (init) and actor pairs (swap) per
    sentence; Python simulates the swap chain and matches the final state
    to the correct option letter.

    Handles BBH tracking_shuffled_objects_three/five/seven_objects.

        t = ObjectTrackingTurnstyle(model, tokenizer, device)
        text, proof = t.generate("Alice, Bob, and Claire each have a ball. At the start...")
    """

    probe_label = "object_tracking"
    sentence_ir_spec = OBJECT_TRACKING_SPEC
    examples = [
        "Alice, Bob, and Claire each have a ball. At the start of the day, Alice has a yellow ball, Bob has a white ball, and Claire has a green ball. Then Alice and Bob swap balls. At the end of the day, what color ball does Alice have?\nOptions:\n(A) yellow\n(B) white\n(C) green",
        "Alice, Bob, and Claire each have a ball. At the start of the day, Alice has a red ball, Bob has a blue ball, and Claire has a purple ball. Then Bob and Claire swap balls. At the end of the day, what color ball does Bob have?\nOptions:\n(A) red\n(B) blue\n(C) purple",
        "Alice, Bob, and Claire each have a ball. At the start of the day, Alice has a green ball, Bob has a orange ball, and Claire has a pink ball. Then Alice and Claire swap balls. Then Alice and Bob swap balls. At the end of the day, what color ball does Alice have?\nOptions:\n(A) green\n(B) orange\n(C) pink",
        "Alice, Bob, Claire, Dave, and Eve each have a toy. At the start of the day, Alice has a piano, Bob has a guitar, Claire has a violin, Dave has a flute, and Eve has a drum. Then Alice and Bob swap toys. Then Claire and Dave swap toys. At the end of the day, what does Alice have?\nOptions:\n(A) piano\n(B) guitar\n(C) violin\n(D) flute\n(E) drum",
        "Alice, Bob, and Claire each get a present. At the start of the day, Alice has a doll, Bob has a robot, and Claire has a ball. Then Bob and Claire swap presents. At the end of the day, what does Claire have?\nOptions:\n(A) doll\n(B) robot\n(C) ball",
        "Alice, Bob, and Claire each have a pet. At the start of the day, Alice has a cat, Bob has a dog, and Claire has a bird. Then Alice and Bob swap pets. Then Bob and Claire swap pets. At the end of the day, what pet does Bob have?\nOptions:\n(A) cat\n(B) dog\n(C) bird",
        "Alice, Bob, Claire, Dave, and Eve each have a book. At the start of the day, Alice has a novel, Bob has a comic, Claire has a textbook, Dave has a magazine, and Eve has a dictionary. Then Bob and Dave trade books. At the end of the day, what book does Dave have?\nOptions:\n(A) novel\n(B) comic\n(C) textbook\n(D) magazine\n(E) dictionary",
        "Alice, Bob, and Claire each have a hat. At the start of the day, Alice has a red hat, Bob has a blue hat, and Claire has a green hat. Then Alice and Claire swap hats. Then Bob and Claire swap hats. At the end of the day, what color hat does Claire have?\nOptions:\n(A) red\n(B) blue\n(C) green",
        "Alice, Bob, Claire, Dave, and Eve each have a coin. At the start of the day, Alice has a penny, Bob has a nickel, Claire has a dime, Dave has a quarter, and Eve has a half-dollar. Then Alice and Eve switch coins. At the end of the day, what coin does Alice have?\nOptions:\n(A) penny\n(B) nickel\n(C) dime\n(D) quarter\n(E) half-dollar",
        "Alice, Bob, and Claire each have a fruit. At the start of the day, Alice has an apple, Bob has a banana, and Claire has a cherry. Then Alice and Bob swap fruits. Then Alice and Claire swap fruits. At the end of the day, what fruit does Alice have?\nOptions:\n(A) apple\n(B) banana\n(C) cherry",
        "Alice, Bob, and Claire each have a gem. At the start of the day, Alice has a ruby, Bob has an emerald, and Claire has a sapphire. Then Bob and Claire trade gems. Then Alice and Bob trade gems. At the end of the day, what gem does Bob have?\nOptions:\n(A) ruby\n(B) emerald\n(C) sapphire",
        "Alice, Bob, Claire, Dave, and Eve each have a card. At the start of the day, Alice has a hearts card, Bob has a diamonds card, Claire has a clubs card, Dave has a spades card, and Eve has a joker. Then Claire and Eve swap cards. At the end of the day, what card does Eve have?\nOptions:\n(A) hearts\n(B) diamonds\n(C) clubs\n(D) spades\n(E) joker",
        "Alice, Bob, and Claire each have a shirt. At the start of the day, Alice has a striped shirt, Bob has a solid shirt, and Claire has a plaid shirt. Then Alice and Claire swap shirts. At the end of the day, what shirt does Claire have?\nOptions:\n(A) striped\n(B) solid\n(C) plaid",
        "Alice, Bob, Claire, Dave, and Eve each have a tool. At the start of the day, Alice has a hammer, Bob has a screwdriver, Claire has a wrench, Dave has a drill, and Eve has a saw. Then Dave and Eve swap tools. Then Bob and Dave swap tools. At the end of the day, what tool does Dave have?\nOptions:\n(A) hammer\n(B) screwdriver\n(C) wrench\n(D) drill\n(E) saw",
        "Alice, Bob, and Claire each have a car. At the start of the day, Alice has a sedan, Bob has a truck, and Claire has a van. Then Bob and Claire trade cars. At the end of the day, what car does Claire have?\nOptions:\n(A) sedan\n(B) truck\n(C) van",
        "Alice, Bob, Claire, Dave, and Eve each have a plant. At the start of the day, Alice has a rose, Bob has a tulip, Claire has a lily, Dave has a daisy, and Eve has a sunflower. Then Alice and Bob swap plants. Then Claire and Eve swap plants. At the end of the day, what plant does Claire have?\nOptions:\n(A) rose\n(B) tulip\n(C) lily\n(D) daisy\n(E) sunflower",
        "Alice, Bob, and Claire each have a food. At the start of the day, Alice has a sandwich, Bob has a salad, and Claire has a soup. Then Alice and Bob switch foods. Then Bob and Claire switch foods. At the end of the day, what food does Claire have?\nOptions:\n(A) sandwich\n(B) salad\n(C) soup",
        "Alice, Bob, Claire, Dave, and Eve each have a key. At the start of the day, Alice has a gold key, Bob has a silver key, Claire has a bronze key, Dave has a iron key, and Eve has a copper key. Then Alice and Claire swap keys. At the end of the day, what key does Alice have?\nOptions:\n(A) gold\n(B) silver\n(C) bronze\n(D) iron\n(E) copper",
        "Alice, Bob, and Claire each have a drink. At the start of the day, Alice has coffee, Bob has tea, and Claire has juice. Then Alice and Claire swap drinks. At the end of the day, what drink does Alice have?\nOptions:\n(A) coffee\n(B) tea\n(C) juice",
        "Alice, Bob, Claire, Dave, and Eve each have a flag. At the start of the day, Alice has a red flag, Bob has a blue flag, Claire has a green flag, Dave has a yellow flag, and Eve has a white flag. Then Bob and Eve trade flags. Then Alice and Dave trade flags. At the end of the day, what flag does Eve have?\nOptions:\n(A) red\n(B) blue\n(C) green\n(D) yellow\n(E) white",
        "Alice, Bob, and Claire each have a musical instrument. At the start of the day, Alice has a piano, Bob has a guitar, and Claire has a drum. Then Bob and Claire swap instruments. Then Alice and Bob swap instruments. At the end of the day, what instrument does Bob have?\nOptions:\n(A) piano\n(B) guitar\n(C) drum",
        "Alice, Bob, Claire, Dave, and Eve each have a vegetable. At the start of the day, Alice has a carrot, Bob has a celery, Claire has a broccoli, Dave has a spinach, and Eve has a kale. Then Alice and Eve switch vegetables. Then Bob and Claire switch vegetables. At the end of the day, what vegetable does Bob have?\nOptions:\n(A) carrot\n(B) celery\n(C) broccoli\n(D) spinach\n(E) kale",
        "Alice, Bob, and Claire each have a ball. At the start of the day, Alice has a soccer ball, Bob has a basketball, and Claire has a tennis ball. Then Bob and Alice trade balls. Then Claire and Bob trade balls. At the end of the day, what ball does Bob have?\nOptions:\n(A) soccer ball\n(B) basketball\n(C) tennis ball",
        "Alice, Bob, Claire, Dave, and Eve each have a ribbon. At the start of the day, Alice has a yellow ribbon, Bob has a red ribbon, Claire has a blue ribbon, Dave has a green ribbon, and Eve has a purple ribbon. Then Alice and Bob swap ribbons. At the end of the day, what ribbon does Bob have?\nOptions:\n(A) yellow\n(B) red\n(C) blue\n(D) green\n(E) purple",
        "Alice, Bob, and Claire each have a jacket. At the start of the day, Alice has a leather jacket, Bob has a denim jacket, and Claire has a wool jacket. Then Alice and Bob swap jackets. Then Alice and Claire swap jackets. At the end of the day, what jacket does Alice have?\nOptions:\n(A) leather\n(B) denim\n(C) wool",
        "Alice, Bob, Claire, Dave, and Eve each have a sport. At the start of the day, Alice is playing tennis, Bob is playing soccer, Claire is playing basketball, Dave is playing baseball, and Eve is playing golf. Then Dave and Eve switch sports. Then Bob and Dave switch sports. At the end of the day, what sport is Bob playing?\nOptions:\n(A) tennis\n(B) soccer\n(C) basketball\n(D) baseball\n(E) golf",
        "Alice, Bob, and Claire each have a snack. At the start of the day, Alice has chips, Bob has cookies, and Claire has crackers. Then Alice and Claire swap snacks. At the end of the day, what snack does Alice have?\nOptions:\n(A) chips\n(B) cookies\n(C) crackers",
        "Alice, Bob, Claire, Dave, and Eve each have a stone. At the start of the day, Alice has a marble, Bob has a granite, Claire has a limestone, Dave has a sandstone, and Eve has a slate. Then Claire and Dave swap stones. Then Alice and Claire swap stones. At the end of the day, what stone does Claire have?\nOptions:\n(A) marble\n(B) granite\n(C) limestone\n(D) sandstone\n(E) slate",
        "Alice, Bob, and Claire each have a dessert. At the start of the day, Alice has a cake, Bob has a pie, and Claire has a brownie. Then Bob and Claire trade desserts. Then Alice and Bob trade desserts. At the end of the day, what dessert does Alice have?\nOptions:\n(A) cake\n(B) pie\n(C) brownie",
        "Alice, Bob, Claire, Dave, and Eve each have a bag. At the start of the day, Alice has a backpack, Bob has a briefcase, Claire has a purse, Dave has a suitcase, and Eve has a tote. Then Alice and Dave swap bags. At the end of the day, what bag does Dave have?\nOptions:\n(A) backpack\n(B) briefcase\n(C) purse\n(D) suitcase\n(E) tote",
    ]

    def parse(self, prompt: str):  # noqa: ARG002
        """No regex fast path — all solving via sentence_ir_spec."""
        return None

    def make_processor(self, parsed, max_new_tokens: int):
        (answer_letter,) = parsed
        answer_ids = self.tokenizer.encode(answer_letter, add_special_tokens=False)
        return SequenceLogitsProcessor(
            self.tokenizer, answer_ids, expression="object_tracking",
            answer_str=answer_letter, bias_strength=self.bias_strength,
            max_new_tokens=max_new_tokens)
