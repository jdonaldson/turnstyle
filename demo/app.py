"""Turnstyle demo — SmolLM2 vanilla vs. SmolLM2 + Turnstyle, side by side.

A neurosymbolic wrapper: the model generates, but turnstyle parses the prompt into a
typed Task, solves it exactly (arithmetic / dates / logic / sorting / tracking) or with
a calibrated hidden-state recognition probe (snarks / movie), and biases generation
toward that answer. The left pane is the raw model; the right pane is the same model
with turnstyle engaged, plus a worked-step proof of what it did.

Runs on a HuggingFace Space (GPU recommended; works on CPU, just slower). The bundled
SmolLM2 profile (probes) ships inside the turnstyle wheel and auto-loads by fingerprint.
"""
import os

import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from turnstyle.dispatch_turnstyle import DispatchTurnstyle

MODEL = os.environ.get("TS_MODEL", "HuggingFaceTB/SmolLM2-1.7B-Instruct")
MAX_NEW = int(os.environ.get("TS_MAX_NEW", "64"))
PROBE_SOURCES = {"choice_probe", "selection_probe", "pmi_floor"}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

print(f"loading {MODEL} on {DEVICE} ({DTYPE})…", flush=True)
TOK = AutoTokenizer.from_pretrained(MODEL)
MDL = AutoModelForCausalLM.from_pretrained(MODEL, dtype=DTYPE).to(DEVICE)
DT = DispatchTurnstyle(MDL, TOK, DEVICE)
print(f"turnstyle ready. probe tasks: {DT.profile_tasks}", flush=True)

AUTO = "(auto — symbolic / deterministic)"
# Friendly labels for the recognition-probe picker. The *value* stays the calibrated
# task key (use_probe() looks it up by that key); only the displayed label is clean.
PROBE_LABELS = {
    "snarks": "sarcasm",
    "movie_recommendation": "movie similarity",
    "ruin_names": "humorous name edit",
    "disambiguation_qa": "pronoun reference",
    "salient_translation_error_detection": "translation error type",
    "temporal_sequences": "temporal ordering",
    "date_understanding": "date selection",
}
PROBE_CHOICES = [(AUTO, AUTO)] + [(PROBE_LABELS.get(t, t), t)
                                  for t in sorted(DT.profile_tasks)]

# (prompt, probe_task). All verified to TRIP vanilla SmolLM2 (wrong answer) while
# turnstyle gets it right. Deterministic tasks use AUTO; probe tasks tag their task.
EXAMPLES = [
    ["((6 * -6 * 8 * 1) * (-1 * 7 * -6 + -2)) =", AUTO],
    ["not True or ( True or False ) is", AUTO],
    ["Complete the rest of the sequence, making sure that the parentheses are closed "
     "properly. Input: [ ] ( [ [ { < { { ( < > [ ] ) } } < > > } ] ] { }", AUTO],
    ["Sort the following words alphabetically: List: regret starlight wallboard "
     "cotyledon more pepperoni", AUTO],
    ["Alice, Bob, and Claire are friends and avid readers who occasionally trade books. "
     "At the start of the semester, they each buy one new book: Alice gets Ulysses, Bob "
     "gets Frankenstein, and Claire gets Lolita.\nAs the semester proceeds, they start "
     "trading around the new books. First, Claire and Bob swap books. Then, Bob and Alice "
     "swap books. Finally, Claire and Bob swap books. At the end of the semester, Bob has\n"
     "Options:\n(A) Ulysses\n(B) Frankenstein\n(C) Lolita", AUTO],
    ["Jane is celebrating the last day of Jan 2012. What is the date yesterday in "
     "MM/DD/YYYY?\nOptions:\n(A) 01/29/2012\n(B) 09/30/2011\n(C) 02/06/2012\n"
     "(D) 01/30/2012\n(E) 01/30/1925", AUTO],
    ["The following paragraphs each describe a set of five objects arranged in a fixed "
     "order. The statements are logically consistent within each paragraph. On a branch, "
     "there are five birds: a quail, an owl, a raven, a falcon, and a robin. The owl is "
     "the leftmost. The robin is to the left of the raven. The quail is the rightmost. "
     "The raven is the third from the left.\nOptions:\n(A) The quail is the rightmost\n"
     "(B) The owl is the rightmost\n(C) The raven is the rightmost\n(D) The falcon is the "
     "rightmost\n(E) The robin is the rightmost", AUTO],
    ["Which statement is sarcastic?\nOptions:\n(A) He's a generous person, trying to "
     "promote a charity stream that has raised millions to help kids in need\n(B) He's a "
     "terrible person, trying to promote a charity stream that has raised millions to help "
     "kids in need", AUTO],
    ["Find a movie similar to Batman, The Mask, The Fugitive, Pretty Woman:\nOptions:\n"
     "(A) The Front Page\n(B) Maelstrom\n(C) The Lion King\n(D) Lamerica", AUTO],
]


def _vanilla(prompt: str) -> str:
    text = TOK.apply_chat_template([{"role": "user", "content": prompt}],
                                   tokenize=False, add_generation_prompt=True)
    inputs = TOK(text, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = MDL.generate(**inputs, max_new_tokens=MAX_NEW, do_sample=False)
    return TOK.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def compare(prompt: str, probe_task: str):
    prompt = (prompt or "").strip()
    if not prompt:
        return "", "", "_Enter a prompt._"
    # Activate the per-task recognition probe (MC tasks), or reset to symbolic-only.
    DT.ctx.choice_artifact = None
    if probe_task and probe_task != AUTO:
        DT.use_probe(probe_task)
    vanilla = _vanilla(prompt)
    ans = DT.parse(prompt)
    grounded, _ = DT.generate(prompt, max_new_tokens=MAX_NEW)
    if ans is None:
        trace = "○ **Turnstyle abstained** — no symbolic or probe route matched; plain generation."
    elif ans.source in PROBE_SOURCES:
        # ⊨ semantic entailment: the answer is *recognized* by a hidden-state probe.
        trace = f"⊨ **recognized** via `{ans.source}` → **{ans.text}**"
        if ans.proof:
            trace += f"\n\n_{ans.proof}_"
    else:
        # ⊢ syntactic turnstile: the answer is *proved* by a deterministic solver.
        trace = f"⊢ **proved** by `{ans.source}` → **{ans.text}**"
        if ans.proof and "\n" in ans.proof:           # worked steps
            trace += f"\n\n```\n{ans.proof}\n```"
        elif ans.proof:
            trace += f"\n\n`{ans.proof}`"
    return vanilla, grounded, trace


with gr.Blocks(title="Turnstyle: neurosymbolic SmolLM2") as demo:
    gr.Markdown(
        "# ⊢ Turnstyle — a 1.7B model that stops guessing\n"
        "Same model, both sides. **Left:** raw SmolLM2-1.7B. **Right:** SmolLM2 with "
        "**Turnstyle** — it parses your prompt into a typed task, **proves** the answer with "
        "an exact solver (arithmetic, dates, logic, sorting, tracking) or **recognizes** it "
        "with a hidden-state probe (sarcasm, movie similarity), and steers the model toward "
        "it. The trace shows the worked steps."
    )
    inp = gr.Textbox(label="Prompt", lines=4,
                     placeholder="Try an arithmetic expression, a date question, a logic puzzle…")
    task = gr.Dropdown(PROBE_CHOICES, value=AUTO, label="Recognition probe (override)",
                       allow_custom_value=True,
                       info="Leave on auto — a route probe detects which recognition probe "
                            "to use (or none) from the prompt's hidden state. Override only "
                            "to force a specific one.")
    with gr.Row():
        btn = gr.Button("Compare", variant="primary")
        clr = gr.Button("Clear")
    trace = gr.Markdown()
    with gr.Row():
        with gr.Column():
            gr.Markdown("### 🤖 SmolLM2 (vanilla)")
            out_v = gr.Textbox(label="", lines=6, show_copy_button=True)
        with gr.Column():
            gr.Markdown("### ⊢ SmolLM2 + Turnstyle")
            out_t = gr.Textbox(label="", lines=6, show_copy_button=True)
    gr.Examples(examples=EXAMPLES, inputs=[inp, task],
                label="Try these (vanilla gets every one wrong)")

    btn.click(compare, [inp, task], [out_v, out_t, trace])
    inp.submit(compare, [inp, task], [out_v, out_t, trace])
    clr.click(lambda: ("", AUTO, "", "", ""), None, [inp, task, out_v, out_t, trace])


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
