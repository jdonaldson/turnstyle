"""Turnstyle demo — SmolLM2 vanilla vs. SmolLM2 + Turnstyle, side by side.

A neurosymbolic wrapper: the model generates, but turnstyle parses the prompt into a
typed Task, solves it deterministically (arithmetic / dates / logic / sorting) or with
a calibrated hidden-state recognition probe (snarks / movie / …), and biases generation
toward that answer. The left pane is the raw model; the right pane is the same model with
turnstyle engaged, plus a trace of what it did.

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

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

print(f"loading {MODEL} on {DEVICE} ({DTYPE})…", flush=True)
TOK = AutoTokenizer.from_pretrained(MODEL)
MDL = AutoModelForCausalLM.from_pretrained(MODEL, dtype=DTYPE).to(DEVICE)
DT = DispatchTurnstyle(MDL, TOK, DEVICE)
print(f"turnstyle ready. probe tasks: {DT.profile_tasks}", flush=True)

AUTO = "(auto — symbolic / deterministic)"
PROBE_TASKS = [AUTO] + sorted(DT.profile_tasks)

# (prompt, probe_task). Deterministic tasks use AUTO; probe tasks (snarks, movie)
# tag their task so the demo activates the right calibrated probe via use_probe().
EXAMPLES = [
    ["((-1 + 2 + 9 * 5) - (-2 + -4 + -4 * -7)) =", AUTO],
    ["not ( True ) and ( True ) is", AUTO],
    ["Complete the rest of the sequence, making sure that the parentheses are closed "
     "properly. Input: [ [", AUTO],
    ["Sort the following words alphabetically: List: tiger apple mango banana cherry", AUTO],
    ["Today is Christmas Eve of 1937. What is the date tomorrow in MM/DD/YYYY?\n"
     "Options:\n(A) 12/11/1937\n(B) 12/25/1937\n(C) 01/04/1938\n(D) 12/04/1937\n"
     "(E) 12/25/2006\n(F) 07/25/1937", AUTO],
    ["The following paragraphs each describe a set of three objects arranged in a fixed "
     "order. The statements are logically consistent within each paragraph. On a branch, "
     "there are three birds: a blue jay, a quail, and a falcon. The falcon is to the right "
     "of the blue jay. The blue jay is to the right of the quail.\nOptions:\n"
     "(A) The blue jay is the second from the left\n(B) The quail is the second from the "
     "left\n(C) The falcon is the second from the left", AUTO],
    ["If you follow these instructions, do you return to the starting point? Always face "
     "forward. Take 1 step backward. Take 9 steps left. Take 2 steps backward. Take 6 steps "
     "forward. Take 4 steps forward. Take 4 steps backward. Take 3 steps right.\n"
     "Options:\n- Yes\n- No", AUTO],
    ["Which statement is sarcastic?\nOptions:\n(A) He's a generous person, trying to "
     "promote a charity stream that has raised millions to help kids in need\n(B) He's a "
     "terrible person, trying to promote a charity stream that has raised millions to help "
     "kids in need", "snarks"],
    ["Find a movie similar to Batman, The Mask, The Fugitive, Pretty Woman:\nOptions:\n"
     "(A) The Front Page\n(B) Maelstrom\n(C) The Lion King\n(D) Lamerica",
     "movie_recommendation"],
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
        trace = "⚪ **Turnstyle abstained** — no symbolic or probe route matched; this is plain generation."
    else:
        trace = f"🔧 **Turnstyle engaged** → parsed as `{ans.source}`, answer **{ans.text}**"
        if ans.proof:
            trace += f"\n\n> {ans.proof}"
    return vanilla, grounded, trace


with gr.Blocks(title="Turnstyle: neurosymbolic SmolLM2") as demo:
    gr.Markdown(
        "# 🌀 Turnstyle — a 1.7B model that stops guessing\n"
        "Same model, both sides. **Left:** raw SmolLM2-1.7B. **Right:** SmolLM2 with "
        "**Turnstyle** — it parses your prompt into a typed task, solves it exactly "
        "(arithmetic, dates, logic, sorting, …) or with a hidden-state recognition probe "
        "(sarcasm, movie similarity, …), and steers the model toward that answer. The "
        "trace shows what it did."
    )
    inp = gr.Textbox(label="Prompt", lines=4,
                     placeholder="Try an arithmetic expression, a date question, a logic puzzle…")
    task = gr.Dropdown(PROBE_TASKS, value=AUTO, label="Recognition probe",
                       allow_custom_value=True,
                       info="Symbolic/deterministic tasks need nothing here. Probe tasks "
                            "(sarcasm, movie) are per-task calibrated — pick one to activate it.")
    with gr.Row():
        btn = gr.Button("Compare", variant="primary")
        clr = gr.Button("Clear")
    trace = gr.Markdown()
    with gr.Row():
        with gr.Column():
            gr.Markdown("### 🤖 SmolLM2 (vanilla)")
            out_v = gr.Textbox(label="", lines=6, show_copy_button=True)
        with gr.Column():
            gr.Markdown("### 🌀 SmolLM2 + Turnstyle")
            out_t = gr.Textbox(label="", lines=6, show_copy_button=True)
    gr.Examples(examples=EXAMPLES, inputs=[inp, task],
                label="Try these (vanilla usually fails)")

    btn.click(compare, [inp, task], [out_v, out_t, trace])
    inp.submit(compare, [inp, task], [out_v, out_t, trace])
    clr.click(lambda: ("", AUTO, "", "", ""), None, [inp, task, out_v, out_t, trace])


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
