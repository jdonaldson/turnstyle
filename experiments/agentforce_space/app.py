"""
requirement-dag — NL requirement → typed action DAG, with zero generated tokens.

A 360M language model (SmolLM2-360M-Instruct) is used purely as a TEXT ENCODER:
one forward pass over the requirement, then 13 tiny logistic heads read the
semantic payload off a mid-stack hidden state —
  * which state keys the requirement DEMANDS (the plan's sinks),
  * whether a conditional gates them, WHICH check, and its POLARITY
    (incl. antonym negations like "on track" that a negation-regex cannot see).
Everything else is symbolic: backward-chaining (regress) inserts dependencies,
a topo-sort orders them, a validator checks guardrails BEFORE anything runs,
and the plan compiles to a Burr state machine. No plan is ever *written* by
the model, so no plan can be hallucinated; low head confidence → abstain.

Probe artifact: probe_artifact.npz (fit offline; see the turnstyle project's
experiments/build_space_probe.py). Held-out numbers on UNSEEN wordings:
keys set-exact .57, guard polarity 9/9 on regex-blind negations, arm
assignment 54/54 via clause-span pooling.
"""
from __future__ import annotations
import html as html_mod
import json, os, re

import numpy as np

# ---------------------------------------------------------------------------
# Closed action library (Agentforce-style: typed reads/writes over state keys).
# ---------------------------------------------------------------------------
ACTIONS: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "LookupOrder":            (("order_ref",),          ("order",)),
    "GetOrderStatus":         (("order",),              ("status",)),
    "CheckShipping":          (("order",),              ("is_late",)),
    "CheckRefundEligibility": (("order",),              ("eligible",)),
    "CheckPriorCredits":      (("order",),              ("already_credited",)),
    "IssueRefund":            (("order", "eligible"),   ("refunded",)),
    "OfferCredit":            (("order",),              ("credit",)),
    "ApplyDiscount":          (("order",),              ("discount",)),
    "EscalateToHuman":        (("order",),              ("escalated",)),
    "CancelOrder":            (("order",),              ("cancelled",)),
    "DraftReply":             (("order",),              ("reply",)),
    "NotifyCustomer":         (("reply",),              ("notified",)),
}
PRODUCERS = {k: n for n, (_, w) in ACTIONS.items() for k in w}
INITIAL_KEYS = {"order_ref", "requested_amount"}

# Verified public Salesforce documentation targets (nominative reference only;
# this demo is not affiliated with Salesforce).
DOC_INDEX = ("https://help.salesforce.com/s/articleView?id=ai.copilot_actions_ref.htm"
             "&language=en_US&type=5")
DOC_CUSTOM = "https://developer.salesforce.com/docs/ai/agentforce/guide/ascript-ref-actions.html"
DOCS = {
    "EscalateToHuman": ("https://help.salesforce.com/s/articleView?id="
                        "ai.service_agent_escalation.htm&language=en_US&type=5",
                        "Agentforce escalation / human handoff docs"),
    "IssueRefund":   (DOC_CUSTOM, "Custom agent actions (Agent Script guide)"),
    "OfferCredit":   (DOC_CUSTOM, "Custom agent actions (Agent Script guide)"),
    "ApplyDiscount": (DOC_CUSTOM, "Custom agent actions (Agent Script guide)"),
}
def doc_for(action: str):
    return DOCS.get(action, (DOC_INDEX, "Standard Agent Action Reference"))


def regress(seed):
    """Backward-chain: pull in the producer of every unmet read, to fixpoint."""
    nodes = set(a for a in seed if a in ACTIONS)
    changed = True
    while changed:
        changed = False
        have = set(INITIAL_KEYS) | {k for a in nodes for k in ACTIONS[a][1]}
        for a in list(nodes):
            for need in ACTIONS[a][0]:
                if need not in have and need in PRODUCERS and PRODUCERS[need] not in nodes:
                    nodes.add(PRODUCERS[need]); changed = True
    return frozenset(nodes)


def dataflow_edges(nodes):
    edges = set()
    for a in nodes:
        for k in ACTIONS[a][0]:
            p = PRODUCERS.get(k)
            if p in nodes and p != a:
                edges.add((p, a))
    return edges


# ---------------------------------------------------------------------------
# Probe runtime: numpy mirror of the sklearn (scaler + logistic) heads.
# ---------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
META = json.load(open(os.path.join(HERE, "probe_meta.json")))
ART = np.load(os.path.join(HERE, "probe_artifact.npz"))
LAYER, KEYS, CHECKS = META["layer"], META["keys"], META["checks"]
PRODUCER_OF_KEY = {k: PRODUCERS[k] for k in KEYS}

def _head(name):
    return {f: ART[f"{name}_{f}"] for f in ("mean", "scale", "coef", "intercept")}

def p_binary(name, x):
    h = _head(name)
    z = (x - h["mean"]) / h["scale"]
    return float(1.0 / (1.0 + np.exp(-(z @ h["coef"][0] + h["intercept"][0]))))

def p_check(x):
    h = _head("check")
    z = (x - h["mean"]) / h["scale"]
    logits = z @ h["coef"].T + h["intercept"]
    e = np.exp(logits - logits.max())
    return e / e.sum()

_model = _tok = None
def load_model():
    global _model, _tok
    if _model is None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        _tok = AutoTokenizer.from_pretrained(META["model"])
        _model = AutoModelForCausalLM.from_pretrained(
            META["model"], torch_dtype=torch.float32).eval()
    return _model, _tok

def encode(text: str):
    """ONE forward pass. Returns (mean-pooled vec at probe layer, token h, offsets)."""
    import torch
    model, tok = load_model()
    enc = tok(text, return_tensors="pt", return_offsets_mapping=True)
    with torch.no_grad():
        out = model(enc["input_ids"], output_hidden_states=True)
    h = out.hidden_states[LAYER][0].float().numpy()
    return h.mean(axis=0), h, enc["offset_mapping"][0].tolist()

def arm_char_spans(req: str):
    lo, semi = req.find(",") + 1, req.find(";")
    oth = req.lower().find("otherwise")
    if semi < 0 or oth < 0:
        return None
    return (lo, semi), (oth + len("otherwise"), len(req))

def span_mean(h, offsets, span):
    a, b = span
    rows = [i for i, (s, e) in enumerate(offsets) if e > a and s < b and e > s]
    return h[rows].mean(axis=0) if rows else h.mean(axis=0)

NEG_RE = re.compile(r"\b(not|never|no|unless)\b|n't", re.I)


# ---------------------------------------------------------------------------
# The pipeline: requirement -> decision table -> plan -> SVG + Burr.
# ---------------------------------------------------------------------------
def analyze(req: str) -> dict:
    req = req.strip()[:400]
    vec, h, offsets = encode(req)
    key_p = {k: p_binary(f"key_{k}", vec) for k in KEYS}
    demanded = [k for k, p in key_p.items() if p > 0.5]
    pres_p = p_binary("presence", vec)
    out = {"req": req, "key_p": key_p, "demanded": demanded, "presence_p": pres_p,
           "conditional": pres_p > 0.5, "check": None, "polarity": None,
           "polarity_p": None, "check_p": None, "arms": None, "regex_pol": None}
    if not demanded:
        out["abstain"] = True
        return out
    out["abstain"] = False

    if out["conditional"]:
        cp = p_check(vec)
        out["check"] = CHECKS[int(np.argmax(cp))]
        out["check_p"] = float(cp.max())
        pol_p = p_binary("polarity", vec)
        out["polarity"], out["polarity_p"] = pol_p > 0.5, pol_p
        out["regex_pol"] = not NEG_RE.search(req.split(",")[0])
        spans = arm_char_spans(req)
        arm_true, arm_false = [], []
        if spans and len(demanded) >= 2:
            x1, x2 = span_mean(h, offsets, spans[0]), span_mean(h, offsets, spans[1])
            for k in demanded:
                in_arm1 = p_binary(f"key_{k}", x1) >= p_binary(f"key_{k}", x2)
                arm1 = in_arm1
                (arm_true if arm1 == out["polarity"] else arm_false).append(k)
        else:  # single-outcome conditional: demanded keys sit in the guard-true world
            arm_true = demanded
        out["arms"] = {"true": arm_true, "false": arm_false}

    # symbolic reconstruction
    goals = {PRODUCER_OF_KEY[k] for k in demanded}
    seed = set(goals) | ({out["check"]} if out["check"] else set())
    plan = regress(seed)
    out["plan"] = plan
    out["edges"] = dataflow_edges(plan)
    if out["check"]:
        t_nodes = regress({PRODUCER_OF_KEY[k] for k in out["arms"]["true"]} | {out["check"]})
        f_nodes = regress({PRODUCER_OF_KEY[k] for k in out["arms"]["false"]} | {out["check"]})
        shared = t_nodes & f_nodes
        out["arm_nodes"] = {"true": plan & (t_nodes - shared),
                            "false": plan & (f_nodes - shared), "shared": shared}
    # sinks (for coloring): guard consumes its check; reads consume the rest
    consumed = set()
    for a in plan:
        consumed |= set(ACTIONS[a][0])
    if out["check"]:
        consumed |= set(ACTIONS[out["check"]][1])
    out["sinks"] = {a for a in plan if not (set(ACTIONS[a][1]) & consumed)}
    return out


def topo_depth(nodes, edges):
    depth = {n: 0 for n in nodes}
    for _ in range(len(nodes)):
        for a, b in edges:
            depth[b] = max(depth[b], depth[a] + 1)
    return depth


# ---------------------------------------------------------------------------
# Solarized SVG renderer with clickable nodes.
# ---------------------------------------------------------------------------
S = {"bg": "#fdf6e3", "line": "#657b83", "text": "#073642", "muted": "#93a1a1",
     "sink": "#6c71c4", "glue": "#eee8d5", "check": "#cb4b16",
     "true": "#859900", "false": "#dc322f"}
NW, NH, XGAP, YGAP = 168, 44, 70, 26

def render_svg(res: dict) -> str:
    plan, edges = res["plan"], set(res["edges"])
    guard_edges = []
    if res.get("check"):
        gk = ACTIONS[res["check"]][1][0]
        for world, arm in (("True", "true"), ("False", "false")):
            arm_nodes = res["arm_nodes"][arm]
            entries = {b for b in arm_nodes
                       if not any((a, b) in edges for a in arm_nodes)} or arm_nodes
            for e in sorted(entries):
                guard_edges.append((res["check"], e, f"{gk} = {world}", world == "True"))
    depth = topo_depth(plan, edges | {(a, b) for a, b, *_ in guard_edges})
    cols: dict[int, list] = {}
    def lane_rank(n):
        if res.get("check"):
            if n in res["arm_nodes"]["true"]: return 1
            if n in res["arm_nodes"]["false"]: return 2
        return 0
    for n in sorted(plan, key=lambda n: (depth[n], lane_rank(n), n)):
        cols.setdefault(depth[n], []).append(n)
    pos = {}
    max_rows = max(len(v) for v in cols.values())
    for d, ns in cols.items():
        for i, n in enumerate(ns):
            x = 24 + d * (NW + XGAP)
            y = 30 + i * (NH + YGAP) + (max_rows - len(ns)) * (NH + YGAP) / 2
            pos[n] = (x, y)
    W = 48 + (max(cols) + 1) * (NW + XGAP) - XGAP
    H = 60 + max_rows * (NH + YGAP)

    def esc(s): return html_mod.escape(str(s), quote=True)
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" '
             f'style="max-width:100%;background:{S["bg"]};border-radius:10px;'
             f'font-family:ui-sans-serif,system-ui">']
    def edge_path(a, b):
        x1, y1 = pos[a][0] + NW, pos[a][1] + NH / 2
        x2, y2 = pos[b][0], pos[b][1] + NH / 2
        mx = (x1 + x2) / 2
        return f"M{x1},{y1} C{mx},{y1} {mx},{y2} {x2},{y2}", (x1+x2)/2, (y1+y2)/2
    parts.append(f'<defs><marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" '
                 f'markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
                 f'<path d="M0,0 L10,5 L0,10 z" fill="{S["line"]}"/></marker></defs>')
    for a, b in sorted(edges):
        d, _, _ = edge_path(a, b)
        parts.append(f'<path d="{d}" fill="none" stroke="{S["line"]}" '
                     f'stroke-width="1.6" marker-end="url(#arr)"/>')
    for a, b, label, is_true in guard_edges:
        d, mx, my = edge_path(a, b)
        c = S["true"] if is_true else S["false"]
        parts.append(f'<path d="{d}" fill="none" stroke="{c}" stroke-width="2" '
                     f'stroke-dasharray="6,3" marker-end="url(#arr)"/>')
        parts.append(f'<text x="{mx}" y="{my - 6}" text-anchor="middle" '
                     f'font-size="11" font-weight="600" fill="{c}">{esc(label)}</text>')
    for n in plan:
        x, y = pos[n]
        is_sink, is_check = n in res["sinks"], n == res.get("check")
        fill = S["check"] if is_check else (S["sink"] if is_sink else S["glue"])
        tcol = "#fdf6e3" if (is_sink or is_check) else S["text"]
        url, doc_label = doc_for(n)
        key = ACTIONS[n][1][0]
        conf = ""
        if is_sink and key in res["key_p"]:
            conf = f' · p={res["key_p"][key]:.2f}'
        parts.append(
            f'<a href="{esc(url)}" target="_blank" rel="noopener">'
            f'<g><title>{esc(n)} — writes `{esc(key)}`\nDocs: {esc(doc_label)}</title>'
            f'<rect x="{x}" y="{y}" rx="9" width="{NW}" height="{NH}" fill="{fill}" '
            f'stroke="{S["line"]}" stroke-opacity="0.35"/>'
            f'<text x="{x + NW/2}" y="{y + 19}" text-anchor="middle" font-size="13" '
            f'font-weight="650" fill="{tcol}">{esc(n)}</text>'
            f'<text x="{x + NW/2}" y="{y + 34}" text-anchor="middle" font-size="10" '
            f'fill="{tcol}" fill-opacity="0.85">→ {esc(key)}{esc(conf)}</text></g></a>')
    ly = H - 14
    parts.append(
        f'<g font-size="11" fill="{S["text"]}">'
        f'<rect x="24" y="{ly-10}" width="12" height="12" rx="3" fill="{S["sink"]}"/>'
        f'<text x="40" y="{ly}">demanded outcome (sink) — recognized by probe</text>'
        f'<rect x="330" y="{ly-10}" width="12" height="12" rx="3" fill="{S["check"]}"/>'
        f'<text x="346" y="{ly}">guard check — recognized</text>'
        f'<rect x="520" y="{ly-10}" width="12" height="12" rx="3" fill="{S["glue"]}" '
        f'stroke="{S["line"]}" stroke-opacity="0.4"/>'
        f'<text x="536" y="{ly}">derived dependency — inserted symbolically</text></g>')
    parts.append("</svg>")
    return "".join(parts)


# ---------------------------------------------------------------------------
# Decision-table + status panels (HTML), Burr codegen.
# ---------------------------------------------------------------------------
def bar(p, color):
    return (f'<div style="background:#eee8d5;border-radius:4px;height:8px;width:120px;'
            f'display:inline-block;vertical-align:middle;margin-left:6px">'
            f'<div style="background:{color};height:8px;border-radius:4px;'
            f'width:{int(p*120)}px"></div></div>')

def table_html(res: dict) -> str:
    esc = html_mod.escape
    css = 'style="border-collapse:collapse;font-family:ui-sans-serif,system-ui;font-size:14px"'
    td = 'style="padding:6px 12px;border-bottom:1px solid #eee8d5;vertical-align:top"'
    rows = []
    def key_chips(keys):
        return " ".join(
            f'<span style="background:#6c71c4;color:#fdf6e3;border-radius:6px;'
            f'padding:2px 8px;font-size:12px">{esc(k)}</span>'
            f'<span style="color:#93a1a1;font-size:11px"> p={res["key_p"][k]:.2f}</span>'
            for k in keys) or '<span style="color:#93a1a1">∅ (no action)</span>'
    if res.get("check"):
        gk = ACTIONS[res["check"]][1][0]
        rows.append(f'<tr><td {td}><code>{esc(gk)} = True</code></td>'
                    f'<td {td}>{key_chips(res["arms"]["true"])}</td></tr>')
        rows.append(f'<tr><td {td}><code>{esc(gk)} = False</code></td>'
                    f'<td {td}>{key_chips(res["arms"]["false"])}</td></tr>')
    else:
        rows.append(f'<tr><td {td}><code>⊤ (always)</code></td>'
                    f'<td {td}>{key_chips(res["demanded"])}</td></tr>')
    guard_info = ""
    if res.get("check"):
        pol = res["polarity_p"] if res["polarity"] else 1 - res["polarity_p"]
        guard_info = (f'<p style="margin:6px 0">guard: <b>{esc(res["check"])}</b> '
                      f'(p={res["check_p"]:.2f}) · polarity read from hidden state '
                      f'(p={pol:.2f}){bar(pol, "#cb4b16")}</p>')
        if res["regex_pol"] is not None and res["regex_pol"] != res["polarity"]:
            guard_info += ('<p style="margin:6px 0;color:#cb4b16"><b>⚠ a negation-regex '
                           'would have read this guard with the opposite polarity</b> — '
                           'no negation token is present; the probe read the antonym '
                           'semantics ("on time", "off the table", …) from the '
                           'mid-stack representation.</p>')
    return (f'<div style="background:#fdf6e3;border-radius:10px;padding:12px 16px;color:#073642">'
            f'<p style="margin:4px 0;color:#586e75">decision table — the full semantic '
            f'payload the probe read (everything else is derived):</p>'
            f'<table {css}><tr><th {td}>world</th><th {td}>demanded keys</th></tr>'
            f'{"".join(rows)}</table>{guard_info}</div>')

def abstain_html(res: dict) -> str:
    tops = sorted(res["key_p"].items(), key=lambda kv: -kv[1])[:3]
    chips = " ".join(f'<code>{html_mod.escape(k)}</code> p={p:.2f}' for k, p in tops)
    return (f'<div style="background:#fdf6e3;border-left:6px solid #b58900;'
            f'border-radius:10px;padding:14px 16px;color:#073642">'
            f'<b>⊘ Abstain.</b> No demanded outcome was recognized above threshold — '
            f'this requirement doesn\'t map onto the action library with confidence. '
            f'In production this routes to human review instead of executing a guessed '
            f'workflow. Top scores: {chips}</div>')


def burr_source(res: dict) -> str:
    plan, edges = res["plan"], res["edges"]
    depth = topo_depth(plan, edges)
    order = sorted(plan, key=lambda n: (depth[n], n))
    lines = ["from burr.core import ApplicationBuilder, State, action, when", ""]
    for n in order:
        r, w = ACTIONS[n]
        lines += [f"@action(reads={list(r)}, writes={list(w)})",
                  f"def {n.lower()}(state: State) -> State:",
                  f"    return state.update(**_{n.lower()}_backend(state))", ""]
    names = {n: n.lower() for n in order}
    tr = []
    if res.get("check"):
        gk = ACTIONS[res["check"]][1][0]
        shared = [n for n in order if n in res["arm_nodes"]["shared"]]
        for a, b in zip(shared, shared[1:]):
            tr.append(f'        ("{names[a]}", "{names[b]}"),')
        for world, arm in (("True", "true"), ("False", "false")):
            chain = [n for n in order if n in res["arm_nodes"][arm]]
            if not chain:
                continue
            tr.append(f'        ("{names[res["check"]]}", "{names[chain[0]]}", '
                      f'when({gk}={world})),')
            for a, b in zip(chain, chain[1:]):
                tr.append(f'        ("{names[a]}", "{names[b]}"),')
    else:
        for a, b in zip(order, order[1:]):
            tr.append(f'        ("{names[a]}", "{names[b]}"),')
    lines += ["app = (", "    ApplicationBuilder()",
              f"    .with_actions({', '.join(f'{v}={v}' for v in names.values())})",
              "    .with_transitions(", *tr, "    )",
              f'    .with_entrypoint("{names[order[0]]}")', "    .build()", ")"]
    return "\n".join(lines)


def run_pipeline(req: str):
    if not req or not req.strip():
        return "<p>Enter a requirement.</p>", "", ""
    res = analyze(req)
    if res["abstain"]:
        return abstain_html(res), "", "# abstained — no workflow emitted"
    return table_html(res), render_svg(res), burr_source(res)


EXAMPLES = [
    "If the shipment is on track, notify the customer; otherwise escalate this to a human agent.",
    "Refund the customer and send the customer a message.",
    "Please knock some money off the order for order OA-1142.",
    "If they are ineligible for a refund, hand the case to a person; otherwise give the customer their money back.",
    "Report where the order stands.",
    "Translate the order confirmation into French.",
]

INTRO = """
# Requirement → DAG — planning without generating

Type a support-desk requirement. A **360M** language model reads it in **one
forward pass** (no tokens are generated, ever); 13 logistic probes on a
mid-stack hidden state recognize the **demanded outcomes** and the **guard
condition + polarity**; everything else — dependencies, ordering, the workflow
graph — is **derived symbolically** from the typed action library and compiled
to a [Burr](https://github.com/apache/burr) state machine.

Nodes link to the relevant public Salesforce Agentforce docs (this demo is a
research prototype from the [turnstyle](https://github.com/jdonaldson/turnstyle)
project and is not affiliated with Salesforce). Try example 1: *"on track"*
flips the branch polarity with **no negation word** — a keyword system cannot
see it; the probe reads it 9/9 on held-out antonym negations.
"""

def build_ui():
    import gradio as gr
    with gr.Blocks(title="Requirement → DAG") as demo:
        gr.Markdown(INTRO)
        with gr.Row():
            req = gr.Textbox(label="Requirement (natural language)",
                             value=EXAMPLES[0], scale=4)
            btn = gr.Button("Plan it", variant="primary", scale=1)
        table = gr.HTML(label="Decision table")
        dag = gr.HTML(label="Reconstructed DAG")
        with gr.Accordion("Emitted Burr workflow (the reviewable artifact)", open=False):
            code = gr.Code(language="python")
        gr.Examples(EXAMPLES, inputs=req)
        with gr.Accordion("How it works / honest numbers", open=False):
            gr.Markdown(
                "- **One forward pass** of SmolLM2-360M-Instruct; mean-pooled hidden "
                f"state at layer {LAYER} (picked by wording-transfer, not fit).\n"
                "- **8 key heads** ('does this requirement demand `refunded`?'), a "
                "conditional-presence head, a 3-way check head, a polarity head; fork-arm "
                "membership via clause-span pooling (54/54 on the calibration corpus).\n"
                "- **Symbolic**: goals = the plan's *sinks* (unique minimal generator — "
                "verified); backward-chaining inserts glue; topo-sort orders; the plan "
                "compiles to Burr.\n"
                "- **Held-out (unseen wordings)**: keys set-exact ≈ .57, polarity 9/9 on "
                "regex-blind antonym negations, decision-table exact ≈ .52 — matching a "
                "3.8B model *generating* plans, at ~10× smaller. Calibrated on a small "
                "template corpus: phrasings far from the examples may abstain or misread; "
                "that is the point of the confidence display.\n"
                "- Docs links: [Standard Agent Action Reference]"
                f"({DOC_INDEX}), [escalation]({DOCS['EscalateToHuman'][0]}), "
                f"[custom actions]({DOC_CUSTOM}).")
        btn.click(run_pipeline, inputs=req, outputs=[table, dag, code])
        req.submit(run_pipeline, inputs=req, outputs=[table, dag, code])
        demo.load(run_pipeline, inputs=req, outputs=[table, dag, code])
    return demo


if __name__ == "__main__":
    build_ui().launch()
