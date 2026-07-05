"""
dag_extract.py — can a SMALL model reliably extract a plan-DAG from an NL requirement?

Thesis under test (turnstyle-flavored): a plan-DAG's EDGES are derived from the
action reads/writes (a topo-sort), so DAG correctness reduces to two things the
model actually emits:
  (1) the right ACTION SET  (nodes)      -> node_f1 / node_exact
  (2) the right BRANCHES    (forks+else) -> branch accuracy
DAG-correct := node_exact AND branch_exact.
We also track "fails-safe": a wrong plan the VALIDATOR rejects (rather than
passing a silently-wrong DAG) — for an enterprise that's nearly as good as right.

Closed action vocabulary + a validator stand in for turnstyle's typed ADT.
plan_from_nl is done by a real small model (chat template, greedy, JSON out).

Run one model per process (MPS stability):
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python experiments/dag_extract.py --model smollm
  ... --model qwen ; ... --model phi
"""
from __future__ import annotations
import argparse, json, re, sys
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Closed ACTION vocabulary: name -> (reads, writes). Edges derive from these.
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

# ---------------------------------------------------------------------------
# Requirements + GROUND-TRUTH plans. A step = (action, branch_on|None, on_false|None).
# branch_on names a boolean key a prior action writes; on_false is where the
# negative case jumps. Sign is phrased positively in every prompt.
# ---------------------------------------------------------------------------
REQUIREMENTS = [
    ("R01", "Look up the order and tell the customer its current status.",
     [("LookupOrder", None, None), ("GetOrderStatus", None, None), ("DraftReply", None, None)]),
    ("R02", "Cancel the order, then draft a confirmation and send it to the customer.",
     [("LookupOrder", None, None), ("CancelOrder", None, None), ("DraftReply", None, None), ("NotifyCustomer", None, None)]),
    ("R03", "Apply a discount to the order and send the customer a confirmation.",
     [("LookupOrder", None, None), ("ApplyDiscount", None, None), ("DraftReply", None, None), ("NotifyCustomer", None, None)]),
    ("R04", "Pull up the order and check whether it shipped late.",
     [("LookupOrder", None, None), ("CheckShipping", None, None)]),
    ("R05", "Refund the customer if they are eligible; otherwise escalate to a human agent.",
     [("LookupOrder", None, None), ("CheckRefundEligibility", "eligible", "EscalateToHuman"), ("IssueRefund", None, None)]),
    ("R06", "Check if the order is running late. If it is, offer the customer a credit; otherwise just apologize.",
     [("LookupOrder", None, None), ("CheckShipping", "is_late", "DraftReply"), ("OfferCredit", None, None)]),
    ("R07", "See if the customer is eligible for a refund and if so refund them, then confirm by message. If not, escalate.",
     [("LookupOrder", None, None), ("CheckRefundEligibility", "eligible", "EscalateToHuman"),
      ("IssueRefund", None, None), ("DraftReply", None, None), ("NotifyCustomer", None, None)]),
    ("R08", "Look up the order, check its shipping status, and give the customer a status update.",
     [("LookupOrder", None, None), ("CheckShipping", None, None), ("GetOrderStatus", None, None), ("DraftReply", None, None)]),
    ("R09", "Escalate the order to a human and let the customer know.",
     [("LookupOrder", None, None), ("EscalateToHuman", None, None), ("DraftReply", None, None), ("NotifyCustomer", None, None)]),
    ("R10", "Cancel the order only if it is running late; if it is on time, apply a discount instead.",
     [("LookupOrder", None, None), ("CheckShipping", "is_late", "ApplyDiscount"), ("CancelOrder", None, None)]),
    ("R11", "Look up the order and issue a refund, then notify the customer.",
     # note: IssueRefund reads `eligible`; a correct plan MUST insert CheckRefundEligibility.
     [("LookupOrder", None, None), ("CheckRefundEligibility", None, None),
      ("IssueRefund", None, None), ("DraftReply", None, None), ("NotifyCustomer", None, None)]),
    ("R12", "Check if the customer was already credited this quarter; if not, offer a credit, otherwise escalate.",
     [("LookupOrder", None, None), ("CheckPriorCredits", "not already_credited", "EscalateToHuman"), ("OfferCredit", None, None)]),
]


def gt_nodes(steps):   return frozenset(a for a, _, _ in steps)
def gt_branches(steps):
    # (guard_action -> on_false_action), sign-agnostic
    return frozenset((a, of) for a, bo, of in steps if of is not None)


# --- sinks: the semantic content of a plan (see sink_check.py for the verified
# formalization: goals = sinks; regress(sinks ∪ checks) = plan; unique minimal
# generator). A fork-check action is consumed by its own guard; everything else
# is consumed only by other actions' reads. Sink-F1 scores the DEMANDED outcomes
# and ignores glue disagreements that regress() renders irrelevant.
def _sinks(items) -> frozenset:
    """items: iterable of (action, is_fork_check)."""
    items = [(a, chk) for a, chk in items if a in ACTIONS]
    consumed = set()
    for a, chk in items:
        consumed |= set(ACTIONS[a][0])          # reads
        if chk:
            consumed |= set(ACTIONS[a][1])      # guard consumes the check's boolean
    return frozenset(a for a, _ in items if not (set(ACTIONS[a][1]) & consumed))

def gold_sinks(steps):
    return _sinks((a, bo is not None or of is not None) for a, bo, of in steps)

def pred_sinks(steps):
    return _sinks((s.get("action"), ("branch_on" in s or "on_false" in s))
                  for s in steps if isinstance(s, dict))


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------
def catalog() -> str:
    rows = [f"  {n}: needs [{', '.join(r) or '-'}] -> produces [{', '.join(w)}]"
            for n, (r, w) in ACTIONS.items()]
    return "\n".join(rows)

# Two exemplars: one LINEAR (no branch), one BRANCHING — so a branch is not the
# default pattern the model copies onto everything.
# Chosen so neither exemplar's node-set exactly matches any test requirement.
SHOT_LINEAR_REQ = "Look up the order and cancel it."
SHOT_LINEAR_JSON = json.dumps({"steps": [
    {"action": "LookupOrder"}, {"action": "CancelOrder"},
]})
SHOT_BRANCH_REQ = "Issue a refund if the customer is eligible, otherwise apply a discount."
SHOT_BRANCH_JSON = json.dumps({"steps": [
    {"action": "LookupOrder"},
    {"action": "CheckRefundEligibility", "branch_on": "eligible", "on_false": "ApplyDiscount"},
    {"action": "IssueRefund"},
]})

SYS = (
    "You convert a support requirement into a plan over a FIXED action library. "
    "Rules: (1) Use ONLY the listed action names. (2) Insert any glue action needed "
    "so every action's inputs are produced earlier — e.g. LookupOrder must come first "
    "to produce `order`. (3) Add \"branch_on\"/\"on_false\" ONLY when the requirement "
    "contains an explicit conditional (\"if\", \"only if\", \"otherwise\", \"unless\"). "
    "A plain sequence of steps has NO branches. "
    'Reply with ONLY JSON: {"steps":[{"action":...,"branch_on"?:...,"on_false"?:...}]}.'
)

def build_prompt(req: str) -> list[dict]:
    user = (f"ACTION LIBRARY:\n{catalog()}\n\n"
            f"Requirement: {SHOT_LINEAR_REQ}\nAnswer: {SHOT_LINEAR_JSON}\n\n"
            f"Requirement: {SHOT_BRANCH_REQ}\nAnswer: {SHOT_BRANCH_JSON}\n\n"
            f"Requirement: {req}\nAnswer:")
    return [{"role": "system", "content": SYS}, {"role": "user", "content": user}]


# ---------------------------------------------------------------------------
# BACKWARD-CHAINING mode: model names only the OUTCOME actions (+ conditionals);
# a symbolic goal-regression over the known signatures fills in all glue and
# derives the edges. The model never has to insert LookupOrder / DraftReply /
# an eligibility check — the dependency demands them.
# ---------------------------------------------------------------------------
PRODUCERS = {k: n for n, (_, w) in ACTIONS.items() for k in w}   # state-key -> action
INITIAL_KEYS = {"order_ref", "requested_amount"}

def regress(seed_actions) -> frozenset:
    """Backward-chain: from a seed set of actions, pull in the producer of every
    unmet input until fixpoint. Returns the full node set."""
    nodes = set(a for a in seed_actions if a in ACTIONS)
    changed = True
    while changed:
        changed = False
        have = set(INITIAL_KEYS) | {k for a in nodes for k in ACTIONS[a][1]}
        for a in list(nodes):
            for need in ACTIONS[a][0]:
                if need not in have and need in PRODUCERS:
                    prod = PRODUCERS[need]
                    if prod not in nodes:
                        nodes.add(prod); changed = True
    return frozenset(nodes)

SHOT_LINEAR_GOALS = json.dumps({"goals": ["CancelOrder"], "branches": []})
SHOT_BRANCH_GOALS = json.dumps({"goals": ["IssueRefund"],
                                "branches": [{"check": "CheckRefundEligibility", "on_false": "ApplyDiscount"}]})
SYS_GOAL = (
    "List the OUTCOME actions a support requirement asks for — the things it wants "
    "DONE. Do NOT list setup/glue: looking up the order, drafting a message before a "
    "notification, and eligibility checks before a refund are added AUTOMATICALLY. "
    "Only when the requirement has an explicit conditional (\"if/only if/otherwise/"
    "unless\"), add a branch naming the CHECK action that produces the gating boolean "
    "and the on_false action to run otherwise. "
    'Reply with ONLY JSON: {"goals":[...],"branches":[{"check":...,"on_false":...}]}.'
)

def build_goal_prompt(req: str) -> list[dict]:
    user = (f"ACTION LIBRARY:\n{catalog()}\n\n"
            f"Requirement: {SHOT_LINEAR_REQ}\nAnswer: {SHOT_LINEAR_GOALS}\n\n"
            f"Requirement: {SHOT_BRANCH_REQ}\nAnswer: {SHOT_BRANCH_GOALS}\n\n"
            f"Requirement: {req}\nAnswer:")
    return [{"role": "system", "content": SYS_GOAL}, {"role": "user", "content": user}]

def reconstruct(obj):
    """Model goal-object -> a Step list (nodes via regress, branches attached)."""
    goals = [g for g in obj.get("goals", []) if g in ACTIONS]
    branches = [(b.get("check"), b.get("on_false")) for b in obj.get("branches", [])
                if b.get("check") in ACTIONS]
    nodes = regress(goals + [c for c, _ in branches])
    order = toposort(nodes)          # the whole point: edges + order are symbolic
    steps = []
    for a in order:
        of = next((of for c, of in branches if c == a), None)
        steps.append({"action": a, "on_false": of} if of else {"action": a})
    return steps

def toposort(nodes) -> list:
    """Kahn over the data-dependency edges producer(read) -> action, within `nodes`."""
    nodes = set(nodes)
    deps = {a: {PRODUCERS[k] for k in ACTIONS[a][0]
                if k in PRODUCERS and PRODUCERS[k] in nodes and PRODUCERS[k] != a}
            for a in nodes}
    order, ready = [], sorted(a for a in nodes if not deps[a])
    while ready:
        a = ready.pop(0); order.append(a)
        for b in sorted(nodes):
            if a in deps[b]:
                deps[b].discard(a)
                if not deps[b] and b not in order and b not in ready:
                    ready.append(b)
    order += [a for a in sorted(nodes) if a not in order]   # any cycle remnant
    return order


# ---------------------------------------------------------------------------
# Parse + validate (the guardrail layer)
# ---------------------------------------------------------------------------
def extract_json(text: str):
    # first balanced {...}
    depth = 0; start = -1
    for i, c in enumerate(text):
        if c == "{":
            if depth == 0: start = i
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0 and start >= 0:
                try: return json.loads(text[start:i + 1])
                except Exception: start = -1
    return None

def pred_nodes(steps):    return frozenset(s.get("action") for s in steps if s.get("action") in ACTIONS)
def pred_branches(steps):
    return frozenset((s["action"], s["on_false"]) for s in steps
                     if s.get("action") in ACTIONS and s.get("on_false") in ACTIONS)

def validate(steps) -> list[str]:
    """Static guardrail check over the typed steps. Returns violations."""
    v = []; available = {"order_ref", "requested_amount"}
    for i, s in enumerate(steps):
        a = s.get("action")
        if a not in ACTIONS:
            v.append(f"step {i}: unknown action {a!r}"); continue
        reads, writes = ACTIONS[a]
        miss = [k for k in reads if k not in available]
        if miss: v.append(f"step {i} {a}: unmet inputs {miss}")
        of = s.get("on_false")
        if of is not None and of not in ACTIONS:
            v.append(f"step {i}: bad on_false {of!r}")
        available |= set(writes)
    return v


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def f1(pred: frozenset, gold: frozenset) -> float:
    if not pred and not gold: return 1.0
    if not pred or not gold: return 0.0
    tp = len(pred & gold)
    if tp == 0: return 0.0
    p, r = tp / len(pred), tp / len(gold)
    return 2 * p * r / (p + r)

def score_one(steps, gt_steps):
    pn, gn = pred_nodes(steps), gt_nodes(gt_steps)
    pb, gb = pred_branches(steps), gt_branches(gt_steps)
    ps, gs = pred_sinks(steps), gold_sinks(gt_steps)
    node_exact = pn == gn
    branch_exact = pb == gb
    viols = validate(steps)
    return {
        "node_f1": f1(pn, gn),
        "node_exact": node_exact,
        "branch_exact": branch_exact,
        "sink_f1": f1(ps, gs),
        "sink_exact": ps == gs,
        "dag_correct": node_exact and branch_exact,
        "sem_correct": ps == gs and branch_exact,   # sinks + forks = the semantic content
        "valid": len(viols) == 0,
        "fails_safe": (not (node_exact and branch_exact)) and len(viols) > 0,
        "pred_nodes": sorted(pn), "gold_nodes": sorted(gn),
        "pred_sinks": sorted(ps), "gold_sinks": sorted(gs),
        "pred_branches": sorted(pb), "gold_branches": sorted(gb),
        "violations": viols,
    }


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
MODELS = {
    "smollm": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    "qwen":   "Qwen/Qwen2.5-1.5B-Instruct",
    "phi":    "microsoft/Phi-4-mini-instruct",
}

def load(model_key):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    name = MODELS[model_key]
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float32).to(dev).eval()
    return model, tok, dev

def generate(model, tok, dev, messages, max_new_tokens=256):
    import torch
    enc = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
    ids = enc["input_ids"] if hasattr(enc, "keys") else enc
    ids = ids.to(dev)
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tok.eos_token_id)
    return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--mode", default="generate", choices=["generate", "regress"])
    ap.add_argument("--limit", type=int, default=len(REQUIREMENTS))
    ap.add_argument("--verbose", type=int, default=4)
    args = ap.parse_args()

    print(f"\n### DAG extraction [{args.mode}] — {MODELS[args.model]}", flush=True)
    model, tok, dev = load(args.model)
    rows = REQUIREMENTS[:args.limit]
    agg = {"node_f1": 0.0, "node_exact": 0, "branch_exact": 0, "dag_correct": 0,
           "sink_f1": 0.0, "sink_exact": 0, "sem_correct": 0,
           "valid": 0, "fails_safe": 0, "parsed": 0}
    n_branch = sum(1 for _, _, s in rows if gt_branches(s))

    for i, (rid, req, gt_steps) in enumerate(rows):
        if args.mode == "regress":
            raw = generate(model, tok, dev, build_goal_prompt(req))
            gobj = extract_json(raw)
            obj = {"steps": reconstruct(gobj)} if gobj else None
        else:
            raw = generate(model, tok, dev, build_prompt(req))
            obj = extract_json(raw)
        if obj is None or "steps" not in obj:
            sc = {"node_f1": 0.0, "node_exact": False, "branch_exact": False,
                  "sink_f1": 0.0, "sink_exact": False, "sem_correct": False,
                  "dag_correct": False, "valid": False, "fails_safe": True,
                  "pred_nodes": [], "gold_nodes": sorted(gt_nodes(gt_steps)),
                  "pred_sinks": [], "gold_sinks": sorted(gold_sinks(gt_steps)),
                  "pred_branches": [], "gold_branches": sorted(gt_branches(gt_steps)),
                  "violations": ["unparseable"]}
        else:
            agg["parsed"] += 1
            sc = score_one(obj["steps"], gt_steps)
        for k in ("node_f1", "node_exact", "branch_exact", "dag_correct",
                  "sink_f1", "sink_exact", "sem_correct", "valid", "fails_safe"):
            agg[k] += sc[k]
        if i < args.verbose:
            mark = "✓" if sc["dag_correct"] else ("~safe" if sc["fails_safe"] else "✗")
            print(f"\n[{rid}] {mark}  node_f1={sc['node_f1']:.2f} "
                  f"node_exact={sc['node_exact']} branch_exact={sc['branch_exact']} valid={sc['valid']}")
            print(f"   req: {req}")
            print(f"   pred nodes: {sc['pred_nodes']}")
            print(f"   gold nodes: {sc['gold_nodes']}")
            if sc["gold_branches"] or sc["pred_branches"]:
                print(f"   pred branch: {sc['pred_branches']}  gold branch: {sc['gold_branches']}")
            if sc["violations"]:
                print(f"   violations: {sc['violations']}")

    N = len(rows)
    print(f"\n=== SUMMARY  {args.model}  (N={N}, {n_branch} branching) ===")
    print(f"  parsed JSON      : {agg['parsed']}/{N}")
    print(f"  node F1 (mean)   : {agg['node_f1']/N:.3f}")
    print(f"  node exact       : {agg['node_exact']}/{N}")
    print(f"  branch exact     : {agg['branch_exact']}/{N}")
    print(f"  sink F1 (mean)   : {agg['sink_f1']/N:.3f}")
    print(f"  sink exact       : {agg['sink_exact']}/{N}")
    print(f"  DAG correct      : {agg['dag_correct']}/{N}   (nodes ∧ branches — glue-sensitive)")
    print(f"  SEM correct      : {agg['sem_correct']}/{N}   <-- sinks ∧ branches (the semantic content)")
    print(f"  validator-valid  : {agg['valid']}/{N}")
    print(f"  fails-safe (rej) : {agg['fails_safe']}/{N}   (wrong but caught)")
    print(f"  unsafe wrong     : {N - agg['dag_correct'] - agg['fails_safe']}/{N}   (wrong AND passed validator)")

if __name__ == "__main__":
    main()
