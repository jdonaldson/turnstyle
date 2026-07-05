"""
agentforce_burr_mock.py — NL requirement -> validated typed Plan -> Burr workflow.

A mock aimed at Salesforce Agentforce. It demonstrates the turnstyle contribution
to an agent platform: the LLM proposes a plan over a CLOSED vocabulary of typed
actions, a total `match`/validator checks it against guardrails BEFORE anything
executes, and only then is it compiled to a Burr state machine that runs with a
full audit trace.

Agentforce mapping
------------------
  Role          -> the app identity / entrypoint
  Topic         -> intent classification (here: which requirement template)
  Actions       -> the ACTION_LIBRARY below (a closed, typed vocabulary)
  Guardrails    -> validate(): static checks that run before execution
  Atlas planner -> plan_from_nl(): NL -> typed Plan
  reflection    -> Burr conditional transitions (eligible? refund : escalate)

What is real vs mocked
----------------------
  * The typed Action ADT, the validator, the guardrails, the Burr codegen, and the
    execution trace are all real and run standalone (no deps, no model).
  * plan_from_nl() is a STUB standing in for turnstyle's IRSpec/LLM extraction.
    The seam is marked; a real planner drops in there and emits the same Plan type.
  * _MiniBurr mirrors Burr's @action/State/Condition/transition contract so this
    runs without burr installed. to_burr_source() emits the equivalent real
    `ApplicationBuilder` code as the shippable artifact.

Run:  python3 experiments/agentforce_burr_mock.py
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable


# ---------------------------------------------------------------------------
# 1. The closed ACTION VOCABULARY (Agentforce "Actions").
#    Each action declares what state it READS and WRITES. That's all the
#    validator needs to check dependencies, ordering, and guardrails — no
#    natural-language keyword lists.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Action:
    name: str
    reads: tuple[str, ...]       # state keys required before this runs
    writes: tuple[str, ...]      # state keys produced
    run: Callable[[dict], dict]  # (state) -> {written keys}


def _lookup_order(s):      return {"order": {"id": s["order_ref"], "total": 240.0, "days_since": 12}}
def _order_status(s):      return {"status": "delivered"}
def _check_eligibility(s): return {"eligible": s["order"]["days_since"] <= 30,
                                   "refund_cap": s["order"]["total"]}
def _issue_refund(s):      return {"refunded": min(s["requested_amount"], s["refund_cap"])}
def _escalate(s):          return {"escalated": True, "reason": s.get("escalation_reason", "policy")}
def _draft_reply(s):       return {"reply": f"(drafted, tone={s.get('tone','warm')})"}
def _notify(s):            return {"notified": True}

ACTION_LIBRARY: dict[str, Action] = {a.name: a for a in [
    Action("LookupOrder",            ("order_ref",),                     ("order",),               _lookup_order),
    Action("GetOrderStatus",         ("order",),                         ("status",),              _order_status),
    Action("CheckRefundEligibility", ("order",),                         ("eligible", "refund_cap"), _check_eligibility),
    Action("IssueRefund",            ("eligible", "refund_cap", "requested_amount"), ("refunded",), _issue_refund),
    Action("EscalateToHuman",        ("order",),                         ("escalated", "reason"),  _escalate),
    Action("DraftReply",             (),                                 ("reply",),               _draft_reply),
    Action("NotifyCustomer",         ("reply",),                         ("notified",),            _notify),
]}


# ---------------------------------------------------------------------------
# 2. The typed PLAN. A validated Plan is the artifact turnstyle hands to Burr.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Step:
    action: str                          # must be a key in ACTION_LIBRARY
    branch_on: str | None = None         # a boolean state key gating the NEXT step
    on_false: str | None = None          # action name to jump to when branch_on is False

@dataclass
class Plan:
    goal: str
    steps: list[Step]
    initial: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# 3. plan_from_nl — THE SEAM. Stub for turnstyle's IarSpec/LLM extraction.
#    A real planner reads the requirement + the ACTION_LIBRARY signatures and
#    emits a Plan. Here we hard-code two decompositions (one deliberately buggy)
#    so the validator has something to catch.
# ---------------------------------------------------------------------------
def plan_from_nl(requirement: str) -> Plan:
    r = requirement.lower()
    if "refund" in r and "eligib" not in r and "check" not in r:
        # NAIVE plan: a black-box planner that jumps straight to a refund.
        # This is exactly what free-form LLM codegen produces and what an
        # enterprise cannot ship. The validator will reject it.
        return Plan(
            goal="refund the customer for their order",
            steps=[Step("LookupOrder"), Step("IssueRefund"),
                   Step("DraftReply"), Step("NotifyCustomer")],
            initial={"order_ref": "OA-1174", "requested_amount": 240.0},
        )
    # CORRECT plan: eligibility-gated refund with an escalation branch.
    return Plan(
        goal="refund the customer if eligible, otherwise escalate",
        steps=[
            Step("LookupOrder"),
            Step("CheckRefundEligibility", branch_on="eligible", on_false="EscalateToHuman"),
            Step("IssueRefund"),
            Step("DraftReply"),
            Step("NotifyCustomer"),
        ],
        initial={"order_ref": "OA-1174", "requested_amount": 500.0,
                 "escalation_reason": "outside 30-day window"},
    )


# ---------------------------------------------------------------------------
# 4. validate — the GUARDRAIL layer. Runs before execution. Every check is
#    structural (over the typed vocabulary), not a prompt.
# ---------------------------------------------------------------------------
# Guardrail policies: (action, predicate over the set of keys guaranteed
# available when it runs, human-readable rule).
GUARDRAILS = [
    ("IssueRefund", lambda avail: "eligible" in avail,
     "IssueRefund requires a prior CheckRefundEligibility (no unchecked refunds)"),
]

def validate(plan: Plan) -> list[str]:
    violations: list[str] = []
    available = set(plan.initial)          # keys known before any step runs

    for i, step in enumerate(plan.steps):
        # (a) closed-vocabulary check
        if step.action not in ACTION_LIBRARY:
            violations.append(f"step {i}: unknown action '{step.action}' (not in the action library)")
            continue
        act = ACTION_LIBRARY[step.action]

        # (b) data-dependency check — every read must be satisfiable
        missing = [k for k in act.reads if k not in available]
        if missing:
            violations.append(f"step {i} {act.name}: unmet inputs {missing} "
                              f"(no earlier action writes them)")

        # (c) guardrail check
        for gname, pred, rule in GUARDRAILS:
            if act.name == gname and not pred(available):
                violations.append(f"step {i} {act.name}: GUARDRAIL — {rule}")

        # (d) branch target must exist in the vocabulary
        if step.on_false and step.on_false not in ACTION_LIBRARY:
            violations.append(f"step {i}: branch target '{step.on_false}' not in the action library")

        available |= set(act.writes)
    return violations


# ---------------------------------------------------------------------------
# 5a. to_burr_source — emit the real Burr ApplicationBuilder code (the artifact
#     a Salesforce dev would review + commit). Codegen, build-time binding.
# ---------------------------------------------------------------------------
def to_burr_source(plan: Plan) -> str:
    lines = ["from burr.core import ApplicationBuilder, State, action, when", ""]
    for step in plan.steps:
        act = ACTION_LIBRARY[step.action]
        lines += [f"@action(reads={list(act.reads)}, writes={list(act.writes)})",
                  f"def {act.name.lower()}(state: State) -> State:",
                  f"    # -> Agentforce action '{act.name}'",
                  f"    return state.update(**_{act.name.lower()}_backend(state))", ""]
    names = [ACTION_LIBRARY[s.action].name.lower() for s in plan.steps]
    lines += ["app = (", "    ApplicationBuilder()",
              f"    .with_actions({', '.join(f'{n}={n}' for n in names)})"]
    tr = []
    for i, step in enumerate(plan.steps[:-1]):
        cur, nxt = names[i], names[i + 1]
        if step.branch_on:
            tr.append(f'        ("{cur}", "{nxt}", when({step.branch_on}=True)),')
            tr.append(f'        ("{cur}", "{step.on_false.lower()}", when({step.branch_on}=False)),')
        else:
            tr.append(f'        ("{cur}", "{nxt}"),')
    lines += ["    .with_transitions(", *tr, "    )",
              f'    .with_entrypoint("{names[0]}")', "    .build()", ")"]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 5b. _MiniBurr — a stand-in that mirrors Burr's execution contract so the mock
#     runs without burr installed. Immutable-ish state, conditional transitions,
#     an audit trace. Swap for real Burr by using to_burr_source() above.
# ---------------------------------------------------------------------------
def run_workflow(plan: Plan) -> tuple[dict, list[str]]:
    state = dict(plan.initial)
    trace: list[str] = []
    idx_by_action = {s.action: i for i, s in enumerate(plan.steps)}
    i = 0
    while i < len(plan.steps):
        step = plan.steps[i]
        act = ACTION_LIBRARY[step.action]
        delta = act.run(state)
        state.update(delta)
        trace.append(f"{act.name:<24} -> {delta}")
        # conditional transition (Burr `when(...)`): jump on a False branch key
        if step.branch_on is not None and not state.get(step.branch_on, False):
            trace.append(f"  ↳ transition: {step.branch_on}=False → jump to {step.on_false}")
            i = idx_by_action[step.on_false]
            continue
        i += 1
    return state, trace


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------
def _demo(requirement: str):
    print("=" * 74)
    print(f"REQUIREMENT (NL):  {requirement!r}")
    plan = plan_from_nl(requirement)
    print(f"PLANNED GOAL:      {plan.goal}")
    print(f"PLAN:              {' → '.join(s.action for s in plan.steps)}")
    violations = validate(plan)
    if violations:
        print("\n  ⛔ VALIDATION FAILED — plan rejected BEFORE any action fired:")
        for v in violations:
            print(f"     • {v}")
        print("\n  (a black-box planner would have executed this; the typed guardrail")
        print("   caught it statically. No refund was issued.)")
        return
    print("\n  ✅ VALIDATION PASSED — compiling to Burr and executing.\n")
    final, trace = run_workflow(plan)
    for line in trace:
        print("   " + line)
    print(f"\n  FINAL STATE: refunded={final.get('refunded')} "
          f"escalated={final.get('escalated')} notified={final.get('notified')}")


if __name__ == "__main__":
    # 1) A naive requirement whose black-box plan violates a guardrail.
    _demo("Refund the customer for order OA-1174.")
    # 2) A requirement that yields an eligibility-gated, escalation-aware plan.
    _demo("Refund the customer if they're eligible, otherwise escalate to a human.")

    # The shippable artifact: the real Burr source for the valid plan.
    print("\n" + "=" * 74)
    print("EMITTED BURR WORKFLOW (the reviewable/committable artifact):\n")
    print(to_burr_source(plan_from_nl("refund if eligible otherwise escalate")))
