"""agentscript_gen.py — recognized decision table -> Salesforce Agent Script.

Second codegen backend next to burr_source(): the same symbolically
reconstructed plan (shared glue + guard + arm chains) emitted as an Agent
Script `.agent` file that compiles with Salesforce's own toolchain
(@sf-agentscript/agentforce).

Agent Script has no `elif`, and its shipped linter ALSO rejects nested `if`
("Combine conditions with 'and'/'or'") — contradicting the repo SPEC's advice.
So an n-ary choice must compile to a FLAT ladder of mutually-exclusive guards,
row i carrying its full path conjunction:

    if @variables.f1: ...
    if not @variables.f1 and @variables.f2: ...
    if not @variables.f1 and not @variables.f2: ...

which is exactly the decision-table normal form the probe recognizes. The
emitter below takes an arbitrary list of (condition-literals, actions) rows, so
k-check chains plug in unchanged when the recognizer grows past one check;
today's single-check tables are the two-row special case. Validated against
@sf-agentscript/agentforce 2.5.34 (language 2.18.0): both example ladders parse
and compile to the runtime spec; `elif` and nested-if controls are rejected.
"""
from __future__ import annotations

# state keys that are naturally boolean flags; everything else is a string
BOOL_KEYS = {"is_late", "eligible", "already_credited", "refunded", "notified",
             "escalated", "cancelled"}

ACTION_DESCS = {
    "LookupOrder":            "Resolve an order reference to an order record",
    "GetOrderStatus":         "Fetch the current status of the order",
    "CheckShipping":          "Check whether the order shipped late",
    "CheckRefundEligibility": "Check whether the order is eligible for a refund",
    "CheckPriorCredits":      "Check whether a credit already went out",
    "IssueRefund":            "Refund the customer for the order",
    "OfferCredit":            "Offer the customer a store credit",
    "ApplyDiscount":          "Apply a discount to the order",
    "EscalateToHuman":        "Hand the case to a human agent",
    "CancelOrder":            "Cancel the order",
    "DraftReply":             "Draft a reply to the customer",
    "NotifyCustomer":         "Send the drafted reply to the customer",
}

IND = "    "


def _topo_depth(nodes, edges):
    depth = {n: 0 for n in nodes}
    for _ in range(len(nodes)):
        for a, b in edges:
            depth[b] = max(depth[b], depth[a] + 1)
    return depth


def _vtype(key: str) -> str:
    return "boolean" if key in BOOL_KEYS else "string"


def _vdefault(key: str) -> str:
    return "False" if key in BOOL_KEYS else '""'


def _emit_run(action: str, actions: dict, depth: int) -> list[str]:
    pad = IND * depth
    reads, writes = actions[action]
    lines = [f"{pad}run @actions.{action}"]
    for k in reads:
        lines.append(f"{pad}{IND}with {k}=@variables.{k}")
    for k in writes:
        lines.append(f"{pad}{IND}set @variables.{k} = @outputs.{k}")
    return lines


def emit_ladder(shared: list[str], rows: list[tuple[list[str], list[str]]],
                actions: dict, agent_name: str, description: str,
                initial_keys: set[str]) -> str:
    """Emit a full .agent file.

    shared: glue actions run unconditionally, in topo order (checks included).
    rows:   n-ary ladder — (condition literals to AND together, arm actions in
            topo order). Empty-arm rows are skipped (the no-op world).
    """
    used = list(shared) + [a for _, arm in rows for a in arm]

    keys_read = {k for a in used for k in actions[a][0]}
    keys_written = {k for a in used for k in actions[a][1]}
    all_keys = sorted(keys_read | keys_written)

    lines = [
        "config:",
        f'{IND}developer_name: "{agent_name}"',
        f'{IND}description: "{description}"',
        "",
        "variables:",
    ]
    for k in all_keys:
        role = ("provided by the caller" if k in initial_keys and k not in keys_written
                else "produced by the workflow")
        lines += [f"{IND}{k}: mutable {_vtype(k)} = {_vdefault(k)}",
                  f'{IND}{IND}description: "State key `{k}` — {role}"']
    lines += ["", f"start_agent {agent_name}:",
              f'{IND}description: "{description}"',
              f"{IND}actions:"]
    for a in sorted(set(used)):
        reads, writes = actions[a]
        lines += [f"{IND}{IND}{a}:",
                  f'{IND}{IND}{IND}description: "{ACTION_DESCS.get(a, a)}"',
                  f"{IND}{IND}{IND}inputs:"]
        for k in reads:
            lines += [f"{IND}{IND}{IND}{IND}{k}: {_vtype(k)}",
                      f"{IND}{IND}{IND}{IND}{IND}is_required: True"]
        lines.append(f"{IND}{IND}{IND}outputs:")
        for k in writes:
            lines.append(f"{IND}{IND}{IND}{IND}{k}: {_vtype(k)}")
        lines.append(f'{IND}{IND}{IND}target: "flow://{a}"')

    lines.append(f"{IND}before_reasoning:")
    body_depth = 2
    for a in shared:
        lines += _emit_run(a, actions, body_depth)
    for cond_lits, arm in rows:
        if not arm:
            continue  # the ∅ arm: leave the order alone
        cond = " and ".join(cond_lits)
        lines.append(f"{IND * body_depth}if {cond}:")
        for a in arm:
            lines += _emit_run(a, actions, body_depth + 1)
    if not shared and all(not arm for _, arm in rows):
        lines.append(f"{IND * body_depth}set @variables.order_ref = @variables.order_ref")

    lines += [
        f"{IND}reasoning:",
        f"{IND}{IND}instructions: ->",
        f"{IND}{IND}{IND}| Report to the customer what the workflow did for order "
        "{!@variables.order_ref}, based on the state variables above.",
        f"{IND}{IND}{IND}| Do not promise any action the workflow did not take.",
    ]
    return "\n".join(lines) + "\n"


def agentscript_source(res: dict, actions: dict, initial_keys: set[str]) -> str:
    """Adapter: the Space's analyze() result -> emit_ladder().

    Today's recognizer reads single-check two-arm tables, so the ladder has two
    rows (flag / not flag); the emitter itself is n-ary for the k-check chains
    the recognizer will grow into.
    """
    plan, edges = res["plan"], set(res["edges"])
    depth = _topo_depth(plan, edges)
    order = sorted(plan, key=lambda n: (depth[n], n))
    desc = res["req"].replace('"', "'")[:160]

    if res.get("check"):
        gk = actions[res["check"]][1][0]
        shared = [n for n in order if n in res["arm_nodes"]["shared"]]
        rows = [
            ([f"@variables.{gk}"],
             [n for n in order if n in res["arm_nodes"]["true"]]),
            ([f"not @variables.{gk}"],
             [n for n in order if n in res["arm_nodes"]["false"]]),
        ]
    else:
        shared, rows = order, []
    return emit_ladder(shared, rows, actions, "Requirement_Plan", desc, initial_keys)
