"""
goal_probe.py — can a small model RECOGNIZE the semantic payload of an NL
requirement: the DEMANDED KEYS + the GUARD TABLE?

v2 of the goal-membership probe, restructured around the sinks formalization
(verified in sink_check.py): a requirement denotes (a) a set of demanded state
KEYS — the writes of the plan's sink actions — and (b) a guard table for
conditionals (which check gates the fork, and its POLARITY). Everything else
(glue, edges, order, producer choice) is symbolic: regress() + toposort.

Why keys, not actions: goals = producers of demanded keys. With unique producers
(this toy) the two are isomorphic — v1's action-labeled results carry over
unchanged — but keys are the target that survives real libraries where several
actions write the same key: producer choice is an engineering decision (cost,
permissions), not a semantic one, and the probe should never have to make it.

New in v2 — guard-table recognition, the NL-bound residue the frpr isolated:
  * presence: does the requirement contain a conditional at all?
  * check: WHICH condition gates it (3-way here)?
  * polarity: does arm-1 run when the check's boolean is TRUE? Polarity is the
    one thing regress() cannot derive. Corpus includes ANTONYM negations
    ("the order is on time", "a refund is off the table") that a negation-regex
    cannot see — the regex-blind subset is where a probe earns its keep.

Method notes carried from v1 (both were live failure modes):
  * layer picked by WORDING-TRANSFER (train v0 → validate v1), never
    in-distribution CV — that picks a lexical layer whose heads collapse on
    unseen wordings; * mean-pooling over tokens, since last-token squeezes
    multi-goal requirements into one position (set-exact 0.07 → 0.46).
Known gap (unbuilt, deliberate): explicit temporal constraints ("then") beyond
data-flow order are not modeled.

Run (one model per process, MPS):
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python experiments/goal_probe.py --model smollm
  ... --model smollm360
"""
from __future__ import annotations
import argparse, os, re, sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dag_extract import ACTIONS, regress  # closed vocabulary + symbolic substrate

# ---------------------------------------------------------------------------
# Goal-able actions, their surface variants, and the demanded key each writes.
# Variant 0 names the action directly; 1-2 are synonyms. Held-out = variant 2.
# ---------------------------------------------------------------------------
GOAL_SURFACES: dict[str, list[str]] = {
    "CancelOrder":     ["cancel the order", "call the whole order off", "terminate the purchase"],
    "IssueRefund":     ["refund the customer", "give the customer their money back", "reimburse them for the purchase"],
    "OfferCredit":     ["offer the customer a credit", "give them store credit", "extend a goodwill credit"],
    "ApplyDiscount":   ["apply a discount to the order", "knock some money off the order", "mark the order down"],
    "EscalateToHuman": ["escalate this to a human agent", "hand the case to a person", "route the case to a live agent"],
    "GetOrderStatus":  ["tell the customer the order status", "give the customer a status update", "report where the order stands"],
    "CheckShipping":   ["check whether the order shipped late", "see if the delivery is running behind", "verify the shipping timeline"],
    "NotifyCustomer":  ["notify the customer", "send the customer a message", "drop the customer a note"],
}
GOALABLE = sorted(GOAL_SURFACES)
KEY_OF = {a: ACTIONS[a][1][0] for a in GOALABLE}      # each writes exactly one key
KEYS = [KEY_OF[a] for a in GOALABLE]                  # aligned with GOALABLE
PRODUCER_OF = {k: a for a, k in KEY_OF.items()}

# Guard table: check action -> {polarity_value: [3 surface variants]}.
# polarity label = arm-1 runs when the check's boolean equals this value.
# False-surfaces deliberately include ANTONYM negations with no negation token.
COND_SURFACES: dict[str, dict[bool, list[str]]] = {
    "CheckRefundEligibility": {
        True:  ["they are eligible for a refund", "they qualify for a refund",
                "refund eligibility comes back positive"],
        False: ["they are ineligible for a refund", "they don't qualify for a refund",
                "a refund is off the table"],
    },
    "CheckShipping": {
        True:  ["the order is running late", "the delivery is behind schedule",
                "the shipment missed its delivery window"],
        False: ["the order is on time", "the delivery is on schedule",
                "the shipment is on track"],
    },
    "CheckPriorCredits": {
        True:  ["they have already been credited this quarter", "a credit already went out this quarter",
                "they already got a credit this quarter"],
        False: ["they have not been credited this quarter", "no credit has gone out this quarter",
                "they missed out on this quarter's credits"],
    },
}
CHECKS = sorted(COND_SURFACES)
ARM_PAIRS = [("IssueRefund", "EscalateToHuman"), ("OfferCredit", "ApplyDiscount"),
             ("CancelOrder", "NotifyCustomer")]

def _cap(s: str) -> str: return s[0].upper() + s[1:]

def build_corpus() -> list[dict]:
    """Deterministic template x variant enumeration; labels by construction.
    Record: req, goals (actions), keys (demanded keys), variant, template,
    guard (None | {check, polarity})."""
    reqs: list[dict] = []
    def add(text, goals, variant, template, guard=None):
        reqs.append({"req": text, "goals": frozenset(goals),
                     "keys": frozenset(KEY_OF[a] for a in goals),
                     "variant": variant, "template": template, "guard": guard})

    for v in range(3):
        for i, a in enumerate(GOALABLE):
            s = GOAL_SURFACES[a][v]
            add(f"{_cap(s)}.", {a}, v, "single")
            add(f"Please {s} for order OA-{1100 + 10*i + v}.", {a}, v, "polite")
        for i, a in enumerate(GOALABLE):
            for b in GOALABLE[i + 1:]:
                add(f"{_cap(GOAL_SURFACES[a][v])} and {GOAL_SURFACES[b][v]}.",
                    {a, b}, v, "pair")
        seq = [("NotifyCustomer", "CancelOrder"), ("GetOrderStatus", "OfferCredit"),
               ("EscalateToHuman", "IssueRefund"), ("ApplyDiscount", "NotifyCustomer"),
               ("CheckShipping", "GetOrderStatus")]
        for a, b in seq:
            add(f"{_cap(GOAL_SURFACES[a][v])}, then {GOAL_SURFACES[b][v]}.",
                {a, b}, v, "then")
        # conditionals: 3 checks x 2 polarities x 3 arm pairs
        for check in CHECKS:
            for pol in (True, False):
                cond = COND_SURFACES[check][pol][v]
                for a, b in ARM_PAIRS:
                    add(f"If {cond}, {GOAL_SURFACES[a][v]}; otherwise {GOAL_SURFACES[b][v]}.",
                        {a, b}, v, "cond", guard={"check": check, "polarity": pol})
    return reqs


# ---------------------------------------------------------------------------
# Cheap baselines (the mandatory middle column).
# ---------------------------------------------------------------------------
_GENERIC = {"order", "get", "to", "the", "customer", "check"}

def _name_words(action: str) -> list[str]:
    return [w.lower() for w in re.findall(r"[A-Z][a-z]+", action)
            if w.lower() not in _GENERIC]

def lexical_key(req: str, key: str) -> bool:
    """Prefix word-match of the producing action's name against the requirement."""
    t = req.lower()
    return any(re.search(r"\b" + re.escape(w[:5]), t)
               for w in _name_words(PRODUCER_OF[key]))

PRESENCE_RE = re.compile(r"\b(if|unless|otherwise|when)\b", re.I)
NEG_RE = re.compile(r"\b(not|never|no|unless)\b|n't", re.I)

def regex_presence(req: str) -> bool: return bool(PRESENCE_RE.search(req))
def regex_polarity(req: str) -> bool:
    """Negation-token heuristic on the guard clause: no negation token -> True."""
    clause = req.split(",")[0]
    return not NEG_RE.search(clause)
def lexical_check(req: str) -> str:
    t = req.lower()
    for c in CHECKS:
        if any(re.search(r"\b" + re.escape(w[:5]), t) for w in _name_words(c)):
            return c
    return CHECKS[0]


# ---------------------------------------------------------------------------
# Hidden states: one forward per requirement; last-token + mean-pooled, all layers.
# ---------------------------------------------------------------------------
MODELS = {
    "smollm":    "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    "smollm360": "HuggingFaceTB/SmolLM2-360M-Instruct",
    "qwen":      "Qwen/Qwen2.5-1.5B-Instruct",
    "phi":       "microsoft/Phi-4-mini-instruct",
}

def collect(model_key: str, reqs: list[dict]) -> dict:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    name = MODELS[model_key]
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32).to(dev).eval()
    last, mean = [], []
    with torch.no_grad():
        for i, r in enumerate(reqs):
            ids = tok(r["req"], return_tensors="pt").input_ids.to(dev)
            out = model(ids, output_hidden_states=True)
            last.append(np.stack([h[0, -1].float().cpu().numpy() for h in out.hidden_states]))
            mean.append(np.stack([h[0].float().mean(dim=0).cpu().numpy() for h in out.hidden_states]))
            if i % 50 == 0:
                print(f"  [{i}/{len(reqs)}] {r['req'][:60]}", flush=True)
    del model
    return {"last": np.asarray(last), "mean": np.asarray(mean)}


# ---------------------------------------------------------------------------
# Probe: key heads + guard-table heads at a wording-transfer-selected layer.
# ---------------------------------------------------------------------------
def fit_eval(feats: np.ndarray, reqs: list[dict], verbose: bool = True):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    n_layers = feats.shape[1]
    y_keys = np.array([[k in r["keys"] for k in KEYS] for r in reqs])
    variant = np.array([r["variant"] for r in reqs])
    tr, te = variant < 2, variant == 2
    is_cond = np.array([r["guard"] is not None for r in reqs])
    y_pres = is_cond.copy()
    y_check = np.array([CHECKS.index(r["guard"]["check"]) if r["guard"] else -1 for r in reqs])
    y_pol = np.array([bool(r["guard"]["polarity"]) if r["guard"] else False for r in reqs])

    def head():
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))

    # layer sweep by wording transfer on the key heads (v0 -> v1)
    v0, v1 = variant == 0, variant == 1
    layer_cv = []
    for L in range(n_layers):
        accs = [head().fit(feats[v0, L], y_keys[v0, j]).score(feats[v1, L], y_keys[v1, j])
                for j in range(len(KEYS))]
        layer_cv.append(float(np.mean(accs)))
        if verbose and (L % 4 == 0 or L == n_layers - 1):
            print(f"  L{L:2d} v0->v1 key transfer {layer_cv[-1]:.3f}", flush=True)
    best_L = int(np.argmax(layer_cv))
    X_tr, X_te = feats[tr, best_L], feats[te, best_L]

    # --- key heads
    pred_keys = np.zeros_like(y_keys[te])
    for j in range(len(KEYS)):
        pred_keys[:, j] = head().fit(X_tr, y_keys[tr, j]).predict(X_te)
    lex_keys = np.array([[lexical_key(r["req"], k) for k in KEYS]
                         for r in reqs if r["variant"] == 2])
    gold_keys = y_keys[te]

    # --- guard heads (presence on all; check/polarity trained on cond rows)
    pred_pres = head().fit(X_tr, y_pres[tr]).predict(X_te)
    ctr, cte = tr & is_cond, te & is_cond
    check_clf = head().fit(feats[ctr, best_L], y_check[ctr])
    pol_clf = head().fit(feats[ctr, best_L], y_pol[ctr])
    pred_check = check_clf.predict(feats[cte, best_L])
    pred_pol = pol_clf.predict(feats[cte, best_L])

    held = [r for r in reqs if r["variant"] == 2]
    held_cond = [r for r in held if r["guard"] is not None]
    gold_check = np.array([CHECKS.index(r["guard"]["check"]) for r in held_cond])
    gold_pol = np.array([bool(r["guard"]["polarity"]) for r in held_cond])
    rx_pres = np.array([regex_presence(r["req"]) for r in held])
    rx_pol = np.array([regex_polarity(r["req"]) for r in held_cond])
    lx_check = np.array([CHECKS.index(lexical_check(r["req"])) for r in held_cond])
    # regex-blind negatives: gold polarity False, no negation token in the guard
    blind = np.array([(not g) and rp for g, rp in zip(gold_pol, rx_pol)])

    def pair_acc(p):  return float((p == gold_keys).mean())
    def set_exact(p): return float((p == gold_keys).all(axis=1).mean())

    lin = [k for k, r in enumerate(held) if r["template"] != "cond"]
    def e2e(p):
        ok = 0
        for k in lin:
            goals_pred = {PRODUCER_OF[KEYS[j]] for j in range(len(KEYS)) if p[k, j]}
            ok += regress(goals_pred) == regress(held[k]["goals"])
        return ok / len(lin)

    # decision-table exact: full semantic payload per held-out requirement
    cond_pos = {id(r): i for i, r in enumerate(held_cond)}
    table_ok = 0
    for k, r in enumerate(held):
        keys_ok = bool((pred_keys[k] == gold_keys[k]).all())
        if r["guard"] is None:
            table_ok += keys_ok and not pred_pres[k]
        else:
            i = cond_pos[id(r)]
            table_ok += (keys_ok and pred_pres[k]
                         and pred_check[i] == gold_check[i]
                         and bool(pred_pol[i]) == gold_pol[i])

    return {"best_layer": best_L, "transfer": layer_cv[best_L],
            "n_train": int(tr.sum()), "n_held": len(held),
            "n_cond": len(held_cond), "n_blind": int(blind.sum()), "n_lin": len(lin),
            "keys": {"pair": pair_acc(pred_keys), "set": set_exact(pred_keys), "e2e": e2e(pred_keys)},
            "lex":  {"pair": pair_acc(lex_keys),  "set": set_exact(lex_keys),  "e2e": e2e(lex_keys)},
            "presence": {"probe": float((pred_pres == np.array([r["guard"] is not None for r in held])).mean()),
                         "regex": float((rx_pres == np.array([r["guard"] is not None for r in held])).mean())},
            "check": {"probe": float((pred_check == gold_check).mean()),
                      "lex":   float((lx_check == gold_check).mean())},
            "polarity": {"probe": float((pred_pol == gold_pol).mean()),
                         "regex": float((rx_pol == gold_pol).mean()),
                         "probe_blind": float((pred_pol[blind] == gold_pol[blind]).mean()) if blind.any() else float("nan"),
                         "regex_blind": float((rx_pol[blind] == gold_pol[blind]).mean()) if blind.any() else float("nan")},
            "table_exact": table_ok / len(held)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS))
    args = ap.parse_args()

    reqs = build_corpus()
    n_cond = sum(r["template"] == "cond" for r in reqs)
    print(f"\n### demanded-keys + guard-table probe — {MODELS[args.model]}")
    print(f"corpus: {len(reqs)} requirements ({n_cond} conditional, half neg-polarity), "
          f"train=variants 0-1, held-out=variant 2 (unseen wordings)")
    for r in (reqs[0], next(r for r in reqs if r["guard"] and not r["guard"]["polarity"])):
        print(f"  e.g. {r['req']!r}\n       keys={sorted(r['keys'])} guard={r['guard']}")

    feats = collect(args.model, reqs)
    for pool in ("last", "mean"):
        print(f"\n--- pooling: {pool} ---")
        res = fit_eval(feats[pool], reqs, verbose=(pool == "last"))
        print(f"=== SUMMARY  {args.model}/{pool}  (train {res['n_train']}, held {res['n_held']}: "
              f"{res['n_cond']} cond / {res['n_blind']} regex-blind-neg / {res['n_lin']} linear) ===")
        print(f"  best layer         : L{res['best_layer']} (v0->v1 {res['transfer']:.3f})")
        print(f"  keys  pair|set|e2e : probe {res['keys']['pair']:.3f}|{res['keys']['set']:.3f}|{res['keys']['e2e']:.3f}"
              f"   lexical {res['lex']['pair']:.3f}|{res['lex']['set']:.3f}|{res['lex']['e2e']:.3f}")
        print(f"  guard presence     : probe {res['presence']['probe']:.3f} | regex {res['presence']['regex']:.3f}")
        print(f"  check identity     : probe {res['check']['probe']:.3f} | lexical {res['check']['lex']:.3f}")
        print(f"  polarity           : probe {res['polarity']['probe']:.3f} | neg-regex {res['polarity']['regex']:.3f}")
        print(f"  polarity (blind)   : probe {res['polarity']['probe_blind']:.3f} | neg-regex {res['polarity']['regex_blind']:.3f}"
              f"   (antonym negations, n={res['n_blind']})")
        print(f"  DECISION-TABLE exact: {res['table_exact']:.3f}   <-- full semantic payload")

if __name__ == "__main__":
    main()
