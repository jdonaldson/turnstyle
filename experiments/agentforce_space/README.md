---
title: Requirement → DAG
emoji: 🧭
colorFrom: indigo
colorTo: yellow
sdk: gradio
sdk_version: 5.49.1
app_file: app.py
pinned: false
license: mit
short_description: NL requirement → typed action DAG via a 360M probe
---

# Requirement → DAG — planning without generating

A research demo from the [turnstyle](https://github.com/jdonaldson/turnstyle)
project: turn a natural-language support requirement into an executable,
auditable workflow DAG — **without the language model generating a single
token**.

## How

1. **One forward pass** of SmolLM2-360M-Instruct over the requirement (the LLM
   is used purely as a text encoder).
2. **13 logistic probes** on a mid-stack hidden state recognize the semantic
   payload: which state keys the requirement *demands* (the plan's sinks),
   whether a conditional gates them, which check, and its **polarity** —
   including antonym negations ("the shipment is *on track*") that carry no
   negation token a regex could see. Fork-arm membership is read by re-scoring
   the same heads on clause-span-pooled features.
3. **Everything else is symbolic**: goals = the plan's *sinks* (the verified
   unique minimal generator of the DAG); backward-chaining inserts every
   dependency; a topo-sort orders them; a validator checks guardrails before
   anything executes; the plan compiles to BOTH a
   [Burr](https://github.com/apache/burr) state machine and a Salesforce
   [Agent Script](https://github.com/salesforce/agentscript) `.agent` file.

### Why the Agent Script comes out as a flat guard ladder

Agent Script has no `elif`, and its shipped linter also rejects **nested**
`if` ("Combine conditions with 'and'/'or'"). So an n-ary conditional must be
emitted as flat, mutually-exclusive guards, each row carrying its full path
conjunction (`if not f1 and f2: ...`) — the decision-table normal form. That
is exactly the object the probe recognizes, so the recognized table maps 1:1
onto Agentforce's own reviewable artifact. Emitted scripts for every non-abstain
demo example were validated against Salesforce's published toolchain
(`@sf-agentscript/agentforce` parser + compiler, 0 diagnostics); `elif` and
nested-`if` control cases are rejected by the same toolchain, confirming the
validator has teeth.

No plan is ever *written* by the model, so no plan can be hallucinated — and
every head returns a calibrated probability, so low confidence becomes an
**abstain** (route to human review) instead of a silently wrong workflow.

## Honest numbers (held-out, unseen wordings)

- demanded-keys set-exact ≈ **.57**; decision-table exact ≈ **.52** — matching a
  3.8B model *generating* full plans, at ~10× smaller
- polarity on regex-blind antonym negations: **9/9** (negation-regex: 0/9)
- fork-arm assignment via span pooling: **54/54** on the calibration corpus

Calibrated on a small template corpus over a 12-action toy library; phrasings
far from the examples may abstain or misread — the confidence display is the
point, not a limitation being hidden.

Action nodes link to public Salesforce Agentforce documentation (Standard
Agent Action Reference, escalation, custom actions). This is an independent
research prototype, not affiliated with or endorsed by Salesforce.
