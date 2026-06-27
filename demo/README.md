---
title: Turnstyle — Neurosymbolic SmolLM2
emoji: 🌀
colorFrom: indigo
colorTo: green
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
short_description: SmolLM2-1.7B, vanilla vs. with turnstyle's symbolic+probe grounding
---

# 🌀 Turnstyle — a 1.7B model that stops guessing

Same model on both sides. **Left:** raw SmolLM2-1.7B-Instruct. **Right:** the same model
with **[turnstyle](https://github.com/jdonaldson/turnstyle)** engaged — it parses your prompt
into a typed task and either:

- **solves it exactly** (arithmetic, dates, boolean logic, bracket matching, sorting,
  spatial navigation, logical deduction), or
- **recognizes the answer with a hidden-state probe** (sarcasm, movie similarity) — decoding
  what the model *knows* but can't reliably *say*,

then biases generation toward that answer. The trace shows what it did.

The bundled SmolLM2 calibration profile (the recognition probes) ships inside the turnstyle
wheel and auto-loads by model fingerprint — no extra setup.

## Notes

- **Hardware**: runs on CPU but each query runs the model twice (vanilla + grounded), so a
  small GPU (T4) makes it snappy. Set in the Space's hardware settings.
- **Env vars**: `TS_MODEL` (default `HuggingFaceTB/SmolLM2-1.7B-Instruct`),
  `TS_MAX_NEW` (default 64).
- The deterministic tasks are where the contrast is starkest — vanilla SmolLM2 will
  confidently miscompute arithmetic or pick the wrong date while turnstyle nails it.
