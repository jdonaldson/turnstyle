"""autoprobe — automated layer × token-finder × mode sweep for hidden-state probes.

Engineering intent: the user provides information sources (training data, label
extractor, model). Everything else — token finder, layer, mode, classifier —
is search over a fixed hypothesis space. No DSL.

Usage:

    from turnstyle.autoprobe import autoprobe

    result = autoprobe(
        examples=load_task("snarks"),
        target_fn=lambda ex: ex["target"].strip(),
        model=model, tokenizer=tokenizer, device=device,
    )

    if result.ship:
        answer = result.predict(new_text, model, tokenizer, device)

The result object carries a fitted probe (or None on no-ship), the full sweep
table for inspection, the cheap baselines, and a verdict explanation.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


# ── Token finders ────────────────────────────────────────────────────────────

# Each finder: (text, tokenizer, encoded) -> list[(token_idx, label)] | None
# label is used for per-option mode (e.g., "A", "B"); "_LAST_" for single mode.
TokenFinder = Callable[[str, object, dict], Optional[list[tuple[int, str]]]]

DEFAULT_OPTION_RE = re.compile(r"\(([A-Z])\)\s+(.+?)(?=\n\(|\Z)", flags=re.S)


def last_token_of_prompt(text, tokenizer, encoded):
    """Single-mode finder: hidden state at the prompt's last token."""
    return [(len(encoded["input_ids"]) - 1, "_LAST_")]


def per_option_last_token(option_re=DEFAULT_OPTION_RE) -> TokenFinder:
    """Per-option finder factory: returns the last token of each option text."""
    def finder(text, tokenizer, encoded):
        offsets = encoded["offset_mapping"]
        positions = []
        for m in option_re.finditer(text):
            opt_text = m.group(2).rstrip()
            opt_end_char = m.start(2) + len(opt_text)
            last_idx = None
            for tok_idx, (s, e) in enumerate(offsets):
                if s < opt_end_char and e >= opt_end_char and s != e:
                    last_idx = tok_idx
            if last_idx is None:
                return None
            positions.append((last_idx, m.group(1)))
        return positions if positions else None
    return finder


# Default registry tried by autoprobe when finders are not specified.
DEFAULT_FINDERS: dict[str, TokenFinder] = {
    "per_option_last_token": per_option_last_token(),
    "last_token_of_prompt": last_token_of_prompt,
}


# ── Result types ─────────────────────────────────────────────────────────────

@dataclass
class ProbeArtifact:
    """A fitted probe ready to predict on new examples."""
    finder: TokenFinder
    layer: int
    mode: str            # "single" | "per_option"
    classes: list        # for single: list of class labels; for per_option: ["positive"]
    scaler: object
    classifier: object
    answer_format: str   # "letter_paren" → "(A)"; "letter" → "A"; "raw" → use class label as-is

    def _format(self, class_label):
        if self.answer_format == "letter_paren":
            return f"({class_label})"
        if self.answer_format == "letter":
            return str(class_label)
        return str(class_label)

    def predict(self, text, model, tokenizer, device):
        """Run the probe on a new example. Returns the formatted answer string,
        or None if the finder fails."""
        encoded = tokenizer(text, return_offsets_mapping=True,
                            add_special_tokens=True)
        positions = self.finder(text, tokenizer, encoded)
        if positions is None:
            return None

        ids = {
            "input_ids": torch.tensor([encoded["input_ids"]]).to(device),
            "attention_mask": torch.tensor([encoded["attention_mask"]]).to(device),
        }
        with torch.no_grad():
            out = model(**ids, output_hidden_states=True)
        h = out.hidden_states[self.layer][0]

        if self.mode == "single":
            tok_idx = positions[0][0]
            vec = h[tok_idx].float().cpu().numpy()
            pred_idx = self.classifier.predict(self.scaler.transform([vec]))[0]
            return self._format(self.classes[pred_idx])
        else:  # per_option
            scores = {}
            for tok_idx, label in positions:
                vec = h[tok_idx].float().cpu().numpy()
                scores[label] = self.classifier.predict_proba(
                    self.scaler.transform([vec])
                )[0, 1]
            return self._format(max(scores, key=scores.get))


@dataclass
class AutoprobeResult:
    n_examples: int
    target_fmt: str
    sweep: list                   # [(finder_name, mode, layer, cv), ...] sorted by CV desc
    cheap_baselines: dict
    chosen: Optional[tuple]       # (finder_name, mode, layer, cv) or None
    fitted: Optional[ProbeArtifact]
    ship: bool
    reason: str
    fit_time_s: float = 0.0

    def predict(self, text, model, tokenizer, device):
        if self.fitted is None:
            return None
        return self.fitted.predict(text, model, tokenizer, device)

    def summary(self) -> str:
        lines = [
            f"AutoprobeResult(N={self.n_examples}, target_fmt={self.target_fmt})",
            f"  cheap baselines: " + ", ".join(
                f"{k}={v:.1%}" for k, v in self.cheap_baselines.items()),
            f"  sweep top 3:",
        ]
        for finder, mode, layer, cv in self.sweep[:3]:
            lines.append(f"    {finder:25s} {mode:10s} L{layer:>2}  {cv:>6.1%}")
        lines.append(f"  → {self.reason}  ({self.fit_time_s:.1f}s)")
        return "\n".join(lines)


# ── Target-format detection ──────────────────────────────────────────────────

_LETTER_RE = re.compile(r"^\(([A-Z])\)$")


def _detect_target_format(examples, target_fn):
    """Returns (fmt, meta).
    fmt='letter': meta = letters string (e.g. 'ABCD')
    fmt='binary': meta = (yes_label, no_label)
    fmt='unknown'
    """
    targets = [target_fn(ex) for ex in examples]
    matched = [_LETTER_RE.match(t) if t else None for t in targets]
    if sum(1 for m in matched if m) / len(targets) >= 0.90:
        letters = sorted({m.group(1) for m in matched if m})
        return "letter", "".join(letters)
    lower = {t.lower() for t in targets if t}
    if lower <= {"yes", "no"}:
        return "binary", ("yes", "no")
    return "unknown", None


# ── Cheap baselines ──────────────────────────────────────────────────────────

def _cheap_baselines(examples, target_fn, target_fmt, target_meta):
    n = len(examples)
    baselines = {}
    if target_fmt == "letter":
        from collections import Counter
        letters = [_LETTER_RE.match(target_fn(ex)) for ex in examples]
        letters = [m.group(1) if m else None for m in letters]
        valid = [l for l in letters if l]
        cnt = Counter(valid)
        baselines["majority"] = max(cnt.values()) / len(valid)

        # Length heuristics on options
        for name, reducer in [("longest", max), ("shortest", min)]:
            correct = 0
            valid_n = 0
            for ex, target_letter in zip(examples, letters):
                if not target_letter:
                    continue
                opts = [(m.group(1), m.group(2).strip())
                        for m in DEFAULT_OPTION_RE.finditer(ex["input"])]
                if not opts:
                    continue
                lens = {l: len(t) for l, t in opts}
                pred = reducer(lens, key=lens.get)
                correct += int(pred == target_letter)
                valid_n += 1
            baselines[name] = correct / valid_n if valid_n else 0
    else:
        from collections import Counter
        targets = [target_fn(ex) for ex in examples if target_fn(ex)]
        cnt = Counter(t.lower() for t in targets)
        baselines["majority"] = max(cnt.values()) / len(targets)

    return baselines


# ── Hidden-state collection ──────────────────────────────────────────────────

def _collect(examples, target_fn, target_fmt, target_meta,
             finders, model, tokenizer, device, verbose=True):
    """One forward pass per example; capture hidden states at all positions
    requested by all finders, across all layers."""
    n_layers = model.config.num_hidden_layers + 1
    records = []
    skipped = 0

    for i, ex in enumerate(examples):
        text = ex["input"]
        target = target_fn(ex)

        # Extract label for this example
        if target_fmt == "letter":
            m = _LETTER_RE.match(target) if target else None
            if not m:
                skipped += 1
                continue
            label = m.group(1)
        else:  # binary
            if target is None:
                skipped += 1
                continue
            label = target.lower()

        encoded = tokenizer(text, return_offsets_mapping=True,
                            add_special_tokens=True)

        # Run each finder and merge positions, keyed by (finder_name, label)
        positions_by_finder = {}
        skip_this = False
        for fname, finder in finders.items():
            pos = finder(text, tokenizer, encoded)
            if pos is None:
                # That finder doesn't apply to this example; record None
                positions_by_finder[fname] = None
                continue
            positions_by_finder[fname] = pos

        if all(v is None for v in positions_by_finder.values()):
            skipped += 1
            continue

        # Forward pass once
        ids = {
            "input_ids": torch.tensor([encoded["input_ids"]]).to(device),
            "attention_mask": torch.tensor([encoded["attention_mask"]]).to(device),
        }
        with torch.no_grad():
            out = model(**ids, output_hidden_states=True)
        hidden = out.hidden_states  # tuple of (1, seq, dim) per layer

        # Capture per (finder, layer, position-label)
        per_finder_layer = {}
        for fname, pos_list in positions_by_finder.items():
            if pos_list is None:
                continue
            per_finder_layer[fname] = {}
            for layer_idx in range(n_layers):
                per_finder_layer[fname][layer_idx] = {
                    plabel: hidden[layer_idx][0, pos].float().cpu().numpy()
                    for pos, plabel in pos_list
                }

        records.append({
            "label": label,
            "per_finder_layer": per_finder_layer,
        })

        if verbose and (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(examples)}] records={len(records)}",
                  flush=True)

    if verbose and skipped:
        print(f"  skipped {skipped}", flush=True)
    return records, n_layers


# ── CV evaluation ────────────────────────────────────────────────────────────

def _cv_single(records, finder_name, layer, target_fmt, n_splits=5, seed=42):
    """Multinomial classifier on records[i].per_finder_layer[finder][L]['_LAST_']."""
    usable = [r for r in records
              if finder_name in r["per_finder_layer"]
              and "_LAST_" in r["per_finder_layer"][finder_name][layer]]
    if len(usable) < 50:
        return None

    if target_fmt == "letter":
        y = np.array([ord(r["label"]) - ord("A") for r in usable])
    else:
        y = np.array([1 if r["label"] == "yes" else 0 for r in usable])
    X = np.array([r["per_finder_layer"][finder_name][layer]["_LAST_"]
                  for r in usable])

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    accs = []
    for tr, te in skf.split(X, y):
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=2000, C=0.1)
        clf.fit(sc.transform(X[tr]), y[tr])
        accs.append((clf.predict(sc.transform(X[te])) == y[te]).mean())
    return float(np.mean(accs))


def _cv_per_option(records, finder_name, layer, letters, n_splits=5, seed=42):
    """Per-option binary classifier; argmax at eval."""
    usable = [r for r in records
              if finder_name in r["per_finder_layer"]
              and all(l in r["per_finder_layer"][finder_name][layer]
                      for l in letters)]
    if len(usable) < 50:
        return None

    n_ex = len(usable)
    y_class = np.array([ord(r["label"]) - ord("A") for r in usable])

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    accs = []
    for tr_idx, te_idx in skf.split(np.arange(n_ex), y_class):
        X_tr, y_tr = [], []
        for i in tr_idx:
            for letter in letters:
                X_tr.append(usable[i]["per_finder_layer"][finder_name][layer][letter])
                y_tr.append(1 if letter == usable[i]["label"] else 0)
        X_tr = np.array(X_tr); y_tr = np.array(y_tr)
        sc = StandardScaler().fit(X_tr)
        clf = LogisticRegression(max_iter=2000, C=0.1, class_weight="balanced")
        clf.fit(sc.transform(X_tr), y_tr)

        correct = 0
        for i in te_idx:
            scores = {}
            for letter in letters:
                h = usable[i]["per_finder_layer"][finder_name][layer][letter]
                scores[letter] = clf.predict_proba(sc.transform(h[None]))[0, 1]
            pred = max(scores, key=scores.get)
            if pred == usable[i]["label"]:
                correct += 1
        accs.append(correct / len(te_idx))
    return float(np.mean(accs))


# ── Final fit ────────────────────────────────────────────────────────────────

def _fit_final(records, finder_name, layer, mode, target_fmt,
               target_meta, finders):
    """Refit the chosen config on ALL records; return a ProbeArtifact."""
    finder = finders[finder_name]
    if mode == "single":
        usable = [r for r in records
                  if finder_name in r["per_finder_layer"]
                  and "_LAST_" in r["per_finder_layer"][finder_name][layer]]
        if target_fmt == "letter":
            classes = sorted({r["label"] for r in usable})
            class_to_idx = {c: i for i, c in enumerate(classes)}
            y = np.array([class_to_idx[r["label"]] for r in usable])
            answer_format = "letter_paren"
        else:
            classes = ["no", "yes"]   # 0=no, 1=yes
            y = np.array([1 if r["label"] == "yes" else 0 for r in usable])
            answer_format = "raw"
        X = np.array([r["per_finder_layer"][finder_name][layer]["_LAST_"]
                      for r in usable])
        sc = StandardScaler().fit(X)
        clf = LogisticRegression(max_iter=2000, C=0.1).fit(sc.transform(X), y)
        return ProbeArtifact(finder=finder, layer=layer, mode="single",
                             classes=classes, scaler=sc, classifier=clf,
                             answer_format=answer_format)

    else:  # per_option
        letters = target_meta
        usable = [r for r in records
                  if finder_name in r["per_finder_layer"]
                  and all(l in r["per_finder_layer"][finder_name][layer]
                          for l in letters)]
        X, y = [], []
        for r in usable:
            for letter in letters:
                X.append(r["per_finder_layer"][finder_name][layer][letter])
                y.append(1 if letter == r["label"] else 0)
        X = np.array(X); y = np.array(y)
        sc = StandardScaler().fit(X)
        clf = LogisticRegression(max_iter=2000, C=0.1,
                                 class_weight="balanced").fit(sc.transform(X), y)
        return ProbeArtifact(finder=finder, layer=layer, mode="per_option",
                             classes=list(letters), scaler=sc, classifier=clf,
                             answer_format="letter_paren")


# ── Main entry point ─────────────────────────────────────────────────────────

def autoprobe(
    examples: list,
    target_fn: Callable,
    model,
    tokenizer,
    device: str,
    finders: Optional[dict[str, TokenFinder]] = None,
    layers: Optional[range] = None,
    ship_threshold_abs: float = 0.60,
    ship_threshold_lift: float = 0.10,
    verbose: bool = True,
) -> AutoprobeResult:
    """Auto-fit a hidden-state probe via 5-fold CV layer × finder × mode sweep.

    Args:
        examples: training examples (list of dicts).
        target_fn: function (ex) -> str or None, extracts target label.
        model, tokenizer, device: HuggingFace model and its tokenizer.
        finders: dict of {name: finder_callable}; defaults to canonical 2.
        layers: range to sweep; defaults to all layers.
        ship_threshold_abs: minimum CV accuracy to ship.
        ship_threshold_lift: minimum lift over best cheap baseline to ship.
        verbose: print sweep progress and decision.

    Returns:
        AutoprobeResult with .fitted (ProbeArtifact or None), .ship, .reason,
        .sweep, .cheap_baselines.
    """
    t0 = time.time()
    finders = finders or DEFAULT_FINDERS
    layers = layers or range(model.config.num_hidden_layers + 1)

    target_fmt, target_meta = _detect_target_format(examples, target_fn)
    if verbose:
        print(f"[autoprobe] N={len(examples)}, target_fmt={target_fmt}, "
              f"meta={target_meta}", flush=True)

    if target_fmt == "unknown":
        return AutoprobeResult(
            n_examples=len(examples), target_fmt=target_fmt,
            sweep=[], cheap_baselines={}, chosen=None, fitted=None,
            ship=False, reason="unknown target format",
            fit_time_s=time.time() - t0,
        )

    cheap = _cheap_baselines(examples, target_fn, target_fmt, target_meta)
    if verbose:
        print(f"[autoprobe] cheap baselines: " + ", ".join(
            f"{k}={v:.1%}" for k, v in cheap.items()), flush=True)

    if verbose:
        print(f"[autoprobe] collecting hidden states across {len(finders)} "
              f"finder(s)…", flush=True)
    records, n_layers = _collect(examples, target_fn, target_fmt, target_meta,
                                 finders, model, tokenizer, device,
                                 verbose=verbose)
    if not records:
        return AutoprobeResult(
            n_examples=len(examples), target_fmt=target_fmt,
            sweep=[], cheap_baselines=cheap, chosen=None, fitted=None,
            ship=False, reason="no usable records",
            fit_time_s=time.time() - t0,
        )

    sweep = []
    layers = list(layers) if layers else list(range(n_layers))
    for fname in finders:
        # single-mode sweep (uses _LAST_ if finder produces it)
        for layer in layers:
            cv = _cv_single(records, fname, layer, target_fmt)
            if cv is not None:
                sweep.append((fname, "single", layer, cv))
        # per-option sweep (only for letter-format tasks)
        if target_fmt == "letter":
            for layer in layers:
                cv = _cv_per_option(records, fname, layer, target_meta)
                if cv is not None:
                    sweep.append((fname, "per_option", layer, cv))

    sweep.sort(key=lambda r: -r[3])

    if not sweep:
        return AutoprobeResult(
            n_examples=len(examples), target_fmt=target_fmt,
            sweep=[], cheap_baselines=cheap, chosen=None, fitted=None,
            ship=False, reason="no probe runnable",
            fit_time_s=time.time() - t0,
        )

    chosen = sweep[0]
    best_cheap = max(cheap.values()) if cheap else 0.0
    abs_ok = chosen[3] >= ship_threshold_abs
    lift = chosen[3] - best_cheap
    lift_ok = lift >= ship_threshold_lift
    ship = abs_ok and lift_ok

    if ship:
        fitted = _fit_final(records, chosen[0], chosen[2], chosen[1],
                            target_fmt, target_meta, finders)
        reason = (f"SHIP: {chosen[0]} {chosen[1]} L{chosen[2]} cv={chosen[3]:.1%} "
                  f"≥ {ship_threshold_abs:.0%} and lift +{lift*100:.1f}pp "
                  f"≥ +{ship_threshold_lift*100:.1f}pp")
    else:
        fitted = None
        why = []
        if not abs_ok:
            why.append(f"absolute {chosen[3]:.1%} < {ship_threshold_abs:.0%}")
        if not lift_ok:
            why.append(f"lift +{lift*100:.1f}pp < +{ship_threshold_lift*100:.1f}pp")
        reason = "NO SHIP: " + " and ".join(why)

    if verbose:
        print(f"[autoprobe] {reason}", flush=True)

    return AutoprobeResult(
        n_examples=len(examples), target_fmt=target_fmt,
        sweep=sweep, cheap_baselines=cheap, chosen=chosen, fitted=fitted,
        ship=ship, reason=reason, fit_time_s=time.time() - t0,
    )
