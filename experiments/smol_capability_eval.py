#!/Users/jdonaldson/Projects/turnstyle/.venv/bin/python
"""SmolLM2 full capability eval with error analysis.

Runs SmolLM2 through each tier of the turnstyle taxonomy with diagnostic capture:
  T0: Logit steering (sanity check)
  T1: Simple field extraction (ExtractionSpec, regex disabled)
  T2: JSON triple extraction (SentenceIRSpec)
  T3: SQL generation

Usage:
    uv run experiments/smol_capability_eval.py [--tier 0] [--tier 1] [--tier 2] [--tier 3]

    If no --tier flags, runs all tiers.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter

sys.path.insert(0, "/Users/jdonaldson/Projects/turnstyle/src")
sys.path.insert(0, "/Users/jdonaldson/Projects/swollm/src")

import torch

# ─── model loading ───────────────────────────────────────────────────────────

def load_model():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model_id = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
    print(f"Loading {model_id}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.float32)
    device = torch.device("cpu")
    model = model.to(device)
    model.eval()
    print(f"  loaded on {device}", flush=True)
    return model, tokenizer, device


def load_bbh(task_name: str, n: int | None = None) -> list[dict]:
    """Load BBH examples. Returns list of {input, target}."""
    from swollm.bench.bbh import load_task
    examples = load_task(task_name)
    if n is not None:
        examples = examples[:n]
    return examples


# ─── output helpers ──────────────────────────────────────────────────────────

def banner(text: str):
    print(f"\n{'═' * 70}")
    print(f"  {text}")
    print(f"{'═' * 70}\n", flush=True)


def example_header(i: int, total: int, task: str, extra: str = ""):
    tag = f"[{i+1}/{total}] {task}"
    if extra:
        tag += f" {extra}"
    print(f"  {tag}", flush=True)


def truncate(s: str, maxlen: int = 120) -> str:
    s = s.replace("\n", " ↵ ")
    return s[:maxlen] + "..." if len(s) > maxlen else s


# ═══════════════════════════════════════════════════════════════════════════════
# TIER 0: Logit Steering (sanity check)
# ═══════════════════════════════════════════════════════════════════════════════

def run_tier0(model, tokenizer, device):
    banner("TIER 0: Logit Steering (sanity check)")

    from turnstyle.boolean import BooleanTurnstyle, parse_boolean
    from turnstyle.sorting import SortingTurnstyle, parse_sorting
    from turnstyle.dyck import DyckTurnstyle, parse_dyck
    from turnstyle.arithmetic import ArithmeticTurnstyle, parse_expression

    tasks = [
        ("boolean_expressions", BooleanTurnstyle, parse_boolean, 10),
        ("word_sorting", SortingTurnstyle, parse_sorting, 10),
        ("dyck_languages", DyckTurnstyle, parse_dyck, 10),
    ]

    tier_results = {}

    for task_name, TurnstyleClass, parse_fn, n in tasks:
        examples = load_bbh(task_name, n)
        ts = TurnstyleClass(model, tokenizer, device)
        correct = 0
        parse_failures = 0

        for i, ex in enumerate(examples):
            prompt = ex["input"]
            target = ex["target"].strip()

            parsed = parse_fn(prompt)
            if parsed is None:
                parse_failures += 1
                if i < 3:
                    example_header(i, n, task_name, "PARSE_FAIL")
                    print(f"    prompt: {truncate(prompt, 80)}")
                continue

            text, proof = ts.generate(prompt)

            # Check: does the output contain the target?
            got = text.strip()
            ok = target.lower() in got.lower()
            if ok:
                correct += 1

            if i < 3:
                status = "OK" if ok else "WRONG"
                example_header(i, n, task_name, status)
                print(f"    target={target}  got={truncate(got, 60)}")
                if proof and proof.any_corrected:
                    print(f"    corrections: {proof.num_corrected}/{len(proof.digits)}")

        acc = correct / len(examples) * 100 if examples else 0
        tier_results[task_name] = {
            "correct": correct, "total": len(examples),
            "parse_failures": parse_failures, "accuracy": acc,
        }
        print(f"\n  {task_name}: {correct}/{len(examples)} = {acc:.1f}%"
              f"  (parse_fail={parse_failures})", flush=True)

    # Arithmetic: use ArithmeticTurnstyle with parse_expression (handles nested BBH exprs)
    task_name = "multistep_arithmetic_two"
    examples = load_bbh(task_name, 10)
    ts = ArithmeticTurnstyle(model, tokenizer, device)
    correct = 0
    parse_failures = 0

    for i, ex in enumerate(examples):
        prompt = ex["input"]
        target = ex["target"].strip()

        parsed = parse_expression(prompt)
        if parsed is None:
            parse_failures += 1
            if i < 3:
                example_header(i, 10, task_name, "PARSE_FAIL")
                print(f"    prompt: {truncate(prompt, 80)}")
            continue

        expr, result = parsed
        # Check parse correctness first
        parse_correct = str(result) == target
        if not parse_correct and i < 3:
            example_header(i, 10, task_name, "PARSE_WRONG")
            print(f"    expr={expr}  result={result}  target={target}")
            continue

        text, proof = ts.generate(prompt)
        got = text.strip()
        ok = target in got
        if ok:
            correct += 1

        if i < 3:
            status = "OK" if ok else "WRONG"
            example_header(i, 10, task_name, status)
            print(f"    target={target}  got={truncate(got, 60)}")
            if proof and hasattr(proof, 'any_corrected') and proof.any_corrected:
                print(f"    corrections: {proof.num_corrected}/{len(proof.digits)}")

    acc = correct / len(examples) * 100 if examples else 0
    tier_results[task_name] = {
        "correct": correct, "total": len(examples),
        "parse_failures": parse_failures, "accuracy": acc,
    }
    print(f"\n  {task_name}: {correct}/{len(examples)} = {acc:.1f}%"
          f"  (parse_fail={parse_failures})", flush=True)

    # Summary
    print(f"\n  --- Tier 0 Summary ---")
    total_correct = sum(r["correct"] for r in tier_results.values())
    total_examples = sum(r["total"] for r in tier_results.values())
    for task, r in tier_results.items():
        print(f"    {task}: {r['accuracy']:.1f}%")
    print(f"    OVERALL: {total_correct}/{total_examples}", flush=True)

    return tier_results


# ═══════════════════════════════════════════════════════════════════════════════
# TIER 1: Simple Field Extraction (ExtractionSpec, regex disabled)
# ═══════════════════════════════════════════════════════════════════════════════

def run_tier1(model, tokenizer, device):
    banner("TIER 1: Simple Field Extraction (regex disabled)")

    from turnstyle.boolean import BooleanTurnstyle, BOOLEAN_EXTRACTION_SPEC
    from turnstyle.sorting import SortingTurnstyle, SORTING_EXTRACTION_SPEC
    from turnstyle.dyck import DyckTurnstyle, DYCK_EXTRACTION_SPEC
    from turnstyle.extract import extract, ExtractionMethod

    tasks = [
        ("boolean_expressions", BooleanTurnstyle, BOOLEAN_EXTRACTION_SPEC, 20),
        ("word_sorting", SortingTurnstyle, SORTING_EXTRACTION_SPEC, 20),
        ("dyck_languages", DyckTurnstyle, DYCK_EXTRACTION_SPEC, 20),
    ]

    tier_results = {}

    for task_name, TurnstyleClass, spec, n in tasks:
        examples = load_bbh(task_name, n)
        ts = TurnstyleClass(model, tokenizer, device)

        # Disable regex parse — force LLM extraction
        original_parse = ts.parse
        ts.parse = lambda _p: None

        categories = Counter()  # correct, partial, garbage, empty, assemble_fail
        raw_outputs = []

        for i, ex in enumerate(examples):
            prompt = ex["input"]
            target = ex["target"].strip()

            # Manually call the extraction pipeline (same as Turnstyle.generate fallback)
            result = extract(prompt, ts, spec)

            if result is None:
                categories["no_result"] += 1
                cat = "no_result"
                raw = "(None returned)"
            elif result.method == ExtractionMethod.FAILED:
                # LLM extracted but assembly failed or confidence too low
                raw = str(result.raw_fields)
                if not result.raw_fields:
                    categories["empty"] += 1
                    cat = "empty"
                else:
                    categories["assemble_fail"] += 1
                    cat = "assemble_fail"
            elif result.parsed is not None:
                # Got a parsed result — check correctness
                raw = str(result.raw_fields)
                # For boolean: parsed is (expr, result_bool, result_str)
                # For sorting: parsed is (words, sorted_words, sorted_str)
                # For dyck: parsed is (open_seq, closing, closing_str)
                try:
                    if task_name == "boolean_expressions":
                        _, _, answer = result.parsed
                    elif task_name == "word_sorting":
                        _, _, answer = result.parsed
                    elif task_name == "dyck_languages":
                        _, _, answer = result.parsed
                    else:
                        answer = str(result.parsed)

                    if answer.strip().lower() == target.strip().lower():
                        categories["correct"] += 1
                        cat = "correct"
                    else:
                        categories["wrong_answer"] += 1
                        cat = "wrong_answer"
                except Exception:
                    categories["assemble_fail"] += 1
                    cat = "assemble_fail"
            else:
                categories["assemble_fail"] += 1
                cat = "assemble_fail"
                raw = str(result.raw_fields) if result else "(None)"

            record = {
                "idx": i, "task": task_name, "category": cat,
                "target": target,
                "raw_fields": result.raw_fields if result else {},
                "confidence": result.confidence if result else 0,
                "method": result.method.name if result else "NONE",
            }
            raw_outputs.append(record)

            # Verbose for first 3
            if i < 3:
                example_header(i, n, task_name, cat.upper())
                print(f"    target: {target}")
                if result and result.raw_fields:
                    for fname, (val, conf) in result.raw_fields.items():
                        print(f"    {fname}: {truncate(val, 80)} (conf={conf:.3f})")
                if result and result.parsed is not None:
                    print(f"    parsed answer: {truncate(str(result.parsed), 80)}")

        # Restore parse
        ts.parse = original_parse

        acc = categories.get("correct", 0) / len(examples) * 100 if examples else 0
        tier_results[task_name] = {
            "total": len(examples),
            "categories": dict(categories),
            "accuracy": acc,
            "examples": raw_outputs,
        }

        print(f"\n  {task_name}: {acc:.1f}% correct")
        for cat, count in sorted(categories.items()):
            print(f"    {cat}: {count}")

        # Show first 3 failures with raw model output
        failures = [r for r in raw_outputs if r["category"] != "correct"]
        if failures:
            print(f"\n  First failures ({task_name}):")
            for r in failures[:3]:
                print(f"    [{r['idx']}] {r['category']} target={r['target']}")
                for fname, (val, conf) in r["raw_fields"].items():
                    print(f"      {fname}: {truncate(str(val), 100)} (conf={conf:.3f})")

    print(f"\n  --- Tier 1 Summary ---")
    for task, r in tier_results.items():
        print(f"    {task}: {r['accuracy']:.1f}% — {r['categories']}")
    print(flush=True)

    return tier_results


# ═══════════════════════════════════════════════════════════════════════════════
# TIER 2: JSON Triple Extraction (SentenceIRSpec)
# ═══════════════════════════════════════════════════════════════════════════════

def run_tier2(model, tokenizer, device):
    banner("TIER 2: JSON Triple Extraction via SentenceIRSpec")

    from turnstyle.ir import sentence_ir_solve, parse_scene
    from turnstyle.comparison_ordering import COMPARISON_ORDERING_SPEC
    from turnstyle.object_tracking import OBJECT_TRACKING_SPEC

    tier2_tasks = [
        ("logical_deduction_three_objects", COMPARISON_ORDERING_SPEC, 5),
        ("tracking_shuffled_objects_three_objects", OBJECT_TRACKING_SPEC, 5),
    ]

    # For navigate and web_of_lies, test basic JSON extraction capability
    # using generate_short directly — these don't have SentenceIRSpec
    from turnstyle.extract import generate_short

    tier_results = {}

    # ── Tasks with SentenceIRSpec ──
    for task_name, spec, n in tier2_tasks:
        examples = load_bbh(task_name, n)
        categories = Counter()
        example_diags = []

        for i, ex in enumerate(examples):
            prompt = ex["input"]
            target = ex["target"].strip()

            diag: dict = {}
            answer = sentence_ir_solve(
                model, tokenizer, device, prompt, spec, diag=diag)

            # Categorize
            sent_extractions = diag.get("sentence_extractions", [])
            n_sentences = len(sent_extractions)
            n_parsed = sum(1 for s in sent_extractions if s.get("parsed"))
            n_failed = n_sentences - n_parsed

            if answer is not None and target.lower() in answer.lower():
                cat = "correct"
            elif answer is not None:
                cat = "wrong_answer"
            elif n_parsed == 0:
                cat = "no_json_at_all"
            elif diag.get("error", "").startswith("aggregate"):
                cat = "valid_json_agg_fail"
            else:
                cat = "partial_json"

            categories[cat] += 1
            record = {
                "idx": i, "task": task_name, "category": cat,
                "target": target, "answer": answer,
                "n_sentences": n_sentences,
                "n_parsed": n_parsed,
                "error": diag.get("error"),
            }

            # Capture per-sentence extraction details
            sent_details = []
            for s in sent_extractions:
                sd = {
                    "sentence": truncate(s.get("sentence", ""), 80),
                    "type": s.get("type"),
                    "parsed": s.get("parsed"),
                    "response": truncate(s.get("response", ""), 120),
                    "confidence": s.get("confidence"),
                }
                sent_details.append(sd)
            record["sentence_details"] = sent_details
            example_diags.append(record)

            # Verbose for first 3
            if i < 3:
                example_header(i, n, task_name, cat.upper())
                print(f"    target={target}  answer={answer}")
                print(f"    sentences: {n_parsed}/{n_sentences} parsed")
                for j, sd in enumerate(sent_details[:5]):
                    status = "OK" if sd["parsed"] else "FAIL"
                    print(f"      [{j}] {sd['type']} {status}: "
                          f"{truncate(sd['response'], 80)}")
                if diag.get("error"):
                    print(f"    error: {diag['error']}")

        acc = categories.get("correct", 0) / len(examples) * 100 if examples else 0
        tier_results[task_name] = {
            "total": len(examples), "categories": dict(categories),
            "accuracy": acc, "examples": example_diags,
        }
        print(f"\n  {task_name}: {acc:.1f}% correct")
        for cat, count in sorted(categories.items()):
            print(f"    {cat}: {count}")

    # ── Navigate & Web of Lies: raw JSON extraction test ──
    for task_name in ["navigate", "web_of_lies"]:
        examples = load_bbh(task_name, 5)
        categories = Counter()
        example_diags = []

        for i, ex in enumerate(examples):
            prompt = ex["input"]
            target = ex["target"].strip()
            scene = parse_scene(prompt)

            # Test: can SmolLM2 produce valid JSON for each body sentence?
            n_sentences = len(scene.body)
            n_json_ok = 0
            sent_details = []

            for j, sent in enumerate(scene.body[:8]):  # cap at 8 sentences
                extraction_prompt = (
                    f"Extract a JSON triple from this sentence: "
                    f'{{\"subj\": \"...\", \"pred\": \"...\", \"obj\": \"...\"}}\n\n'
                    f"sentence: {sent}\n"
                )
                response, conf = generate_short(
                    model, tokenizer, device, extraction_prompt, max_tokens=60)

                # Try to parse JSON
                json_ok = False
                try:
                    # Find JSON in response
                    start = response.find("{")
                    end = response.rfind("}")
                    if start >= 0 and end > start:
                        parsed = json.loads(response[start:end+1])
                        if isinstance(parsed, dict):
                            json_ok = True
                except (json.JSONDecodeError, ValueError):
                    pass

                if json_ok:
                    n_json_ok += 1
                sent_details.append({
                    "sentence": truncate(sent, 80),
                    "json_ok": json_ok,
                    "response": truncate(response, 120),
                    "confidence": conf,
                })

            if n_json_ok == n_sentences and n_sentences > 0:
                cat = "all_json_valid"
            elif n_json_ok > 0:
                cat = "partial_json"
            else:
                cat = "no_json_at_all"

            categories[cat] += 1
            record = {
                "idx": i, "task": task_name, "category": cat,
                "target": target,
                "n_sentences": n_sentences, "n_json_ok": n_json_ok,
                "sentence_details": sent_details,
            }
            example_diags.append(record)

            if i < 3:
                example_header(i, 5, task_name, cat.upper())
                print(f"    target={target}  json_ok={n_json_ok}/{n_sentences}")
                for j, sd in enumerate(sent_details[:4]):
                    status = "JSON_OK" if sd["json_ok"] else "JSON_FAIL"
                    print(f"      [{j}] {status}: {sd['response']}")

        tier_results[task_name] = {
            "total": len(examples), "categories": dict(categories),
            "examples": example_diags,
        }
        print(f"\n  {task_name} (raw JSON test):")
        for cat, count in sorted(categories.items()):
            print(f"    {cat}: {count}")

    print(f"\n  --- Tier 2 Summary ---")
    for task, r in tier_results.items():
        print(f"    {task}: {r['categories']}")
    print(flush=True)

    return tier_results


# ═══════════════════════════════════════════════════════════════════════════════
# TIER 3: SQL Generation
# ═══════════════════════════════════════════════════════════════════════════════

def run_tier3(model, tokenizer, device):
    banner("TIER 3: SQL Generation")

    from turnstyle.sql import SQLTurnstyle

    tier_results = {}

    # ── Penguins ──
    task_name = "penguins_in_a_table"
    n = 10
    examples = load_bbh(task_name, n)
    sql_ts = SQLTurnstyle(model, tokenizer, device)

    categories = Counter()
    example_diags = []

    for i, ex in enumerate(examples):
        prompt = ex["input"]
        target = ex["target"].strip()

        diag: dict = {}
        result = sql_ts._sql_solve(prompt, diag=diag)

        tables_ok = diag.get("tables_parsed", False)
        raw_sql = diag.get("raw_sql")
        repaired = diag.get("repaired_sql")
        sql_err = diag.get("sql_error")
        sql_result = diag.get("sql_result")

        if result is not None:
            _, answer = result
            if target.lower() in answer.lower():
                cat = "correct"
            else:
                cat = "wrong_result"
        elif not tables_ok:
            cat = "no_tables"
        elif raw_sql is None:
            cat = "no_sql_output"
        elif sql_err:
            cat = "syntax_error"
        else:
            cat = "no_match"

        categories[cat] += 1
        record = {
            "idx": i, "task": task_name, "category": cat,
            "target": target,
            "tables_parsed": tables_ok,
            "raw_sql": raw_sql,
            "repaired_sql": repaired,
            "sql_error": sql_err,
            "sql_result": str(sql_result) if sql_result is not None else None,
            "question": diag.get("question"),
        }
        example_diags.append(record)

        if i < 3:
            example_header(i, n, task_name, cat.upper())
            print(f"    target={target}  tables_ok={tables_ok}")
            if raw_sql:
                print(f"    SQL: {truncate(raw_sql, 100)}")
            if repaired:
                print(f"    repaired: {truncate(repaired, 100)}")
            if sql_err:
                print(f"    error: {truncate(sql_err, 100)}")
            if sql_result is not None:
                print(f"    result: {sql_result}")
            if result:
                print(f"    answer: {result[1]}")

    acc = categories.get("correct", 0) / n * 100
    tier_results[task_name] = {
        "total": n, "categories": dict(categories),
        "accuracy": acc, "examples": example_diags,
    }
    print(f"\n  {task_name}: {acc:.1f}% correct")
    for cat, count in sorted(categories.items()):
        print(f"    {cat}: {count}")

    # Show failure details
    failures = [r for r in example_diags if r["category"] != "correct"]
    if failures:
        print(f"\n  Failure details ({task_name}):")
        for r in failures[:3]:
            print(f"    [{r['idx']}] {r['category']}")
            if r["raw_sql"]:
                print(f"      SQL: {truncate(r['raw_sql'], 100)}")
            if r["sql_error"]:
                print(f"      error: {truncate(r['sql_error'], 100)}")

    # ── Object counting ──
    task_name = "object_counting"
    n = 5
    examples = load_bbh(task_name, n)
    sql_ts2 = SQLTurnstyle(model, tokenizer, device)

    categories2 = Counter()
    example_diags2 = []

    for i, ex in enumerate(examples):
        prompt = ex["input"]
        target = ex["target"].strip()

        diag = {}
        result = sql_ts2._sql_solve(prompt, diag=diag)

        tables_ok = diag.get("tables_parsed", False)
        raw_sql = diag.get("raw_sql")
        sql_err = diag.get("sql_error")
        sql_result = diag.get("sql_result")

        if result is not None:
            _, answer = result
            if target.lower() in answer.lower():
                cat = "correct"
            else:
                cat = "wrong_result"
        elif not tables_ok:
            cat = "no_tables"
        elif raw_sql is None:
            cat = "no_sql_output"
        elif sql_err:
            cat = "syntax_error"
        else:
            cat = "no_match"

        categories2[cat] += 1
        record = {
            "idx": i, "task": task_name, "category": cat,
            "target": target,
            "tables_parsed": tables_ok,
            "raw_sql": raw_sql,
            "sql_error": sql_err,
            "sql_result": str(sql_result) if sql_result is not None else None,
            "question": diag.get("question"),
        }
        example_diags2.append(record)

        if i < 3:
            example_header(i, n, task_name, cat.upper())
            print(f"    target={target}  tables_ok={tables_ok}")
            if raw_sql:
                print(f"    SQL: {truncate(raw_sql, 100)}")
            if sql_err:
                print(f"    error: {truncate(sql_err, 100)}")
            if sql_result is not None:
                print(f"    result: {sql_result}")

    acc2 = categories2.get("correct", 0) / n * 100
    tier_results[task_name] = {
        "total": n, "categories": dict(categories2),
        "accuracy": acc2, "examples": example_diags2,
    }
    print(f"\n  {task_name}: {acc2:.1f}% correct")
    for cat, count in sorted(categories2.items()):
        print(f"    {cat}: {count}")

    print(f"\n  --- Tier 3 Summary ---")
    for task, r in tier_results.items():
        print(f"    {task}: {r['accuracy']:.1f}% — {r['categories']}")
    print(flush=True)

    return tier_results


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="SmolLM2 capability eval")
    parser.add_argument("--tier", type=int, action="append",
                        help="Which tier(s) to run (0-3). Omit for all.")
    args = parser.parse_args()

    tiers = set(args.tier) if args.tier else {0, 1, 2, 3}

    model, tokenizer, device = load_model()

    all_results = {}
    t0 = time.time()

    if 0 in tiers:
        all_results["tier0"] = run_tier0(model, tokenizer, device)
    if 1 in tiers:
        all_results["tier1"] = run_tier1(model, tokenizer, device)
    if 2 in tiers:
        all_results["tier2"] = run_tier2(model, tokenizer, device)
    if 3 in tiers:
        all_results["tier3"] = run_tier3(model, tokenizer, device)

    elapsed = time.time() - t0

    # ── Grand summary ──
    banner("GRAND SUMMARY")
    print(f"  Elapsed: {elapsed:.1f}s\n")

    if "tier0" in all_results:
        print("  Tier 0 (Logit Steering):")
        for task, r in all_results["tier0"].items():
            print(f"    {task}: {r['accuracy']:.1f}%")

    if "tier1" in all_results:
        print("\n  Tier 1 (Field Extraction, no regex):")
        for task, r in all_results["tier1"].items():
            print(f"    {task}: {r['accuracy']:.1f}% — {r['categories']}")

    if "tier2" in all_results:
        print("\n  Tier 2 (JSON Triple Extraction):")
        for task, r in all_results["tier2"].items():
            cats = r["categories"]
            acc = r.get("accuracy", "N/A")
            if isinstance(acc, (int, float)):
                print(f"    {task}: {acc:.1f}% — {cats}")
            else:
                print(f"    {task}: {cats}")

    if "tier3" in all_results:
        print("\n  Tier 3 (SQL Generation):")
        for task, r in all_results["tier3"].items():
            print(f"    {task}: {r['accuracy']:.1f}% — {r['categories']}")

    print(flush=True)


if __name__ == "__main__":
    main()
