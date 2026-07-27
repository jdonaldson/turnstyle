"""Date-stamp results files, and make staleness loud when they are READ.

Motivated by a real error: `experiments/data/bbh_native_report.json` was dated
2026-06-21 and got cited as current five weeks later, producing a wrong premise
(11 BBH tasks "have no solver" when 9 of them had been solved in the interim).

The asymmetry that caused it is mechanical, not a lapse of attention. Memory files
print an age warning automatically when recalled; results JSONs print nothing. So a
five-week-old number and a five-minute-old number look identical at the point of
use — which is exactly the point where the difference matters.

WRITE side stamps `_provenance` (timestamp, git SHA, dirty flag). READ side is the
one that earns its keep: `load_json` warns when a file is older than `max_age_days`,
and `require_fresh` raises instead, for cases where acting on a stale number would
be worse than stopping.

    from turnstyle.provenance import write_json, load_json
    write_json(path, report)
    report, prov = load_json(path)        # warns on stderr if stale
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_MAX_AGE_DAYS = 14
_KEY = "_provenance"


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(("git", *args), capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or None if out.returncode == 0 else None
    except Exception:
        return None


def stamp(payload: dict) -> dict:
    """Return payload with a `_provenance` block. Non-destructive on the input."""
    sha = _git("rev-parse", "--short", "HEAD")
    dirty = _git("status", "--porcelain")
    return {**payload, _KEY: {
        "written_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_sha": sha,
        # a result produced from a dirty tree cannot be reproduced from its SHA
        "git_dirty": bool(dirty) if dirty is not None else None,
    }}


def write_json(path, payload: dict, indent: int = 2) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(stamp(payload), indent=indent, default=str))
    return p


def age_days(payload: dict) -> float | None:
    """Age from the stamp. None when unstamped — which must NOT read as fresh."""
    ts = (payload.get(_KEY) or {}).get("written_at")
    if not ts:
        return None
    try:
        when = datetime.fromisoformat(ts)
    except ValueError:
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    return (datetime.now(timezone.utc) - when).total_seconds() / 86400.0


def describe(payload: dict) -> str:
    prov = payload.get(_KEY)
    if not prov:
        return "UNSTAMPED (age unknown — do not assume current)"
    a = age_days(payload)
    bits = [f"{a:.1f}d old" if a is not None else "age unknown"]
    if prov.get("git_sha"):
        bits.append(prov["git_sha"] + ("-dirty" if prov.get("git_dirty") else ""))
    return ", ".join(bits)


def load_json(path, max_age_days: float = DEFAULT_MAX_AGE_DAYS, warn: bool = True):
    """Load a results file and WARN on stderr if it is stale or unstamped.

    Returns (payload, provenance_dict). Unstamped files warn too: an unknown age is
    not evidence of freshness, and every file predating this module is unstamped.
    """
    p = Path(path)
    payload = json.loads(p.read_text())
    prov = payload.get(_KEY) or {}
    a = age_days(payload)
    if warn:
        if a is None:
            print(f"[provenance] {p.name}: UNSTAMPED — age unknown, verify before "
                  f"citing as current", file=sys.stderr)
        elif a > max_age_days:
            print(f"[provenance] {p.name}: {a:.0f} DAYS OLD "
                  f"(sha {prov.get('git_sha') or '?'}) — regenerate before citing",
                  file=sys.stderr)
    return payload, prov


def require_fresh(path, max_age_days: float = DEFAULT_MAX_AGE_DAYS) -> dict:
    """Like load_json but RAISES on stale/unstamped. Use where acting on an
    out-of-date number is worse than stopping."""
    payload, _ = load_json(path, max_age_days, warn=False)
    a = age_days(payload)
    if a is None:
        raise ValueError(f"{Path(path).name} is unstamped — cannot confirm freshness")
    if a > max_age_days:
        raise ValueError(f"{Path(path).name} is {a:.0f} days old "
                         f"(limit {max_age_days}) — regenerate it")
    return payload
