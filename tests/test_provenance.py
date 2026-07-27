"""Tests for results-file provenance.

The failure this prevents: a results JSON dated five weeks earlier was read as
current, producing a wrong premise about which BBH tasks had solvers. The tests
below pin the properties that make that hard to repeat — chiefly that an UNSTAMPED
or STALE file announces itself at read time, and that unknown age never reads as
fresh.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from turnstyle.provenance import (age_days, describe, load_json, require_fresh,
                                  stamp, write_json)


def _aged(payload, days):
    """Same payload with its stamp backdated."""
    out = json.loads(json.dumps(payload))
    when = datetime.now(timezone.utc) - timedelta(days=days)
    out["_provenance"]["written_at"] = when.isoformat(timespec="seconds")
    return out


def test_stamp_records_time_and_is_non_destructive():
    src = {"acc": 0.9}
    out = stamp(src)
    assert out["acc"] == 0.9
    assert "_provenance" not in src                 # input untouched
    assert out["_provenance"]["written_at"]
    assert age_days(out) < 1.0


def test_round_trip_through_disk(tmp_path):
    p = write_json(tmp_path / "r.json", {"acc": 0.5})
    payload, prov = load_json(p)
    assert payload["acc"] == 0.5 and prov["written_at"]


def test_unstamped_age_is_none_not_zero():
    """An unknown age must not be indistinguishable from a fresh one — that
    equivalence is the whole bug."""
    assert age_days({"acc": 1.0}) is None
    assert "UNSTAMPED" in describe({"acc": 1.0})


def test_stale_file_warns_on_read(tmp_path, capsys):
    p = tmp_path / "old.json"
    p.write_text(json.dumps(_aged(stamp({"acc": 1.0}), 40)))
    load_json(p, max_age_days=14)
    err = capsys.readouterr().err
    assert "DAYS OLD" in err and "old.json" in err


def test_unstamped_file_also_warns(tmp_path, capsys):
    p = tmp_path / "bare.json"
    p.write_text(json.dumps({"acc": 1.0}))
    load_json(p)
    assert "UNSTAMPED" in capsys.readouterr().err


def test_fresh_file_is_quiet(tmp_path, capsys):
    p = write_json(tmp_path / "new.json", {"acc": 1.0})
    load_json(p, max_age_days=14)
    assert capsys.readouterr().err == ""


def test_require_fresh_raises_on_stale_and_on_unstamped(tmp_path):
    stale = tmp_path / "s.json"
    stale.write_text(json.dumps(_aged(stamp({"a": 1}), 40)))
    with pytest.raises(ValueError, match="days old"):
        require_fresh(stale, max_age_days=14)

    bare = tmp_path / "b.json"
    bare.write_text(json.dumps({"a": 1}))
    with pytest.raises(ValueError, match="unstamped"):
        require_fresh(bare)

    ok = write_json(tmp_path / "ok.json", {"a": 1})
    assert require_fresh(ok, max_age_days=14)["a"] == 1


def test_describe_surfaces_dirty_tree():
    """A result produced from a dirty tree is not reproducible from its SHA, so the
    flag has to survive into the human-readable summary."""
    s = stamp({"a": 1})
    s["_provenance"].update(git_sha="abc1234", git_dirty=True)
    assert "abc1234-dirty" in describe(s)
