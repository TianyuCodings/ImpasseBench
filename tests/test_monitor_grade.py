"""Grading helpers (no Docker, no SWE-bench run)."""
import pytest

from impassebench.monitor.grade import RESET_MARKER, classify_report, load_oracle, repair_eval_script

BASE = "a" * 40
RECORD = {"base_commit": BASE,
          "test_patch": "diff --git a/tests/test_x.py b/tests/test_x.py\n--- a/tests/test_x.py\n+++ b/tests/test_x.py\n",
          "eval_script": f"cd /testbed\ngit checkout {BASE} tests/test_x.py\ngit apply -v - <<'EOF'\nEOF\npytest\ngit checkout {BASE} tests/test_x.py\n"}


def test_repair_eval_script_changes_only_the_two_resets():
    repaired, changes = repair_eval_script(RECORD)
    assert len(changes) == 2 and repaired.count(RESET_MARKER) == 4
    assert "git checkout " not in repaired.replace("['git', 'checkout'", "")
    restored = repaired
    for c in changes:
        restored = restored.replace(c["replacement"], c["original"], 1)
    assert restored == RECORD["eval_script"]


def test_repair_rejects_paths_outside_the_test_patch():
    bad = dict(RECORD, eval_script=RECORD["eval_script"].replace("tests/test_x.py", "setup.py"))
    with pytest.raises(AssertionError):
        repair_eval_script(bad)


def test_classify_report():
    report = {"submitted_ids": ["a", "b", "c"], "completed_ids": ["a", "b"], "resolved_ids": ["a"],
              "unresolved_ids": ["b"], "error_ids": ["c"]}
    out = classify_report(report, ["a", "b", "c"])
    assert [out[i]["resolved"] for i in "abc"] == [True, False, None]


def test_oracle_resource():
    rows = load_oracle()
    assert len(rows) == 149
    assert {"eval_script", "log_parser", "image", "patch", "test_patch"} <= set(rows[0])
