"""Bounded retry policy."""
import pytest

from impassebench.monitor.evidence import ToolObservation
from impassebench.monitor.retry import RetryController, failure_kind, observation_notice, safe_query

GIT_IO = "fatal: unable to read config file: Input/output error"  # as printed by the git_flaky fault
DEAD = "Cannot connect to the Docker daemon at unix:///var/run/docker.sock. Is the docker daemon running?"


def bash(step, cmd, out, rc):
    return ToolObservation(step, "bash", {"command": cmd}, out, rc)


def test_failure_kinds():
    assert failure_kind(bash(1, "git status", GIT_IO, 128)) == "git_io"
    assert failure_kind(bash(1, "echo fatal", GIT_IO, 128)) is None  # not a git command
    assert failure_kind(bash(1, "ls", DEAD, 1)) == "executor"
    assert failure_kind(bash(1, "ls", DEAD + "\nmore", 1)) is None  # whole output must match
    assert failure_kind(bash(1, "ls", "", 0)) is None


def test_safe_queries():
    assert safe_query("git status --short")
    assert safe_query("cd /testbed && git log --oneline -5")
    assert not safe_query("git status > out.txt")
    assert not safe_query("git commit -m x")
    assert not safe_query("git diff $REV")


def test_replay_then_recover():
    rc = RetryController("bounded", max_attempts=3, backoff_s=1.0)
    assert rc.observe(bash(1, "git status --short", GIT_IO, 128), remaining_steps=79) == "retry_scheduled"
    a = rc.take()
    assert (a.command, a.role, a.attempt, a.delay_s) == ("git status --short", "replay", 2, 1.0)
    assert rc.observe(bash(2, a.command, " M a.py\n", 0), current=a, remaining_steps=78) == "recovered"
    assert rc.meta()["recovered"] == 1 and rc.take() is None


def test_probe_and_exhaustion():
    rc = RetryController("bounded", max_attempts=3, backoff_s=1.0)
    rc.observe(bash(1, "python -m pytest", DEAD, 1), remaining_steps=10)
    a = rc.take()
    assert (a.command, a.role) == ("pwd", "probe")
    assert rc.observe(bash(2, "pwd", DEAD, 1), current=a, remaining_steps=9) == "retry_scheduled"
    b = rc.take()
    assert (b.attempt, b.delay_s) == (3, 2.0)
    assert rc.observe(bash(3, "pwd", DEAD, 1), current=b, remaining_steps=8) == "exhausted"
    # the same kind is not retried again until the channel works
    assert rc.observe(bash(4, "ls", DEAD, 1), remaining_steps=7) is None
    rc.observe(bash(5, "ls", "", 0), remaining_steps=6)
    assert rc.observe(bash(6, "ls", DEAD, 1), remaining_steps=5) == "retry_scheduled"


def test_budget_and_off_mode():
    rc = RetryController("bounded")
    assert rc.observe(bash(80, "git status", GIT_IO, 128), remaining_steps=0) == "budget_exhausted"
    assert RetryController("off").observe(bash(1, "git status", GIT_IO, 128), remaining_steps=5) is None
    with pytest.raises(ValueError):
        RetryController("always")


def test_observation_notice_mentions_probe():
    rc = RetryController("bounded")
    rc.observe(bash(1, "python x.py", DEAD, 1), remaining_steps=10)
    a = rc.take()
    text = observation_notice(a, 2, {"exit_code": 0, "output": "/testbed\n"}, "recovered", 3)
    assert text.startswith("Harness recovery attempt (counts toward the existing step budget):")
    assert "Your original failed operation was not repeated." in text
    assert "This recovery command succeeded." in text
