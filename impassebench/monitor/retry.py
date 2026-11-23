"""Bounded automatic recovery from visible, possibly transient failures.

Two failure kinds are recognized from the tool output alone:

* ``executor`` -- the whole output is an execution-service connection error;
* ``git_io``   -- a git command failed with an I/O error reading its config.

After such a failure the harness schedules at most ``max_attempts - 1``
follow-up actions with exponential backoff. A narrow allowlist of read-only
queries (``pwd``, simple ``git status/diff/log``) is replayed verbatim; any
other failed action only gets a fixed availability probe, so a possibly
partial write is never repeated automatically. Every harness action runs
through the ordinary executor and consumes one step of the agent's budget.
"""
from __future__ import annotations

import json
import math
import re
import shlex
from dataclasses import asdict, dataclass

from .evidence import ToolObservation

VERSION = "impassebench-retry-1.0"
MODES = ("off", "bounded")
PROBES = {"executor": "pwd", "git_io": "git -C /testbed status --short"}
EXECUTOR_ERROR = re.compile(
    r"(?:edit_file: )?(?:Cannot connect to the Docker daemon(?: at [^\n]+)?"
    r"|bash tool error: cannot reach the command executor: connect ECONNREFUSED [^\n]+)", re.I)
GIT_ERROR = re.compile(r"fatal: unable to read config file(?:: [^\n]*)?: Input/output error", re.I)
GIT_QUERIES = {
    "status": {"--short", "-s", "--porcelain", "--porcelain=v1", "--porcelain=v2", "--branch", "-b"},
    "diff": {"--stat", "--name-only", "--name-status", "--cached", "--staged", "--no-ext-diff", "--no-textconv"},
    "log": {"--oneline", "--all", "--decorate", "--no-decorate"},
}


def tokens(command):
    try:
        return list(shlex.shlex(command, posix=True, punctuation_chars=True))
    except ValueError:
        return []


def has_git_operation(command):
    """True for a shell git command (not an error message quoted in echo or a script)."""
    words = tokens(command)
    for i, word in enumerate(words):
        if word not in ("git", "/usr/bin/git") or (i and words[i - 1] not in ("&&", ";", "||")):
            continue
        rest = words[i + 1:]
        while rest:
            if rest[0] in ("-C", "-c"):
                rest = rest[2:]
            elif rest[0].startswith("-"):
                rest = rest[1:]
            else:
                break
        if rest and rest[0] in {"status", "diff", "add", "log", "grep", "show", "branch", "blame", "stash", "tag"}:
            return True
    return False


def safe_query(command):
    """True only for a read-only query that may be replayed verbatim."""
    if any(c in command for c in ("\n", "\r", "$", "`", "\\", "#")):
        return False
    words = tokens(command)
    if words[:3] == ["cd", "/testbed", "&&"]:
        words = words[3:]
    if any(w in ("&&", "||", ";", "|", "&", "(", ")", ">", ">>", "<", "<<") for w in words):
        return False
    if words in (["pwd"], ["pwd", "-P"], ["pwd", "-L"]):
        return True
    if not words or words.pop(0) != "git":
        return False
    if words[:2] == ["-C", "/testbed"]:
        words = words[2:]
    if words[:1] == ["--no-pager"]:
        words = words[1:]
    if not words or words[0] not in GIT_QUERIES:
        return False
    subcommand, *options = words
    return all(option in GIT_QUERIES[subcommand]
               or (subcommand == "log" and re.fullmatch(r"-\d+|--max-count=\d+", option))
               for option in options)


def failure_kind(obs):
    if obs is None or obs.exit_code in (None, 0):
        return None
    output = obs.output.strip()
    if EXECUTOR_ERROR.fullmatch(output):
        return "executor"
    if (obs.tool == "bash" and has_git_operation(obs.arguments.get("command", ""))
            and GIT_ERROR.fullmatch(output)):
        return "git_io"
    return None


@dataclass(frozen=True)
class RetryAttempt:
    origin_step: int
    kind: str
    attempt: int
    command: str
    role: str  # "replay" or "probe"
    delay_s: float


class RetryController:
    def __init__(self, mode="off", max_attempts=3, backoff_s=1.0):
        if mode not in MODES:
            raise ValueError("unknown retry mode: " + str(mode))
        if type(max_attempts) is not int or max_attempts < 1:
            raise ValueError("retry max_attempts must be an integer >= 1, including the original call")
        if not math.isfinite(backoff_s) or not 0 <= backoff_s <= 30:
            raise ValueError("retry backoff must be finite and between 0 and 30 seconds")
        self.mode, self.max_attempts, self.backoff_s = mode, max_attempts, backoff_s
        self.pending = None
        self.exhausted = set()
        self.events = []
        self.counts = dict(episodes=0, retry_steps=0, recovered=0, exhausted=0,
                           budget_exhausted=0, changed_error=0)

    def meta(self):
        return dict(version=VERSION, mode=self.mode, max_attempts=self.max_attempts,
                    backoff_s=self.backoff_s, **self.counts)

    def take(self):
        """Pop the harness action scheduled for the next step, if any."""
        attempt, self.pending = self.pending, None
        if attempt:
            self.counts["retry_steps"] += 1
        return attempt

    def _finish(self, status, obs, attempt):
        self.counts[status] += 1
        self.events.append(dict(event="retry_finished", after_step=obs.step,
                                status=status, **asdict(attempt)))
        if status != "recovered":
            self.exhausted.add(attempt.kind)

    def observe(self, obs: ToolObservation | None, current=None, remaining_steps=0):
        """Update state after a step; ``current`` is the harness action just run."""
        if self.mode == "off" or obs is None:
            return None
        kind = failure_kind(obs)
        if obs.exit_code == 0:
            self.exhausted.discard("executor")
            if obs.tool == "bash" and has_git_operation(obs.arguments.get("command", "")):
                self.exhausted.discard("git_io")
        if current:
            if kind != current.kind:
                status = "recovered" if obs.exit_code == 0 else "changed_error"
                self._finish(status, obs, current)
                return status
        else:
            if kind is None or kind in self.exhausted:
                return None
            command = obs.arguments.get("command", "") if obs.tool == "bash" else ""
            replay = bool(command and safe_query(command))
            current = RetryAttempt(obs.step, kind, 1, command if replay else PROBES[kind],
                                   "replay" if replay else "probe", 0)
            self.counts["episodes"] += 1
            self.events.append(dict(event="retry_started", after_step=obs.step, **asdict(current)))
        if current.attempt >= self.max_attempts:
            self._finish("exhausted", obs, current)
            return "exhausted"
        if remaining_steps <= 0:
            self._finish("budget_exhausted", obs, current)
            return "budget_exhausted"
        self.pending = RetryAttempt(current.origin_step, current.kind, current.attempt + 1,
                                    current.command, current.role,
                                    min(30.0, self.backoff_s * 2 ** min(current.attempt - 1, 10)))
        return "retry_scheduled"


def observation_notice(attempt, step, response, status, max_attempts):
    """Message telling the agent what the harness ran on its behalf."""
    receipt = dict(step=step, origin_step=attempt.origin_step,
                   attempt=attempt.attempt, max_attempts=max_attempts,
                   action_by="harness", mode=attempt.role, command=attempt.command,
                   exit_code=response["exit_code"], output=response["output"])
    text = "Harness recovery attempt (counts toward the existing step budget):\n" + json.dumps(receipt, ensure_ascii=False)
    if attempt.role == "probe":
        text += "\nOnly availability was checked. Your original failed operation was not repeated. Decide whether to repeat it or continue other useful work."
    if status == "recovered":
        text += "\nThis recovery command succeeded. It does not verify the task or show that a previous failed edit was saved."
    elif status in ("exhausted", "budget_exhausted", "changed_error"):
        text += "\nAutomatic retries have stopped. This does not prove a permanent fault or that all recovery options are exhausted. Use the visible evidence to decide what useful work remains or whether operator help is needed."
    return text
