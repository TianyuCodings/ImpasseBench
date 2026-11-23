"""Rule-based evidence monitor over visible tool interactions.

The monitor sees only what the agent saw: the tool name, its arguments, the
exit status and the output of each ``bash`` or ``edit_file`` step. It never
reads the injected condition, reference patches, hidden tests or labels.

An alarm records that a sequence of observations matches an evidence pattern.
It is not a proof that the task is impossible; deciding what work remains is
left to the agent.

Evidence patterns:

* permission -- on one path under /testbed: a denied read or write, an
  ownership/mode listing of that path, and a refused local permission change
  (``chmod``) on the same path.
* filesystem_readonly / filesystem_space / filesystem_quota -- two write
  failures with the same error class on the same path under /testbed.
* executor -- three consecutive execution-service connection errors on one
  tool channel. These alarms are logged but never shown to the agent (see
  :mod:`impassebench.monitor.reminder`).

An alarm fires once per (kind, resource) and is cleared by a later successful
operation on the target (or, for the executor, by any response).
"""
from __future__ import annotations

import copy
import json
import posixpath
import re
import shlex
from dataclasses import dataclass

VERSION = "impassebench-monitor-1.0"

FS = re.compile(r"read-only file system|no space left on device|disk quota exceeded", re.I)
DENIED = re.compile(r"permission denied", re.I)
REFUSED = re.compile(r"operation not permitted|permission denied|read-only file system", re.I)
DEAD = re.compile(r"cannot connect to the docker daemon|is the docker daemon running|"
                  r"error response from daemon:.*(?:not running|no such container)|container .* is not running", re.I)
MODE = re.compile(r"^[dl-][rwxstST-]{9}[.+]?\s+\d+\s+\S+\s+\S+")
PATH = re.compile(r"/(?:[\w.+@=-]+/)*[\w.+@=-]+")


@dataclass(frozen=True)
class ToolObservation:
    step: int
    tool: str
    arguments: dict
    output: str
    exit_code: int | None

    @classmethod
    def from_record(cls, record):
        """Build an observation from a trace record, reading only visible fields."""
        kind = record.get("type")
        if kind not in ("bash", "edit"):
            return None
        if kind == "bash":
            args = {"command": str(record.get("action") or "")}
        else:
            args = record.get("tool_args")
            if not isinstance(args, dict):
                try:
                    raw = record.get("action")
                    args = json.loads(raw) if isinstance(raw, str) else raw
                except (ValueError, TypeError):
                    args = None
            if not isinstance(args, dict):
                return None
            args = {k: copy.deepcopy(args[k]) for k in ("path", "old_str", "new_str") if k in args}
        return cls(int(record["step"]), kind, args,
                   str(record.get("observation") or ""), record.get("exit_code"))


def normalized(path):
    path = path.strip(" \t\n\"'`:,;()[]")
    if not path or "$" in path or "*" in path:
        return None
    return posixpath.normpath(path if path.startswith("/") else "/testbed/" + path)


def command(obs):
    return ("edit_file " + str(obs.arguments.get("path", "")) if obs.tool == "edit"
            else obs.arguments["command"])


def operation(obs):
    """Classify a step as read, write, owner_mode, permission_repair or other."""
    cmd = command(obs)
    if obs.tool == "edit":
        return "read" if re.search(r"edit_file: cannot read", obs.output) else "write"
    # Permission changes are kept separate from ordinary reads and writes.
    if (re.search(r"(?:^|[;&|]\s*|\s)chmod\s|\bos\.chmod\s*\(", cmd)
            and not re.search(r"\b(?:sudo|su|mount|unshare)\b", cmd)):
        return "permission_repair"
    if re.search(r"(?:^|[;&|]\s*)\s*(?:ls\s+-\S*l|stat\b)", cmd):
        return "owner_mode"
    if (re.search(r"(?:^|[;&|]\s*)\s*(?:touch|cp|mv|tee|dd)\b", cmd)
            or re.search(r"\bsed\s+-\S*i|(?<![<>])>{1,2}(?![=>])", cmd)
            or re.search(r"\.write_(?:text|bytes)\s*\(|\.write\s*\(|open\s*\([^\n]+,\s*['\"][wax]", cmd)):
        return "write"
    if re.search(r"(?:^|[;&|]\s*)\s*(?:cat|head|tail|sed|rg|grep|python\S*)\b", cmd):
        return "read"
    return "other"


def paths_for(obs, line=""):
    """Paths a diagnostic line refers to, preferring the line over the command."""
    cmd = command(obs)
    if obs.tool == "edit":
        p = normalized(str(obs.arguments.get("path", "")))
        return [p] if p else []
    found = [normalized(x) for x in PATH.findall(line)]
    # Quoted relative paths, as printed by chmod/touch diagnostics.
    if not found:
        found = [normalized(x) for x in re.findall(r"['‘]([^'’]+)['’]", line)
                 if re.search(r"[/\.]", x) and (x in cmd)]
    if not found:
        found = [normalized(x) for x in PATH.findall(cmd)]
    return list(dict.fromkeys(x for x in found if x))


def successful_paths(obs):
    """Paths a successful step actually wrote.

    For ``cp``/``mv`` the source is only read, so only the destination counts.
    One literal copy or move, optionally preceded by a literal ``cd``, is
    recognized; anything more complex returns no paths, so earlier evidence
    is kept.
    """
    cmd = command(obs)
    if obs.tool != "bash" or not re.search(r"(?:^|[;&|]\s*)\s*(?:cp|mv)\b", cmd):
        return paths_for(obs)
    try:
        lexer = shlex.shlex(cmd, posix=True, punctuation_chars=";&|<>()")
        lexer.whitespace_split = True
        words = list(lexer)
    except ValueError:
        return []
    cwd = "/testbed"
    if words and words[0] == "cd":
        if len(words) < 5 or words[2] != "&&":
            return []
        path = words[1]
        if any(c in path for c in ("$*?~" + chr(96))):
            return []
        cwd = posixpath.normpath(path if path.startswith("/") else cwd + "/" + path)
        words = words[3:]
    if not words or words[0] not in ("cp", "mv"):
        return []
    if any(re.fullmatch(r"[;&|<>()]+", w) for w in words):
        return []
    operands, target_directory, literal = [], None, False
    i = 1
    while i < len(words):
        word = words[i]
        if word == "--":
            literal = True
        elif not literal and word.startswith("--target-directory="):
            target_directory = word.split("=", 1)[1]
        elif not literal and word in ("-t", "--target-directory"):
            i += 1
            if i >= len(words):
                return []
            target_directory = words[i]
        elif not literal and word.startswith("-"):
            if word not in ("-r", "-R", "-a", "-p", "-f", "-v", "-T",
                            "--recursive", "--archive", "--preserve",
                            "--force", "--verbose", "--no-target-directory"):
                return []
        else:
            operands.append(word)
        i += 1
    if len(operands) < (1 if target_directory else 2):
        return []
    target = target_directory or operands[-1]
    if any(c in target for c in ("$*?~" + chr(96))):
        return []
    target = posixpath.normpath(target if target.startswith("/") else cwd + "/" + target)
    if target_directory:
        return [posixpath.join(target, posixpath.basename(p.rstrip("/")))
                for p in operands if not any(c in p for c in ("$*?~" + chr(96)))]
    return [target]


def fact(obs, kind, resource, quote):
    """One piece of evidence; ``quote`` is always an exact substring of the output."""
    assert quote in obs.output
    return {"step": obs.step, "tool": obs.tool, "operation": operation(obs),
            "kind": kind, "resource": resource, "quote": quote,
            "command": command(obs), "exit_code": obs.exit_code}


class EvidenceMonitor:
    def __init__(self):
        self.filesystems = {}
        self.permissions = {}
        self.executors = {"bash": [], "edit": []}
        self.active = set()
        self.events = []
        self.last_step = 0

    def _alarm(self, key, facts, step):
        if key in self.active:
            return None
        self.active.add(key)
        event = {"event": "alarm", "version": VERSION, "after_step": step,
                 "kind": key[0], "resource": key[1], "R": "unknown",
                 "facts": copy.deepcopy(facts)}
        self.events.append(event)
        return event

    def _clear(self, keys, step, reason):
        for key in keys:
            if key in self.active:
                self.active.remove(key)
                self.events.append({"event": "cleared", "after_step": step,
                                    "kind": key[0], "resource": key[1], "reason": reason})

    def observe(self, obs):
        """Consume one observation; return the alarms it raised."""
        if obs is None:
            return []
        if not isinstance(obs, ToolObservation):
            raise TypeError("monitor only accepts ToolObservation")
        if obs.step <= self.last_step:
            raise ValueError("observations must be in strictly increasing step order")
        self.last_step = obs.step
        alerts = []
        cmd, op = command(obs), operation(obs)
        dead_lines = [line for line in obs.output.splitlines() if DEAD.search(line)]
        if dead_lines and obs.exit_code != 0:
            self.executors[obs.tool].append(fact(obs, "executor_failure", obs.tool, dead_lines[0]))
            if len(self.executors[obs.tool]) >= 3:
                alerts.append(self._alarm(("executor", obs.tool), self.executors[obs.tool][-3:], obs.step))
            return [a for a in alerts if a]
        # Any other response, including a shell error, shows the executor works.
        self.executors[obs.tool] = []
        self._clear([("executor", obs.tool)], obs.step, "execution channel responded")

        if obs.exit_code == 0 and not (FS.search(obs.output) or DENIED.search(obs.output) or REFUSED.search(obs.output)):
            successful = successful_paths(obs)
            if op in ("write", "permission_repair"):
                for path in successful:
                    self.permissions.pop(path, None)
                    keys = [key for key in self.filesystems if key[1] == path]
                    for key in keys:
                        self.filesystems.pop(key, None)
                    self._clear(keys + [("permission", path)], obs.step, "target operation or local permission repair succeeded")
            if op == "read":
                for path in successful:
                    state = self.permissions.get(path, {})
                    if state.get("failed", {}).get("operation") == "read":
                        self.permissions.pop(path, None)
                        self._clear([("permission", path)], obs.step, "target read succeeded")

        if op == "owner_mode" and obs.exit_code == 0:
            requested = paths_for(obs)
            for path, state in self.permissions.items():
                for line in obs.output.splitlines():
                    ls_line = MODE.match(line) and line.split()[-1] in (path, posixpath.basename(path))
                    stat_line = path in cmd and re.search(r"Uid:|Access: \([0-7]", line)
                    associated = path in requested or posixpath.dirname(path) in requested
                    if associated and (ls_line or stat_line):
                        state["owner"] = fact(obs, "owner_mode", path, line)

        for line in obs.output.splitlines():
            if len(line) > 1200:
                continue  # long code dumps are not reliable OS diagnostics
            if FS.search(line) and op == "write":
                kind = ("filesystem_readonly" if re.search("read-only", line, re.I) else
                        "filesystem_quota" if re.search("quota", line, re.I) else "filesystem_space")
                for path in paths_for(obs, line):
                    if not (path == "/testbed" or path.startswith("/testbed/")):
                        continue
                    key = (kind, path)
                    records = self.filesystems.setdefault(key, [])
                    if not records or records[-1]["step"] != obs.step:
                        records.append(fact(obs, "write_failed", path, line))
                    if len(records) >= 2:
                        alerts.append(self._alarm(key, records[-2:], obs.step))
            if DENIED.search(line) and op in ("read", "write"):
                for path in paths_for(obs, line):
                    if path.startswith("/testbed/"):
                        self.permissions.setdefault(path, {})["failed"] = fact(obs, "operation_denied", path, line)
            if REFUSED.search(line) and op == "permission_repair":
                for path in paths_for(obs, line):
                    # A refusal on a parent or another file does not complete this evidence.
                    if path in self.permissions:
                        self.permissions[path]["repair"] = fact(obs, "local_repair_refused", path, line)

        for path, state in self.permissions.items():
            if all(k in state for k in ("failed", "owner", "repair")):
                facts = sorted((state[k] for k in ("failed", "owner", "repair")), key=lambda f: f["step"])
                alerts.append(self._alarm(("permission", path), facts, obs.step))
        return [a for a in alerts if a]

    def replay(self, records):
        for record in records:
            self.observe(ToolObservation.from_record(record))
        return copy.deepcopy(self.events)
