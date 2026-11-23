"""Faults injected between the agent and the sandbox.

For ``sandbox_lost_early``, ``sandbox_lost_late`` and ``sandbox_flaky`` the
container stays healthy; the rollout loop asks a :class:`RuntimeFaultController`
before dispatching each bash/edit call and, when the fault is active, returns
the injected error instead of executing the call. Setup and patch extraction
are therefore unaffected.

Spec keys: ``kind`` (``sandbox_dead`` | ``sandbox_flaky``), ``trigger`` for
``sandbox_dead`` (``first_bash`` | ``first_source_write``), and ``at_step`` /
``window`` for ``sandbox_flaky``. An optional ``visible_message`` overrides
the injected error text.
"""

from __future__ import annotations

import re

SANDBOX_DEAD_MSG = ("Cannot connect to the Docker daemon at "
                    "unix:///var/run/docker.sock. Is the docker daemon running?")

# A test-runner invocation at command position (env prefixes and wrappers
# allowed); inspection flags such as --version or --collect-only do not count.
_TEST_RUN_TRIGGER_RX = re.compile(
    r"(?:^\s*|[;&|(]\s*)(?:[A-Za-z_][A-Za-z0-9_]*=\S*\s+)*"
    r"(?:timeout\s+(?:-\S+\s+)*\d+\s+|time\s+|xvfb-run\s+)?"
    r"(?:(?:\S*/)?python[0-9.]*\s+(?:-[a-zA-Z]\s+)*"
    r"(?:-m\s+(?:pytest|unittest)\b|\S*runtests?\.py\b|\S*setup\.py\s+test\b)"
    r"|(?:\S*/)?(?:pytest|py\.test|nosetests)\b|tox\b|\.?/?\S*runtests?\.py\b)"
    r"(?![^|;&\n]*(?:--version|--help|--collect-only|--co\b))", re.M)

# Modified tracked files relative to the sealed baseline (root-side check).
SRC_WRITE_CHECK_CMD = ("git -C /testbed --git-dir=/opt/t2e_baseline.git "
                       "--work-tree=/testbed -c core.fileMode=false "
                       "diff --name-only --diff-filter=M HEAD")

# Test files at any depth, and files that patch extraction strips, do not
# count as a source write.
_TRIGGER_TEST_PATH_RX = re.compile(
    r"(?:^|/)(?:tests?|testing)/|(?:^|/)test_[^/]*\.py$|_test\.py$")
_TRIGGER_STRIPPED_RX = re.compile(
    r"(?:^|/)(?:conftest\.py|sitecustomize\.py|usercustomize\.py)$|\.pth$"
    r"|\.pyc$|(?:^|/)(?:__pycache__|\.pytest_cache|\.mypy_cache|"
    r"site-packages|node_modules)/|(?:^|/)\.?venv/")


def src_write_hits(output: str) -> list[str]:
    """Modified tracked paths that count as deliverable source."""
    return [p for p in output.splitlines()
            if p.strip() and not _TRIGGER_TEST_PATH_RX.search(p)
            and not _TRIGGER_STRIPPED_RX.search(p)]


class RuntimeFaultController:
    """Per-rollout state machine.

    ``intercept()`` is asked before a bash/edit call; a non-None return is the
    injected observation and the call must not reach the container.
    ``observe()`` is told about every call that really executed and advances
    the trigger, so the call that satisfies the trigger completes normally and
    the fault applies from the next call on.
    """

    def __init__(self, spec: dict):
        self.spec = dict(spec)
        self.armed_step: int | None = None
        self.first_failed_step: int | None = None
        self.n_failed = 0

    def intercept(self, tool: str, step: int) -> str | None:
        """The injected error if this call must fail, else None."""
        kind = self.spec["kind"]
        if kind == "sandbox_flaky":
            # the window counts failed tool calls, not loop steps
            active = step >= self.spec["at_step"] \
                and self.n_failed < self.spec["window"]
        else:
            active = self.armed_step is not None
        if not active:
            return None
        if self.first_failed_step is None:
            self.first_failed_step = step
        self.n_failed += 1
        msg = self.spec.get("visible_message") or SANDBOX_DEAD_MSG
        return f"edit_file: {msg}" if tool == "edit" else msg

    def pre_dispatch(self, tool: str, action: str, step: int,
                     src_written=None) -> None:
        """If a source change is already on disk when a test run is
        requested, arm before dispatch so that run already fails."""
        if (self.armed_step is None
                and self.spec.get("trigger") == "first_source_write"
                and tool == "bash"
                and _TEST_RUN_TRIGGER_RX.search(action or "")
                and src_written is not None and src_written()):
            self.armed_step = step

    def observe(self, tool: str, action: str, step: int, exit_code=None,
                observation: str = "", src_written=None) -> None:
        """Advance the trigger after a call that executed for real."""
        if self.armed_step is not None:
            return
        trig = self.spec.get("trigger")
        if trig == "first_bash" and tool == "bash" and exit_code == 0:
            # one successful call first, so the loss is clearly mid-run
            self.armed_step = step
        elif trig == "first_source_write" and src_written is not None \
                and src_written():
            self.armed_step = step

    def meta(self) -> dict:
        return {"kind": self.spec["kind"], "trigger": self.spec.get("trigger"),
                "armed_step": self.armed_step,
                "first_failed_step": self.first_failed_step,
                "n_failed_calls": self.n_failed}
