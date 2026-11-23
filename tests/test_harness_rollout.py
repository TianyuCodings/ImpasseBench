"""Rollout loop against an in-memory container (no Docker, no network)."""
import base64
import json
import re
from dataclasses import dataclass

import pytest

import impassebench.harness.rollout as rollout
from impassebench.harness import run as runner
from impassebench.harness.agents import ScriptedAgent
from impassebench.harness.rollout import RolloutHooks, run_rollout
from impassebench.harness.runtime_faults import SANDBOX_DEAD_MSG
from impassebench.harness.tasks import load_tasks

TASK = next(t for t in load_tasks() if t["repo"].endswith("sqlparse.e57923b3"))


@dataclass
class R:
    exit_code: int
    output: str
    truncated: bool = False
    wall_time_s: float = 0.0


BASE = {"/testbed/pkg/mod.py": "def f(x):\n    return x + 1\n",
        "/testbed/tests/test_mod.py": "def test_f():\n    pass\n"}


class FakeSession:
    """Answers the commands the loop itself issues; agent commands exit 0
    unless they contain FAIL, and MODIFY:<path> appends to a file."""

    def __init__(self, image, name, extra_run_args=None, agent_env=None, agent_user=None):
        self.image, self.name = image, name
        self.extra_run_args = extra_run_args or []
        self.agent_env, self.agent_user = agent_env or {}, agent_user
        self.files = dict(BASE)
        self.agent_cmds = []

    def start(self):
        pass

    def stop(self):
        pass

    def exec_agent(self, command, timeout_s=180, limit=8000, stdin_data=None):
        m = re.match(r"test -e '?([^' ;]+)'? \|\| exit 66; echo __T2E_B64_BEGIN__", command)
        if m:
            if m.group(1) not in self.files:
                return R(66, "")
            b = base64.b64encode(self.files[m.group(1)].encode()).decode()
            return R(0, f"__T2E_B64_BEGIN__\n{b}\n__T2E_B64_END__\n")
        m = re.match(r"base64 -d > '?([^' ]+)'?$", command)
        if m:
            self.files[m.group(1)] = base64.b64decode(stdin_data).decode()
            return R(0, "")
        self.agent_cmds.append(command)
        for p in re.findall(r"MODIFY:(\S+)", command):
            self.files[p] = self.files.get(p, "") + "# changed\n"
        return R(1 if "FAIL" in command else 0, f"out:{command}\n")

    def exec_setup(self, command, timeout_s=120):
        return R(0, "")

    def exec_root(self, command, timeout_s=60, limit=100_000, merge_stderr=True):
        if "--diff-filter=M" in command:
            changed = [p[len("/testbed/"):] for p in BASE if self.files[p] != BASE[p]]
            return R(0, "".join(c + "\n" for c in changed))
        return R(0, "")

    def git_diff(self):
        changed = [p for p in sorted(self.files) if BASE.get(p) != self.files[p]]
        return "".join(f"+++ b{p[8:]}\n" for p in changed), None


@pytest.fixture(autouse=True)
def no_docker(monkeypatch):
    monkeypatch.setattr(rollout, "ContainerSession", FakeSession)
    monkeypatch.setattr(rollout, "prepare_swesmith_image", lambda im, iid, checkout=True: "img")
    monkeypatch.setattr(rollout, "prepare_verified_image", lambda im: "img")
    monkeypatch.setattr(rollout, "image_digest_of", lambda im: None)


def bash(cmd):
    return {"tool": "bash", "args": {"command": cmd}, "text": "", "call_id": "c"}


def edit(path, old, new):
    return {"tool": "edit_file", "args": {"path": path, "old_str": old, "new_str": new},
            "text": "", "call_id": "c"}


DONE = {"tool": "task_complete", "args": {"summary": "s", "verified": False}}
ESC = {"tool": "escalate_to_human", "args": {"summary": "blocked"}}
TEXT = {"tool": None, "text": "hmm"}


def go(tmp_path, condition, script, interface="escalate", max_steps=80, hooks=None, live=False):
    agent = ScriptedAgent(script, live=live)
    meta = run_rollout(TASK, condition, agent, interface=interface,
                       max_steps=max_steps, out_dir=tmp_path, hooks=hooks)
    d = next(tmp_path.iterdir())
    trace = [json.loads(l) for l in (d / "trace.jsonl").read_text().splitlines()]
    assert json.loads((d / "meta.json").read_text())["termination"] == meta["termination"]
    return meta, trace


def test_clean_edit_and_complete(tmp_path):
    meta, trace = go(tmp_path, "clean", [
        bash("ls"), edit("pkg/mod.py", "x + 1", "x + 2"), edit("pkg/mod.py", "nope", "y"),
        edit("pkg/none.py", "a", "b"), DONE])
    assert meta["termination"] == "task_complete" and meta["n_steps"] == 5
    assert [s["type"] for s in trace] == ["bash", "edit", "edit", "edit", "task_complete"]
    assert trace[1]["observation"] == "edited /testbed/pkg/mod.py (+0 lines)"
    assert "old_str not found" in trace[2]["observation"] and trace[2]["exit_code"] == 1
    assert "does not exist" in trace[3]["observation"]
    assert trace[1]["tool_args"]["new_str"] == "x + 2"
    assert meta["model_patch"] == "+++ b/pkg/mod.py\n"
    assert meta["interface"] == "escalate" and meta["condition"] == "clean"
    assert set(meta) == {
        "run_id", "instance_id", "repo", "condition", "interface", "with_escalate",
        "with_edit", "task_preview", "max_steps", "started_at", "termination", "n_steps",
        "total_tokens", "model_patch", "model", "temperature", "sampling", "base_image",
        "base_image_digest", "docker_run_args", "setup_cmds", "runtime_fault_spec",
        "agent_env", "agent_user", "system_prompt", "patch_extraction_error",
        "runtime_fault", "ended_at"}


def test_sandbox_lost_early(tmp_path):
    meta, trace = go(tmp_path, "sandbox_lost_early", [
        bash("FAIL"), bash("ls"), bash("ls"), edit("pkg/mod.py", "x", "y"), ESC])
    assert meta["termination"] == "escalate"
    assert trace[0]["exit_code"] == 1 and "fault_injected" not in trace[1]
    assert trace[2]["observation"] == SANDBOX_DEAD_MSG and trace[2]["fault_injected"]
    assert trace[3]["observation"] == "edit_file: " + SANDBOX_DEAD_MSG
    assert meta["runtime_fault"] == {"kind": "sandbox_dead", "trigger": "first_bash",
                                     "armed_step": 2, "first_failed_step": 3,
                                     "n_failed_calls": 2}


def test_sandbox_lost_late_after_source_write(tmp_path):
    meta, trace = go(tmp_path, "sandbox_lost_late", [
        bash("MODIFY:/testbed/tests/test_mod.py"), bash("python -m pytest"),
        edit("pkg/mod.py", "x + 1", "x + 2"), bash("python -m pytest"), DONE])
    assert "fault_injected" not in trace[1]          # test edits do not arm it
    assert trace[3]["fault_injected"] and meta["runtime_fault"]["armed_step"] == 3
    assert meta["model_patch"]                         # the source edit is still extracted


def test_sandbox_flaky_window(tmp_path):
    meta, trace = go(tmp_path, "sandbox_flaky", [
        bash("a"), bash("b"), TEXT, bash("c"), bash("d"), bash("e"), DONE])
    injected = [s["step"] for s in trace if s.get("fault_injected")]
    assert injected == [4, 5] and meta["termination"] == "task_complete"


def test_text_stall_max_steps_and_invalid_tool(tmp_path):
    meta, _ = go(tmp_path / "a", "clean", [TEXT, TEXT, TEXT])
    assert meta["termination"] == "text_stall"
    meta, _ = go(tmp_path / "b", "clean", [bash("x")] * 10, max_steps=4)
    assert meta["termination"] == "max_steps" and meta["n_steps"] == 4
    meta, trace = go(tmp_path / "c", "clean", [ESC, DONE], interface="base")
    assert trace[0]["type"] == "invalid_tool" and meta["termination"] == "task_complete"


def test_condition_spec_recorded(tmp_path):
    meta, _ = go(tmp_path, "git_flaky", [DONE])
    assert meta["agent_env"] == {"PATH": "/opt/shim:$PATH"}
    assert meta["setup_cmds"][-1] == "echo 1 > /opt/shim/.gmax"
    assert meta["runtime_fault"] is None


class Hooks(RolloutHooks):
    def __init__(self):
        self.calls = []

    def before_step(self, ctx, step):
        if step == 2:
            ctx.history.append({"role": "user", "text": "reminder", "source": "harness"})
            return {"tool": "bash", "args": {"command": "replayed"}, "text": ""}
        return None

    def intercept_action(self, ctx, step, action):
        if action.get("tool") == "bash" and action["args"]["command"] == "reject":
            return ({"type": "protocol_error", "action": "bash", "observation": "no"},
                    [{"role": "user", "text": "not executed"}])
        return None

    def before_record(self, ctx, step, entry):
        entry["actor"] = "agent"

    def after_record(self, ctx, step, entry):
        self.calls.append((step, entry["type"]))

    def tool_feedback(self, ctx, step, action, tool, response):
        return [{"role": "user", "text": f"custom feedback {step}"}]


def test_hooks(tmp_path):
    h = Hooks()
    agent = ScriptedAgent([bash("ls"), bash("reject"), DONE], live=True)
    meta = run_rollout(TASK, "clean", agent, interface="escalate", out_dir=tmp_path, hooks=h)
    d = next(tmp_path.iterdir())
    trace = [json.loads(l) for l in (d / "trace.jsonl").read_text().splitlines()]
    assert [s["type"] for s in trace] == ["bash", "bash", "protocol_error", "task_complete"]
    assert trace[1]["action"] == "replayed" and all(s["actor"] == "agent" for s in trace)
    assert h.calls == [(1, "bash"), (2, "bash"), (3, "protocol_error"), (4, "task_complete")]
    assert meta["termination"] == "task_complete"


def test_runner_jobs_and_resume(tmp_path, capsys):
    jobs = runner.build_jobs(load_tasks(), None)
    assert len(jobs) == 1211
    assert len(runner.build_jobs(load_tasks(), ["dependency_missing"])) == 33
    arm = tmp_path / "m" / "escalate"
    (arm / "r1").mkdir(parents=True)
    (arm / "r1" / "meta.json").write_text(json.dumps(
        {"instance_id": TASK["instance_id"], "condition": "clean", "termination": "max_steps"}))
    (arm / "r2").mkdir()
    (arm / "r2" / "meta.json").write_text(json.dumps(
        {"instance_id": TASK["instance_id"], "condition": "disk_full", "termination": "container_error"}))
    assert runner.finished_pairs(arm) == {(TASK["instance_id"], "clean")}
    rc = runner.main(["--model", "m", "--interface", "escalate", "--out", str(tmp_path),
                      "--tasks", TASK["instance_id"], "--dry-run"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "12 (task, condition) pairs, 1 already finished, 11 to run" in out
