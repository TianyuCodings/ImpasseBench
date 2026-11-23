"""The rollout loop: one (task, condition, interface) -> trace + meta + patch.

Each trajectory directory holds

* ``trace.jsonl`` one JSON object per step (bash, edit, task_complete,
  escalate_to_human, text, invalid_tool),
* ``meta.json``   task, condition, model, fault spec, termination and the
  extracted patch (``model_patch``),
* ``diff.patch``  the extracted patch, when non-empty.

Termination reasons: ``task_complete``, ``escalate`` (escalate_to_human),
``max_steps`` (budget exhausted), ``text_stall`` (three consecutive replies
without a tool call) and ``container_error`` (harness-side failure; such
trajectories are not part of the benchmark statistics).

Every model reply counts as one step, including invalid tool calls and
text-only replies. ``meta.json`` is written once when the run starts
(``termination`` null) and again when it ends.
"""

from __future__ import annotations

import base64
import json
import os
import shlex
import time
from dataclasses import dataclass, field
from pathlib import Path

from . import calibration, prompts
from .agents import new_call_id
from .container import (ContainerSession, container_name, image_digest_of,
                        prepare_swesmith_image, prepare_verified_bug_absent_image,
                        prepare_verified_image)
from .runtime_faults import (SRC_WRITE_CHECK_CMD, RuntimeFaultController,
                             src_write_hits)

MAX_TEXT_STALLS = 3
TERMINATIONS = ("task_complete", "escalate", "max_steps", "text_stall")
HARNESS_FAILURES = ("container_error", "harness_error")


def task_pool(task: dict) -> str:
    """``swesmith`` (bug on a per-task branch) or ``verified`` (SWE-bench
    Verified image with the bug in the working tree)."""
    if task.get("pool"):
        return task["pool"]
    return "swesmith" if "swesmith" in (task.get("image") or "") else "verified"


def image_for(task: dict) -> str:
    if task.get("image"):
        return task["image"]
    iid = task["instance_id"]
    return f"swebench/sweb.eval.x86_64.{iid.replace('__', '_1776_')}:latest"


@dataclass
class RolloutContext:
    """State shared with hooks during one rollout."""
    task: dict
    condition: object
    interface: str
    max_steps: int
    run_dir: Path
    meta: dict
    agent: object
    live: bool
    history: list = field(default_factory=list)
    session: ContainerSession | None = None
    runtime_fault: RuntimeFaultController | None = None


class RolloutHooks:
    """Extension points of the rollout loop; every method is a no-op here.

    With ``hooks=None`` the loop runs exactly the benchmark harness. Order
    within one step: ``before_step`` -> agent call -> ``intercept_action`` ->
    dispatch -> ``before_record``/``after_record`` -> ``tool_feedback``.
    """

    def make_session(self, **kwargs) -> ContainerSession:
        """Construct the sandbox (override to customize patch extraction)."""
        return ContainerSession(**kwargs)

    def system_prompt(self, system: str) -> str:
        """Transform the system prompt before the run starts."""
        return system

    def on_meta(self, ctx: RolloutContext) -> None:
        """Called once ``ctx.meta`` is built, before it is first written."""

    def on_session_start(self, ctx: RolloutContext) -> None:
        """Called after the container starts, before preflight and setup."""

    def after_setup(self, ctx: RolloutContext) -> None:
        """Called after fault setup, before the first step."""

    def before_step(self, ctx: RolloutContext, step: int) -> dict | None:
        """May append messages to ``ctx.history``. Return an action dict to
        execute it instead of asking the agent, ``{"terminate": reason}`` to
        end the run, or None."""
        return None

    def intercept_action(self, ctx: RolloutContext, step: int,
                         action: dict) -> tuple[dict, list[dict]] | None:
        """Return ``(trace_entry, history_entries)`` to record the action
        without executing it, or None to dispatch it normally."""
        return None

    def wrap_exec(self, exec_fn):
        """Wrap the executor used for agent bash commands."""
        return exec_fn

    def before_record(self, ctx: RolloutContext, step: int, entry: dict) -> None:
        """May add fields to a trace entry before it is written."""

    def after_record(self, ctx: RolloutContext, step: int, entry: dict) -> None:
        """Called after a trace entry is written."""

    def tool_feedback(self, ctx: RolloutContext, step: int, action: dict,
                      tool: str, response: dict) -> list[dict] | None:
        """History entries to append for an executed bash/edit call instead
        of the default assistant/tool pair, or None for the default."""
        return None

    def on_finish(self, ctx: RolloutContext) -> None:
        """Called after patch extraction, before the final meta write."""


_NO_HOOKS = RolloutHooks()


def _apply_edit(session, args: dict) -> tuple[str, int]:
    """Exact-string replacement through the container filesystem.

    Uses only coreutils (no Python in the container), base64 round-trips the
    content, reads are delimited by markers so stray output cannot corrupt
    them, and the new content is streamed over stdin and written by a plain
    shell redirect, so the same permissions and filesystem limits apply as
    to a bash write. An empty ``old_str`` appends (and may create the file).
    """
    path = str(args.get("path", "")).strip()
    old, new = str(args.get("old_str", "")), str(args.get("new_str", ""))
    if not path:
        return "edit_file: path is required", 2
    if not path.startswith("/"):
        path = "/testbed/" + path.lstrip("./")

    r = session.exec_agent(
        f"test -e {shlex.quote(path)} || exit 66; "
        f"echo __T2E_B64_BEGIN__; base64 -w0 < {shlex.quote(path)}; rc=$?; "
        f"echo; echo __T2E_B64_END__; exit $rc", timeout_s=60, limit=None)
    if r.exit_code == 66 and not old:
        content = ""
    elif r.exit_code == 66:
        return f"edit_file: {path} does not exist", 1
    elif r.exit_code != 0:
        # the shell's diagnostic follows the closing marker
        err = r.output.split("__T2E_B64_END__")[-1].strip()
        return (f"edit_file: cannot read {path}: "
                f"{err[:300] or f'read failed (exit {r.exit_code})'}"), r.exit_code
    else:
        try:
            payload_in = r.output.split("__T2E_B64_BEGIN__", 1)[1] \
                                 .split("__T2E_B64_END__", 1)[0]
        except IndexError:
            return (f"edit_file: could not delimit the contents of {path}; the read "
                    f"was disturbed: {r.output.strip()[:300]}"), 2
        payload_in = "".join(payload_in.split())
        try:
            content = base64.b64decode(payload_in, validate=True).decode(
                "utf-8", errors="replace")
        except Exception as e:  # noqa: BLE001
            return f"edit_file: {path} is not decodable text ({e})", 2

    n = content.count(old)
    if old and n == 0:
        return (f"edit_file: old_str not found in {path}. It must match the file "
                f"exactly, including indentation."), 1
    if old and n > 1:
        return (f"edit_file: old_str appears {n} times in {path}; it must be "
                f"unique. Include surrounding lines to disambiguate."), 1

    updated = content.replace(old, new, 1) if old else content + new
    payload = base64.b64encode(updated.encode()).decode()
    w = session.exec_agent(
        f"base64 -d > {shlex.quote(path)}",
        timeout_s=60, limit=None, stdin_data=payload)
    if w.exit_code != 0:
        return f"edit_file: write to {path} failed: {w.output.strip()[:300]}", w.exit_code
    delta = updated.count("\n") - content.count("\n")
    return f"edited {path} ({delta:+d} lines)", 0


def _prepare_image(task: dict, cond, condition_name: str) -> str:
    iid = task["instance_id"]
    pool = task_pool(task)
    image = image_for(task)
    if pool == "swesmith":
        # bug_absent stays on the healthy main branch
        return prepare_swesmith_image(image, iid, checkout=not cond.skip_bug_checkout)
    if pool != "verified":
        raise RuntimeError(f"unknown task pool '{pool}' for {iid}")
    if cond.skip_bug_checkout:
        gold = task.get("patch") or ""
        if not gold.strip():
            raise RuntimeError(
                f"condition '{condition_name}' on a Verified task needs the "
                f"gold patch, and task {iid} carries none.")
        if not gold.endswith("\n"):
            gold += "\n"
        return prepare_verified_bug_absent_image(image, gold)
    return prepare_verified_image(image)


def run_rollout(task: dict, condition: str, agent, *, interface: str,
                max_steps: int = 80, out_dir: Path,
                hooks: RolloutHooks | None = None) -> dict:
    """Run one trajectory and return its meta dict.

    ``task`` is a task record (see ``resources/tasks.json``); ``out_dir`` is
    the directory that receives the trajectory's own subdirectory.
    """
    hooks = hooks or _NO_HOOKS
    with_escalate = interface == "escalate"
    live = getattr(agent, "live", True)
    cond = calibration.specialize(condition, task["repo"])
    iid = task["instance_id"]
    pool = task_pool(task)
    run_id = (f"{time.strftime('%Y%m%d_%H%M%S')}_{iid}_{condition}"
              f"_{os.urandom(3).hex()}")
    out = Path(out_dir) / run_id
    out.mkdir(parents=True, exist_ok=False)

    system = hooks.system_prompt(prompts.system_prompt(interface))
    task_text = prompts.task_prompt(task["problem_statement"], max_steps)

    official_image = image_for(task)
    official_digest = image_digest_of(official_image)
    try:
        image = _prepare_image(task, cond, condition)
    except Exception as e:
        (out / "meta.json").write_text(json.dumps({
            "run_id": run_id, "instance_id": iid, "repo": task["repo"],
            "condition": condition, "interface": interface,
            "termination": "harness_error", "n_steps": 0, "total_tokens": 0,
            "error": f"image preparation failed: {e!r}"}, indent=2))
        raise

    run_args = list(cond.docker_run_args)
    if cond.skip_bug_checkout and pool == "verified":
        # without network, upstream history cannot reveal the applied fix
        run_args += ["--network", "none"]
    session = hooks.make_session(
        image=image,
        name=container_name(iid, condition),
        extra_run_args=run_args,
        agent_env=cond.agent_env,
        agent_user=cond.agent_user,
    )
    rf = RuntimeFaultController(cond.runtime_fault) if cond.runtime_fault else None

    def _src_written() -> bool:
        # Has any tracked non-test source file been modified on disk, through
        # any channel? A failed probe returns False and is retried next step.
        try:
            r = session.exec_root(SRC_WRITE_CHECK_CMD, timeout_s=60)
        except Exception:
            return False
        if r.exit_code != 0:
            return False
        return bool(src_write_hits(r.output))

    trace_f = (out / "trace.jsonl").open("w")
    meta = {
        "run_id": run_id,
        "instance_id": iid, "repo": task["repo"], "condition": condition,
        "interface": interface, "with_escalate": with_escalate, "with_edit": True,
        "task_preview": task["problem_statement"],
        "max_steps": max_steps, "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "termination": None, "n_steps": 0, "total_tokens": 0, "model_patch": None,
        "model": getattr(agent, "model", "unknown"),
        "temperature": getattr(agent, "temperature", None),
        "sampling": getattr(agent, "sampling_note", None),
        "base_image": official_image,
        "base_image_digest": official_digest,
        "docker_run_args": list(run_args),
        "setup_cmds": list(cond.setup_cmds),
        "runtime_fault_spec": dict(cond.runtime_fault) if cond.runtime_fault else None,
        "agent_env": dict(cond.agent_env),
        "agent_user": cond.agent_user,
        "system_prompt": system,
        "patch_extraction_error": None,
    }
    if getattr(agent, "endpoint_meta", None) is not None:
        meta["endpoint"] = agent.endpoint_meta
    ctx = RolloutContext(task=task, condition=cond, interface=interface,
                         max_steps=max_steps, run_dir=out, meta=meta,
                         agent=agent, live=live, session=session,
                         runtime_fault=rf)
    history = ctx.history
    hooks.on_meta(ctx)
    (out / "meta.json").write_text(json.dumps(meta, indent=2))

    if live:
        agent.set_system(system)
        history.append({"role": "user", "text": task_text})

    exec_agent = hooks.wrap_exec(session.exec_agent)

    def record(step, entry):
        hooks.before_record(ctx, step, entry)
        entry["step"] = step
        trace_f.write(json.dumps(entry) + "\n")
        trace_f.flush()
        hooks.after_record(ctx, step, entry)

    def feedback(step, action, tool, response, assistant_text):
        if not live:
            return
        custom = hooks.tool_feedback(ctx, step, action, tool, response)
        if custom is not None:
            history.extend(custom)
            return
        cid = action.get("call_id") or new_call_id()
        history.append({"role": "assistant", "text": assistant_text,
                        "tool": tool, "args": action.get("args") or {},
                        "call_id": cid,
                        "provider_meta": action.get("provider_meta")})
        history.append({"role": "tool", "name": tool, "call_id": cid,
                        "response": response})

    termination = None
    try:
        session.start()
        hooks.on_session_start(ctx)
        if pool == "verified":
            # tracked files the image build modified (see prepare_verified_image)
            r = session.exec_root("cat /opt/t2e_dirt_paths 2>/dev/null",
                                  timeout_s=30)
            meta["baseline_dirt_paths"] = (
                [p for p in r.output.splitlines() if p.strip()]
                if r.exit_code == 0 else None)
        # A failing preflight or setup means the condition cannot be built
        # on this image; no trajectory is produced.
        for cmd in cond.preflight_cmds:
            r = session.exec_setup(cmd, timeout_s=180)
            if r.exit_code != 0:
                raise RuntimeError(
                    f"preflight failed for condition '{condition}' "
                    f"({cmd[:60]}...): {r.output[-300:]}")
        for cmd in cond.setup_cmds:
            r = session.exec_setup(cmd, timeout_s=600)
            if r.exit_code != 0:
                raise RuntimeError(
                    f"fault setup failed ({cmd[:60]}...): {r.output[-300:]}")
        hooks.after_setup(ctx)
        stalls = 0
        for step in range(1, max_steps + 1):
            action = hooks.before_step(ctx, step)
            if action is not None and "terminate" in action:
                termination = action["terminate"]
                break
            if action is None:
                action = agent.next_action(history)
            meta["n_steps"] = step

            assistant_text = action.get("text", "")

            intercepted = hooks.intercept_action(ctx, step, action)
            if intercepted is not None:
                entry, extra_history = intercepted
                record(step, entry)
                if live:
                    history.extend(extra_history)
                continue

            if action["tool"] is None:
                stalls += 1
                record(step, {"type": "text", "action": assistant_text})
                if stalls >= MAX_TEXT_STALLS:
                    termination = "text_stall"
                    break
                if live:
                    history.append({"role": "assistant", "text": assistant_text,
                                    "tool": None, "args": None})
                    history.append({"role": "user", "text":
                                    "Respond with a tool call (bash / task_complete"
                                    + (" / escalate_to_human" if with_escalate else "")
                                    + ")."})
                continue
            stalls = 0

            if action["tool"] == "task_complete":
                record(step, {"type": "task_complete", "action": action["args"],
                              "assistant_text": assistant_text})
                termination = "task_complete"
                break

            if action["tool"] == "escalate_to_human" and with_escalate:
                record(step, {"type": "escalate_to_human", "action": action["args"],
                              "assistant_text": assistant_text})
                termination = "escalate"
                break

            if action["tool"] == "edit_file":
                inj = rf.intercept("edit", step) if rf else None
                if inj is not None:
                    obs, rc = inj, 1
                else:
                    obs, rc = _apply_edit(session, action.get("args") or {})
                    if rf:
                        rf.observe("edit", "", step, src_written=_src_written)
                # `action` is the argument JSON cut to 2000 characters;
                # `tool_args` keeps the full arguments
                record(step, {"type": "edit", "action": json.dumps(
                    action.get("args") or {}, ensure_ascii=False)[:2000],
                    "tool_args": action.get("args") or {},
                    "observation": obs, "exit_code": rc,
                    **({"fault_injected": True} if inj is not None else {}),
                    "assistant_text": assistant_text})
                feedback(step, action, "edit_file",
                         {"exit_code": rc, "output": obs}, assistant_text)
                continue

            if action["tool"] != "bash":
                # unknown or unavailable tool: recorded, never executed
                record(step, {"type": "invalid_tool", "action": action["tool"],
                              "args": action.get("args"), "assistant_text": assistant_text})
                if live:
                    cid = action.get("call_id") or new_call_id()
                    history.append({"role": "assistant", "text": assistant_text,
                                    "tool": action["tool"],
                                    "args": action.get("args") or {},
                                    "call_id": cid,
                                    "provider_meta": action.get("provider_meta")})
                    history.append({"role": "tool", "name": action["tool"],
                                    "call_id": cid,
                                    "response": {"error": f"unknown tool '{action['tool']}'"}})
                continue

            cmd = str(action["args"].get("command", ""))
            if rf:
                rf.pre_dispatch("bash", cmd, step, src_written=_src_written)
            inj = rf.intercept("bash", step) if rf else None
            if inj is not None:
                record(step, {
                    "type": "bash", "action": cmd, "observation": inj,
                    "exit_code": 1, "wall_time_s": 0.0, "truncated": False,
                    "fault_injected": True, "assistant_text": assistant_text,
                })
                feedback(step, action, "bash", {"exit_code": 1, "output": inj},
                         assistant_text)
                continue
            result = exec_agent(cmd)
            if rf:
                rf.observe("bash", cmd, step, exit_code=result.exit_code,
                           observation=result.output, src_written=_src_written)
            record(step, {
                "type": "bash", "action": cmd, "observation": result.output,
                "exit_code": result.exit_code, "wall_time_s": round(result.wall_time_s, 2),
                "truncated": result.truncated, "assistant_text": assistant_text,
            })
            feedback(step, action, "bash", {"exit_code": result.exit_code,
                                            "output": result.output}, assistant_text)
        else:
            termination = "max_steps"
    except Exception as e:
        termination = "container_error"
        meta["error"] = repr(e)

    # extraction cannot change the termination decided above
    try:
        patch, patch_err = session.git_diff()
        meta["model_patch"] = patch
        meta["patch_extraction_error"] = patch_err
    except Exception as e:
        meta["model_patch"] = None
        meta["patch_extraction_error"] = repr(e)
    finally:
        trace_f.close()
        session.stop()

    meta["termination"] = termination
    meta["runtime_fault"] = rf.meta() if rf else None
    meta["total_tokens"] = getattr(agent, "total_tokens", 0)
    if getattr(agent, "endpoint_meta", None) is not None:
        meta["endpoint"] = agent.endpoint_meta
    meta["sampling"] = getattr(agent, "sampling_note", None)
    meta["ended_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    hooks.on_finish(ctx)
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    if meta["model_patch"]:
        (out / "diff.patch").write_text(meta["model_patch"])
    return meta
