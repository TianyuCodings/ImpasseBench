"""The monitored harness, expressed as :class:`~impassebench.harness.rollout.RolloutHooks`.

Compared with the benchmark harness (``hooks=None``) this adds, with the
model, tools, prompts and sampling unchanged:

* strict tool calls -- a malformed, truncated or multiple call is recorded as
  a ``protocol_error`` step and never executed (both arms);
* the evidence monitor -- always running; in ``soft`` mode its file-evidence
  alarms produce a reminder with an optional ticket template, appended as a
  separate user message before the next model call;
* bounded retry (``retry_mode="bounded"``) -- harness-scheduled probes or
  replays after a transient-looking failure, each consuming one step;
* filesystem snapshots after every step (observer only), and a patch
  extraction that also works when the faulted tree's git index is not
  writable.

Harness events are written to ``<run_dir>/harness_events.jsonl``.
"""
from __future__ import annotations

import json
import time

from ..harness import prompts
from ..harness.rollout import RolloutHooks
from .artifacts import ArtifactObserver
from .extraction import SnapshotExtractionSession
from .session import MonitorSession
from .strict_calls import error_history, protocol_error, validate_call

AUDIT_KEYS = ("raw_arguments", "finish_reason", "protocol_error", "tool_call_count")


class MonitorHooks(RolloutHooks):
    def __init__(self, monitor_mode="off", retry_mode="off", retry_max_attempts=3,
                 retry_backoff_s=1.0, audit_artifacts=True):
        self.monitor_mode = monitor_mode
        self.retry_mode = retry_mode
        self.retry_max_attempts = retry_max_attempts
        self.retry_backoff_s = retry_backoff_s
        self.audit_artifacts = audit_artifacts
        self.session = None
        self.observer = None
        self._events_f = None
        self._flushed = 0
        self._audit = {}
        self.n_model_calls = 0

    # -- setup ------------------------------------------------------------
    def make_session(self, **kwargs):
        return SnapshotExtractionSession(**kwargs)

    def on_meta(self, ctx):
        if ctx.interface != "escalate":
            raise ValueError("monitor comparisons use the escalate interface in every arm")
        self.session = MonitorSession(self.monitor_mode, self.retry_mode, self.retry_max_attempts,
                                      self.retry_backoff_s, ctx.max_steps)
        self._events_f = (ctx.run_dir / "harness_events.jsonl").open("w")
        ctx.meta["monitor"] = self.session.meta()
        ctx.meta["n_model_calls"] = 0
        if self.audit_artifacts:
            ctx.meta["artifact_observer_ok"] = True

    def _observe_files(self, ctx, actor, label):
        if not self.audit_artifacts:
            return None
        try:
            event = self.observer.snapshot(actor, label)
            if event.get("errors"):
                ctx.meta["artifact_observer_ok"] = False
            return event
        except Exception as exc:
            # an observer failure disqualifies the record, not the agent's task
            ctx.meta["artifact_observer_ok"] = False
            ctx.meta.setdefault("artifact_observer_errors", []).append({"label": label, "error": repr(exc)})
            return None

    def on_session_start(self, ctx):
        if self.audit_artifacts:
            self.observer = ArtifactObserver(ctx.session.name, ctx.run_dir / "artifacts")
            self._observe_files(ctx, "baseline", "before_fault_setup")

    def after_setup(self, ctx):
        self._observe_files(ctx, "fault_injection", "after_fault_setup")

    # -- per step -----------------------------------------------------------
    def _flush(self):
        for event in self.session.log[self._flushed:]:
            self._events_f.write(json.dumps(event) + "\n")
        self._flushed = len(self.session.log)
        self._events_f.flush()

    def before_step(self, ctx, step):
        retry = self.session.begin_step()
        text = self.session.reminder_before_step(step, live=ctx.live)
        if text:
            ctx.history.append({"role": "user", "text": text, "source": "harness_notice"})
        self._flush()
        if retry:
            if retry.delay_s:
                time.sleep(retry.delay_s)
            return {"tool": "bash", "args": {"command": retry.command}, "text": ""}
        return None

    def intercept_action(self, ctx, step, action):
        if not self.session.current_retry:
            self.n_model_calls += 1
            ctx.meta["n_model_calls"] = self.n_model_calls
        # validate every call before anything is executed
        if action.get("tool") and not action.get("protocol_error"):
            errors = validate_call(action["tool"], action.get("args"),
                                   prompts.tool_declarations(ctx.interface))
            if errors:
                action["protocol_error"] = protocol_error(errors)
        self._audit = {k: action[k] for k in AUDIT_KEYS if k in action}
        if not action.get("protocol_error"):
            return None
        entry = {"type": "protocol_error", "action": action.get("tool"),
                 "args": action.get("args"), "assistant_text": action.get("text", ""),
                 "observation": action["protocol_error"]["feedback"],
                 "environment_action_executed": False}
        return entry, error_history(action)

    def before_record(self, ctx, step, entry):
        self.session.annotate(entry)
        if entry.get("type") in ("bash", "edit"):
            actor = "harness_retry" if self.session.current_retry else "agent"
            event = self._observe_files(ctx, actor, f"step_{step}")
            if self.audit_artifacts:
                entry["artifact_event_index"] = event["index"] if event else None
        entry.update(self._audit)

    def after_record(self, ctx, step, entry):
        self.session.observe(entry)
        ctx.meta["monitor"] = self.session.meta()
        self._flush()

    def tool_feedback(self, ctx, step, action, tool, response):
        if not self.session.current_retry:
            return None
        text = self.session.retry_feedback(step, response)
        self._flush()
        return [{"role": "user", "text": text, "source": "harness_retry"}]

    def on_finish(self, ctx):
        ctx.meta["monitor"] = self.session.meta()
        ctx.meta["n_model_calls"] = self.n_model_calls
        ctx.meta["patch_extraction"] = getattr(ctx.session, "patch_extraction_meta", None)
        if self._events_f:
            self._flush()
            self._events_f.close()
