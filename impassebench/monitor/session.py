"""Per-trajectory monitor state shared by the live rollout loop and offline replay.

:class:`MonitorSession` owns the evidence monitor, the bounded-retry
controller and the order in which their events are logged. The rollout loop
calls, for every step:

1. :meth:`begin_step` -- the harness action scheduled by the retry
   controller, if any (it replaces the model call for that step);
2. :meth:`reminder_before_step` -- the reminder to append to the history
   before the model is asked for its next action (``soft`` mode only);
3. :meth:`annotate` and :meth:`observe` on the step's trace record;
4. :meth:`retry_feedback` -- for a harness action, the message shown to the
   agent in place of an ordinary tool result.

Replaying a recorded trace through the same calls reproduces the reminders
and retry messages the agent saw (see :func:`replay`).
"""
from __future__ import annotations

from .evidence import EvidenceMonitor, ToolObservation, VERSION as MONITOR_VERSION
from .reminder import notice, select_alarms
from .retry import RetryController, observation_notice

MONITOR_MODES = ("off", "soft")


class MonitorSession:
    def __init__(self, monitor_mode="off", retry_mode="off", retry_max_attempts=3,
                 retry_backoff_s=1.0, max_steps=80):
        if monitor_mode not in MONITOR_MODES:
            raise ValueError(f"unknown monitor mode: {monitor_mode}")
        self.monitor_mode = monitor_mode
        self.monitor = EvidenceMonitor()
        self.retry = RetryController(retry_mode, retry_max_attempts, retry_backoff_s)
        self.retry_max_attempts = retry_max_attempts
        self.max_steps = max_steps
        self.pending_alarms = []
        self.current_retry = None
        self.retry_status = None
        self.log = []            # harness events, in the order they occurred
        self.alarms = 0
        self.notices = 0
        self.errors = []

    # -- step lifecycle ---------------------------------------------------
    def begin_step(self):
        self.current_retry = self.retry.take()
        return self.current_retry

    def reminder_before_step(self, step, live=True):
        """Reminder text to append before the model call, or None.

        Alarms raised by earlier steps are consumed here. While a harness
        retry is pending the alarms are kept for the next model step.
        """
        if not self.pending_alarms or self.current_retry:
            return None
        text = None
        if self.monitor_mode == "soft" and live:
            active = list({(e["kind"], e["resource"]): e for e in self.monitor.events
                           if e["event"] == "alarm" and (e["kind"], e["resource"]) in self.monitor.active}.values())
            selected_pending = select_alarms(self.pending_alarms)
            current = select_alarms(active) if selected_pending else []
            text = notice(current) or None
            if text:
                self.log.append({"event": "harness_notice", "before_step": step, "text": text,
                                 "alarm_steps": [a["after_step"] for a in selected_pending]})
                self.notices += 1
            else:
                self.log.append({"event": "notice_suppressed", "before_step": step,
                                 "reason": "execution-channel alarms are log-only",
                                 "alarm_steps": [a["after_step"] for a in self.pending_alarms]})
        self.pending_alarms.clear()
        return text

    def annotate(self, entry):
        """Mark a trace record produced by a harness retry action."""
        r = self.current_retry
        if r:
            entry.update(actor="harness_retry", retry_origin_step=r.origin_step,
                         retry_attempt=r.attempt, retry_kind=r.kind,
                         retry_role=r.role, retry_delay_s=r.delay_s)
        return entry

    def observe(self, entry):
        """Feed one finished trace record (with its ``step``) to monitor and retry."""
        step = entry["step"]
        before = len(self.monitor.events)
        try:
            alarms = self.monitor.observe(ToolObservation.from_record(entry))
        except Exception as exc:  # a monitor failure must not stop the task
            self.errors.append({"step": step, "error": repr(exc)})
            alarms = []
        self.pending_alarms.extend(alarms)
        self.alarms += len(alarms)
        self.log.extend(self.monitor.events[before:])
        before_retry = len(self.retry.events)
        self.retry_status = self.retry.observe(ToolObservation.from_record(entry),
                                               current=self.current_retry,
                                               remaining_steps=self.max_steps - step)
        self.log.extend(self.retry.events[before_retry:])

    def retry_feedback(self, step, response):
        """Message replacing the tool result of a harness retry action."""
        text = observation_notice(self.current_retry, step, response, self.retry_status,
                                  self.retry_max_attempts)
        self.log.append({"event": "harness_retry_observation", "before_step": step + 1,
                         "text": text, "source": "harness_retry"})
        return text

    def meta(self):
        return {"mode": self.monitor_mode, "version": MONITOR_VERSION,
                "alarms": self.alarms, "notices": self.notices, "errors": list(self.errors),
                "retry": self.retry.meta()}


def replay(records, *, monitor_mode, retry_mode="off", retry_max_attempts=3,
           retry_backoff_s=1.0, max_steps=80):
    """Re-run a session over a recorded trace.

    Returns ``(log, problems)``: the reconstructed harness events and any
    disagreement between the trace and the retry schedule (a harness retry
    record where none was scheduled, or the reverse).
    """
    s = MonitorSession(monitor_mode, retry_mode, retry_max_attempts, retry_backoff_s, max_steps)
    problems = []
    for entry in records:
        step = entry["step"]
        r = s.begin_step()
        s.reminder_before_step(step)
        harness_step = entry.get("actor") == "harness_retry"
        if bool(r) != harness_step:
            problems.append({"step": step, "problem": "retry schedule mismatch"})
        elif r and str(entry.get("action")) != r.command:
            problems.append({"step": step, "problem": "retry command mismatch"})
        s.observe(entry)
        if r and entry.get("type") in ("bash", "edit"):
            s.retry_feedback(step, {"exit_code": entry.get("exit_code"),
                                    "output": entry.get("observation")})
    return s.log, problems
