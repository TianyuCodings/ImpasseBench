"""Harness monitor: evidence reminders, an optional ticket template and bounded retry.

The second scaffold of the paper (Section 4). Only the execution harness
changes; model, tools, prompts and sampling stay the same. See
``docs/monitor.md``.
"""
from .evidence import EvidenceMonitor, ToolObservation
from .reminder import notice, select_alarms, template
from .retry import RetryController
from .session import MonitorSession, replay

__all__ = ["EvidenceMonitor", "ToolObservation", "MonitorSession", "RetryController",
           "notice", "select_alarms", "template", "replay"]
