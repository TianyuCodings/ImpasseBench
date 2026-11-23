"""Rollout harness: fault conditions, Docker sandbox, model backends, loop.

Main entry points:

* :data:`conditions.CONDITIONS`  the 12 fault conditions and the clean control
* :func:`rollout.run_rollout`    run one trajectory
* :class:`rollout.RolloutHooks`  extension points (used by the monitor)
* :func:`run.main`               the ``impassebench run`` command
"""

from .conditions import BY_CATEGORY, CATEGORIES, CONDITIONS, FaultCondition
from .prompts import INTERFACES, system_prompt, task_prompt, tool_declarations
from .rollout import RolloutContext, RolloutHooks, run_rollout
from .tasks import load_tasks, task_index

__all__ = [
    "BY_CATEGORY", "CATEGORIES", "CONDITIONS", "FaultCondition", "INTERFACES",
    "RolloutContext", "RolloutHooks", "load_tasks", "run_rollout",
    "system_prompt", "task_index", "task_prompt", "tool_declarations",
]
