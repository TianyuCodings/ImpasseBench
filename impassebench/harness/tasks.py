"""Task records.

The benchmark ships 102 SWE-smith tasks in ``resources/tasks.json``. Each
record carries ``instance_id``, ``repo``, ``image``, ``problem_statement``,
the gold ``patch``, ``FAIL_TO_PASS`` / ``PASS_TO_PASS`` (used only for
grading, never shown to the agent), a ``difficulty`` tier and the list of
applicable ``conditions``.

Other task files (e.g. SWE-bench Verified tasks) may be supplied as a JSON
list of records or ``{"tasks": [...]}``; records then need at least
``instance_id``, ``repo`` and ``problem_statement`` (and ``image``, ``pool``
or a ``patch`` where the condition needs one).
"""

from __future__ import annotations

import json
from pathlib import Path

from .calibration import applicable_conditions

TASKS_FILE = Path(__file__).resolve().parent.parent / "resources" / "tasks.json"


def load_tasks(path: str | Path | None = None) -> list[dict]:
    payload = json.loads(Path(path or TASKS_FILE).read_text())
    rows = payload["tasks"] if isinstance(payload, dict) else payload
    out = []
    for r in rows:
        r = dict(r)
        if not (r.get("problem_statement") or "").strip():
            raise ValueError(f"task {r.get('instance_id')} has no problem_statement")
        if "conditions" not in r:
            r["conditions"] = applicable_conditions(r["repo"])
        out.append(r)
    return out


def task_index(path: str | Path | None = None) -> dict[str, dict]:
    return {t["instance_id"]: t for t in load_tasks(path)}
