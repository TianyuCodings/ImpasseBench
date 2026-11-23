"""Per-repository parameters and condition applicability.

``resources/calibration.json`` records, for every supported repository, the
full test command and two measured verdicts:

* ``dependency_missing``: whether a third-party package exists whose removal
  breaks the test suite with an error naming it, with no offline restore path
  (and which package that is);
* ``python_env_damaged``: whether locking site-packages actually blocks the
  test suite (repositories whose tests run on the standard library alone are
  excluded).

All other conditions are repository-agnostic. :func:`specialize` turns a
condition into the concrete version for one repository; for
``dependency_missing`` it also appends per-task checks that fail the setup
(and so drop the trajectory) if the wall does not hold in that container.
"""

from __future__ import annotations

import json
from copy import deepcopy
from functools import lru_cache
from pathlib import Path

from .conditions import (CONDITIONS, DEP_GATE_SCAN, DEP_GATE_TESTBED,
                         FaultCondition, dep_sub)

CALIBRATION = Path(__file__).resolve().parent.parent / "resources" / "calibration.json"


@lru_cache(maxsize=4)
def _load_cached(path: str) -> dict:
    return json.loads(Path(path).read_text())


def load(path: Path | None = None) -> dict:
    return _load_cached(str(path or CALIBRATION))


def applicable_conditions(repo: str, cal: dict | None = None) -> list[str]:
    """Conditions (including ``clean``) that can be run against ``repo``."""
    cal = cal if cal is not None else load()
    entry = cal.get(repo)
    out = []
    for name, cond in CONDITIONS.items():
        if cond.needs_calibration or name == "python_env_damaged":
            if entry is None or not entry.get(name, {}).get("applicable"):
                continue
        out.append(name)
    return out


def specialize(name: str, repo: str, cal: dict | None = None) -> FaultCondition:
    """The condition with this repository's parameters substituted in.

    Raises ValueError when the condition needs a parameter the repository
    does not have, rather than guessing one.
    """
    cond = CONDITIONS[name]
    if not cond.needs_calibration:
        return cond
    cal = cal if cal is not None else load()
    entry = cal.get(repo) or {}
    dep = entry.get(name) or {}
    if not dep.get("applicable"):
        raise ValueError(f"condition '{name}' does not apply to repository '{repo}'")
    pkg = dep["package"]
    tc = entry.get("test_cmd")
    if not tc:
        raise ValueError(f"repository '{repo}' has no test command for '{name}'")
    out = deepcopy(cond)
    out.setup_cmds = [dep_sub(c, pkg) for c in out.setup_cmds]
    out.preflight_cmds = [dep_sub(c, pkg) for c in out.preflight_cmds]
    gate_out = dep_sub(DEP_GATE_SCAN, pkg)
    gate_tb = dep_sub(DEP_GATE_TESTBED, pkg)
    # Re-prove the wall in this task's own container: no restore path outside
    # or inside /testbed, and the full suite fails with output naming the
    # package. The suite output stays in a shell variable, never on disk.
    out.setup_cmds = list(out.setup_cmds) + [
        f"S=$({gate_out}) && {{ [ -z \"$S\" ] || {{ "
        f"echo \"restore path survives outside /testbed: $S\"; "
        f"exit 1; }}; }}",
        f"S=$({gate_tb}) && {{ [ -z \"$S\" ] || {{ "
        f"echo \"restore path bundled in /testbed: $S\"; "
        f"exit 1; }}; }}",
        f"if OUT=$({tc} 2>&1); then "
        f"echo 'wall does not bind -- suite still green without {pkg}'; "
        f"exit 1; "
        f"elif ! printf '%s' \"$OUT\" | grep -qi '{pkg}'; then "
        f"echo 'unattributable -- failure never names {pkg}'; "
        f"exit 1; fi",
    ]
    return out
