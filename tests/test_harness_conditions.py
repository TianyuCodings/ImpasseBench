"""Condition registry, calibration and task applicability (no Docker)."""
import hashlib
import json

import pytest

from impassebench.harness import calibration
from impassebench.harness.conditions import (BY_CATEGORY, CATEGORIES, CONDITIONS,
                                             FAULT_CONDITIONS)
from impassebench.harness.tasks import load_tasks

PAPER_CATEGORIES = {
    "persistent_barrier": ["readonly_fs", "disk_full", "source_unreadable",
                           "source_write_denied", "dependency_missing", "sandbox_lost_early"],
    "verification_limit": ["tests_unreadable", "python_env_damaged", "sandbox_lost_late"],
    "temporary_fault": ["git_flaky", "sandbox_flaky"],
    "task_mismatch": ["bug_absent"],
    "control": ["clean"],
}


def test_registry_matches_paper_taxonomy():
    assert len(FAULT_CONDITIONS) == 12
    assert set(CATEGORIES) == set(PAPER_CATEGORIES)
    for cat, names in PAPER_CATEGORIES.items():
        assert BY_CATEGORY[cat] == names


def test_task_set_and_applicability():
    tasks = load_tasks()
    assert len(tasks) == 102
    assert len({t["repo"] for t in tasks}) == 10
    counts = {c: sum(c in t["conditions"] for t in tasks) for c in CONDITIONS}
    assert counts["dependency_missing"] == 33
    assert counts["python_env_damaged"] == 56
    assert all(counts[c] == 102 for c in CONDITIONS
               if c not in ("dependency_missing", "python_env_damaged"))
    assert sum(counts.values()) == 1211
    for t in tasks:
        assert t["conditions"] == calibration.applicable_conditions(t["repo"])
    tiers = {d: sum(t["difficulty"] == d for t in tasks) for d in ("easy", "medium", "hard")}
    assert tiers == {"easy": 34, "medium": 34, "hard": 34}


def test_dependency_specialization():
    repo = "swesmith/cantools__cantools.0c6a7871"
    c = calibration.specialize("dependency_missing", repo)
    joined = "\n".join(c.setup_cmds)
    assert "pytz" not in joined and "bitstruct" in joined
    assert c.setup_cmds[0] == "pip uninstall -y bitstruct"
    assert len(c.setup_cmds) == len(CONDITIONS["dependency_missing"].setup_cmds) + 3
    assert c.docker_run_args == ["--network", "none"]
    with pytest.raises(ValueError):
        calibration.specialize("dependency_missing", "swesmith/andialbrecht__sqlparse.e57923b3")
    # repository-agnostic conditions are returned unchanged
    assert calibration.specialize("readonly_fs", repo) is CONDITIONS["readonly_fs"]


def _spec(c):
    return {"docker_run_args": c.docker_run_args, "preflight_cmds": c.preflight_cmds,
            "setup_cmds": c.setup_cmds, "agent_env": c.agent_env,
            "agent_user": c.agent_user, "skip_bug_checkout": c.skip_bug_checkout,
            "runtime_fault": c.runtime_fault}


def test_fault_specs_are_frozen():
    """The executed fault definitions are part of the benchmark; any change
    to them must be deliberate (update this digest together with a version bump)."""
    tasks = load_tasks()
    specs = {}
    for repo in sorted({t["repo"] for t in tasks}):
        for name in calibration.applicable_conditions(repo):
            specs[f"{repo}/{name}"] = _spec(calibration.specialize(name, repo))
    digest = hashlib.sha256(json.dumps(specs, sort_keys=True).encode()).hexdigest()
    assert digest == FROZEN_SPEC_DIGEST


FROZEN_SPEC_DIGEST = "aa821baad2a5260ce199ba794f7be0de508b4119e5f4b2fbbde8a32f0125d868"
