"""Repair success A: does the final patch pass the held-out target tests?

For one trajectory, in a fresh container started from the task's BASE
SWE-smith image (not the fault-prepared one):

  1. check out the task branch (the tree the agent worked on);
  2. apply the agent's final patch (`git apply`, falling back to `patch -p1`);
  3. restore the FAIL_TO_PASS test files from `main`, overwriting any agent
     edit to them (the held-out tests take precedence, as in the official
     SWE-bench harness);
  4. run the FAIL_TO_PASS tests with pytest.

A = 1 iff the tests pass (status "resolved"). An empty patch is "empty_patch"
(A = 0). PASS_TO_PASS is not run, so A is FAIL_TO_PASS resolution: necessary
for success, not sufficient against regressions elsewhere. Infrastructure
failures (docker_error, checkout_failed, apply_failed, oracle_restore_failed,
test_timeout, eval_error) leave A undefined.

    python -m impassebench.annotate.grade jobs.json results.json --parallel 8

`jobs.json` is a list of {"instance_id", "patch"[, "traj_id"]} rows.
"""

import argparse
import json
import shlex
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .io import load_tasks

ACT = "source /opt/miniconda3/etc/profile.d/conda.sh && conda activate testbed"
GRADED = ("resolved", "f2p_failed", "empty_patch")


def _sh(args, timeout, **kw):
    return subprocess.run(args, capture_output=True, text=True, errors="replace",
                          timeout=timeout, **kw)


def a_label(status):
    """A from a grading status: 1, 0, or None when grading did not complete."""
    if status == "resolved":
        return 1
    if status in ("f2p_failed", "empty_patch"):
        return 0
    return None


def grade_one(job, task, tag="", run=_sh):
    """Grade one {"instance_id", "patch"} job against its task record.
    `run(args, timeout, **kw)` executes a host command (subprocess.run by
    default; injectable for tests)."""
    iid = job["instance_id"]
    patch = job.get("patch") or ""
    f2p = task["FAIL_TO_PASS"]
    out = {"instance_id": iid, "n_f2p": len(f2p), "status": None, "detail": ""}
    if "traj_id" in job:
        out["traj_id"] = job["traj_id"]
    if not patch.strip():
        out["status"] = "empty_patch"
        out["A"] = 0
        return out
    files = sorted({t.split("::")[0] for t in f2p})
    name = "impasse_grade_" + (tag + "_" if tag else "") + iid.replace("/", "_").replace(".", "_")[-40:]
    try:
        run(["docker", "rm", "-f", name], 60)
        r = run(["docker", "run", "-d", "--name", name, "--platform", "linux/amd64",
                 task["image"], "sleep", "1800"], 300)
        if r.returncode != 0:
            out["status"], out["detail"] = "docker_error", r.stderr[-200:]
            return out
        r = run(["docker", "exec", name, "bash", "-c",
                 f"cd /testbed && git checkout -q {shlex.quote(iid)}"], 120)
        if r.returncode != 0:
            out["status"], out["detail"] = "checkout_failed", (r.stdout + r.stderr)[-200:]
            return out
        p = run(["docker", "exec", "-i", name, "bash", "-c",
                 "cd /testbed && git apply --whitespace=nowarn -"], 120, input=patch)
        if p.returncode != 0:
            p2 = run(["docker", "exec", "-i", name, "bash", "-c",
                      "cd /testbed && patch -p1 --forward --silent"], 120, input=patch)
            if p2.returncode != 0:
                out["status"] = "apply_failed"
                out["detail"] = (p.stdout + p.stderr)[-200:]
                return out
        r = run(["docker", "exec", name, "bash", "-c",
                 "cd /testbed && git checkout -q main -- "
                 + " ".join(shlex.quote(f) for f in files)], 120)
        if r.returncode != 0:
            out["status"], out["detail"] = "oracle_restore_failed", (r.stdout + r.stderr)[-200:]
            return out
        # test ids go through stdin + xargs: some tasks carry enough ids to
        # exceed ARG_MAX on the command line. Install pytest if the image's
        # environment lacks it, so a missing runner is not scored as a failure.
        try:
            t = run(["docker", "exec", "-i", name, "bash", "-c",
                     f"{ACT} && cd /testbed && "
                     "(python -m pytest --version >/dev/null 2>&1 "
                     "|| python -m pip install -q pytest) && xargs -d '\\n' "
                     "python -m pytest -q --no-header"],
                    900, input="\n".join(f2p))
        except subprocess.TimeoutExpired:
            out["status"], out["detail"] = "test_timeout", ""
            return out
        tail = (t.stdout + t.stderr).strip().splitlines()
        out["detail"] = tail[-1][:160] if tail else "(no output)"
        out["status"] = "resolved" if t.returncode == 0 else "f2p_failed"
        return out
    except Exception as e:  # noqa: BLE001 - recorded as an infrastructure failure
        out["status"], out["detail"] = "eval_error", repr(e)[:200]
        return out
    finally:
        out["A"] = a_label(out["status"])
        run(["docker", "rm", "-f", name], 60)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("jobs", help='JSON list of {"instance_id", "patch"[, "traj_id"]}')
    ap.add_argument("out", help="results JSON (rewritten after every job)")
    ap.add_argument("--parallel", type=int, default=8)
    ap.add_argument("--tasks", default=None, help="tasks.json (default: bundled)")
    ap.add_argument("--tag", default="", help="container-name suffix for concurrent graders")
    args = ap.parse_args(argv)

    tasks = load_tasks(args.tasks)
    jobs = json.loads(Path(args.jobs).read_text())
    results, lock = [], threading.Lock()

    def one(j):
        res = grade_one(j, tasks[j["instance_id"]], tag=args.tag)
        with lock:
            results.append(res)
            Path(args.out).write_text(json.dumps(results, indent=1))
            print(f"[{len(results)}/{len(jobs)}] {res['instance_id'][-40:]} -> {res['status']}", flush=True)
        return res

    with ThreadPoolExecutor(max_workers=max(1, args.parallel)) as pool:
        for f in as_completed([pool.submit(one, j) for j in jobs]):
            f.result()
    n = len(results)
    resolved = sum(1 for r in results if r["status"] == "resolved")
    graded = sum(1 for r in results if r["status"] in GRADED)
    print(f"resolved {resolved}/{n}; graded {graded}/{n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
