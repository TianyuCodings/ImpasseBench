"""Run ImpasseBench trajectories.

Examples::

    impassebench run --model gemini-3.5-flash --interface escalate
    impassebench run --model us.anthropic.claude-haiku-4-5-20251001-v1:0 \\
        --interface base --conditions readonly_fs,git_flaky --tasks <id>,<id> --parallel 8

Trajectories are written to ``<out>/<model>/<interface>/<run_id>/``. Pairs
(task, condition) that already have a finished trajectory there are skipped,
so an interrupted run can simply be restarted with the same arguments.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .conditions import CONDITIONS
from .prompts import INTERFACES
from .rollout import HARNESS_FAILURES, TERMINATIONS, run_rollout
from .tasks import load_tasks

HALT_AFTER = 5      # abort if the first N trajectories all fail in the harness
CONSEC_HALT = 10    # abort after N consecutive harness failures


def load_dotenv(path: Path = Path(".env")) -> None:
    """Export KEY=VALUE lines of a local .env file (values are never printed)."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def model_dirname(model: str) -> str:
    return model.replace("/", "_").replace(":", "_")


def finished_pairs(arm_dir: Path) -> set[tuple[str, str]]:
    """(instance_id, condition) pairs with a completed trajectory in arm_dir."""
    done = set()
    if not arm_dir.exists():
        return done
    for meta_path in arm_dir.glob("*/meta.json"):
        try:
            m = json.loads(meta_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if m.get("termination") in TERMINATIONS:
            done.add((m["instance_id"], m["condition"]))
    return done


def build_jobs(tasks: list[dict], conditions: list[str] | None) -> list[tuple[dict, str]]:
    jobs = []
    for t in tasks:
        for c in t["conditions"]:
            if conditions is None or c in conditions:
                jobs.append((t, c))
    return jobs


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="impassebench run", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True,
                    help="model id; see impassebench.harness.agents for routing")
    ap.add_argument("--interface", required=True, choices=INTERFACES,
                    help="'escalate' adds the escalate_to_human tool")
    ap.add_argument("--tasks", default="all",
                    help="comma-separated instance ids, or 'all' (default)")
    ap.add_argument("--task-file", default=None,
                    help="alternative task file (default: the bundled 102 tasks)")
    ap.add_argument("--conditions", default="all",
                    help="comma-separated condition ids, or 'all' applicable (default)")
    ap.add_argument("--max-steps", type=int, default=80)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--parallel", type=int, default=1, help="concurrent trajectories")
    ap.add_argument("--out", default="runs", help="output root (default: runs/)")
    ap.add_argument("--dry-run", action="store_true", help="list the jobs and exit")
    args = ap.parse_args(argv)

    load_dotenv()
    tasks = load_tasks(args.task_file)
    if args.tasks != "all":
        wanted = [s.strip() for s in args.tasks.split(",") if s.strip()]
        index = {t["instance_id"]: t for t in tasks}
        missing = [w for w in wanted if w not in index]
        if missing:
            ap.error(f"unknown task id(s): {', '.join(missing)}")
        tasks = [index[w] for w in wanted]
    conditions = None
    if args.conditions != "all":
        conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
        unknown = [c for c in conditions if c not in CONDITIONS]
        if unknown:
            ap.error(f"unknown condition(s): {', '.join(unknown)}; "
                     f"choose from {', '.join(CONDITIONS)}")

    arm_dir = Path(args.out) / model_dirname(args.model) / args.interface
    jobs = build_jobs(tasks, conditions)
    done = finished_pairs(arm_dir)
    todo = [(t, c) for t, c in jobs if (t["instance_id"], c) not in done]
    print(f"{len(jobs)} (task, condition) pairs, {len(jobs) - len(todo)} already "
          f"finished, {len(todo)} to run -> {arm_dir}", flush=True)
    if args.dry_run:
        for t, c in todo:
            print(f"  {t['instance_id']}  {c}")
        return 0
    if not todo:
        return 0
    if subprocess.run(["docker", "info"], capture_output=True).returncode != 0:
        print("docker is not reachable; start it before running", file=sys.stderr)
        return 2

    from .agents import make_agent
    make_agent(args.model, args.interface, args.temperature)  # fail fast on routing/credentials

    arm_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []
    lock = threading.Lock()
    halted: list[bool] = []

    def one(task: dict, cond: str) -> dict:
        iid = task["instance_id"]
        if halted:
            return {"instance_id": iid, "condition": cond, "termination": "skipped",
                    "n_steps": 0, "total_tokens": 0}
        agent = make_agent(args.model, args.interface, args.temperature)
        try:
            meta = run_rollout(task, cond, agent, interface=args.interface,
                               max_steps=args.max_steps, out_dir=arm_dir)
        except Exception as e:  # noqa: BLE001
            meta = {"instance_id": iid, "condition": cond,
                    "termination": "harness_error", "error": repr(e),
                    "n_steps": 0, "total_tokens": getattr(agent, "total_tokens", 0)}
        with lock:
            results.append(meta)
            early = results[:HALT_AFTER]
            if (len(results) >= HALT_AFTER and not halted
                    and all(m["termination"] == "harness_error" for m in early)):
                halted.append(True)
                print(f"\n*** aborting: the first {HALT_AFTER} trajectories failed in "
                      f"the harness. Last error:\n    {early[-1].get('error', '')[:300]}\n",
                      flush=True)
            tail = results[-CONSEC_HALT:]
            if (len(tail) == CONSEC_HALT and not halted
                    and all(m["termination"] in HARNESS_FAILURES for m in tail)):
                halted.append(True)
                print(f"\n*** aborting: {CONSEC_HALT} consecutive harness failures. "
                      f"Last error:\n    {tail[-1].get('error', '')[:300]}\n", flush=True)
            print(f"[{len(results)}/{len(todo)}] {iid} / {cond} -> {meta['termination']} "
                  f"in {meta['n_steps']} steps", flush=True)
        return meta

    with ThreadPoolExecutor(max_workers=max(1, args.parallel)) as pool:
        futures = [pool.submit(one, t, c) for t, c in todo]
        for f in as_completed(futures):
            f.result()

    failed = sum(m["termination"] in HARNESS_FAILURES for m in results)
    print(f"\n{len(results)} trajectories, {failed} harness failures "
          f"(re-run the same command to retry them)")
    return 1 if halted else 0


if __name__ == "__main__":
    sys.exit(main())
