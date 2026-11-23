"""Command line: annotate trajectories.

    python -m impassebench.annotate.cli RUNS_DIR [RUNS_DIR ...] -o labels.jsonl
    python -m impassebench.annotate.cli --dataset data/trajectories -o labels.jsonl

A RUNS_DIR is a harness output directory: either one run (meta.json +
trace.jsonl) or a directory whose sub-directories are runs. `--dataset`
reads released trajectory files through `impassebench.data`.
Each output line is {"traj_id", "condition", **labels}.
"""

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from .core import NOT_A_TRAJECTORY, annotate
from .io import load_run_dir, load_tasks

_TASKS = None


def _run_dirs(paths):
    for p in map(Path, paths):
        if (p / "meta.json").exists():
            yield p
        else:
            yield from sorted(d for d in p.iterdir() if (d / "meta.json").exists())


def _label_run_dir(args):
    path, tasks_path = args
    global _TASKS
    if _TASKS is None:
        _TASKS = load_tasks(tasks_path)
    rec = load_run_dir(path)
    if rec.get("termination") in NOT_A_TRAJECTORY:
        return dict(traj_id=rec.get("traj_id"), condition=rec.get("condition"),
                    skipped=f"termination={rec.get('termination')!r}")
    return _label(rec, _TASKS)


def _label(rec, tasks):
    try:
        lab = annotate(rec, tasks[rec["instance_id"]])
    except (KeyError, ValueError) as e:
        return dict(traj_id=rec.get("traj_id"), condition=rec.get("condition"), error=str(e))
    return dict(traj_id=rec.get("traj_id"), condition=rec.get("condition"), **lab)


def _dataset_records(path):
    try:
        from .. import data  # released-dataset loader
    except ImportError:  # pragma: no cover
        sys.exit("impassebench.data is not available; annotate run directories instead")
    loader = getattr(data, "iter_trajectories", None) or getattr(data, "load_trajectories", None)
    if loader is None:  # pragma: no cover
        sys.exit("impassebench.data exposes no trajectory loader")
    yield from loader(path)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Rule-based ImpasseBench trajectory annotator.")
    ap.add_argument("runs", nargs="*", help="harness run directories (or parents of them)")
    ap.add_argument("--dataset", help="released trajectory file or directory (via impassebench.data)")
    ap.add_argument("--tasks", default=None, help="tasks.json (default: the bundled task list)")
    ap.add_argument("-o", "--out", default="-", help="output JSONL (default: stdout)")
    ap.add_argument("-j", "--jobs", type=int, default=1, help="worker processes")
    args = ap.parse_args(argv)
    if not args.runs and not args.dataset:
        ap.error("give run directories or --dataset")

    out = sys.stdout if args.out == "-" else open(args.out, "w")
    n = bad = 0

    def emit(row):
        nonlocal n, bad
        n += 1
        bad += int("error" in row or "skipped" in row)
        out.write(json.dumps(row, ensure_ascii=False) + "\n")

    if args.runs:
        jobs = [(str(p), args.tasks) for p in _run_dirs(args.runs)]
        if args.jobs > 1:
            with ProcessPoolExecutor(args.jobs) as pool:
                for row in pool.map(_label_run_dir, jobs, chunksize=8):
                    emit(row)
        else:
            for j in jobs:
                emit(_label_run_dir(j))
    if args.dataset:
        tasks = load_tasks(args.tasks)
        for rec in _dataset_records(args.dataset):
            emit(_label(rec, tasks))
    if out is not sys.stdout:
        out.close()
    print(f"annotated {n} trajectories ({bad} skipped or failed)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
