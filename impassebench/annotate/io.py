"""Loading tasks and harness run directories into annotator records."""

import json
from pathlib import Path

from . import common as cc

TASKS_PATH = Path(__file__).resolve().parent.parent / "resources" / "tasks.json"

# metadata keys copied from a run directory's meta.json into the record
_META_KEYS = ("traj_id", "model", "model_id", "interface", "with_escalate", "with_edit",
              "instance_id", "repo", "condition", "category", "max_steps", "n_steps",
              "termination", "total_tokens", "sampling", "started_at", "ended_at")


def load_tasks(path=None):
    """{instance_id: task} from tasks.json (default: the bundled task list)."""
    payload = json.loads(Path(path or TASKS_PATH).read_text())
    rows = payload["tasks"] if isinstance(payload, dict) else payload
    return {r["instance_id"]: r for r in rows}


def load_run_dir(run_dir):
    """A trajectory record from one harness output directory
    (meta.json + trace.jsonl [+ diff.patch])."""
    run_dir = Path(run_dir)
    meta = json.loads((run_dir / "meta.json").read_text())
    trace_p = run_dir / "trace.jsonl"
    steps = []
    if trace_p.exists():
        with open(trace_p, errors="replace") as fh:
            steps = cc.read_trace_lines(fh)
    record = {k: meta[k] for k in _META_KEYS if k in meta}
    patch = meta.get("final_patch")
    if patch is None:
        patch = meta.get("model_patch")
    if patch is None and (run_dir / "diff.patch").exists():
        patch = (run_dir / "diff.patch").read_text(errors="replace")
    record["final_patch"] = patch or ""
    record["steps"] = steps
    record.setdefault("traj_id", run_dir.name)
    return record
