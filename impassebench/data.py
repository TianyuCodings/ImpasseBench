"""Load ImpasseBench trajectories, labels, tasks and conditions.

The released dataset (Hugging Face ``C-Tianyu/ImpasseBench``) has this layout::

    tasks.jsonl                                   102 base tasks
    conditions.json                               13 conditions (12 faults + clean)
    models.json                                   the 8 evaluated models
    fault_setups.jsonl                            injected environment per (task, condition)
    data/labels.parquet                           one row per trajectory, no steps
    data/trajectories/<model>/<interface>.parquet full trajectories with labels

Each trajectory is a *record* (a plain dict)::

    traj_id        "<model>/<interface>/<instance_id>/<condition>"
    model, model_id, interface ("base" | "escalate"), with_escalate, with_edit
    instance_id, repo, condition, category
    max_steps, n_steps, termination, total_tokens, sampling, started_at, ended_at
    system_prompt, final_patch, runtime_fault (dict or None),
    base_image, base_image_digest
    steps          list of step dicts, exactly as the harness recorded them
    labels         dict of annotation labels (see docs/data.md)

In the parquet files every step is stored in a fixed struct
(``STEP_FIELDS`` plus ``extra``); :func:`decode_step` restores the recorded
dict exactly, so ``iter_trajectories`` yields the same steps the harness wrote
to ``trace.jsonl``.

``pyarrow`` (and ``huggingface_hub`` for remote loading) are needed only by
the functions that read parquet files: ``pip install "impassebench[data]"``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Iterator

DEFAULT_REPO = "C-Tianyu/ImpasseBench"

INTERFACES = ("base", "escalate")

#: Condition id -> category.
CONDITION_CATEGORY = {
    "clean": "control",
    "readonly_fs": "persistent_barrier",
    "disk_full": "persistent_barrier",
    "source_unreadable": "persistent_barrier",
    "source_write_denied": "persistent_barrier",
    "dependency_missing": "persistent_barrier",
    "sandbox_lost_early": "persistent_barrier",
    "tests_unreadable": "verification_limit",
    "python_env_damaged": "verification_limit",
    "sandbox_lost_late": "verification_limit",
    "git_flaky": "temporary_fault",
    "sandbox_flaky": "temporary_fault",
    "bug_absent": "task_mismatch",
}

#: Terminal tools; their call arguments are stored JSON-encoded in ``action``.
TERMINAL_TOOLS = ("task_complete", "escalate_to_human")

#: Step fields in the order the harness writes them.
STEP_FIELDS = ("type", "action", "args", "observation", "exit_code",
               "wall_time_s", "truncated", "fault_injected", "assistant_text",
               "step")

_STR_FIELDS = {"type", "observation", "assistant_text"}
_INT_FIELDS = {"exit_code", "step"}
_FLOAT_FIELDS = {"wall_time_s"}
_BOOL_FIELDS = {"truncated", "fault_injected"}

# Fields of a record other than ``steps`` and ``labels``, in column order.
RECORD_FIELDS = ("traj_id", "model", "model_id", "interface", "with_escalate",
                 "with_edit", "instance_id", "repo", "condition", "category",
                 "max_steps", "n_steps", "termination", "total_tokens",
                 "sampling", "started_at", "ended_at", "system_prompt",
                 "final_patch", "runtime_fault", "base_image",
                 "base_image_digest")


# ---------------------------------------------------------------------------
# Step encoding
# ---------------------------------------------------------------------------

def _fits(name: str, value: Any) -> bool:
    if name in _STR_FIELDS:
        return type(value) is str
    if name in _INT_FIELDS:
        return type(value) is int
    if name in _FLOAT_FIELDS:
        return type(value) is float
    if name in _BOOL_FIELDS:
        return type(value) is bool
    return True


def encode_step(step: dict) -> dict:
    """Map a recorded step to the fixed parquet struct (lossless).

    ``action`` holds the command text for ``bash``, the edit description for
    ``edit``, the reply text for ``text`` and, JSON-encoded, the call
    arguments for terminal tools. ``args`` (invalid tool calls) is
    JSON-encoded. Anything that does not fit its typed field, and any key
    outside ``STEP_FIELDS``, goes to ``extra`` (a JSON object) so that
    :func:`decode_step` can restore it.
    """
    out: dict[str, Any] = {k: None for k in STEP_FIELDS}
    extra: dict[str, Any] = {}
    terminal = step.get("type") in TERMINAL_TOOLS
    for key, value in step.items():
        if key not in out or value is None:
            extra[key] = value
        elif key == "action":
            if terminal:
                out[key] = json.dumps(value, ensure_ascii=False)
            elif type(value) is str:
                out[key] = value
            else:
                extra[key] = value
        elif key == "args":
            out[key] = json.dumps(value, ensure_ascii=False)
        elif _fits(key, value):
            out[key] = value
        else:
            extra[key] = value
    out["extra"] = json.dumps(extra, ensure_ascii=False) if extra else None
    return out


def decode_step(enc: dict) -> dict:
    """Inverse of :func:`encode_step`."""
    terminal = enc.get("type") in TERMINAL_TOOLS
    step: dict[str, Any] = {}
    for key in STEP_FIELDS:
        value = enc.get(key)
        if value is None:
            continue
        if key == "args" or (key == "action" and terminal):
            value = json.loads(value)
        step[key] = value
    if enc.get("extra"):
        step.update(json.loads(enc["extra"]))
    return step


def encode_record(record: dict) -> dict:
    """Record -> parquet row (steps encoded, runtime_fault JSON-encoded)."""
    row = {k: record.get(k) for k in RECORD_FIELDS}
    rf = record.get("runtime_fault")
    row["runtime_fault"] = None if rf is None else json.dumps(rf, sort_keys=True)
    row["steps"] = [encode_step(s) for s in record.get("steps") or []]
    row["labels"] = record.get("labels")
    return row


def decode_record(row: dict) -> dict:
    """Parquet row -> record (inverse of :func:`encode_record`)."""
    rec = dict(row)
    rf = rec.get("runtime_fault")
    if isinstance(rf, str):
        rec["runtime_fault"] = json.loads(rf)
    if rec.get("steps") is not None:
        rec["steps"] = [decode_step(s) for s in rec["steps"]]
    return rec


# ---------------------------------------------------------------------------
# Locating the dataset
# ---------------------------------------------------------------------------

def dataset_path(source: str | Path | None = None,
                 allow_patterns: Iterable[str] | None = None,
                 revision: str | None = None) -> Path:
    """Return a local directory holding (part of) the dataset.

    ``source`` is a local directory, a Hugging Face dataset repo id, or None
    for :data:`DEFAULT_REPO`. Remote files are fetched with
    ``huggingface_hub.snapshot_download`` (only ``allow_patterns`` if given)
    and cached by it.
    """
    if source is not None and Path(source).expanduser().is_dir():
        return Path(source).expanduser()
    from huggingface_hub import snapshot_download
    path = snapshot_download(
        repo_id=str(source or DEFAULT_REPO), repo_type="dataset",
        allow_patterns=list(allow_patterns) if allow_patterns else None,
        revision=revision)
    return Path(path)


def _trajectory_files(root: Path, models=None, interfaces=None) -> list[Path]:
    files = sorted((root / "data" / "trajectories").glob("*/*.parquet"))
    if models:
        files = [f for f in files if f.parent.name in set(models)]
    if interfaces:
        files = [f for f in files if f.stem in set(interfaces)]
    return files


def _patterns(models=None, interfaces=None) -> list[str]:
    ms = list(models) if models else ["*"]
    its = list(interfaces) if interfaces else ["*"]
    return [f"data/trajectories/{m}/{i}.parquet" for m in ms for i in its]


# ---------------------------------------------------------------------------
# Readers
# ---------------------------------------------------------------------------

def load_tasks(source=None) -> dict[str, dict]:
    """instance_id -> task dict (problem statement, image, tests, ...)."""
    root = dataset_path(source, ["tasks.jsonl"])
    with open(root / "tasks.jsonl", encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    return {r["instance_id"]: r for r in rows}


def load_conditions(source=None) -> list[dict]:
    root = dataset_path(source, ["conditions.json"])
    return json.loads((root / "conditions.json").read_text(encoding="utf-8"))


def load_models(source=None) -> list[dict]:
    root = dataset_path(source, ["models.json"])
    return json.loads((root / "models.json").read_text(encoding="utf-8"))


def load_fault_setups(source=None) -> dict[tuple[str, str], dict]:
    """(instance_id, condition) -> injected environment configuration."""
    root = dataset_path(source, ["fault_setups.jsonl"])
    out = {}
    with open(root / "fault_setups.jsonl", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                r = json.loads(line)
                out[(r["instance_id"], r["condition"])] = r
    return out


def load_labels(source=None, as_pandas: bool = False):
    """One row per trajectory (ids, outcome and labels, no steps).

    Returns a list of dicts, or a ``pandas.DataFrame`` if ``as_pandas``.
    """
    import pyarrow.parquet as pq
    root = dataset_path(source, ["data/labels.parquet"])
    table = pq.read_table(root / "data" / "labels.parquet")
    return table.to_pandas() if as_pandas else table.to_pylist()


def iter_trajectories(source=None, *, models=None, interfaces=None,
                      conditions=None, instances=None,
                      with_steps: bool = True) -> Iterator[dict]:
    """Yield trajectory records, optionally filtered.

    ``models`` are model slugs (e.g. ``"gpt-5.6-terra"``), ``interfaces`` are
    ``"base"`` / ``"escalate"``. Only the matching parquet files are
    downloaded. ``with_steps=False`` skips the (large) steps column.
    """
    import pyarrow.parquet as pq
    root = dataset_path(source, _patterns(models, interfaces))
    conds = set(conditions) if conditions else None
    insts = set(instances) if instances else None
    for path in _trajectory_files(root, models, interfaces):
        pf = pq.ParquetFile(path)
        columns = None
        if not with_steps:
            columns = [n for n in pf.schema_arrow.names if n != "steps"]
        for batch in pf.iter_batches(batch_size=64, columns=columns):
            for row in batch.to_pylist():
                if conds and row["condition"] not in conds:
                    continue
                if insts and row["instance_id"] not in insts:
                    continue
                yield decode_record(row)


def load_trajectory(traj_id: str, source=None) -> dict:
    """Fetch a single record by ``traj_id``."""
    model, interface, instance_id, condition = traj_id.split("/")
    for rec in iter_trajectories(source, models=[model], interfaces=[interface],
                                 conditions=[condition], instances=[instance_id]):
        return rec
    raise KeyError(traj_id)


# ---------------------------------------------------------------------------
# Run directories (the harness output format)
# ---------------------------------------------------------------------------

def record_to_raw(record: dict) -> tuple[dict, list[dict]]:
    """Record -> (meta, trace rows) in the harness's run-directory format."""
    meta = {
        "run_id": record["traj_id"].replace("/", "__"),
        "instance_id": record["instance_id"],
        "repo": record["repo"],
        "condition": record["condition"],
        "with_escalate": bool(record["with_escalate"]),
        "with_edit": bool(record.get("with_edit", True)),
        "max_steps": record["max_steps"],
        "n_steps": record["n_steps"],
        "termination": record["termination"],
        "total_tokens": record.get("total_tokens"),
        "model": record.get("model_id"),
        "sampling": record.get("sampling"),
        "started_at": record.get("started_at"),
        "ended_at": record.get("ended_at"),
        "system_prompt": record.get("system_prompt"),
        "model_patch": record.get("final_patch") or "",
        "runtime_fault": record.get("runtime_fault"),
        "base_image": record.get("base_image"),
        "base_image_digest": record.get("base_image_digest"),
    }
    return meta, [dict(s) for s in record.get("steps") or []]


def write_run_dir(record: dict, out_dir: str | Path) -> Path:
    """Write ``trace.jsonl``, ``meta.json`` (and ``diff.patch`` when the
    patch is non-empty) for one record; returns the run directory."""
    meta, steps = record_to_raw(record)
    run_dir = Path(out_dir) / meta["run_id"]
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "trace.jsonl", "w", encoding="utf-8") as fh:
        for s in steps:
            fh.write(json.dumps(s) + "\n")
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    if meta["model_patch"]:
        (run_dir / "diff.patch").write_text(meta["model_patch"], encoding="utf-8")
    return run_dir


def read_run_dir(run_dir: str | Path, *, model: str | None = None,
                 category: str | None = None, labels: dict | None = None) -> dict:
    """Build a record from a run directory written by the harness."""
    run_dir = Path(run_dir)
    meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
    with open(run_dir / "trace.jsonl", encoding="utf-8") as fh:
        steps = [json.loads(line) for line in fh if line.strip()]
    interface = "escalate" if meta.get("with_escalate") else "base"
    model = model or meta.get("model")
    rec = {
        "traj_id": f"{model}/{interface}/{meta['instance_id']}/{meta['condition']}",
        "model": model,
        "model_id": meta.get("model"),
        "interface": interface,
        "with_escalate": bool(meta.get("with_escalate")),
        "with_edit": bool(meta.get("with_edit", True)),
        "instance_id": meta["instance_id"],
        "repo": meta.get("repo"),
        "condition": meta["condition"],
        "category": category or CONDITION_CATEGORY.get(meta["condition"]),
        "max_steps": meta.get("max_steps"),
        "n_steps": meta.get("n_steps", len(steps)),
        "termination": meta.get("termination"),
        "total_tokens": meta.get("total_tokens"),
        "sampling": meta.get("sampling"),
        "started_at": meta.get("started_at"),
        "ended_at": meta.get("ended_at"),
        "system_prompt": meta.get("system_prompt"),
        "final_patch": meta.get("model_patch") or "",
        "runtime_fault": meta.get("runtime_fault"),
        "base_image": meta.get("base_image"),
        "base_image_digest": meta.get("base_image_digest"),
        "steps": steps,
        "labels": labels,
    }
    return rec


def iter_run_dirs(root: str | Path, **kwargs) -> Iterator[dict]:
    """Records for every ``*/meta.json`` run directory under ``root``."""
    for meta_path in sorted(Path(root).glob("**/meta.json")):
        if (meta_path.parent / "trace.jsonl").exists():
            yield read_run_dir(meta_path.parent, **kwargs)
