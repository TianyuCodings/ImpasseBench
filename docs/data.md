# Trajectory data

The trajectories and labels behind the paper are released as the Hugging Face
dataset `C-Tianyu/ImpasseBench`. `impassebench.data` reads it (install the
extra with `pip install "impassebench[data]"`).

## What is in it

19,371 trajectories: 8 models × 2 interfaces (`base`, `escalate`) × 102
SWE-smith tasks × the 13 conditions applicable to each task (12 faults + the
clean control), one run per cell, 80-step budget. Five Gemini 3.5 Flash
`bug_absent` cells are missing, so 9,684 pairs are complete and 3 baseline runs
have no pair.

```
tasks.jsonl                                   102 tasks (same content as data/tasks.parquet)
conditions.json                               13 conditions with category and expected response
models.json                                   8 models: slug, request id, access route, sampling
fault_setups.jsonl                            injected environment per (instance_id, condition)
data/labels.parquet                           one row per trajectory, labels as flat columns, no steps
data/trajectories/<model>/<interface>.parquet full trajectories (16 files, ~0.5 GB in total)
```

`fault_setups.jsonl` records, for every (task, condition), the container flags,
setup commands, agent user, agent environment and runtime-fault spec that the
harness applied; `impassebench.harness` regenerates the same configuration.

## Loading

```python
from impassebench import data

tasks = data.load_tasks()                     # instance_id -> task
labels = data.load_labels(as_pandas=True)     # 19,371 rows

for rec in data.iter_trajectories(models=["claude-haiku-4.5"], interfaces=["base"],
                                  conditions=["disk_full"]):
    rec["traj_id"], rec["termination"], rec["labels"]["Q"], rec["labels"]["t_star"]
    for step in rec["steps"]:
        step["type"], step["action"], step.get("observation")

rec = data.load_trajectory("gpt-5.6-terra/escalate/<instance_id>/source_write_denied")
```

Every function takes an optional `source`: a local directory with the layout
above, or a Hugging Face repo id (default `C-Tianyu/ImpasseBench`). Remote files
are fetched with `huggingface_hub.snapshot_download`, and only the parquet files
that match the `models` / `interfaces` filters are downloaded.
`iter_trajectories(..., with_steps=False)` skips the step column.

The `datasets` library works too:

```python
from datasets import load_dataset
labels = load_dataset("C-Tianyu/ImpasseBench", "labels", split="main")
trajs = load_dataset("C-Tianyu/ImpasseBench", "trajectories", split="main", streaming=True)
tasks = load_dataset("C-Tianyu/ImpasseBench", "tasks", split="test")
```

With `datasets`, steps come back in their parquet encoding (see below); pass a
row through `data.decode_record` to get the recorded form.

## Records

A record is a plain dict:

| field | description |
|---|---|
| `traj_id` | `<model>/<interface>/<instance_id>/<condition>` |
| `model`, `model_id` | model slug and request id |
| `interface`, `with_escalate`, `with_edit` | `base` / `escalate`; `edit_file` is always offered |
| `instance_id`, `repo`, `condition`, `category` | task and injected condition |
| `max_steps`, `n_steps`, `termination` | budget, rounds used, how the run ended (`task_complete`, `escalate`, `max_steps`, `text_stall`) |
| `total_tokens`, `sampling`, `started_at`, `ended_at` | run metadata |
| `system_prompt` | the system prompt the agent saw |
| `final_patch` | repository diff at termination (`""` if nothing changed) |
| `runtime_fault` | sandbox conditions only: when the fault armed and fired, and how many calls failed |
| `base_image`, `base_image_digest` | SWE-smith image |
| `steps` | the rounds, exactly as the harness wrote them to `trace.jsonl` |
| `labels` | annotation labels (definitions in the dataset card and [metrics.md](metrics.md)) |

Step `type` is one of `bash`, `edit`, `task_complete`, `escalate_to_human`,
`text` (reply without a tool call) and `invalid_tool`. Shell and edit steps carry
`observation`, `exit_code` and (shell) `wall_time_s` / `truncated`;
`fault_injected` marks a call answered by an injected runtime fault; terminal
steps carry the call arguments as `action`.

### Parquet encoding

In the parquet files each step is a struct with the fixed fields
`type, action, args, observation, exit_code, wall_time_s, truncated,
fault_injected, assistant_text, step, extra`. Terminal-tool arguments (`action`)
and invalid-call arguments (`args`) are JSON strings; any value that does not fit
its typed field lands in `extra` (always null in the released data).
`data.decode_step` / `data.decode_record` invert this exactly: for all 19,371
released trajectories the decoded steps equal the recorded rows, and
`json.dumps(step)` per line reproduces each original `trace.jsonl`
byte for byte.

## Run directories

The harness writes one directory per run (`trace.jsonl`, `meta.json`,
`diff.patch` when the patch is non-empty). To move between the two forms:

```python
meta, trace = data.record_to_raw(rec)       # harness-style meta dict + trace rows
run_dir = data.write_run_dir(rec, "runs/")  # materialise a record as a run directory
rec2 = data.read_run_dir(run_dir, model="my-model")   # and back
recs = list(data.iter_run_dirs("runs/my_experiment", model="my-model"))
```

`read_run_dir` is the way to turn new rollouts into records that the annotator
and the analysis code accept alongside the released ones.
