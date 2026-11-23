# ImpasseBench

**Do coding agents know when to stop?** ImpasseBench places coding agents at
controlled environment impasses and scores each trajectory against the evidence
the agent has already collected.

It starts from solvable software-repair tasks, injects one of twelve realistic
environment faults into the execution sandbox, and marks in every trajectory the
step `t*` at which the agent's own observations become sufficient evidence of
the fault. Scoring actions relative to `t*` separates appropriate effort from
over-persistence (continuing after the evidence says outside help is needed) and
premature handoff (stopping before the evidence is in).

This repository contains the benchmark harness, the fault injection, the
rule-based annotator, patch grading, and the evidence monitor from the paper
*Knowing When to Stop: The Evidence–Action Gap in Coding Agents*. All 19,371
trajectories behind the paper's results, plus the monitor study, are in the
Hugging Face dataset [`C-Tianyu/ImpasseBench`](https://huggingface.co/datasets/C-Tianyu/ImpasseBench).

## Benchmark at a glance

| | |
|---|---|
| Tasks | 102 SWE-smith repair tasks from 10 Python repositories, 34 per difficulty tercile |
| Conditions | 12 injected faults in 4 categories, plus a clean control |
| Tools | `bash`, `edit_file`, `task_complete(summary, verified, unverified_reason)`; the `escalate` interface adds `escalate_to_human` |
| Budget | 80 steps |
| Labels | `C`, `Q`, `t*`, `N_post`, `S`, `L`, `U`, `U_post`, `Y`, `Z` (rule-based), `A` (held-out tests) |

| Category | Conditions | Appropriate response |
|---|---|---|
| Persistent barriers | `readonly_fs`, `disk_full`, `source_unreadable`, `source_write_denied`, `dependency_missing`, `sandbox_lost_early` | check recovery options, then report that outside help is needed |
| Verification limits | `tests_unreadable`, `python_env_damaged`, `sandbox_lost_late` | keep the useful changes, state what is unverified |
| Temporary faults | `git_flaky`, `sandbox_flaky` | retry and continue |
| Task mismatch | `bug_absent` | check the issue's behavior, report the mismatch |

Ten conditions apply to all 102 tasks; `dependency_missing` applies to 33 and
`python_env_damaged` to 56, which gives 1,211 (task, condition) pairs per model
and interface. See [docs/conditions.md](docs/conditions.md) for how each fault
is built.

## Install

```bash
git clone https://github.com/TianyuCodings/ImpasseBench.git
cd ImpasseBench
pip install -e ".[data]"            # analysis and the released trajectories
pip install -e ".[all]"             # + every model SDK, Bedrock and SWE-bench grading
```

Python 3.10+. Labeling and data loading need no Docker; running agents and
grading patches need a Linux x86-64 host with Docker.

## Quick start

**Load the released trajectories and labels.**

```python
from impassebench import data

labels = data.load_labels(as_pandas=True)          # 19,371 rows, no steps
for rec in data.iter_trajectories(models=["gpt-5.6-terra"], interfaces=["escalate"],
                                  conditions=["source_write_denied"]):
    print(rec["traj_id"], rec["termination"], rec["labels"]["t_star"])
```

or with `datasets`:

```python
from datasets import load_dataset
labels = load_dataset("C-Tianyu/ImpasseBench", "labels", split="main")
```

**Label trajectories** (released ones, or your own run directories):

```bash
impassebench annotate runs/ -o labels.jsonl -j 8
impassebench annotate --dataset /path/to/ImpasseBench -o labels.jsonl -j 8
```

**Run a model on the benchmark** (needs Docker and provider credentials):

```bash
impassebench pull-images
impassebench run --model gemini-3.5-flash --interface base     --parallel 8
impassebench run --model gemini-3.5-flash --interface escalate --parallel 8
impassebench annotate runs/ -o labels.jsonl
impassebench grade --help            # A: final patch vs held-out tests in a healthy container
```

See [docs/running.md](docs/running.md) for providers, output format and the
Python API.

## Repository layout

```
impassebench/
  harness/     conditions, fault injection, container control, prompts, model backends, rollout loop, runner
  annotate/    rule-based annotator (C, Q, t*, N_post, S, L, U, Y, Z) and patch grading (A)
  monitor/     evidence monitor, ticket template, bounded retry, restoration (paper Section 4)
  data.py      loaders for the released dataset and for run directories
  resources/   the 102 tasks and per-repository fault parameters
docs/          conditions, running, metrics, data format, monitor
tests/         unit tests (no Docker or network needed): pytest -q
```

## Documentation

* [docs/conditions.md](docs/conditions.md): the 12 faults and the clean control
* [docs/running.md](docs/running.md): running agents, output format
* [docs/metrics.md](docs/metrics.md): label definitions
* [docs/data.md](docs/data.md): the released dataset and its record format
* [docs/monitor.md](docs/monitor.md): the evidence monitor and its study

## Reproducibility

The annotator in this repository reproduces the labels of all 19,371
released trajectories exactly. The two system prompts match the recorded runs
byte for byte, and so does the injected fault configuration of 19,272 runs.
The other 99 are `dependency_missing` runs that ran three fewer read-only setup
checks; the environment they built is the same. The paper's tables can be
recomputed from `labels.parquet`: for example, the escalation tool cuts
framework-limit endings under persistent barriers from 34.0% to 6.7%.
Model outputs vary between runs and provider versions, so new runs reproduce
the protocol, not identical trajectories.

## Citation

```bibtex
@article{chen2026knowing,
  title   = {Knowing When to Stop: The Evidence--Action Gap in Coding Agents},
  author  = {Chen, Tianyu and Zhao, Xinran and Viswanathan, Vijay and Yu, Shoubin and Zhou, Mingyuan and Wu, Tongshuang},
  journal = {arXiv preprint},
  year    = {2026}
}
```

## License

MIT (see [LICENSE](LICENSE)). The tasks are derived from
[SWE-smith](https://github.com/SWE-bench/SWE-smith) and
[SWE-bench Verified](https://www.swebench.com/) (MIT).
