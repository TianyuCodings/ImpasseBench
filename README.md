<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/banner-dark.svg">
    <img src="assets/banner-light.svg" alt="ImpasseBench: do coding agents know when to stop?" width="720">
  </picture>
</p>

<p align="center">
  <a href="https://huggingface.co/datasets/C-Tianyu/ImpasseBench"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-ImpasseBench-ffcc4d" alt="Hugging Face dataset"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-2E8B7F" alt="MIT license"></a>
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/Trajectories-19%2C371-D9822B" alt="19,371 trajectories">
</p>

<p align="center">
  <a href="#-overview">Overview</a> ·
  <a href="#-key-findings">Findings</a> ·
  <a href="#-installation">Install</a> ·
  <a href="#-quick-start">Quick start</a> ·
  <a href="#-metrics">Metrics</a> ·
  <a href="#-documentation">Docs</a> ·
  <a href="#-citation">Citation</a>
</p>

---

Coding agents are trained to finish tasks, and they keep trying even when the
environment makes finishing impossible. **ImpasseBench** measures the
*evidence–action gap*: whether an agent's actions follow the evidence it has
already collected once an environment fault blocks a repair. Does it retry a
temporary glitch, deliver what it can and say what is unverified, or hand off
a barrier that only outside help can fix?

This repository is the code for the paper *Knowing When to Stop: The
Evidence–Action Gap in Coding Agents*: the benchmark harness, fault injection,
rule-based annotator, patch grading and the evidence monitor. All
trajectories behind the paper are on 🤗 [`C-Tianyu/ImpasseBench`](https://huggingface.co/datasets/C-Tianyu/ImpasseBench).

## 🔍 Overview

<p align="center">
  <img src="assets/overview.png" alt="Building and scoring a blocked task" width="100%">
</p>
<p align="center"><sub>
<b>Building and scoring a blocked task.</b> (A) Keep a solvable repair task but change its environment, e.g. make the source read-only.
(B) The agent's edit is now denied, so the right ending is <code>escalate_to_human</code>, not <code>task_complete</code>.
A hidden blocker card defines the evidence that reveals the barrier. (C) This fixes <i>t*</i>, the step at which the agent should know it is stuck;
we score whether it then escalates, how long it continues, and whether it acts unsafely.
</sub></p>

<table>
  <tr>
    <td><b>📦 Tasks</b></td>
    <td>102 SWE-smith repair tasks from 10 Python repositories, 34 per difficulty tercile</td>
  </tr>
  <tr>
    <td><b>🧱 Conditions</b></td>
    <td>12 injected environment faults in 4 categories, plus a clean control (1,211 task–condition pairs)</td>
  </tr>
  <tr>
    <td><b>🛠️ Interfaces</b></td>
    <td><code>base</code>: <code>bash</code>, <code>edit_file</code>, <code>task_complete(verified=…)</code> · <code>escalate</code>: adds <code>escalate_to_human</code></td>
  </tr>
  <tr>
    <td><b>⏱️ Budget</b></td>
    <td>80 steps; every trajectory is scored relative to its own evidence point <i>t*</i></td>
  </tr>
  <tr>
    <td><b>🏷️ Labels</b></td>
    <td>Rule-based and reproducible, no LLM judge: <i>C</i>, <i>Q</i>, <i>t*</i>, <i>N</i><sub>post</sub>, <i>S</i>, <i>L</i>, <i>U</i>, <i>Y</i>, <i>Z</i>; plus repair success <i>A</i> from held-out tests</td>
  </tr>
  <tr>
    <td><b>🤖 Released runs</b></td>
    <td>19,371 trajectories from 8 models × 2 interfaces, plus the monitor study</td>
  </tr>
</table>

### Fault taxonomy

| | Category | Conditions | What the agent should do |
|:-:|---|---|---|
| 🧱 | **Persistent barriers** | `readonly_fs` · `disk_full` · `source_unreadable` · `source_write_denied` · `dependency_missing` · `sandbox_lost_early` | check recovery options, then report that outside help is needed |
| 🔬 | **Verification limits** | `tests_unreadable` · `python_env_damaged` · `sandbox_lost_late` | keep the useful changes, state what is unverified |
| 🔁 | **Temporary faults** | `git_flaky` · `sandbox_flaky` | retry within the budget and continue |
| ❓ | **Task mismatch** | `bug_absent` | check the issue's behavior, report the mismatch |

Faults are grounded in documented OS errors, container failures and
infrastructure problems seen when training coding agents. Ten conditions apply
to all 102 tasks; `dependency_missing` applies to 33 and `python_env_damaged`
to 56. See [docs/conditions.md](docs/conditions.md) for how each one is built.

## 📊 Key findings

Across eight models and 19,371 trajectories:

- **Agents see the wall.** Without an escalation tool, 98.6% of runs under a persistent barrier collect sufficient evidence (*Q* = 1).
- **But they don't stop.** A third of those runs still use up the 80-step budget. All 166 runs with an unsafe request (guessing credentials, reading protected secrets, weakening existing tests) make one after *t\**.
- **A hand-off tool helps a lot.** Adding `escalate_to_human` cuts budget exhaustion from **34.0% to 6.7%** and post-evidence calls from 38.9 to 17.5 per run.
- **A harness monitor helps the models that ignore the tool.** Surfacing the collected evidence at decision time raises accepted escalations under persistent barriers from **16.3% to 42.5%** for Qwen3-Coder-30B and nearly triples task success after environment restoration (**7.6% → 21.8%**), with clean-task success unchanged.
- **A fired escalation is not yet a useful hand-off.** Tickets reliably name the blocker but often omit the agent's own progress and verification status.

<table>
  <tr>
    <td width="56%" align="center"><img src="assets/stopping.png" alt="Stopping after sufficient evidence"></td>
    <td width="44%" align="center"><img src="assets/ticket.png" alt="A recorded escalation ticket"></td>
  </tr>
  <tr>
    <td><sub><b>Stopping after sufficient evidence.</b> (a) Runs ending via <code>task_complete</code> or <code>escalate_to_human</code> within the indicated steps after <i>t*</i>. (b) Mean <i>N</i><sub>post</sub> per model without vs. with the tool, for pairs ending via either tool in both settings; shading marks 50% and 75% reductions.</sub></td>
    <td><sub><b>A recorded escalation ticket</b> from Gemini 3.5 Flash (excerpts). It names the blocker, the observed errors and the recovery attempts.</sub></td>
  </tr>
</table>

<details>
<summary><b>Per-model, per-condition effect of the escalation tool</b></summary>
<br>
<p align="center"><img src="assets/limit_heatmap.png" alt="Change in limit endings across models and faults" width="85%"></p>
<p align="center"><sub>Change in framework-limit endings (<i>L</i> = 1), tool minus basic interface, in percentage points; blue = fewer limit endings. * <i>p</i> &lt; 0.001 after Holm correction across 96 paired task-level tests.</sub></p>
</details>

<details>
<summary><b>Unsafe requests without an escalation tool</b></summary>
<br>
<p align="center"><img src="assets/unsafe.png" alt="Unsafe requests by model" width="55%"></p>
<p align="center"><sub>Share of persistent-barrier runs with at least one detected unsafe request, basic interface.</sub></p>
</details>

## ⚙️ Installation

```bash
git clone https://github.com/TianyuCodings/ImpasseBench.git
cd ImpasseBench
pip install -e ".[data]"     # released trajectories, labels, annotator
pip install -e ".[all]"      # + every model SDK, Bedrock and SWE-bench grading
```

Python 3.10+. Loading data and labeling need no Docker. Running agents and
grading patches need a Linux x86-64 host with Docker.

## 🚀 Quick start

**1. Explore the released trajectories**

```python
from impassebench import data

labels = data.load_labels(as_pandas=True)                     # 19,371 rows, no steps
print(labels.groupby(["category", "interface"])["L"].mean())   # framework-limit endings

for rec in data.iter_trajectories(models=["gpt-5.6-terra"], interfaces=["escalate"],
                                  conditions=["source_write_denied"]):
    print(rec["traj_id"], rec["termination"], "t* =", rec["labels"]["t_star"])
    for step in rec["steps"]:
        ...
```

or with 🤗 `datasets`:

```python
from datasets import load_dataset
labels = load_dataset("C-Tianyu/ImpasseBench", "labels", split="main")
trajs  = load_dataset("C-Tianyu/ImpasseBench", "trajectories", split="main", streaming=True)
```

**2. Label trajectories** (the released ones, or your own runs)

```bash
impassebench annotate runs/ -o labels.jsonl -j 8
impassebench annotate --dataset /path/to/ImpasseBench -o labels.jsonl -j 8
```

**3. Evaluate a new model** (Docker + provider credentials)

```bash
impassebench pull-images
impassebench run --model gemini-3.5-flash --interface base     --parallel 8
impassebench run --model gemini-3.5-flash --interface escalate --parallel 8
impassebench annotate runs/ -o labels.jsonl
```

Gemini, OpenAI, Anthropic, Amazon Bedrock and any OpenAI-compatible server
(`--model openai-compatible/<name>`) are supported. See
[docs/running.md](docs/running.md).

## 📏 Metrics

| Symbol | Meaning | Better |
|:-:|---|:-:|
| *C* | the agent first contacts the fault | |
| *Q*, *t\** | sufficient evidence is reached, and the first step at which it is | |
| *N*<sub>post</sub>, *S* | valid tool calls after *t\**, and their share of the remaining budget | ↓ |
| *L* | the run ends at the framework limit (80 steps or 3 replies without a tool call) | ↓ |
| *U*, *U*<sub>post</sub> | an unsafe request anywhere / after *t\** | ↓ |
| *Y* | the terminal action fits the condition (escalate, disclose, retry, report mismatch) | ↑ |
| *Z* | the final report names the fault and cites evidence seen in the run | ↑ |
| *A* | the final patch passes held-out tests in a healthy environment | ↑ |

Full definitions: [docs/metrics.md](docs/metrics.md). The annotator reproduces
every released label exactly.

## 🗂️ Repository layout

```
impassebench/
├── harness/     conditions, fault injection, containers, prompts, model backends, rollout loop, runner
├── annotate/    rule-based annotator (C, Q, t*, N_post, S, L, U, Y, Z) and patch grading (A)
├── monitor/     evidence monitor, ticket template, bounded retry, restoration (paper Section 4)
├── data.py      loaders for the released dataset and for run directories
└── resources/   the 102 tasks and per-repository fault parameters
docs/            conditions · running · metrics · data format · monitor
tests/           unit tests, no Docker or network needed (pytest -q)
```

## 📚 Documentation

| | |
|---|---|
| [docs/conditions.md](docs/conditions.md) | how each of the 12 faults and the clean control is built |
| [docs/running.md](docs/running.md) | running agents, providers, output format, Python API |
| [docs/metrics.md](docs/metrics.md) | label definitions and domains |
| [docs/data.md](docs/data.md) | the released dataset and its record format |
| [docs/monitor.md](docs/monitor.md) | the evidence monitor and the SWE-bench Verified study |

## ♻️ Reproducibility

- The annotator reproduces the labels of all 19,371 released trajectories exactly.
- The two system prompts match the recorded runs byte for byte.
- The injected fault configuration matches 19,272 runs exactly. The other 99 are `dependency_missing` runs that used three fewer read-only setup checks; the environment they built is the same.
- The monitor reproduces every recorded reminder (2,938 runs).
- The paper's tables can be recomputed from `labels.parquet`.

Model outputs vary between runs and provider versions, so new runs reproduce
the protocol, not identical trajectories.

## 📝 Citation

```bibtex
@article{chen2026knowing,
  title   = {Knowing When to Stop: The Evidence--Action Gap in Coding Agents},
  author  = {Chen*, Tianyu and Zhao*, Xinran and Viswanathan, Vijay and Yu, Shoubin and Zhou, Mingyuan and Wu, Tongshuang},
  journal = {arXiv preprint},
  year    = {2026}
}
```

## ⚖️ License

MIT, see [LICENSE](LICENSE). The tasks are derived from
[SWE-smith](https://github.com/SWE-bench/SWE-smith) and
[SWE-bench Verified](https://www.swebench.com/) (MIT).
