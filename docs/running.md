# Running the benchmark

## Requirements

* A Linux x86-64 host with Docker (images are `linux/amd64`). Budget about
  2 CPU cores and 2-4 GB RAM per concurrent trajectory, and ~40 GB of disk for
  the ten repository images plus one prepared image per task.
* Python 3.10+, `pip install -e .` plus the SDK of your model provider
  (`google-genai`, `openai`, `anthropic` or `boto3`).
* API credentials in the environment or in a `.env` file in the working
  directory: `GEMINI_API_KEY`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`,
  `AWS_BEARER_TOKEN_BEDROCK` (or standard AWS credentials) for Bedrock, or
  `IMPASSEBENCH_OPENAI_BASE_URL` (+ optional `IMPASSEBENCH_OPENAI_API_KEY`)
  for a self-hosted OpenAI-compatible server.

## Steps

```bash
impassebench pull-images                       # 10 SWE-smith images, retried with backoff

# all 1,211 (task, condition) pairs, with and without the escalation tool
impassebench run --model gemini-3.5-flash --interface base     --parallel 8
impassebench run --model gemini-3.5-flash --interface escalate --parallel 8

# a subset
impassebench run --model openai-compatible/Qwen3-Coder-30B-A3B-Instruct \
    --interface escalate --conditions readonly_fs,sandbox_flaky --tasks <id>,<id>

impassebench run ... --dry-run                 # list what would run
```

Model routing: `gemini-*` (Google), `gpt-*` / `o*` (OpenAI), `claude-*`
(Anthropic), Bedrock ids such as `us.anthropic.claude-haiku-4-5-20251001-v1:0`,
`us.openai.gpt-5.6-terra`, `qwen.qwen3-coder-30b-a3b-v1:0` or
`moonshotai.kimi-k2.5`, and `openai-compatible/<served-name>`. Sampling is
temperature 0.7 (where a provider rejects it, its default is used and recorded
in `meta.json`), at most 4,096 output tokens per call, 80 steps.

## Output

Each trajectory is written to `runs/<model>/<interface>/<run_id>/`:

* `trace.jsonl`: one record per step with `type` (`bash`, `edit`,
  `task_complete`, `escalate_to_human`, `text`, `invalid_tool`), `action`,
  `observation`, `exit_code`, `assistant_text`, `step`, and `fault_injected`
  on calls that the runtime fault answered. `edit` records also carry
  `tool_args`, the full edit arguments (`action` holds them as JSON cut to
  2,000 characters, as in the released dataset).
* `meta.json`: task, condition, interface, model and sampling, the fault spec
  (`docker_run_args`, `setup_cmds`, `runtime_fault_spec`, `agent_env`,
  `agent_user`), `termination`, `n_steps`, `total_tokens`, the extracted
  `model_patch`, and `runtime_fault` (when a runtime fault armed and first
  fired).
* `diff.patch`: the extracted patch, if non-empty.

`termination` is one of `task_complete`, `escalate`, `max_steps`,
`text_stall` (three replies in a row without a tool call) or
`container_error` (harness failure). Re-running the same command skips pairs
that already have a finished trajectory and retries the rest. Trajectories
can then be scored with `impassebench annotate` (see `docs/metrics.md`).

## Using the loop directly

```python
from impassebench.harness import run_rollout, load_tasks
from impassebench.harness.agents import make_agent

task = load_tasks()[0]
agent = make_agent("gemini-3.5-flash", "escalate")
meta = run_rollout(task, "readonly_fs", agent, interface="escalate",
                   max_steps=80, out_dir="runs/custom")
```

`run_rollout(..., hooks=...)` accepts a `RolloutHooks` subclass to add
behavior to the loop without changing it (the evidence monitor in
`impassebench.monitor` is built this way); with no hooks the loop is the
benchmark harness.
