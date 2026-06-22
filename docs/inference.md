# Inference

Two CLI commands run trained checkpoints:

- `generate` — one model, one set of sampling parameters, many prompts.
- `batch-generate` — a self-describing JSONL where each record can name
  its own model and sampling parameters (eval sweeps, multi-checkpoint
  comparisons).

Both also exist as Python functions in `gpt_simple` (`generate`,
`load_for_inference`, `validate_checkpoint`).

## Pointing at a checkpoint

Anywhere a model location is accepted, you can pass either:

- a **run directory** (`output_dir`) — the latest checkpoint under
  `checkpoints/` is selected automatically (same rule as `resume: auto`);
  or
- a **specific checkpoint** directory (`.../checkpoint-N`).

The tokenizer is found automatically next to the checkpoint or at the run
root; override it with `--tokenizer`. Paths may contain `~` and
environment variables (e.g. `$WORK`), which are expanded even inside a
JSONL.

## `generate`

```bash
gpt-simple generate \
  --output-dir ./outputs \
  --prompt "Once upon a time" \
  --max-new-tokens 200
```

Output is JSONL on stdout by default (pipe to `jq -r .completion` for just
the text); `--output FILE.jsonl` writes to a file. Feed many prompts with
`--prompts-file prompts.jsonl` (one `{"prompt": ...}` per line). Each
output record echoes the input and adds `completion` plus a `generation`
block recording the sampling parameters used.

## `batch-generate`

Use this when records need **different** models or sampling. Each line is
self-describing; only `prompt` is required, everything else falls back to
the CLI defaults:

```jsonl
{"id": "ex-1", "prompt": "Once upon a time", "model": {"checkpoint": "./runs/r1/checkpoints/checkpoint-12000", "dtype": "bf16"}, "generation": {"max_new_tokens": 200, "temperature": 0.8, "top_k": 50}}
{"id": "ex-2", "prompt": "def fibonacci(n):", "model": {"output_dir": "./runs/code-model"}, "generation": {"greedy": true}}
```

```bash
gpt-simple batch-generate \
  --input setup.jsonl \
  --output completions.jsonl \
  --output-dir ./runs/r1 \
  --temperature 0.8 --top-k 50
```

Records that share the same `(model, dtype, tokenizer)` are grouped so
each model is **loaded once** (a multi-billion-parameter checkpoint costs
minutes to load), and only one model is held in memory at a time. Output
order always matches input order.

Per-record `model` accepts `checkpoint` or `output_dir`, plus optional
`dtype` and `tokenizer`. The `generation` block accepts the sampling
parameters below.

## Sampling parameters

| Parameter | Meaning |
| --------- | ------- |
| `max_new_tokens` | Number of tokens to generate. |
| `temperature` | Softmax temperature; lower is more deterministic. |
| `top_k` | Keep only the top-k logits before sampling. |
| `top_p` | Nucleus sampling: keep the smallest set of tokens with cumulative probability ≥ p. |
| `greedy` / `do_sample` | Greedy (argmax) decoding instead of sampling. |
| `repetition_penalty` | Penalize already-generated tokens (1.0 = off). |
| `seed` | Seed for reproducible sampling. |
| `dtype` | Load/compute dtype: `fp32`, `fp16`/`half`, `bf16`. |
| `return_full_text` | Include the prompt in the output (default: completion only). |

## Pre-flight validation (`--dry-run`)

Before any weights load, `batch-generate` parses every record, checks the
sampling-parameter ranges, and confirms each distinct checkpoint resolves
(config parses, weights and tokenizer present). `--dry-run` stops there
and prints the execution plan without touching a GPU — safe to run on a
login node to gate a job submission:

```bash
gpt-simple batch-generate --input setup.jsonl --output-dir ./runs/r1 --dry-run
```

Structural problems (bad JSON, missing `prompt`, out-of-range parameters,
a missing checkpoint) are **hard errors**: the job exits non-zero with
nothing loaded. Failures that only surface *during* generation are
**soft**: that record gets an `error` field instead of `completion` and
the run continues.

The same checks are available programmatically via
`gpt_simple.validate_checkpoint(path, ...)`, which returns the resolved
locations without loading weights or touching CUDA.

## Scaling out

`batch-generate` is a plain CLI entry point with no orchestrator
assumptions. To fan out across a job array, shard the input **by model**
(one model per task) so no checkpoint is loaded twice. See
[`examples/orchestrators/`](../examples/orchestrators/) for inference job
templates.

---

Authoritative source: `src/gpt_simple/generate.py`,
`src/gpt_simple/cli/batch_generate_cmd.py`.
