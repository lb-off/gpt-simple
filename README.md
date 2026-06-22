# GPT-Simple

[![CI](https://github.com/lb-off/gpt-simple/actions/workflows/ci.yml/badge.svg)](https://github.com/lb-off/gpt-simple/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A clean, efficient framework for pretraining language models from scratch.

GPT-Simple handles the full LLM pretraining workflow — tokenization,
streaming data loading, multi-GPU training, checkpointing, and inference —
through a single YAML config and a small CLI. It ships with a modern GPT
architecture ready to train out of the box.

## Features

- **Single YAML config + CLI** — `init` / `tokenize` / `train` / `status`
  / `stop` / `validate` / `generate` / `batch-generate`.
- **Multi-GPU out of the box** — `--nproc_per_node N` launches `torchrun`
  automatically (Accelerate, bf16, `torch.compile`, gradient
  checkpointing).
- **Pretokenized streaming** — memory-mapped `.bin/.idx` shards with
  sequence packing; a raw-JSONL fallback for quick experiments.
- **Deterministic stop/resume** — walltime- and signal-aware checkpoints
  with topology-agnostic data cursors, so *N* short jobs equal one long
  job (every document seen exactly once, even if `world_size` /
  `num_workers` change between restarts).
- **Orchestrator-friendly** — runs under SLURM, Kubernetes, or a local
  loop; templates in [`examples/orchestrators/`](examples/orchestrators/).
- **Curriculum learning** — phase-based mixing across named data buckets.
- **Modern architecture** — pre-norm decoder with RoPE, RMSNorm, and a
  gated (SwiGLU) MLP; also expresses GQA/MQA, vanilla MLPs, and untied
  heads via config.
- **Python API** — `import gpt_simple; gpt_simple.train(config="config.yaml")`.

## Installation

```bash
pip install -e ".[dev]"     # from source (development)
pip install .               # core only
pip install ".[wandb]"      # optional: Weights & Biases logging
pip install ".[cli]"        # optional: rich-formatted CLI output
```

## Quick start

### 1. Generate a config

```bash
gpt-simple init -o config.yaml
gpt-simple init --preset small -o config.yaml    # ~125M  (small | medium | large)
```

### 2. Pretokenize your data

```bash
gpt-simple tokenize \
  --input_dir ./raw_data \
  --output_dir ./data/tokenized \
  --tokenizer_path gpt2 \
  --max_length 2048 \
  --num_workers 8
```

Converts `.jsonl`/`.txt` into memory-mapped `.bin/.idx` shards. See the
[data pipeline guide](docs/data.md).

### 3. Train

```bash
gpt-simple train --config config.yaml                     # single GPU
gpt-simple train --config config.yaml --nproc_per_node 4  # 4 GPUs

# override any config value; start fresh with --force
gpt-simple train --config config.yaml --training.max_steps 5000 --force
```

See the [training guide](docs/training.md).

### 4. Monitor and control

```bash
gpt-simple status                 # training progress
gpt-simple stop                   # graceful shutdown (saves a checkpoint)
gpt-simple stop --force           # immediate SIGKILL
```

### 5. Generate

```bash
gpt-simple generate --output-dir ./outputs --prompt "Once upon a time" --max-new-tokens 200
```

`--output-dir` auto-picks the latest checkpoint. For multi-model /
multi-sampling batches and a `--dry-run` submission gate, use
`batch-generate` — see the [inference guide](docs/inference.md).

## Long runs with stop/resume

The trainer targets clusters with a hard per-job wall-clock cap. With
`resume: auto` (the default), re-running the same command resumes the
latest checkpoint, and the trainer saves and exits cleanly before a
walltime deadline or on `SIGTERM`/`SIGUSR1` — so an orchestrator just
re-queues the job.

```bash
gpt-simple train --config config.yaml   # resume is automatic on every restart
gpt-simple status
gpt-simple stop                          # or let walltime/SIGUSR1 do it
```

Templates: [`slurm_resume_chain.sh`](examples/orchestrators/slurm_resume_chain.sh),
[`kubernetes_job.yaml`](examples/orchestrators/kubernetes_job.yaml),
[`local_loop.sh`](examples/orchestrators/local_loop.sh). See the
[checkpointing & resume](docs/checkpointing-and-resume.md) and
[orchestration](docs/orchestration.md) guides.

## Configuration

All settings live in one YAML file with four sections — `model`, `data`,
`optimizer`, `training`:

```yaml
model:
  n_embd: 768
  n_layer: 12
  n_head: 12
  n_positions: 2048

data:
  path: ./data/tokenized
  tokenizer: gpt2
  format: pretokenized       # pretokenized | jsonl
  max_length: 2048

optimizer:
  learning_rate: 3.0e-4
  warmup_steps: 100

training:
  per_device_batch_size: 4
  gradient_accumulation_steps: 4
  max_steps: 1000
  output_dir: ./outputs
  # wandb_project: my-project   # uncomment to enable W&B
```

`gpt-simple init` writes a fully commented template. Every field is
documented in the [configuration reference](docs/configuration.md), and
curriculum learning in the [data pipeline guide](docs/data.md).

## Python API

```python
import gpt_simple

result = gpt_simple.train(
    model=gpt_simple.ModelConfig(n_embd=768, n_layer=12, n_head=12),
    data=gpt_simple.DataConfig(path="./data/tokenized", tokenizer="gpt2"),
    optimizer=gpt_simple.OptimizerConfig(learning_rate=3e-4),
    training=gpt_simple.TrainingConfig(max_steps=1000, output_dir="./outputs"),
)
print(result.final_loss, result.total_tokens, result.checkpoint_path)
```

Or `gpt_simple.train(config="config.yaml")`; sub-configs passed
explicitly override the matching section from the file.

## Documentation

Full guides live in [`docs/`](docs/README.md):

- [Architecture](docs/architecture.md) — the built-in model.
- [Configuration](docs/configuration.md) — every config field.
- [Data pipeline](docs/data.md) — tokenization, packing, curriculum.
- [Training](docs/training.md) — multi-GPU, precision, compile.
- [Checkpointing & resume](docs/checkpointing-and-resume.md) — the
  stop/resume model.
- [Orchestration](docs/orchestration.md) — running under any scheduler.
- [Inference](docs/inference.md) — `generate` / `batch-generate`.
- [Hardware tuning](docs/hardware-tuning.md) — peak GPU throughput.
- [Performance](docs/performance.md) — measured 2.8B throughput and MFU.

## Development

```bash
pip install -e ".[dev]"
pytest tests/
ruff check src/ tests/
```

## License

MIT — see [LICENSE](LICENSE).
