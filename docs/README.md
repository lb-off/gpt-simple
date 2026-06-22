# GPT-Simple documentation

These guides go deeper than the [project README](https://github.com/lb-off/gpt-simple/blob/main/README.md), which
covers installation and a quick start. Each page owns a single concern.

## Guides

| Guide | What it covers |
| ----- | -------------- |
| [Architecture](architecture.md) | The built-in model: decoder block, RoPE, normalization, MLP variants, attention backends, weight tying, KV-cache. |
| [Configuration](configuration.md) | Every config field — meaning and valid values — for the `model` / `data` / `optimizer` / `training` sections. |
| [Data pipeline](data.md) | Tokenization, the `.bin/.idx` format, pretokenized vs JSONL, sequence packing, document windowing, curriculum buckets. |
| [Training](training.md) | Running single- and multi-GPU training, mixed precision, `torch.compile`, gradient checkpointing, W&B. |
| [Checkpointing & resume](checkpointing-and-resume.md) | On-disk layout, the deterministic stop/resume model, walltime budgets, signals, topology-agnostic data cursors. |
| [Orchestration](orchestration.md) | Running long, chained jobs under any orchestrator (SLURM, Kubernetes, a local loop). |
| [Inference](inference.md) | `generate` and `batch-generate`, sampling parameters, the batch JSONL schema, dry-run validation. |
| [Hardware tuning](hardware-tuning.md) | Getting peak throughput from your GPUs: precision, attention backend selection, batch size, dataloader workers. |
| [Performance](performance.md) | Measured throughput on a real 2.8B run, MFU/HFU methodology, how to measure FLOPs, and how it compares to other libraries. |

## Source of truth

These guides describe behavior and intent. When a default value or an
exact field list matters, the authoritative source is always the code:

- Config fields and defaults: `src/gpt_simple/config.py` (or run
  `gpt-simple init` for a commented template).
- Model architecture: `src/gpt_simple/model.py`.
- Public Python API: `src/gpt_simple/__init__.py`.
