# GPT-Simple

A clean, readable framework for pretraining language models from scratch.
GPT-Simple handles the full LLM pretraining workflow — tokenization, streaming
data loading, multi-GPU training, deterministic stop/resume, and inference —
through a single YAML config and a small CLI.

## Install

```bash
pip install gpt-simple-lm
```

The distribution is named `gpt-simple-lm`; you import it as `gpt_simple` and run
the `gpt-simple` CLI. For source installs and optional extras, see the
[project README](https://github.com/lb-off/gpt-simple/blob/main/README.md).

## Quick start

```bash
gpt-simple init -o config.yaml                    # write a commented config template
gpt-simple tokenize --input_dir ./raw_data \
  --output_dir ./data/tokenized --tokenizer_path gpt2
gpt-simple train --config config.yaml             # add --nproc_per_node N for multi-GPU
gpt-simple generate --output-dir ./outputs --prompt "Once upon a time"
```

New here? Start with the [Training guide](training.md); every config field is
documented in the [Configuration reference](configuration.md).

## Guides

These pages go deeper than the quick start above — each owns a single concern.

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
