# Configuration

A run is described by a single YAML (or JSON) file with four sections:
`model`, `data`, `optimizer`, `training`. Generate a commented starting
point with:

```bash
gpt-simple init -o config.yaml            # defaults
gpt-simple init --preset small -o config.yaml
```

This page documents the **meaning and valid values** of each field.
Default values are intentionally not duplicated here — they live in
`src/gpt_simple/config.py` and the `gpt-simple init` template, so they
can never drift out of sync with the code. CLI flags of the form
`--section.field value` override any file value at launch.

Any field omitted from the file falls back to its default. Unknown
top-level keys are ignored with a warning. Validation runs at load time
and raises `ConfigError` on invalid combinations.

## `model`

Defines the architecture — and therefore the checkpoint format. See
[Architecture](architecture.md) for what each knob does to the model.

| Field | Type / values | Notes |
| ----- | ------------- | ----- |
| `vocab_size` | int or null | Null infers from the tokenizer and pads to a multiple of 128. |
| `n_embd` | int | Hidden size. Must be divisible by `n_head`. |
| `n_layer` | int | Number of transformer blocks. |
| `n_head` | int | Number of attention (query) heads. |
| `n_kv_head` | int or null | Null = `n_head` (MHA). `<n_head` = grouped-query (must divide `n_head`); `1` = multi-query. |
| `n_positions` | int | Maximum sequence length the model is built for. |
| `dropout` | float | Dropout probability (0.0 disables). |
| `use_bias` | bool | Global default for linear-layer biases. |
| `qkv_bias` / `attn_out_bias` / `mlp_bias` | bool or null | Per-group bias override; null inherits `use_bias`. |
| `activation` | `swish` \| `gelu` \| `relu` | Activation; with `mlp_type: gated` this selects SwiGLU/GeGLU/ReGLU. |
| `mlp_type` | `gated` \| `mlp` | Gated MLP or vanilla FFN. |
| `intermediate_size` | int or null | Explicit FFN inner width; null derives it from `mlp_type`. |
| `norm` | `rmsnorm` \| `layernorm` | Normalization layer. |
| `norm_eps` | float | Normalization epsilon. |
| `tie_word_embeddings` | bool | Tie the output head to the token embedding. |
| `rope_base` | float | RoPE base frequency (theta). |
| `rope_scaling_type` | null \| `linear` \| `ntk` | Optional context-length scaling. |
| `rope_scaling_factor` | float | Scaling factor (≥ 1.0) for the chosen RoPE scaling. |
| `attention_mode` | `causal` \| `sdpa_mask` \| `flex` | Attention backend (see [Architecture](architecture.md)). |

The architecture-defining subset of these fields is hashed into each
checkpoint; resuming with a changed value is rejected to prevent loading
weights into an incompatible model.

## `data`

What to train on and how to load it. See [Data pipeline](data.md).

| Field | Type / values | Notes |
| ----- | ------------- | ----- |
| `path` | str | Root data directory (required). For pretokenized data, the parent of `{train,val}/<bucket>/`. |
| `tokenizer` | str | HuggingFace name (e.g. `gpt2`) or a local tokenizer directory. |
| `format` | `pretokenized` \| `jsonl` | Memory-mapped `.bin/.idx` shards, or raw JSONL tokenized on the fly. |
| `max_length` | int | Training sequence length. |
| `overlap_size` | int | Overlap (in tokens) between windows of long documents. Must be ≤ `max_length / 2`. |
| `packing` | bool | Pack multiple short documents into each sequence. |
| `num_workers` | int | DataLoader worker processes per rank. |
| `curriculum` | list or null | Phased bucket mixing; null = uniform sampling. Pretokenized only. |
| `allow_bucket_exhaustion` | bool | Permit a curriculum bucket to run out (mix renormalizes) instead of erroring. |
| `allow_budget_mismatch` | bool | Permit the curriculum token budget to differ from what training consumes. |

### `data.curriculum`

A list of phases, each trained in order for a token budget:

```yaml
data:
  curriculum:
    - duration_tokens: 9_000_000_000
      mix: {web: 0.50, wiki: 0.25, code: 0.15, math: 0.05, qa: 0.05}
    - duration_tokens: 6_000_000_000
      mix: {web: 0.30, code: 0.35, wiki: 0.10, math: 0.15, qa: 0.10}
```

- `duration_tokens` — tokens to train in this phase (must be positive).
- `mix` — relative weights per bucket name; normalized to sum to 1.0.
  Bucket names must match subdirectory names under `path`.

## `optimizer`

AdamW with a cosine learning-rate schedule and linear warmup.

| Field | Type | Notes |
| ----- | ---- | ----- |
| `learning_rate` | float | Peak LR (after warmup). |
| `weight_decay` | float | AdamW weight decay. |
| `beta1` / `beta2` | float | AdamW betas. |
| `eps` | float | AdamW epsilon. |
| `max_grad_norm` | float | Gradient-norm clipping threshold. |
| `warmup_steps` | int | Linear warmup length. Must be `< max_steps`. |
| `decay_steps` | int or null | Cosine decay length; null = `max_steps - warmup_steps` (decay finishes exactly at the end of training). |
| `min_lr_ratio` | float | Floor LR as a fraction of peak. |

Set `decay_steps` explicitly to decouple the schedule from the run
length — for example, to leave a min-LR cooldown tail at the end.

## `training`

How the run executes — batch sizes, logging, output, resume, runtime.

| Field | Type / values | Notes |
| ----- | ------------- | ----- |
| `per_device_batch_size` | int | Sequences per optimizer micro-step, per GPU. |
| `gradient_accumulation_steps` | int | Micro-steps accumulated per optimizer step. |
| `max_steps` | int | Total optimizer steps (the unit for all `*_steps` fields). |
| `gradient_checkpointing` | bool | Recompute activations in the backward pass to save memory. |
| `compile` | bool | Enable `torch.compile`. |
| `seed` | int | Global RNG seed. |
| `logging_steps` | int | Metric logging interval. |
| `eval_steps` | int | Validation interval. |
| `save_steps` | int | Checkpoint interval. |
| `max_eval_batches` | int or null | Cap validation batches; null = full validation set. |
| `output_dir` | str | Run directory (checkpoints, tokenizer, logs, state). |
| `resume` | `auto` \| `scratch` \| `<path>` | Resume policy (see [Checkpointing & resume](checkpointing-and-resume.md)). |
| `keep_last_k` | int or null | Keep only the latest K checkpoints; null = keep all. |
| `keep_milestone_every` | int or null | Never delete checkpoints whose step is a multiple of this. |
| `max_runtime_seconds` | int or null | Wall-clock budget; null = auto-detect (`SLURM_JOB_END_TIME` / `GPT_SIMPLE_MAX_RUNTIME`) or disabled. |
| `walltime_reserve_seconds` | int | Safety margin before the deadline, reserved to save a final checkpoint. |
| `mixed_precision` | null \| `bf16` \| `fp16` \| `no` | Null auto-detects per device (see [Hardware tuning](hardware-tuning.md)). |
| `wandb_project` | str or null | Null disables W&B. |
| `wandb_run_name` | str or null | Null auto-generates a name. |

### Cross-section validation

Some constraints span sections and are checked at load time:

- `optimizer.warmup_steps` must be `< training.max_steps`.
- `data.max_length > model.n_positions` warns (sequences would exceed the
  model's positional range).
- If the LR schedule length (`warmup_steps + decay_steps`) differs from
  `max_steps`, a warning explains the consequence (LR holds at min, or
  decay never completes).

---

Authoritative source: `src/gpt_simple/config.py`.
