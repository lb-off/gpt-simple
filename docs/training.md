# Training

Training is driven by `gpt-simple train` (or `gpt_simple.train(...)` from
Python) and configured entirely through the `training`, `optimizer`, and
`data` config sections. Multi-GPU is handled with
[Accelerate](https://github.com/huggingface/accelerate).

## Running

Single GPU:

```bash
gpt-simple train --config config.yaml
```

Multi-GPU (launches `torchrun` automatically):

```bash
gpt-simple train --config config.yaml --nproc_per_node 4
```

Override config values at launch, and start fresh if needed:

```bash
gpt-simple train --config config.yaml \
  --training.max_steps 5000 \
  --optimizer.learning_rate 1e-4 \
  --force                       # discard existing checkpoints in output_dir
```

From Python:

```python
import gpt_simple

result = gpt_simple.train(config="config.yaml")
print(result.final_loss, result.total_tokens, result.checkpoint_path)
```

## Effective batch size

The number of tokens per optimizer step is:

```
per_device_batch_size × gradient_accumulation_steps × world_size × max_length
```

Per-rank batch size is independent of `world_size`: adding GPUs scales the
global batch rather than shrinking each GPU's work. The learning-rate
schedule advances once per optimizer step regardless of accumulation or
GPU count.

## Mixed precision

`training.mixed_precision` selects the compute precision. Left as `null`
(the default) it auto-detects per device: `bf16` on Ampere and newer,
`fp16` on older CUDA GPUs, and no mixed precision on CPU. Prefer `bf16`
where available — it has the dynamic range of fp32 and needs no loss
scaling. See [Hardware tuning](hardware-tuning.md).

## torch.compile

`training.compile: true` (default) wraps the model with `torch.compile`
for a meaningful throughput gain. Compilation happens once, on the first
step (so step 1 is slow). The attention call is treated as an opaque
op and is not decomposed by the compiler.

### compile + DDP + gradient checkpointing

When **all three** of multi-GPU (DDP), `compile: true`, and
`gradient_checkpointing: true` are on, the trainer disables Dynamo's DDP
graph-splitter (`torch._dynamo.config.optimize_ddp = False`) before
compiling. The graph-splitter does not support the higher-order ops that
gradient checkpointing introduces, and recent PyTorch hard-errors on the
combination ([pytorch/pytorch#104674](https://github.com/pytorch/pytorch/issues/104674)).

`torch.compile` itself stays fully enabled — only the graph splitting is
turned off, so the module compiles as a single graph. The only cost is
slightly less communication/compute overlap (the whole graph is one
gradient bucket), which is minor on a single NVLink node and more
noticeable across multiple nodes. This setting is a no-op without DDP.

> A compile-compatible bucketed reducer
> (`torch._dynamo.config.optimize_ddp = "python_reducer"`) can recover
> most of that overlap on newer PyTorch. It is less battle-tested than the
> safe default; validate on a short multi-GPU run before adopting it for a
> long job.

## Gradient checkpointing

`training.gradient_checkpointing: true` (default) recomputes each block's
activations during the backward pass instead of storing them, trading
compute for memory. Turn it off if you have memory headroom and want
maximum throughput.

## Logging and evaluation

- `logging_steps` — interval for loss, learning rate, gradient norm, and
  throughput.
- `eval_steps` — interval for validation over the `val/` data
  (`max_eval_batches` caps the work).
- `save_steps` — checkpoint interval; see
  [Checkpointing & resume](checkpointing-and-resume.md).

### Weights & Biases

Set `training.wandb_project` to enable W&B (install the extra:
`pip install ".[wandb]"`, then `wandb login`). The run id is persisted in
the checkpoint, so a stop/resume chain reports as one continuous run.
Leave `wandb_project` unset to disable logging entirely.

## Validation before training

`gpt-simple validate --config config.yaml` checks a config (and,
optionally, runtime/memory feasibility) without starting a run — useful
as a submission gate. The trainer also runs validation automatically at
startup.

---

Authoritative source: `src/gpt_simple/train.py`.
