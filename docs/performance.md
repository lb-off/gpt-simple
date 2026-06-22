# Performance

This page records the throughput GPT-Simple actually reached on a real
pretraining run, and — more importantly — *how that number was derived*,
so it can be reproduced and judged honestly. The headline metric is
**MFU (Model FLOPs Utilization)**, not raw tokens/second: tokens/second
is meaningless without the model size and the GPU.

## TL;DR

- A 2.8B-parameter model trained on **8× A100 80GB** (single node, DDP)
  at **~65,000 tokens/second**.
- That is **~44% MFU** (≈47% if attention FLOPs are counted), and
  **~59% HFU** including the gradient-checkpointing recompute.
- For plain DDP + gradient checkpointing in a from-scratch library, this
  sits in professional-library territory: PaLM reported 46% MFU,
  Megatron-LM lands ~50–55% with tensor/pipeline parallelism.

## The run

The configuration is [`examples/configs/pretrain_2.8b_15b.yaml`](https://github.com/lb-off/gpt-simple/blob/main/examples/configs/pretrain_2.8b_15b.yaml).

| Property | Value |
| --- | --- |
| Parameters | ~2.8B (`n_embd=2560`, `n_layer=34`, gated SwiGLU, tied head) |
| Hardware | 8× A100 80GB, single node |
| Parallelism | DDP (data parallel only — no TP/PP/FSDP) |
| Precision | bf16 mixed precision |
| Sequence length | 2048 |
| Global batch | 4 × 8 grad-accum × 8 GPUs × 2048 = 524,288 tok/step |
| Memory features | `gradient_checkpointing: true`, `compile: true` |
| Measured throughput | **~65,000 tok/s** (≈8,100 tok/s/GPU) |

## How the numbers are computed

### FLOPs per token

The standard analytical estimate counts matmul work: each parameter does
one multiply + one add per token (2 FLOPs).

- **Forward** ≈ `2N`
- **Backward** ≈ `4N` (gradients w.r.t. both activations and weights)
- **Total** ≈ `6N`

For `N ≈ 2.8B`, that is `6 × 2.8e9 ≈ 1.68e10` FLOPs/token.

`6N` deliberately ignores attention score computation (`QK^T`,
softmax·`V`), which is not parameter-bound — it scales with `seq_len`.
The fuller Kaplan/Chinchilla form adds it back:

```
FLOPs/token ≈ 6N + 6 · n_layer · seq_len · d_model
            = 6N + 6 · 34 · 2048 · 2560  ≈  6N × 1.06
```

So attention is ~6% here; `6N` is a known-conservative undercount.

### MFU and HFU

```
Model FLOP/s = 65,000 tok/s × 1.68e10  = 1.10 PFLOP/s
Per GPU      = 1.10e15 / 8             = 138 TFLOP/s
A100 bf16 peak (dense tensor cores)    = 312 TFLOP/s
```

| Metric | Value | Definition |
| --- | --- | --- |
| **MFU** | **~44%** | useful `6N` work ÷ peak (138 / 312) |
| MFU (with attention term) | ~47% | `6N × 1.06` ÷ peak |
| **HFU** | **~59%** | hardware FLOPs ÷ peak — `8N`, because gradient checkpointing recomputes the forward (+`2N`) |

The gap between MFU (44%) and HFU (59%) is *entirely* the recompute:
gradient checkpointing trades ~25% extra FLOPs for the activation memory
needed to fit the model. See [Levers](#levers) below.

## How to measure it yourself

MFU as reported above — and in published papers — is a **hybrid**:
analytical FLOPs/token × *measured* tokens/second. Nobody puts
hardware-counter FLOPs in the numerator, because `6N` is the agreed
definition of "useful work" (it excludes recompute and padding by
construction) and keeps numbers comparable across projects.

```
MFU = (analytical FLOPs/token × measured tok/s) / (num_GPUs × peak FLOP/s)
```

Three tiers of fidelity, in increasing cost:

| Tier | Tool | Gives |
| --- | --- | --- |
| Analytical | `6N` (or the fuller formula) | FLOPs/token from shapes — for reporting MFU |
| Op-level count | `torch.utils.flop_counter.FlopCounterMode`, PyTorch Profiler `with_flops=True` | exact FLOPs per op, incl. SDPA/attention — to verify the formula |
| Hardware counters | NVIDIA DCGM (`DCGM_FI_PROF_PIPE_TENSOR_ACTIVE`), Nsight Compute | what the silicon actually executed — for kernel-level debugging |

To get an exact (attention-inclusive) count for one step:

```python
from torch.utils.flop_counter import FlopCounterMode

counter = FlopCounterMode(display=False)
with counter:
    loss = model(batch).loss
    loss.backward()
total_flops = counter.get_total_flops()          # fwd + bwd, real shapes
mfu = total_flops / step_time / (8 * 312e12)      # 8× A100
```

It will come out ~6% above `6N` from the attention term.

> **GPU utilization is not MFU.** The `nvidia-smi` / wandb "GPU util"
> figure (≈86% on this run) only measures *whether a kernel was running*,
> not how efficiently the tensor cores were fed. 86% util with 44% MFU is
> fully consistent — they are different axes. High util does tell us the
> GPU is compute-bound (only ~14% idle bubble), which means the
> checkpointing recompute is real, billed GPU time rather than free slack.

## How this compares

For a 2.8B-class model on A100-generation hardware:

| System | MFU | Notes |
| --- | --- | --- |
| **GPT-Simple (this run)** | **~44%** | plain DDP + gradient checkpointing |
| PaLM (Google) | 46% | the paper that coined MFU; considered excellent |
| Megatron-LM | 50–55% | full tensor + pipeline parallelism |
| MosaicML LLM Foundry / MPT | 50–55% | heavily tuned A100 stack |
| nanoGPT | 35–45% | comparable single-node scope |
| "Typical" large-scale runs | 30–50% | GPT-3 era reported range |

Reaching ~44% with data parallelism alone — no tensor, pipeline, or
sharded-optimizer parallelism — is a credible result. The remaining gap
to Megatron is mostly what those frameworks buy with parallelism
strategies GPT-Simple intentionally does not implement.

## Levers

In rough order of payoff, to push the realized MFU higher:

1. **Selective gradient checkpointing.** Today checkpointing is
   all-or-nothing (a single flag applied to every layer — see
   `src/gpt_simple/model.py`). On 80GB you likely do not need to
   checkpoint all 34 layers. Checkpointing only the first *k* converts
   recompute FLOPs into throughput, moving MFU from ~44% toward the ~59%
   HFU ceiling. The realistic landing spot is ~50–55%, since fitting
   without full checkpointing may force a smaller batch (smaller GEMMs
   are less efficient).
2. **`python_reducer` for DDP + compile** — better overlap of the
   gradient all-reduce with the backward pass (see the compile + DDP
   notes in [Training](training.md)).
3. **Fused AdamW** (`fused=True`) — a cheap memory-bound-step win.
4. **Faster GPUs.** On H100 the same code is ~2.5–3× faster in absolute
   throughput; see [Hardware tuning](hardware-tuning.md) for the
   generation factors and fp8 notes.

## Source of truth

These figures describe one specific run. The authoritative inputs are the
code and config:

- Run configuration: [`examples/configs/pretrain_2.8b_15b.yaml`](https://github.com/lb-off/gpt-simple/blob/main/examples/configs/pretrain_2.8b_15b.yaml).
- Model architecture and parameter count: `src/gpt_simple/model.py`.
- A100 bf16 peak (312 TFLOPS dense) is NVIDIA's published spec; substitute
  your GPU's peak when computing MFU on other hardware
  ([Hardware tuning](hardware-tuning.md) has the table).
