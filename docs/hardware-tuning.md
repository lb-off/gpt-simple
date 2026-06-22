# Hardware tuning

This guide is about getting peak throughput from whatever GPUs you have.
It focuses on NVIDIA data-center GPUs (V100 / A100 / H100), which are the
common targets for LLM pretraining, but the config advice generalizes.

## TL;DR

- GPT-Simple already picks a good attention kernel and precision via
  PyTorch SDPA + Accelerate. On modern hardware the main wins are *free*
  given a recent PyTorch and a sensible config.
- Keep `mixed_precision: bf16` (or `null` for auto), `attention_mode:
  causal`, and `compile: true`, then push `per_device_batch_size` up.
- fp8 (Hopper) is a further ~1.5–2× but is **not** free — it needs extra
  integration that GPT-Simple does not ship today.

## The hardware hierarchy

For LLM pretraining (bf16/fp16, seq_len ≈ 2k–8k, matmul-dominated):

| GPU | Arch | bf16 tensor-core throughput | HBM bandwidth | Memory | Practical training factor |
| --- | ---- | --------------------------- | ------------- | ------ | ------------------------- |
| V100 (32GB) | Volta (sm_70) | ~125 TFLOPS (fp16) | 0.9 TB/s | 32 GB | ~0.35–0.5× A100 |
| A100 (80GB) | Ampere (sm_80) | ~312 TFLOPS | 2.0 TB/s | 80 GB | **1.0× (baseline)** |
| H100 (80GB) | Hopper (sm_90) | ~756 TFLOPS dense | 3.35 TB/s | 80 GB | **2.5–3× A100** (bf16) |
| | | ~1500 TFLOPS (fp8) | | | **3–5× A100** (fp8) |

The "practical training factor" is end-to-end on realistic workloads
(attention, optimizer step, communication, dataloader), not isolated
kernel benchmarks.

Notes:

- **V100 lacks FlashAttention-2** — it falls back to memory-efficient
  attention, which is meaningfully slower for long sequences.
- **A100 → H100 is the largest single-generation jump** in the LLM era,
  in both tensor-core throughput and bandwidth.
- **fp8 is Hopper-only** — Ampere has no native fp8 tensor cores.

## Why newer GPUs are faster

In rough order of impact for LLM training:

1. **More tensor-core throughput per cycle** — Hopper produces ~2.4× more
   bf16 FLOPS/clock than Ampere; this dominates compute-bound kernels.
2. **Higher HBM bandwidth** — the memory-bound ops (optimizer step,
   normalization, residual adds) speed up roughly linearly with
   bandwidth.
3. **FlashAttention-3** (Hopper, opt-in) — uses Hopper-specific hardware
   (TMA, WGMMA). Not engaged by default (see below).
4. **fp8 tensor cores** (Hopper) — double the bf16 throughput at the cost
   of per-tensor scaling.
5. **Faster interconnect and larger L2** — cheaper all-reduce and better
   attention block reuse.

## Config knobs that matter

GPT-Simple picks good defaults; these are the ones worth thinking about.

### Mixed precision — bf16 on Ampere and newer
```yaml
training:
  mixed_precision: bf16   # or null to auto-detect
```
Auto-detect (`null`) picks bf16 on Ampere+, fp16 on V100/T4, no mixed
precision on CPU. Prefer bf16 over fp16 where available: it has the same
exponent range as fp32 and needs no loss scaling.

### Attention — keep `causal` unless you need a custom mask
```yaml
model:
  attention_mode: causal
```
`causal` calls SDPA with `is_causal=True`, which dispatches to
FlashAttention-2 on both A100 and H100 — no mask tensor materialized.
`sdpa_mask` and `flex` are slower fast-path-wise and only worth it when
you genuinely need per-document or custom masking (see
[Architecture](architecture.md)).

### Compile — more impactful on faster GPUs
```yaml
training:
  compile: true
```
Framework overhead is a *larger fraction* of step time when kernels are
faster, so `compile` typically gains ~10–20% on A100 and ~20–30% on H100.

### Batch size — push it up on faster GPUs
With a faster step, the same 80 GB can feed a larger batch:
```yaml
training:
  per_device_batch_size: 32       # vs 16 for the same model on A100
  gradient_accumulation_steps: 1  # compensate to keep global batch constant
```
Aim for ~85% peak memory. Below ~70% you leave throughput on the table;
above ~90% you risk OOM during evaluation (which runs at a larger
effective batch).

### Dataloader workers — scale with step speed
```yaml
data:
  num_workers: 8
```
A faster step gives the dataloader less time per batch. If
`tokens_per_second` plateaus as you raise `per_device_batch_size`, the
dataloader is the bottleneck — raise `num_workers`, then prefer
`pretokenized` over `jsonl` (no on-the-fly tokenization).

## FlashAttention-3 (Hopper, advanced)

FA-3 is **not** engaged by default, even on H100. On a modern PyTorch,
`scaled_dot_product_attention(..., is_causal=True)` dispatches to bundled
FA-2 on both Ampere and Hopper. The Hopper FA-3 kernel lives in cuDNN
under the `CUDNN_ATTENTION` SDPA backend, but `FLASH_ATTENTION` outranks
it in SDPA's selection priority, so cuDNN only fires when explicitly
forced.

Indicative per-call cost at `B=2, H=12, S=2048, D=64, bf16, is_causal`
(PyTorch 2.6):

| Setup | µs/call | vs A100 |
| ----- | ------- | ------- |
| A100, default (FA-2) | ~149 | 1.00× |
| H100, default (FA-2 on better silicon) | ~76 | 1.96× |
| H100, forced cuDNN (FA-3) | ~52 | 2.84× |

The headline 2.5–3× end-to-end H100 win comes from *FA-2 on Hopper
hardware*, not FA-3. The marginal FA-3 win (~1.45× at the attention call)
is roughly 10–15% end-to-end at 125M/seq=2048 — bigger at longer
sequences. Engaging it requires patching the SDPA call site in
`src/gpt_simple/model.py` to wrap it in
`sdpa_kernel([SDPBackend.CUDNN_ATTENTION])` (with a probe-once-at-init or
try/except fallback, since `sdpa_kernel` takes a *set* of allowed
backends and FLASH still outranks CUDNN within it). GPT-Simple does not
do this by default.

## fp8 — the bigger Hopper win, with engineering cost

fp8 (E4M3/E5M2) offers another ~1.5–2× over bf16 on H100. The silicon is
free; the software is not. It requires per-tensor scaling factors, fp8-aware
layer wrapping, recipe management (which layers run fp8 vs bf16), and
careful numerics validation. The two routes are NVIDIA's
[Transformer Engine](https://github.com/NVIDIA/TransformerEngine)
(production-grade — swap `nn.Linear`/norm/attention for `te` equivalents
and wrap the step in `te.fp8_autocast`) or PyTorch's native fp8 dtypes
(experimental). GPT-Simple ships neither integration today.

It is worth the effort only for long, expensive runs on large models
(≥1B, ideally ≥7B) where attention + matmul dominate step time; skip it
for models below ~350M or while still iterating on architecture/data.

## Verifying you're on the fast path

It's easy to *think* you're running FA + bf16 + compile while a flag is
wrong. Quick checks:

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda)"
```
Use PyTorch ≥ 2.3 for current SDPA features.

Confirm SDPA picks a FlashAttention backend for your shapes:

```python
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

q = k = v = torch.randn(2, 16, 2048, 64, device="cuda", dtype=torch.bfloat16)
with sdpa_kernel([SDPBackend.FLASH_ATTENTION]):
    out = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
print("FA backend works:", out.shape)
```

Identify which kernel actually fires by profiling and reading the CUDA
trace (`torch.profiler`). Kernel-name fragments:

| Fragment | Backend | Where |
| -------- | ------- | ----- |
| `pytorch_flash::flash_fwd_kernel` | FA-2 | A100/H100 default |
| `cudnn_generated_..._sdpa_sm90_...` | FA-3 (cuDNN) | H100, only when forced |
| `fmha_cutlassF_...` | memory-efficient (CUTLASS) | FA disabled/ineligible |
| anything with `math` | naive eager | last-resort — investigate |

### Throughput sanity ranges
At step ~50 (past warmup), for a 125M model at seq=2048, 8 GPUs,
bf16 + FA + compile:

- 8× V100: ~100–150 k tok/s
- 8× A100: ~400–600 k tok/s
- 8× H100: ~1.2–1.8 M tok/s

Far below these? Check, in order: GPU SM utilization (`nvidia-smi`),
`num_workers`/dataloader timing, whether compile actually engaged, then
SDPA backend selection.

## Migrating a run between GPU generations

The checkpoint is portable, so migrating (e.g. A100 → H100) is mostly a
config and launch change:

1. **Don't change `model.*` or `data.*`** — keep the architecture and
   dataset identical so the checkpoint stays compatible.
2. **Update the launch/scheduler script** for the new partition and GPU
   count per node (your scheduler's account/constraint/queue and
   `--gres`).
3. **Bump `per_device_batch_size`** toward ~85% memory.
4. **Adjust `gradient_accumulation_steps`** to keep the global batch
   constant.
5. **Increase `num_workers`.**
6. **Keep `mixed_precision: bf16`, `attention_mode: causal`,
   `compile: true`.**
7. **Resume from the same checkpoint** — `resume: auto` with the same
   `output_dir` continues the run; a change in GPU count per node is
   handled by the topology-agnostic data cursors (see
   [Checkpointing & resume](checkpointing-and-resume.md)).
8. **Watch `tokens_per_second`** — expect roughly the per-generation
   factor from the table; if you see less, work down the verification
   checklist above.

For a concrete, site-specific worked example (partition names, module
loads, scheduler headers), see the notes alongside the templates in
[`examples/orchestrators/`](https://github.com/lb-off/gpt-simple/tree/main/examples/orchestrators).

## Further reading

- [FlashAttention-3 paper](https://arxiv.org/abs/2407.08608)
- [NVIDIA Transformer Engine docs](https://docs.nvidia.com/deeplearning/transformer-engine/)
- [PyTorch SDPA backend selection](https://pytorch.org/docs/stable/generated/torch.nn.attention.sdpa_kernel.html)
