# Roadmap

Where GPT-Simple is, what it deliberately is *not*, and what I'd build next.
For day-to-day usage see the [README](README.md); for the architecture see the
source under `src/gpt_simple/`.

## Scope

GPT-Simple is a single-author, end-to-end LLM **pretraining** stack built to be
*readable and correct* before it is maximally scaled. The goal was to implement
the full from-scratch workflow — tokenization, streaming data, multi-GPU
training, deterministic stop/resume, inference — as one coherent, legible
codebase rather than to compete with production training frameworks on raw
scale. The scope below is a deliberate choice, not an unfinished to-do list.

It is stable for what it covers:

- Single YAML config + CLI (`init` / `tokenize` / `train` / `status` / `stop`
  / `generate` / `batch-generate`).
- Pretokenized memory-mapped shards with sequence packing, plus an on-the-fly
  JSONL fallback.
- Multi-GPU training via Accelerate — bf16, `torch.compile`, gradient
  checkpointing, and SDPA/FlashAttention-eligible attention kernels.
- Deterministic, topology-agnostic stop/resume driven by walltime and signals
  (every document seen once, even when `world_size` / `num_workers` change).
- Curriculum learning across named data buckets.
- Built-in GPT architecture (RoPE, RMSNorm, SwiGLU, GQA/MQA) and a small
  Python API.

## Where this sits vs. production stacks

The honest gap. Frameworks like NVIDIA [Megatron-LM][megatron],
PyTorch-native [torchtitan][torchtitan], Microsoft [DeepSpeed][deepspeed],
Hugging Face [nanotron][nanotron], EleutherAI [GPT-NeoX][neox], and Lightning
[litgpt][litgpt] exist to push models past the single-GPU memory wall and to
squeeze clusters of hundreds-to-thousands of GPUs. GPT-Simple intentionally
stops short of that. The main axis of difference is **multi-dimensional
parallelism**: GPT-Simple is data-parallel only (DDP — every GPU holds a full
model replica), whereas the production stacks shard the model itself.

| Capability | GPT-Simple | Production stacks |
| --- | --- | --- |
| Data parallelism (DDP) | ✅ | ✅ |
| Sharded data parallel (FSDP2 / ZeRO-3) | ❌ — full replica per GPU | Core to torchtitan, DeepSpeed |
| Tensor / pipeline / context parallelism | ❌ | Megatron-LM, torchtitan, nanotron (3D/4D) |
| fp8 / sub-bf16 precision | ❌ — bf16 only | torchtitan (float8) |
| CPU/NVMe optimizer offload | ❌ | DeepSpeed ZeRO-Infinity |
| Readable single-author codebase | ✅ | Partial — nanotron is the closest in spirit |
| Full from-scratch CLI + Python API in one package | ✅ | Varies |

The takeaway: for models that fit in a single GPU's memory (replicated across
however many GPUs you have), GPT-Simple is a complete, legible pretraining lane.
For 10B+ parameter models that must be sharded across devices, reach for one of
the frameworks above — that's their reason to exist.

## Roadmap (no fixed timeline)

This is an educational/portfolio project I maintain in my own time, so the list
below is ordered by what I'd find most interesting and most in-character to
build next — not by any release schedule. **Contributions are genuinely welcome**;
these make good, self-contained entry points.

Closest to the existing design:

- **FSDP2 / sharded data parallel.** The natural next rung past DDP. The
  checkpoint layer already gathers sharded state dicts; the real work is
  preserving the topology-agnostic stop/resume guarantee under sharding.
- **Memory-efficient (chunked / fused) cross-entropy** for the LM head, so
  large vocabularies stop dominating activation memory.
- **8-bit / fused optimizers** (e.g. `AdamW8bit`, fused AdamW) behind a config
  flag.

Larger, would extend the project meaningfully:

- **Evaluation harness** — wire trained checkpoints to downstream/perplexity
  benchmarks instead of leaving evaluation to the user.
- **Hugging Face-format export** — convert checkpoints to a
  `transformers`-loadable format to plug into the inference ecosystem.
- **Online dedup (MinHash)** as a streaming data-prep stage.
- **RoPE context extension** (NTK / YaRN scaling) for longer sequences.

Exploration — real engineering projects, listed honestly as "maybe someday":

- **Tensor / pipeline / context parallelism.** These are the heart of the
  production frameworks above and a large undertaking; out of scope for now.
- **Mixture-of-Experts layers.**

Suggestions and contributions are welcome — open an issue to start a discussion.

[megatron]: https://github.com/NVIDIA/Megatron-LM
[torchtitan]: https://github.com/pytorch/torchtitan
[deepspeed]: https://github.com/deepspeedai/DeepSpeed
[nanotron]: https://github.com/huggingface/nanotron
[neox]: https://github.com/EleutherAI/gpt-neox
[litgpt]: https://github.com/Lightning-AI/litgpt
