#!/usr/bin/env python3
"""
Benchmark the three attention backends: causal, sdpa_mask, flex.

Generates synthetic packed-sequence data (with document boundaries) and
runs forward + backward passes for each mode, reporting step time,
peak VRAM, and throughput.

Usage:
    python scripts/benchmark_attention.py
    python scripts/benchmark_attention.py --modes causal sdpa_mask --steps 50
    python scripts/benchmark_attention.py --n_layer 12 --n_embd 768 --n_head 12  # smaller model
    python scripts/benchmark_attention.py --compile  # with torch.compile
"""

import argparse
import gc
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

# Allow imports from the project root (append to avoid shadowing pip packages
# like `tokenizers` which has a local directory with the same name)
sys.path.append(str(Path(__file__).resolve().parent.parent))

from gpt_simple.config import ModelConfig
from gpt_simple.model import SimpleLLM


# ---------------------------------------------------------------------------
# Synthetic data generation
# ---------------------------------------------------------------------------

def make_synthetic_batch(
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    num_docs_per_seq: int = 4,
    device: torch.device = torch.device("cuda"),
    need_doc_ids: bool = True,
):
    """Create a batch that mimics real packed-sequence training data.

    Each sequence is filled with ``num_docs_per_seq`` documents of roughly
    equal length, separated by an EOD token (id=0).  Labels mask padding
    and EOD positions with -100.
    """
    input_ids = torch.randint(2, vocab_size, (batch_size, seq_len), device=device)
    labels = input_ids.clone()
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long, device=device)

    doc_ids = None
    if need_doc_ids:
        doc_ids = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)

    eod_token_id = 0
    doc_len = seq_len // num_docs_per_seq

    for b in range(batch_size):
        for d in range(num_docs_per_seq):
            start = d * doc_len
            end = min((d + 1) * doc_len, seq_len)
            if end < seq_len:
                input_ids[b, end - 1] = eod_token_id
                labels[b, end - 1] = -100
                if end < seq_len:
                    labels[b, end] = -100
            if doc_ids is not None:
                doc_ids[b, start:end] = d

    batch = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "position_ids": torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1),
    }
    if doc_ids is not None:
        batch["doc_ids"] = doc_ids
    return batch


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def benchmark_mode(
    mode: str,
    n_embd: int,
    n_layer: int,
    n_head: int,
    seq_len: int,
    batch_size: int,
    vocab_size: int,
    steps: int,
    warmup_steps: int,
    device: torch.device,
    compile_model: bool = False,
):
    """Run forward+backward for *steps* iterations and return metrics."""
    config = ModelConfig(
        vocab_size=vocab_size,
        n_embd=n_embd,
        n_layer=n_layer,
        n_head=n_head,
        n_positions=seq_len,
        attention_mode=mode,
        dropout=0.0,
    )
    model = SimpleLLM(config, gradient_checkpointing=True).to(device).train()

    if compile_model:
        print("  Compiling model with torch.compile (this may take a few minutes)...")
        model = torch.compile(model)

    need_doc_ids = mode != "causal"
    batch = make_synthetic_batch(
        batch_size=batch_size,
        seq_len=seq_len,
        vocab_size=vocab_size,
        device=device,
        need_doc_ids=need_doc_ids,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Warmup (also triggers compilation on first pass when --compile is used)
    for _ in range(warmup_steps):
        optimizer.zero_grad(set_to_none=True)
        out = model(**batch, return_dict=True)
        out.loss.backward()
        optimizer.step()
    torch.cuda.synchronize(device)

    torch.cuda.reset_peak_memory_stats(device)

    step_times = []
    for _ in range(steps):
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()

        optimizer.zero_grad(set_to_none=True)
        out = model(**batch, return_dict=True)
        out.loss.backward()
        optimizer.step()

        torch.cuda.synchronize(device)
        step_times.append(time.perf_counter() - t0)

    peak_vram_bytes = torch.cuda.max_memory_allocated(device)

    # Non-pad tokens per step
    non_pad = (batch["labels"] != -100).sum().item()

    mean_t = sum(step_times) / len(step_times)
    std_t = (sum((t - mean_t) ** 2 for t in step_times) / len(step_times)) ** 0.5
    tokens_per_sec = non_pad / mean_t
    peak_vram_gb = peak_vram_bytes / (1024 ** 3)

    # Cleanup
    del model, optimizer, batch, out
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "mode": mode,
        "mean_step_s": mean_t,
        "std_step_s": std_t,
        "peak_vram_gb": peak_vram_gb,
        "tokens_per_sec": tokens_per_sec,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Benchmark attention backends")
    parser.add_argument("--modes", nargs="+", default=["causal", "sdpa_mask", "flex"],
                        choices=["causal", "sdpa_mask", "flex"])
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--warmup_steps", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seq_len", type=int, default=3072)
    parser.add_argument("--n_embd", type=int, default=2304)
    parser.add_argument("--n_layer", type=int, default=35)
    parser.add_argument("--n_head", type=int, default=18)
    parser.add_argument("--vocab_size", type=int, default=45056)
    parser.add_argument("--compile", action="store_true",
                        help="Wrap model in torch.compile before benchmarking")
    parser.add_argument("--output", type=str, default=None,
                        help="Path to write markdown results (e.g. benchmarks/attention_results.md)")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA is required for this benchmark.")
        sys.exit(1)

    device = torch.device("cuda")
    compile_str = " (torch.compile enabled)" if args.compile else ""
    print(f"GPU: {torch.cuda.get_device_name(device)}")
    print(f"Model: n_embd={args.n_embd}, n_layer={args.n_layer}, n_head={args.n_head}{compile_str}")
    print(f"Sequence length: {args.seq_len}, batch size: {args.batch_size}")
    print(f"Steps: {args.steps} (+ {args.warmup_steps} warmup)\n")

    results = []
    for mode in args.modes:
        print(f"--- Benchmarking mode: {mode} ---")
        try:
            r = benchmark_mode(
                mode=mode,
                n_embd=args.n_embd,
                n_layer=args.n_layer,
                n_head=args.n_head,
                seq_len=args.seq_len,
                batch_size=args.batch_size,
                vocab_size=args.vocab_size,
                steps=args.steps,
                warmup_steps=args.warmup_steps,
                device=device,
                compile_model=args.compile,
            )
            results.append(r)
            print(f"  mean step: {r['mean_step_s']:.3f}s  "
                  f"std: {r['std_step_s']:.3f}s  "
                  f"peak VRAM: {r['peak_vram_gb']:.2f} GB  "
                  f"tok/s: {r['tokens_per_sec']:.0f}\n")
        except Exception as e:
            print(f"  SKIPPED ({e})\n")

    if not results:
        print("No results to report.")
        sys.exit(0)

    # Build markdown table
    header = "| Mode | Mean step (s) | Std (s) | Peak VRAM (GB) | Tokens/sec |"
    sep    = "|------|--------------|---------|----------------|------------|"
    rows = [header, sep]
    for r in results:
        rows.append(
            f"| {r['mode']:<10} "
            f"| {r['mean_step_s']:<12.4f} "
            f"| {r['std_step_s']:<7.4f} "
            f"| {r['peak_vram_gb']:<14.2f} "
            f"| {r['tokens_per_sec']:<10.0f} |"
        )
    table = "\n".join(rows)
    print("\n## Attention Benchmark Results\n")
    print(table)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            f.write("# Attention Benchmark Results\n\n")
            f.write(f"GPU: {torch.cuda.get_device_name(device)}\n\n")
            f.write(f"Model: n_embd={args.n_embd}, n_layer={args.n_layer}, n_head={args.n_head}{compile_str}\n\n")
            f.write(f"Sequence length: {args.seq_len}, batch size: {args.batch_size}\n\n")
            f.write(f"Steps: {args.steps} (+ {args.warmup_steps} warmup)\n\n")
            f.write(table + "\n")
        print(f"\nResults written to {out_path}")


if __name__ == "__main__":
    main()
