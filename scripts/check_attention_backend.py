#!/usr/bin/env python3
"""
Verify the FlashAttention codepath is available and selected for your
real workload shape on the current GPU.

What this checks, in order:

  1.  PyTorch + CUDA version (FA-3 requires PyTorch >= 2.3 on H100).
  2.  GPU model and compute capability — tells you whether FA-2 (Ampere)
      or FA-3 (Hopper) is the applicable kernel.
  3.  Whether SDPA can select the FLASH_ATTENTION backend for your
      actual q/k/v shape and dtype. Forces the backend via
      sdpa_kernel() so the call errors loudly if FA isn't selectable.
  4.  Optionally, profiles the SDPA call and prints the underlying CUDA
      kernel name so you can tell FA-2 from FA-3 in the trace.

Usage:
    python scripts/check_attention_backend.py
    python scripts/check_attention_backend.py --seq 4096 --dtype fp16
    python scripts/check_attention_backend.py --profile         # also print kernel name
    python scripts/check_attention_backend.py --n_head 16 --n_embd 1024  # custom model
"""

import argparse
import sys

import torch

# Cap → architecture label.  Only the first hit matters; we fall back to
# "other" if PyTorch ever runs on something exotic.
_ARCH_BY_CAP = {
    (7, 0): ("Volta",   "FA-2: not supported; SDPA falls back to mem-efficient"),
    (7, 5): ("Turing",  "FA-2: limited; SDPA usually picks mem-efficient"),
    (8, 0): ("Ampere",  "FA-2: supported (the right kernel here)"),
    (8, 6): ("Ampere",  "FA-2: supported"),
    (8, 9): ("Ada",     "FA-2: supported"),
    (9, 0): ("Hopper",  "FA-3: supported on PyTorch >= 2.3"),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seq",    type=int, default=2048, help="sequence length (default: 2048)")
    p.add_argument("--batch",  type=int, default=2,    help="batch size (default: 2)")
    p.add_argument("--n_head", type=int, default=12,   help="number of attention heads (default: 12, small preset)")
    p.add_argument("--n_embd", type=int, default=768,  help="hidden size (default: 768, small preset)")
    p.add_argument("--dtype",  choices=["bf16", "fp16", "fp32"], default="bf16")
    p.add_argument("--profile", action="store_true",
                   help="profile a SDPA call and print the underlying CUDA kernel name")
    return p.parse_args()


def section(title: str) -> None:
    print()
    print("=" * 64)
    print(f" {title}")
    print("=" * 64)


def check_version_and_device() -> tuple[int, int]:
    """Return the compute capability of GPU 0."""
    section("PyTorch & device")
    print(f"  PyTorch version : {torch.__version__}")
    print(f"  CUDA version    : {torch.version.cuda}")

    if not torch.cuda.is_available():
        print("  No CUDA device — run this on a GPU node.")
        sys.exit(2)

    name = torch.cuda.get_device_name(0)
    cap  = torch.cuda.get_device_capability(0)
    arch, note = _ARCH_BY_CAP.get(cap, ("other", "(unknown architecture)"))

    print(f"  Device          : {name}")
    print(f"  Compute cap     : sm_{cap[0]}{cap[1]} ({arch})")
    print(f"  Flash kernel    : {note}")

    # Hopper + old PyTorch is the one common 'silently leaves perf on the table' case.
    if cap == (9, 0):
        major, minor = torch.__version__.split(".")[:2]
        if (int(major), int(minor)) < (2, 3):
            print()
            print("  ⚠ PyTorch < 2.3 on Hopper — SDPA will pick FA-2, not FA-3.")
            print("    Upgrade to >= 2.3 to engage the Hopper-native kernel.")
    return cap


def check_flash_selectable(args: argparse.Namespace) -> None:
    """Force the FA backend and run a SDPA call. Errors loudly if not selectable."""
    section("FLASH_ATTENTION backend selectable for your workload?")

    head_dim = args.n_embd // args.n_head
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]

    print(f"  Workload        : B={args.batch}, H={args.n_head}, S={args.seq}, "
          f"D={head_dim}, dtype={args.dtype}, is_causal=True")

    if head_dim > 256:
        print(f"  ⚠ head_dim={head_dim} exceeds FA's 256 limit — FA path will not be selected.")
        sys.exit(1)
    if args.dtype == "fp32":
        print(f"  ⚠ fp32 inputs — FA requires fp16/bf16. FA path will not be selected.")
        sys.exit(1)

    from torch.nn.attention import SDPBackend, sdpa_kernel

    q = torch.randn(args.batch, args.n_head, args.seq, head_dim, device="cuda", dtype=dtype)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    try:
        with sdpa_kernel([SDPBackend.FLASH_ATTENTION]):
            out = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
            torch.cuda.synchronize()
    except RuntimeError as e:
        print()
        print("  ✗ FLASH_ATTENTION backend rejected this workload.")
        print(f"    PyTorch said: {e}")
        print("    SDPA in your real training run is silently falling back to")
        print("    memory-efficient or math attention. Check head_dim, dtype,")
        print("    mask shape, and tensor contiguity.")
        sys.exit(1)

    print(f"  ✓ FA backend ran cleanly. Output shape: {tuple(out.shape)}")


def profile_kernel(args: argparse.Namespace) -> None:
    """Run SDPA under torch.profiler and print the CUDA kernel name."""
    section("Underlying CUDA kernel (look for 'sm90' / 'v3' = FA-3, else FA-2)")

    head_dim = args.n_embd // args.n_head
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    q = torch.randn(args.batch, args.n_head, args.seq, head_dim, device="cuda", dtype=dtype)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    # Warmup so we don't profile compile / first-call overhead
    for _ in range(3):
        torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
    torch.cuda.synchronize()

    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(10):
            torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
    torch.cuda.synchronize()

    # Print the top CUDA kernels by total time — the FA kernel is almost
    # always #1 for this microbenchmark.
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=8))

    # Convenience: explicit FA-3 hint
    events = [e for e in prof.events() if e.device_type.name == "CUDA"]
    fa_names = {e.name for e in events if "flash" in e.name.lower()}
    if any(("sm90" in n) or ("v3" in n) or ("hopper" in n.lower()) for n in fa_names):
        print("  → kernel name suggests FA-3 (Hopper)")
    elif fa_names:
        print("  → kernel name suggests FA-2 (Ampere/Ada)")
    else:
        print("  → no 'flash' kernel observed — SDPA used a different backend.")


def main() -> None:
    args = parse_args()
    check_version_and_device()
    check_flash_selectable(args)
    if args.profile:
        profile_kernel(args)
    section("Summary")
    print("  All checks passed. Your SDPA path is using FlashAttention for")
    print(f"  the requested workload (seq={args.seq}, dtype={args.dtype}).")


if __name__ == "__main__":
    main()
