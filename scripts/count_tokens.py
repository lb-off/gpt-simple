#!/usr/bin/env python3
"""Count tokens per bucket (train/val) in a pretokenized dataset.

Reads the .idx sentinel offset of every shard (exact token count, no
re-tokenization, no .bin read). Layout assumed:

    <root>/train/<bucket>/*.idx
    <root>/val/<bucket>/*.idx

Usage:
    python scripts/count_tokens.py $SCRATCH/dataset/final-tokenized-v2
"""
import sys
from pathlib import Path

from gpt_simple.pretokenize import read_idx


def shard_tokens(idx_path: Path) -> int:
    # offsets[-1] is the total-token sentinel for the shard.
    _dtype_code, offsets, _overlap = read_idx(idx_path)
    return int(offsets[-1])


def count_split(split_dir: Path) -> dict:
    out = {}
    for bucket in sorted(p for p in split_dir.iterdir() if p.is_dir()):
        idx_files = sorted(bucket.glob("*.idx"))
        toks = sum(shard_tokens(p) for p in idx_files)
        out[bucket.name] = (toks, len(idx_files))
    return out


def main():
    root = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
    for split in ("train", "val"):
        split_dir = root / split
        if not split_dir.is_dir():
            print(f"[skip] no {split}/ under {root}")
            continue
        print(f"\n=== {split} ===")
        buckets = count_split(split_dir)
        grand = 0
        for name, (toks, n) in buckets.items():
            print(f"  {name:<20} {toks:>18,} tokens  ({n} shards)")
            grand += toks
        print(f"  {'TOTAL':<20} {grand:>18,} tokens")


if __name__ == "__main__":
    main()
