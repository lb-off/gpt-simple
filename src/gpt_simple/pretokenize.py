#!/usr/bin/env python3
"""
Offline pretokenization: JSONL -> binary (.bin / .idx) format.

Converts raw JSONL text shards into a compact binary representation that can
be memory-mapped at training time, eliminating on-the-fly tokenization and
JSON parsing overhead.

Usage:
    python -m src.data.pretokenize \
        --input_dir  /data/raw/web/ \
        --output_dir /data/tokenized/web/ \
        --tokenizer_path /path/to/tokenizer \
        --max_length 3072 \
        --overlap_size 256 \
        --num_workers 32

Binary format
-------------
.bin  – flat numpy array of uint16 (or uint32) token IDs.
        Documents/windows are concatenated back-to-back, each terminated
        with an EOD token.

.idx  – binary index with the following layout:
        Header (16 bytes):
            magic       4 bytes   b"GPTS"
            version     uint32    currently 1
            dtype_code  uint32    2 = uint16, 4 = uint32
            num_docs    uint32    number of document/window entries
        Offsets array:
            (num_docs + 1) x int64   token-level start offsets; last entry
                                     is the total token count (sentinel).
        Overlap prefix lengths:
            num_docs x uint16        number of overlap tokens to mask in
                                     labels for each entry (0 for first
                                     windows and short documents).
"""

import argparse
import gzip
import json
import logging
import random
import struct
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple

import numpy as np

from gpt_simple.tokenizer import SimpleLLMTokenizer

logger = logging.getLogger("gpt_simple")

IDX_MAGIC = b"GPTS"
IDX_VERSION = 1

DTYPE_UINT16 = 2
DTYPE_UINT32 = 4


# ------------------------------------------------------------------
# Windowing (ported from streaming_data.py, deterministic variant)
# ------------------------------------------------------------------

def create_windows(
    tokens: List[int],
    eod_token_id: int,
    max_length: int,
    overlap_size: int,
    probabilistic_overlap: bool,
    overlap_probability: float,
    doc_rng: random.Random,
) -> List[Tuple[List[int], int]]:
    """Split a tokenized document into windows.

    Returns a list of ``(token_ids, overlap_prefix_len)`` tuples.
    Each window is terminated with *eod_token_id*.  The overlap prefix
    length indicates how many leading tokens are repeated from the
    previous window and should be masked (-100) during training.
    """
    if len(tokens) < max_length:
        return [(tokens + [eod_token_id], 0)]

    if probabilistic_overlap and doc_rng.random() > overlap_probability:
        actual_overlap = 0
    else:
        actual_overlap = overlap_size

    stride = max_length - actual_overlap
    stride = max(stride, max_length // 2)

    windows: List[Tuple[List[int], int]] = []
    start = 0
    window_idx = 0

    while start < len(tokens):
        end = min(start + max_length, len(tokens))
        window_tokens = list(tokens[start:end])

        if window_idx > 0 and actual_overlap > 0:
            overlap_prefix_len = min(actual_overlap, len(window_tokens))
        else:
            overlap_prefix_len = 0

        is_last = end >= len(tokens)
        if is_last:
            window_tokens.append(eod_token_id)

        windows.append((window_tokens, overlap_prefix_len))

        start += stride
        window_idx += 1

        if window_idx > 10_000:
            logger.warning(
                f"Excessive windows ({window_idx}) for doc with {len(tokens)} tokens, breaking"
            )
            break

    return windows


# ------------------------------------------------------------------
# .idx file I/O helpers
# ------------------------------------------------------------------

def write_idx(
    path: Path,
    offsets: np.ndarray,
    overlap_lengths: np.ndarray,
    dtype_code: int,
) -> None:
    """Write an .idx file with header, offsets, and overlap lengths."""
    num_docs = len(overlap_lengths)
    with open(path, "wb") as f:
        f.write(IDX_MAGIC)
        f.write(struct.pack("<I", IDX_VERSION))
        f.write(struct.pack("<I", dtype_code))
        f.write(struct.pack("<I", num_docs))
        f.write(offsets.astype(np.int64).tobytes())
        f.write(overlap_lengths.astype(np.uint16).tobytes())


def read_idx(path: Path):
    """Read an .idx file.  Returns (dtype_code, offsets, overlap_lengths)."""
    with open(path, "rb") as f:
        magic = f.read(4)
        if magic != IDX_MAGIC:
            raise ValueError(f"Bad magic in {path}: {magic!r}")
        version = struct.unpack("<I", f.read(4))[0]
        if version != IDX_VERSION:
            raise ValueError(f"Unsupported .idx version {version} in {path}")
        dtype_code = struct.unpack("<I", f.read(4))[0]
        num_docs = struct.unpack("<I", f.read(4))[0]

        offsets = np.frombuffer(f.read((num_docs + 1) * 8), dtype=np.int64).copy()
        overlap_lengths = np.frombuffer(f.read(num_docs * 2), dtype=np.uint16).copy()

    return dtype_code, offsets, overlap_lengths


# ------------------------------------------------------------------
# Per-shard worker
# ------------------------------------------------------------------

def tokenize_shard(
    jsonl_path: str,
    output_dir: str,
    tokenizer_path: str,
    max_length: int,
    overlap_size: int,
    probabilistic_overlap: bool,
    overlap_probability: float,
    min_text_length: int,
) -> dict:
    """Tokenize a single JSONL shard and write .bin + .idx files.

    Designed to run inside a ``ProcessPoolExecutor`` worker.
    Returns a summary dict with statistics.
    """
    jsonl_path = Path(jsonl_path)
    output_dir = Path(output_dir)

    tokenizer = SimpleLLMTokenizer(tokenizer_path)
    eod_token_id = tokenizer.eod_token_id
    vocab_size = tokenizer.vocab_size

    np_dtype = np.uint16 if vocab_size < 65536 else np.uint32
    dtype_code = DTYPE_UINT16 if np_dtype == np.uint16 else DTYPE_UINT32

    offsets: List[int] = [0]
    overlap_lengths: List[int] = []

    docs_read = 0
    docs_skipped = 0
    windows_total = 0

    # Strip .jsonl or .jsonl.gz so "00000.jsonl.gz" -> "00000.bin/.idx".
    stem = jsonl_path.name
    for _suf in (".gz", ".jsonl"):
        if stem.endswith(_suf):
            stem = stem[: -len(_suf)]
    bin_path = output_dir / f"{stem}.bin"
    idx_path = output_dir / f"{stem}.idx"

    # Stream tokens straight to the .bin as we go, instead of buffering every
    # token of the shard in a Python list (that OOMs with many workers on large
    # shards). Memory stays bounded to one window; offsets/overlap_lengths hold
    # only one small int per window.
    total_tokens = 0
    opener = gzip.open if str(jsonl_path).endswith(".gz") else open
    with opener(jsonl_path, "rt", encoding="utf-8") as f, open(bin_path, "wb") as bin_f:
        for line_num, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                docs_skipped += 1
                continue

            text = obj.get("text", "")
            if len(text) < min_text_length:
                docs_skipped += 1
                continue

            tokens = tokenizer.encode(text, add_special_tokens=False)
            if not tokens:
                docs_skipped += 1
                continue

            docs_read += 1

            doc_rng = random.Random(hash((jsonl_path.name, line_num)) % (2**32))
            windows = create_windows(
                tokens,
                eod_token_id=eod_token_id,
                max_length=max_length,
                overlap_size=overlap_size,
                probabilistic_overlap=probabilistic_overlap,
                overlap_probability=overlap_probability,
                doc_rng=doc_rng,
            )

            for window_tokens, overlap_prefix_len in windows:
                arr = np.asarray(window_tokens, dtype=np_dtype)
                bin_f.write(arr.tobytes())
                total_tokens += int(arr.size)
                offsets.append(total_tokens)
                overlap_lengths.append(overlap_prefix_len)
                windows_total += 1

    offsets_arr = np.array(offsets, dtype=np.int64)
    overlap_arr = np.array(overlap_lengths, dtype=np.uint16)
    write_idx(idx_path, offsets_arr, overlap_arr, dtype_code)

    expected_bin_size = offsets_arr[-1] * np_dtype().itemsize
    actual_bin_size = bin_path.stat().st_size
    if actual_bin_size != expected_bin_size:
        raise RuntimeError(
            f"Integrity check failed for {bin_path}: "
            f"expected {expected_bin_size} bytes, got {actual_bin_size}"
        )

    return {
        "shard": jsonl_path.name,
        "docs_read": docs_read,
        "docs_skipped": docs_skipped,
        "windows": windows_total,
        "tokens": int(offsets_arr[-1]),
    }


# ------------------------------------------------------------------
# CLI entry-point
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Pre-tokenize JSONL shards into binary .bin/.idx format"
    )
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Directory containing .jsonl files")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to write .bin/.idx files")
    parser.add_argument("--tokenizer_path", type=str, default="gpt2",
                        help="Tokenizer name or path")
    parser.add_argument("--max_length", type=int, default=3072,
                        help="Maximum sequence length for windowing")
    parser.add_argument("--overlap_size", type=int, default=256,
                        help="Overlap size between windows of long documents")
    parser.add_argument("--probabilistic_overlap", action="store_true",
                        help="Apply overlap probabilistically (70%% overlap, 30%% none)")
    parser.add_argument("--overlap_probability", type=float, default=0.7,
                        help="Probability of applying overlap when probabilistic")
    parser.add_argument("--min_text_length", type=int, default=200,
                        help="Skip documents shorter than this many characters")
    parser.add_argument("--num_workers", type=int, default=1,
                        help="Number of parallel workers (1 = sequential)")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.is_dir():
        print(f"ERROR: input_dir does not exist: {input_dir}", file=sys.stderr)
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    jsonl_files = sorted([*input_dir.glob("*.jsonl"), *input_dir.glob("*.jsonl.gz")])
    if not jsonl_files:
        print(f"ERROR: no .jsonl/.jsonl.gz files found in {input_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(jsonl_files)} JSONL shard(s) in {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Tokenizer: {args.tokenizer_path}")
    print(f"Max length: {args.max_length}, overlap: {args.overlap_size}")
    print(f"Workers: {args.num_workers}")
    print()

    worker_kwargs = dict(
        output_dir=str(output_dir),
        tokenizer_path=args.tokenizer_path,
        max_length=args.max_length,
        overlap_size=args.overlap_size,
        probabilistic_overlap=args.probabilistic_overlap,
        overlap_probability=args.overlap_probability,
        min_text_length=args.min_text_length,
    )

    total_tokens = 0
    total_docs = 0
    total_windows = 0
    failed = 0

    if args.num_workers <= 1:
        for jf in jsonl_files:
            stats = tokenize_shard(jsonl_path=str(jf), **worker_kwargs)
            total_tokens += stats["tokens"]
            total_docs += stats["docs_read"]
            total_windows += stats["windows"]
            print(
                f"  {stats['shard']}: {stats['docs_read']} docs, "
                f"{stats['windows']} windows, {stats['tokens']:,} tokens"
                f" ({stats['docs_skipped']} skipped)"
            )
    else:
        with ProcessPoolExecutor(max_workers=args.num_workers) as pool:
            futures = {
                pool.submit(tokenize_shard, jsonl_path=str(jf), **worker_kwargs): jf
                for jf in jsonl_files
            }
            for future in as_completed(futures):
                jf = futures[future]
                try:
                    stats = future.result()
                except Exception as exc:
                    print(f"  ERROR processing {jf.name}: {exc}", file=sys.stderr)
                    failed += 1
                    continue
                total_tokens += stats["tokens"]
                total_docs += stats["docs_read"]
                total_windows += stats["windows"]
                print(
                    f"  {stats['shard']}: {stats['docs_read']} docs, "
                    f"{stats['windows']} windows, {stats['tokens']:,} tokens"
                    f" ({stats['docs_skipped']} skipped)"
                )

    print()
    print(f"Done. {total_docs:,} documents -> {total_windows:,} windows -> {total_tokens:,} tokens")
    if failed:
        print(f"ERROR: {failed} shard(s) failed — output is INCOMPLETE.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
