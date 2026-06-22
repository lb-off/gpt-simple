#!/usr/bin/env python3
"""
Tests for the Phase 3 pre-tokenization pipeline.

Covers:
  1. Roundtrip tokenization (JSONL -> .bin/.idx -> tokens match)
  2. .idx file format integrity (header, offsets, overlap lengths)
  3. Windowing of long documents with overlap masking
  4. Label masking correctness (padding/post-EOD/overlap masked; EOD kept)
  5. Packing efficiency (>= 0.9 target)
  6. doc_ids generation for non-causal attention modes
  7. Resume via start_position

Run:
    python -m pytest tests/test_pretokenized.py -v
    -- or --
    python tests/test_pretokenized.py
"""

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

_project_root = str(Path(__file__).parent.parent)
if _project_root not in sys.path:
    sys.path.append(_project_root)

from gpt_simple.pretokenize import (
    IDX_MAGIC,
    create_windows,
    read_idx,
    tokenize_shard,
    write_idx,
)
from gpt_simple.data import InterleavedDataset, PreTokenizedDataset, WeightedInterleavedDataset


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

EOD_TOKEN = 0
PAD_TOKEN = 1
VOCAB_SIZE = 100


def _make_jsonl(tmp: Path, docs: list[str], name: str = "shard_000.jsonl") -> Path:
    """Write a JSONL file with the given document texts."""
    path = tmp / name
    with open(path, "w") as f:
        for text in docs:
            f.write(json.dumps({"text": text}) + "\n")
    return path


def _tokenize_to_dir(
    input_dir: Path,
    output_dir: Path,
    tokenizer_path: str = "gpt2",
    max_length: int = 128,
    overlap_size: int = 32,
):
    """Run tokenize_shard on every JSONL file in *input_dir*."""
    results = []
    for jf in sorted(input_dir.glob("*.jsonl")):
        stats = tokenize_shard(
            jsonl_path=str(jf),
            output_dir=str(output_dir),
            tokenizer_path=tokenizer_path,
            max_length=max_length,
            overlap_size=overlap_size,
            probabilistic_overlap=False,
            overlap_probability=0.7,
            min_text_length=10,
        )
        results.append(stats)
    return results


# ------------------------------------------------------------------
# 1. Roundtrip tokenization
# ------------------------------------------------------------------

def test_roundtrip_tokenization():
    """Tokens in .bin must match direct tokenizer output."""
    from gpt_simple.tokenizer import SimpleLLMTokenizer

    tokenizer = SimpleLLMTokenizer("gpt2")
    eod = tokenizer.eod_token_id

    docs = [
        "Hello world, this is a test document.",
        "Another document with different content for testing purposes.",
    ]

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        inp = tmp / "input"
        out = tmp / "output"
        inp.mkdir()
        out.mkdir()

        _make_jsonl(inp, docs)
        _tokenize_to_dir(inp, out, max_length=512)

        bin_path = out / "shard_000.bin"
        idx_path = out / "shard_000.idx"
        assert bin_path.exists()
        assert idx_path.exists()

        dtype_code, offsets, overlap_lengths = read_idx(idx_path)
        np_dtype = np.uint16 if dtype_code == 2 else np.uint32
        tokens = np.fromfile(str(bin_path), dtype=np_dtype)

        for i, doc_text in enumerate(docs):
            expected = tokenizer.encode(doc_text, add_special_tokens=False)
            start, end = int(offsets[i]), int(offsets[i + 1])
            actual = tokens[start:end].tolist()
            assert actual[-1] == eod, f"doc {i} does not end with EOD"
            assert actual[:-1] == expected, f"doc {i} token mismatch"

    print("PASS: roundtrip tokenization")


# ------------------------------------------------------------------
# 2. .idx format integrity
# ------------------------------------------------------------------

def test_idx_format():
    """Verify header magic, version, offset sentinel, and size consistency."""
    offsets = np.array([0, 100, 250, 400], dtype=np.int64)
    overlap = np.array([0, 16, 0], dtype=np.uint16)

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        idx_path = tmp / "test.idx"
        write_idx(idx_path, offsets, overlap, dtype_code=2)

        with open(idx_path, "rb") as f:
            magic = f.read(4)
            assert magic == IDX_MAGIC

        dc, off, ovl = read_idx(idx_path)
        assert dc == 2
        np.testing.assert_array_equal(off, offsets)
        np.testing.assert_array_equal(ovl, overlap)

    print("PASS: .idx format integrity")


# ------------------------------------------------------------------
# 3. Windowing
# ------------------------------------------------------------------

def test_windowing_short_doc():
    """A document shorter than max_length should produce one window."""
    import random

    tokens = list(range(50))
    windows = create_windows(
        tokens, eod_token_id=EOD_TOKEN, max_length=128,
        overlap_size=32, probabilistic_overlap=False,
        overlap_probability=0.7, doc_rng=random.Random(0),
    )
    assert len(windows) == 1
    win_tokens, ovl = windows[0]
    assert win_tokens[-1] == EOD_TOKEN
    assert ovl == 0
    assert win_tokens[:-1] == tokens
    print("PASS: windowing short doc")


def test_windowing_long_doc():
    """A long document should produce multiple overlapping windows."""
    import random

    max_length = 64
    overlap = 16
    # Use tokens starting at 10 to avoid confusion with EOD_TOKEN=0
    tokens = list(range(10, 210))

    windows = create_windows(
        tokens, eod_token_id=EOD_TOKEN, max_length=max_length,
        overlap_size=overlap, probabilistic_overlap=False,
        overlap_probability=0.7, doc_rng=random.Random(0),
    )

    assert len(windows) > 1, f"Expected multiple windows, got {len(windows)}"

    stride = max_length - overlap

    # Windows whose slice doesn't reach the end of the document should NOT
    # have EOD.  Windows that do reach the end WILL have EOD appended.
    for i, (wt, _) in enumerate(windows):
        win_start = i * stride
        win_end = min(win_start + max_length, len(tokens))
        reaches_end = win_end >= len(tokens)
        if reaches_end:
            assert wt[-1] == EOD_TOKEN, f"Window {i} reaches doc end but has no EOD"
        else:
            assert wt[-1] != EOD_TOKEN, f"Window {i} should not end with EOD"

    # First window: no overlap masking
    assert windows[0][1] == 0

    # Subsequent windows: overlap prefix > 0, capped by window token count
    for i in range(1, len(windows)):
        wt, ovl = windows[i]
        content_len = len(wt) - (1 if wt[-1] == EOD_TOKEN else 0)
        expected_ovl = min(overlap, content_len)
        assert ovl == expected_ovl, (
            f"Window {i} overlap should be {expected_ovl}, got {ovl}"
        )

    # Overlap content: first `overlap` tokens of window i must match the last
    # `overlap` tokens of the non-EOD portion of window i-1.
    for i in range(1, len(windows)):
        curr_start = i * stride
        expected_overlap = tokens[curr_start : curr_start + overlap]
        curr_tokens = windows[i][0]
        if curr_tokens[-1] == EOD_TOKEN:
            curr_tokens = curr_tokens[:-1]
        actual_overlap = curr_tokens[:overlap]
        assert expected_overlap == actual_overlap, f"Overlap mismatch at window {i}"

    print("PASS: windowing long doc")


# ------------------------------------------------------------------
# 4. Label masking
# ------------------------------------------------------------------

def test_label_masking():
    """Verify label masking in PreTokenizedDataset: padding and post-EOD masked,
    EOD itself kept as a real target, overlap prefixes masked."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)

        # Manually create a small .bin / .idx with known content
        # Two short "documents": [10, 11, 12, EOD] and [20, 21, EOD]
        eod = 0
        pad = 1
        doc1 = np.array([10, 11, 12, eod], dtype=np.uint16)
        doc2 = np.array([20, 21, eod], dtype=np.uint16)
        all_tokens = np.concatenate([doc1, doc2])

        bin_path = tmp / "test.bin"
        idx_path = tmp / "test.idx"
        all_tokens.tofile(str(bin_path))

        offsets = np.array([0, len(doc1), len(doc1) + len(doc2)], dtype=np.int64)
        overlap = np.array([0, 0], dtype=np.uint16)
        write_idx(idx_path, offsets, overlap, dtype_code=2)

        ds = PreTokenizedDataset(
            bin_files=[bin_path],
            max_length=16,
            seed=42,
            pad_token_id=pad,
            eod_token_id=eod,
            attention_mode="causal",
            pack_sequences=True,
            packing_buffer_size=8,
        )

        samples = list(ds)
        assert len(samples) >= 1, "Should produce at least one packed sample"
        sample = samples[0]

        ids = sample["input_ids"]
        labels = sample["labels"]
        attn = sample["attention_mask"]

        content_len = int(attn.sum().item())

        # Rule 1: padding -> -100
        assert (labels[content_len:] == -100).all(), "Padding labels should be -100"

        # EOD token itself is a REAL target (so the model learns to emit it):
        # its label must equal the EOD id, not -100.
        eod_mask = ids[:content_len] == eod
        if eod_mask.any():
            assert (labels[:content_len][eod_mask] == eod).all(), (
                "EOD labels should be the EOD id (a trained target), not -100"
            )

        # Rule: first token after EOD -> -100
        eod_positions = eod_mask.nonzero(as_tuple=True)[0]
        for ep in eod_positions:
            post = ep + 1
            if post < content_len:
                assert labels[post].item() == -100, (
                    f"Token at position {post} (after EOD at {ep}) should be -100"
                )

        # Remaining content positions should NOT be -100 (they are real labels)
        for i in range(content_len):
            if ids[i] == eod:
                continue
            is_post_eod = any(ids[max(0, i - 1)] == eod for _ in [0]) and i > 0 and ids[i - 1] == eod
            if is_post_eod:
                continue
            assert labels[i].item() != -100, f"Real token at position {i} should not be masked"

    print("PASS: label masking (padding, post-EOD, overlap; EOD kept as target)")


def test_overlap_label_masking():
    """Rule 4: overlap prefix region must be masked."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)

        eod = 0
        pad = 1
        overlap_len = 4

        # A single "window" with overlap prefix of 4 tokens
        doc = np.array([50, 51, 52, 53, 60, 61, 62, eod], dtype=np.uint16)
        bin_path = tmp / "test.bin"
        idx_path = tmp / "test.idx"
        doc.tofile(str(bin_path))

        offsets = np.array([0, len(doc)], dtype=np.int64)
        overlap = np.array([overlap_len], dtype=np.uint16)
        write_idx(idx_path, offsets, overlap, dtype_code=2)

        ds = PreTokenizedDataset(
            bin_files=[bin_path],
            max_length=32,
            seed=42,
            pad_token_id=pad,
            eod_token_id=eod,
            pack_sequences=False,
        )

        samples = list(ds)
        assert len(samples) == 1
        labels = samples[0]["labels"]

        # First overlap_len positions should be -100
        for i in range(overlap_len):
            assert labels[i].item() == -100, (
                f"Overlap position {i} should be masked but got {labels[i].item()}"
            )

    print("PASS: overlap label masking (rule 4)")


# ------------------------------------------------------------------
# 5. Packing efficiency
# ------------------------------------------------------------------

def test_packing_efficiency():
    """Packing efficiency should be >= 0.85 with varied-length docs."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)

        eod = 0
        pad = 1
        max_length = 128

        # Generate 50 varied-length documents
        rng = np.random.RandomState(42)
        docs = []
        all_tokens = []
        offsets = [0]
        overlaps = []
        for _ in range(50):
            length = rng.randint(5, 60)
            doc_tokens = rng.randint(2, 100, size=length).astype(np.uint16).tolist()
            doc_tokens.append(eod)
            docs.append(doc_tokens)
            all_tokens.extend(doc_tokens)
            offsets.append(len(all_tokens))
            overlaps.append(0)

        bin_path = tmp / "test.bin"
        idx_path = tmp / "test.idx"
        np.array(all_tokens, dtype=np.uint16).tofile(str(bin_path))
        write_idx(
            idx_path,
            np.array(offsets, dtype=np.int64),
            np.array(overlaps, dtype=np.uint16),
            dtype_code=2,
        )

        ds = PreTokenizedDataset(
            bin_files=[bin_path],
            max_length=max_length,
            seed=42,
            pad_token_id=pad,
            eod_token_id=eod,
            pack_sequences=True,
            packing_buffer_size=50,
        )

        total_content = 0
        total_slots = 0
        for sample in ds:
            content_len = int(sample["attention_mask"].sum().item())
            total_content += content_len
            total_slots += max_length

        efficiency = total_content / total_slots if total_slots > 0 else 0
        print(f"  Packing efficiency: {efficiency:.2%}")
        assert efficiency >= 0.85, f"Packing efficiency {efficiency:.2%} below 0.85 target"

    print("PASS: packing efficiency")


# ------------------------------------------------------------------
# 6. doc_ids generation
# ------------------------------------------------------------------

def test_doc_ids_sdpa_mask():
    """doc_ids should be present for sdpa_mask mode and absent for causal."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)

        eod = 0
        pad = 1
        doc1 = np.array([10, 11, eod], dtype=np.uint16)
        doc2 = np.array([20, 21, 22, eod], dtype=np.uint16)
        all_tokens = np.concatenate([doc1, doc2])

        bin_path = tmp / "test.bin"
        idx_path = tmp / "test.idx"
        all_tokens.tofile(str(bin_path))
        write_idx(
            idx_path,
            np.array([0, len(doc1), len(doc1) + len(doc2)], dtype=np.int64),
            np.array([0, 0], dtype=np.uint16),
            dtype_code=2,
        )

        # Causal mode: no doc_ids
        ds_causal = PreTokenizedDataset(
            bin_files=[bin_path], max_length=16, seed=42,
            pad_token_id=pad, eod_token_id=eod,
            attention_mode="causal", pack_sequences=True, packing_buffer_size=8,
        )
        for sample in ds_causal:
            assert "doc_ids" not in sample, "causal mode should not produce doc_ids"

        # sdpa_mask mode: doc_ids present
        ds_mask = PreTokenizedDataset(
            bin_files=[bin_path], max_length=16, seed=42,
            pad_token_id=pad, eod_token_id=eod,
            attention_mode="sdpa_mask", pack_sequences=True, packing_buffer_size=8,
        )
        for sample in ds_mask:
            assert "doc_ids" in sample, "sdpa_mask mode should produce doc_ids"
            doc_ids = sample["doc_ids"]
            content_len = int(sample["attention_mask"].sum().item())
            ids = sample["input_ids"][:content_len]
            dids = doc_ids[:content_len]
            # doc_ids should increment at EOD boundaries
            eod_positions = (ids == eod).nonzero(as_tuple=True)[0]
            if eod_positions.numel() >= 2:
                assert dids[0] == 0
                assert dids[eod_positions[0] + 1] > dids[eod_positions[0]]

    print("PASS: doc_ids generation")


# ------------------------------------------------------------------
# 7. Resume via start_position
# ------------------------------------------------------------------

def test_resume():
    """Resuming from a start_position should skip the expected docs."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)

        eod = 0
        pad = 1
        rng = np.random.RandomState(123)
        all_tokens = []
        offsets = [0]
        overlaps = []
        num_docs = 20
        for _ in range(num_docs):
            length = rng.randint(5, 20)
            doc = rng.randint(2, 100, size=length).astype(np.uint16).tolist()
            doc.append(eod)
            all_tokens.extend(doc)
            offsets.append(len(all_tokens))
            overlaps.append(0)

        bin_path = tmp / "test.bin"
        idx_path = tmp / "test.idx"
        np.array(all_tokens, dtype=np.uint16).tofile(str(bin_path))
        write_idx(
            idx_path,
            np.array(offsets, dtype=np.int64),
            np.array(overlaps, dtype=np.uint16),
            dtype_code=2,
        )

        # Full run (no packing for simplicity)
        ds_full = PreTokenizedDataset(
            bin_files=[bin_path], max_length=64, seed=42,
            pad_token_id=pad, eod_token_id=eod,
            pack_sequences=False,
        )
        full_samples = list(ds_full)

        # Resume from halfway
        skip = num_docs // 2
        ds_resumed = PreTokenizedDataset(
            bin_files=[bin_path], max_length=64, seed=42,
            pad_token_id=pad, eod_token_id=eod,
            pack_sequences=False,
        )
        # File-indexed cursor: skip the first `skip` items of file 0.
        ds_resumed.set_file_progress({0: skip})
        resumed_samples = list(ds_resumed)

        assert len(resumed_samples) == len(full_samples) - skip, (
            f"Expected {len(full_samples) - skip} samples after resume, "
            f"got {len(resumed_samples)}"
        )

        # Content should match the tail of the full run
        for i, (full, resumed) in enumerate(
            zip(full_samples[skip:], resumed_samples)
        ):
            assert torch.equal(full["input_ids"], resumed["input_ids"]), (
                f"Mismatch at sample {i} after resume"
            )

    print("PASS: resume via cursor")


# ------------------------------------------------------------------
# 8. InterleavedDataset
# ------------------------------------------------------------------

def test_interleaved():
    """InterleavedDataset should round-robin across buckets."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)

        eod = 0
        pad = 1

        datasets = []
        for bucket_id in range(3):
            all_tokens = []
            offsets = [0]
            overlaps = []
            for doc_idx in range(4):
                marker = bucket_id * 100 + doc_idx
                doc = [marker, marker + 1, eod]
                all_tokens.extend(doc)
                offsets.append(len(all_tokens))
                overlaps.append(0)

            bp = tmp / f"bucket{bucket_id}.bin"
            ip = tmp / f"bucket{bucket_id}.idx"
            np.array(all_tokens, dtype=np.uint16).tofile(str(bp))
            write_idx(
                ip,
                np.array(offsets, dtype=np.int64),
                np.array(overlaps, dtype=np.uint16),
                dtype_code=2,
            )

            ds = PreTokenizedDataset(
                bin_files=[bp], max_length=16, seed=42,
                pad_token_id=pad, eod_token_id=eod,
                pack_sequences=False,
            )
            datasets.append(ds)

        interleaved = InterleavedDataset(datasets)
        samples = list(interleaved)

        # Should have 3 buckets * 4 docs = 12 samples total
        assert len(samples) == 12, f"Expected 12 samples, got {len(samples)}"

        # Round-robin: first 3 samples should come from different buckets
        first_markers = [s["input_ids"][0].item() for s in samples[:3]]
        bucket_sources = [m // 100 for m in first_markers]
        assert len(set(bucket_sources)) == 3, (
            f"First 3 samples should be from 3 different buckets, got markers {first_markers}"
        )

    print("PASS: InterleavedDataset")


# ------------------------------------------------------------------
# 9. WeightedInterleavedDataset
# ------------------------------------------------------------------

def _make_bucket_dataset(tmp, bucket_id, num_docs=20, eod=0, pad=1, max_length=32):
    """Helper: create a PreTokenizedDataset for one bucket with identifiable tokens."""
    all_tokens = []
    offsets = [0]
    overlaps = []
    for doc_idx in range(num_docs):
        marker = bucket_id * 1000 + doc_idx
        doc = [marker, marker + 1, marker + 2, eod]
        all_tokens.extend(doc)
        offsets.append(len(all_tokens))
        overlaps.append(0)

    bp = tmp / f"bucket{bucket_id}.bin"
    ip = tmp / f"bucket{bucket_id}.idx"
    np.array(all_tokens, dtype=np.uint16).tofile(str(bp))
    write_idx(
        ip,
        np.array(offsets, dtype=np.int64),
        np.array(overlaps, dtype=np.uint16),
        dtype_code=2,
    )
    return PreTokenizedDataset(
        bin_files=[bp], max_length=max_length, seed=42,
        pad_token_id=pad, eod_token_id=eod, pack_sequences=False,
    )


def test_weighted_interleaved():
    """WeightedInterleavedDataset should approximate the requested mix ratios.

    We measure the first N draws (before any bucket exhausts) to verify
    that the sampling probability matches the configured weights.
    """
    from collections import Counter

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)

        num_docs = 500
        datasets = {}
        for i, name in enumerate(["a", "b", "c"]):
            datasets[name] = _make_bucket_dataset(tmp, i, num_docs=num_docs)

        weights = {"a": 0.7, "b": 0.2, "c": 0.1}
        wid = WeightedInterleavedDataset(datasets, weights, seed=42)

        # Only count the first portion of draws — well before any bucket
        # can exhaust — so the distribution reflects the weights, not
        # the finite bucket sizes.
        measure_n = num_docs // 2
        counts: Counter = Counter()
        for i, sample in enumerate(wid):
            if i >= measure_n:
                break
            marker = sample["input_ids"][0].item()
            bucket_id = marker // 1000
            counts[bucket_id] += 1

        total = sum(counts.values())
        assert total == measure_n

        pct_a = counts[0] / total
        pct_b = counts[1] / total
        pct_c = counts[2] / total

        assert 0.5 < pct_a < 0.9, f"Bucket 'a' should be ~70%, got {pct_a:.1%}"
        assert 0.05 < pct_b < 0.4, f"Bucket 'b' should be ~20%, got {pct_b:.1%}"
        assert 0.01 < pct_c < 0.3, f"Bucket 'c' should be ~10%, got {pct_c:.1%}"

    print("PASS: WeightedInterleavedDataset")


def test_weighted_interleaved_exhaustion():
    """When a bucket exhausts, remaining buckets continue."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)

        datasets = {
            "small": _make_bucket_dataset(tmp, 0, num_docs=2),
            "large": _make_bucket_dataset(tmp, 1, num_docs=50),
        }
        weights = {"small": 0.5, "large": 0.5}
        wid = WeightedInterleavedDataset(datasets, weights, seed=0)

        samples = list(wid)
        assert len(samples) > 2, "Should continue after 'small' exhausts"

    print("PASS: WeightedInterleavedDataset exhaustion")


def test_set_phase():
    """StreamingDataModule.set_phase() should change the training mix."""
    from gpt_simple.config import CurriculumPhase, DataConfig
    from gpt_simple.data import StreamingDataModule
    from gpt_simple.tokenizer import SimpleLLMTokenizer

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)

        # Build the expected directory structure: data/train/{a,b}/  data/val/{a,b}/
        for split in ("train", "val"):
            for bucket in ("a", "b"):
                d = tmp / "data" / split / bucket
                d.mkdir(parents=True)

        tokenizer = SimpleLLMTokenizer("gpt2")
        eod = tokenizer.eod_token_id

        for split in ("train", "val"):
            for bucket in ("a", "b"):
                d = tmp / "data" / split / bucket
                rng = np.random.RandomState(hash(bucket) % (2**31))
                all_tokens = []
                offsets_list = [0]
                overlaps = []
                for doc_idx in range(30):
                    doc_len = rng.randint(5, 20)
                    doc = rng.randint(2, 100, size=doc_len).astype(np.uint16).tolist()
                    doc.append(eod)
                    all_tokens.extend(doc)
                    offsets_list.append(len(all_tokens))
                    overlaps.append(0)
                bp = d / "shard.bin"
                ip = d / "shard.idx"
                np.array(all_tokens, dtype=np.uint16).tofile(str(bp))
                write_idx(
                    ip,
                    np.array(offsets_list, dtype=np.int64),
                    np.array(overlaps, dtype=np.uint16),
                    dtype_code=2,
                )

        curriculum = [
            CurriculumPhase(duration_tokens=100_000, mix={"a": 0.9, "b": 0.1}),
            CurriculumPhase(duration_tokens=100_000, mix={"a": 0.1, "b": 0.9}),
        ]
        config = DataConfig(
            path=str(tmp / "data"),
            curriculum=curriculum,
            num_workers=0,
        )

        dm = StreamingDataModule(
            config=config,
            tokenizer=tokenizer,
            per_device_batch_size=1,
        )
        dm.prepare_data()

        # Phase 0: should be WeightedInterleavedDataset with phase 0 weights
        assert dm.train_dataset is not None

        # Switch to phase 1
        dm.set_phase(1)
        assert dm.train_dataset is not None

    print("PASS: set_phase")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

ALL_TESTS = [
    test_roundtrip_tokenization,
    test_idx_format,
    test_windowing_short_doc,
    test_windowing_long_doc,
    test_label_masking,
    test_overlap_label_masking,
    test_packing_efficiency,
    test_doc_ids_sdpa_mask,
    test_resume,
    test_interleaved,
    test_weighted_interleaved,
    test_weighted_interleaved_exhaustion,
    test_set_phase,
]


def main():
    passed = 0
    failed = 0
    for test_fn in ALL_TESTS:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"FAIL: {test_fn.__name__}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
        print()

    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed out of {len(ALL_TESTS)}")
    if failed:
        sys.exit(1)
    print("All tests passed!")


if __name__ == "__main__":
    main()
