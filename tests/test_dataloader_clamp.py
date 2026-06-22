"""
Regression tests for under-sharded-bucket safety in StreamingDataModule.

Background
----------

``PreTokenizedDataset._assigned_file_indices`` distributes shards across
``world_size * num_workers`` slots using load-balanced bin-packing.  When
the total number of slots exceeds the number of shards, later slots get
ZERO shards.  When ALL slots of a given rank are empty, that rank's
DataLoader raises ``StopIteration`` on the very first batch.  The
trainer's coordinated-stop all-reduce (``MAX``) then propagates that
"exhausted" flag to every rank, and the entire job stops at step 0.

This was the failure mode we saw on the 8x V100 smoke test:
8 ranks * 2 workers = 16 slots, only 8 shards, so 8 slots were empty
and ranks 4..7 had no data.

``StreamingDataModule._prepare_pretokenized`` now:

  - Raises ``DataError`` when ``num_shards < world_size`` (no clamp can
    rescue a rank that has zero shards).
  - Silently clamps ``num_workers`` down to ``num_shards // world_size``
    when slots would otherwise outnumber shards (with a warning log).
  - Stores the effective value on ``self._effective_num_workers`` so
    ``train_dataloader`` builds the DataLoader with the safe count.

This file exercises all three paths without spinning up CUDA / actually
running training.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pytest

from gpt_simple.config import DataConfig
from gpt_simple.data import StreamingDataModule
from gpt_simple.errors import DataError
from gpt_simple.pretokenize import write_idx
from gpt_simple.tokenizer import SimpleLLMTokenizer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_tiny_shard(
    bin_path: Path, idx_path: Path, num_docs: int, doc_len: int, seed: int
) -> None:
    """Write a deterministic .bin / .idx pair with ``num_docs`` documents."""
    rng = np.random.RandomState(seed)
    tokens = []
    offsets = [0]
    overlaps = []
    for _ in range(num_docs):
        d = rng.randint(2, 100, size=doc_len).astype(np.uint16).tolist()
        tokens.extend(d)
        offsets.append(len(tokens))
        overlaps.append(0)
    np.array(tokens, dtype=np.uint16).tofile(str(bin_path))
    write_idx(
        idx_path,
        np.array(offsets, dtype=np.int64),
        np.array(overlaps, dtype=np.uint16),
        dtype_code=2,
    )


def _build_data_dir(root: Path, n_train_shards: int, n_val_shards: int = 1) -> Path:
    """Build ``root/{train,val}/default/shard_*.{bin,idx}`` tree."""
    for split, n in [("train", n_train_shards), ("val", n_val_shards)]:
        d = root / split / "default"
        d.mkdir(parents=True, exist_ok=True)
        for s in range(n):
            _write_tiny_shard(
                d / f"shard_{s:04d}.bin",
                d / f"shard_{s:04d}.idx",
                num_docs=10,
                doc_len=16,
                seed=hash((split, s)) & 0x7FFFFFFF,
            )
    return root


def _make_dm(data_path: Path, *, num_workers: int, world_size: int):
    """Build a StreamingDataModule with the given dist topology."""
    tokenizer = SimpleLLMTokenizer("gpt2")
    cfg = DataConfig(
        path=str(data_path),
        format="pretokenized",
        num_workers=num_workers,
        max_length=64,
        overlap_size=8,
        packing=False,
    )
    return StreamingDataModule(
        config=cfg,
        tokenizer=tokenizer,
        world_size=world_size,
        per_device_batch_size=1,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestUnderShardedBucket:
    """Behavior when a train bucket has fewer shards than (world_size * nw)."""

    def test_num_workers_clamped_when_undersharded(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Config asks for nw=4 with world_size=2 and 4 shards.

        16 slots > 4 shards would leave 12 slots empty.  We expect the
        module to clamp ``num_workers`` to ``4 // 2 == 2`` and log a
        warning.
        """
        data = _build_data_dir(tmp_path / "data", n_train_shards=4)
        dm = _make_dm(data, num_workers=4, world_size=2)
        with caplog.at_level(logging.WARNING, logger="gpt_simple"):
            dm.prepare_data()

        assert dm._effective_num_workers == 2
        assert any(
            "Clamping num_workers from 4 to 2" in r.getMessage()
            for r in caplog.records
        ), f"expected clamp warning, got: {[r.getMessage() for r in caplog.records]}"

        # The DataLoader factory should pick up the clamped value.
        dl = dm.train_dataloader()
        assert dl.num_workers == 2

    def test_num_workers_unchanged_when_sufficient_shards(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """16 shards across world_size=4 means each rank has 4 shards;
        num_workers=2 is fine (each worker gets 2)."""
        data = _build_data_dir(tmp_path / "data", n_train_shards=16)
        dm = _make_dm(data, num_workers=2, world_size=4)
        with caplog.at_level(logging.WARNING, logger="gpt_simple"):
            dm.prepare_data()

        assert dm._effective_num_workers == 2
        assert not any(
            "Clamping num_workers" in r.getMessage() for r in caplog.records
        )
        dl = dm.train_dataloader()
        assert dl.num_workers == 2

    def test_undersharded_raises_when_fewer_shards_than_ranks(
        self, tmp_path: Path
    ) -> None:
        """4 shards across world_size=8 is hopeless — at least one rank
        must have zero shards.  Should fail loudly at prepare_data."""
        data = _build_data_dir(tmp_path / "data", n_train_shards=4)
        dm = _make_dm(data, num_workers=2, world_size=8)
        with pytest.raises(DataError, match="needs at least one shard"):
            dm.prepare_data()

    def test_num_workers_clamps_to_at_least_one(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Edge: 2 shards, world_size=2, num_workers=2.
        ``num_shards // world_size == 1``, so num_workers should clamp to 1
        (not 0 — we want at least one worker if any was requested)."""
        data = _build_data_dir(tmp_path / "data", n_train_shards=2)
        dm = _make_dm(data, num_workers=2, world_size=2)
        with caplog.at_level(logging.WARNING, logger="gpt_simple"):
            dm.prepare_data()
        assert dm._effective_num_workers == 1
        dl = dm.train_dataloader()
        assert dl.num_workers == 1

    def test_zero_num_workers_stays_zero(self, tmp_path: Path) -> None:
        """``num_workers=0`` is a valid main-process-only configuration —
        the clamp must not turn it into 1."""
        data = _build_data_dir(tmp_path / "data", n_train_shards=4)
        dm = _make_dm(data, num_workers=0, world_size=2)
        dm.prepare_data()
        assert dm._effective_num_workers == 0
        dl = dm.train_dataloader()
        assert dl.num_workers == 0
