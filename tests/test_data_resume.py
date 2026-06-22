"""
Unit tests for the deterministic, topology-agnostic dataloader-resume protocol.

Covers:
  - ``PreTokenizedDataset`` emits ``_cursor`` (a ``PerBucketCursor`` with
    a ``file_progress`` dict) on every item, monotonically advancing.
  - Mid-shard and shard-boundary resume via ``set_file_progress``
    re-produces the exact same items.
  - ``WeightedInterleavedDataset`` is deterministic given
    ``(seed, worker_id, counter)``: bucket choices match bit-for-bit
    across runs.
  - Resuming a ``WeightedInterleavedDataset`` mid-stream produces an
    identical tail vs. a fresh single-shot run.
  - The cursor-aware collator strips per-item cursors and attaches the
    LAST one to the batch.
  - ``apply_dataloader_state`` accepts a list of per-rank dicts and:
      * same topology -> direct restore (counters preserved).
      * different topology -> unions file-progress, redistributes,
        counters restart at 0.  The exactly-once invariant holds.
  - ``CheckpointManager.load_all_dataloader_states`` reads every
    ``rank_*.pkl`` it finds and returns them as a list.

Tiny pretokenized shards are generated on the fly so the tests run in
a few hundred milliseconds without any external data.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import pytest
import torch

import gpt_simple.data as data_mod
from gpt_simple.data import (
    DATALOADER_STATE_SCHEMA_VERSION,
    DONE,
    PerBucketCursor,
    PreTokenizedDataset,
    StreamingDataModule,
    WeightedInterleavedDataset,
    WorkerDataState,
    _counter_rng,
    _merge_file_progress,
    cursor_aware_collate,
)


# ---------------------------------------------------------------------------
# Helpers to write tiny .bin/.idx shards in the format pretokenize.py uses.
# ---------------------------------------------------------------------------


def _write_shard(
    path_bin: Path,
    docs: List[List[int]],
    overlap_lengths: List[int],
    dtype_code: int = 2,  # 2 -> uint16
) -> None:
    """Write a (.bin, .idx) pair using the format in ``pretokenize.write_idx``."""
    from gpt_simple.pretokenize import write_idx

    np_dtype = np.uint16 if dtype_code == 2 else np.uint32

    flat = np.concatenate([np.asarray(d, dtype=np_dtype) for d in docs])
    flat.tofile(path_bin)

    offsets = np.zeros(len(docs) + 1, dtype=np.int64)
    cursor = 0
    for i, d in enumerate(docs):
        cursor += len(d)
        offsets[i + 1] = cursor

    idx_path = path_bin.with_suffix(".idx")
    write_idx(idx_path, offsets, np.asarray(overlap_lengths, dtype=np.uint16), dtype_code)


def _make_shard(tmp_path: Path, name: str, n_docs: int, doc_len: int = 8) -> Path:
    docs = []
    for i in range(n_docs):
        start = name.__hash__() & 0xFFF
        docs.append(list(range(start + i, start + i + doc_len)))
    overlaps = [0] * n_docs
    bin_path = tmp_path / f"{name}.bin"
    _write_shard(bin_path, docs, overlaps)
    return bin_path


def _materialise(ds, limit: int = 50) -> List[Dict]:
    """Run an iterable dataset and return up to ``limit`` items."""
    out = []
    for i, item in enumerate(ds):
        out.append(item)
        if i + 1 >= limit:
            break
    return out


def _items_equal(a: Dict, b: Dict) -> bool:
    if a is None or b is None:
        return a is b
    for k in a:
        if k == "_cursor":
            continue
        av = a.get(k)
        bv = b.get(k)
        if isinstance(av, torch.Tensor) and isinstance(bv, torch.Tensor):
            if not torch.equal(av, bv):
                return False
        else:
            if av != bv:
                return False
    return True


# ---------------------------------------------------------------------------
# PerBucketCursor / merge helpers
# ---------------------------------------------------------------------------


class TestMergeFileProgress:
    def test_merge_into_empty(self):
        dst = {}
        _merge_file_progress(dst, {1: 5, 2: 10})
        assert dst == {1: 5, 2: 10}

    def test_done_wins_over_count(self):
        dst = {1: 5}
        _merge_file_progress(dst, {1: DONE})
        assert dst[1] == DONE
        dst2 = {1: DONE}
        _merge_file_progress(dst2, {1: 5})
        assert dst2[1] == DONE

    def test_max_count_wins(self):
        dst = {1: 5}
        _merge_file_progress(dst, {1: 8})
        assert dst[1] == 8
        _merge_file_progress(dst, {1: 3})
        assert dst[1] == 8


# ---------------------------------------------------------------------------
# PreTokenizedDataset
# ---------------------------------------------------------------------------


@pytest.fixture
def small_dataset(tmp_path):
    """A PreTokenizedDataset with 2 shards × 12 docs of 8 tokens each."""
    a = _make_shard(tmp_path, "shardA", n_docs=12, doc_len=8)
    b = _make_shard(tmp_path, "shardB", n_docs=12, doc_len=8)
    return PreTokenizedDataset(
        bin_files=[a, b],
        max_length=32,
        seed=42,
        pad_token_id=0,
        eod_token_id=0,
        attention_mode="causal",
        pack_sequences=True,
        packing_buffer_size=4,
    )


def _fresh_dataset_like(src: PreTokenizedDataset) -> PreTokenizedDataset:
    return PreTokenizedDataset(
        bin_files=src.bin_files,
        max_length=src.max_length,
        seed=src.seed,
        pad_token_id=src.pad_token_id,
        eod_token_id=src.eod_token_id,
        attention_mode=src.attention_mode,
        pack_sequences=src.pack_sequences,
        packing_buffer_size=src.packing_buffer_size,
    )


class TestPreTokenizedCursors:
    def test_each_item_has_cursor(self, small_dataset):
        items = _materialise(small_dataset)
        assert len(items) > 0
        for it in items:
            assert "_cursor" in it
            assert isinstance(it["_cursor"], PerBucketCursor)
            assert isinstance(it["_cursor"].file_progress, dict)

    def test_cursor_monotone_within_file(self, small_dataset):
        items = _materialise(small_dataset)
        last_per_file: Dict[int, int] = {}
        for it in items:
            for fi, val in it["_cursor"].file_progress.items():
                prev = last_per_file.get(fi)
                if prev is None or prev == DONE:
                    last_per_file[fi] = val
                    continue
                # Monotone: either advance items, or transition to DONE.
                if val == DONE:
                    last_per_file[fi] = DONE
                else:
                    assert val >= prev, f"items emitted regressed for file {fi}"
                    last_per_file[fi] = val

    def test_resume_mid_shard_replays_tail(self, small_dataset):
        full = _materialise(small_dataset)
        assert len(full) >= 4
        cut = max(1, len(full) // 3)
        resume_progress = dict(full[cut - 1]["_cursor"].file_progress)

        ds2 = _fresh_dataset_like(small_dataset)
        ds2.set_file_progress(resume_progress)
        resumed = _materialise(ds2)

        assert len(resumed) == len(full) - cut
        for orig, new in zip(full[cut:], resumed):
            assert _items_equal(orig, new)

    def test_skipping_done_shard(self, small_dataset):
        """A DONE marker for a shard makes the iterator skip it entirely."""
        full = _materialise(small_dataset)
        assert any(it["_cursor"].file_progress.get(0) == DONE for it in full), (
            "expected first shard to be marked DONE somewhere in the stream"
        )
        # Pick the first item that comes from shard 1 (second shard).
        first_shard1_idx = None
        for i, it in enumerate(full):
            if it["_cursor"].file_progress.get(0) == DONE and 1 in it["_cursor"].file_progress:
                first_shard1_idx = i
                break
        assert first_shard1_idx is not None

        ds2 = _fresh_dataset_like(small_dataset)
        ds2.set_file_progress({0: DONE})
        resumed = _materialise(ds2)
        # Resumed must match the original tail starting at first_shard1_idx.
        for orig, new in zip(full[first_shard1_idx:], resumed):
            assert _items_equal(orig, new)

    def test_fresh_iteration_is_deterministic(self, small_dataset):
        a = _materialise(small_dataset)
        b = _materialise(small_dataset)
        assert len(a) == len(b)
        for x, y in zip(a, b):
            assert _items_equal(x, y)

    def test_exactly_once_round_trip(self, small_dataset):
        """Split a full run into two halves; union must equal the full run."""
        full = _materialise(small_dataset)
        cut = len(full) // 2
        # Phase 1: emulate run 1 — yield first cut items, capture cursor.
        first_half = full[:cut]
        progress_after_cut = dict(full[cut - 1]["_cursor"].file_progress)

        # Phase 2: fresh dataset with that cursor — should yield exactly the tail.
        ds2 = _fresh_dataset_like(small_dataset)
        ds2.set_file_progress(progress_after_cut)
        second_half = _materialise(ds2)

        # Combine and verify against the unsplit run.
        combined = first_half + second_half
        assert len(combined) == len(full)
        for a, b in zip(combined, full):
            assert _items_equal(a, b)


# ---------------------------------------------------------------------------
# Counter-based RNG
# ---------------------------------------------------------------------------


class TestCounterRng:
    def test_same_inputs_same_output(self):
        r1 = _counter_rng(42, 0, 100)
        r2 = _counter_rng(42, 0, 100)
        assert r1.random() == r2.random()

    def test_different_counter_different_output(self):
        r1 = _counter_rng(42, 0, 100).random()
        r2 = _counter_rng(42, 0, 101).random()
        assert r1 != r2

    def test_different_worker_different_output(self):
        r1 = _counter_rng(42, 0, 100).random()
        r2 = _counter_rng(42, 1, 100).random()
        assert r1 != r2


# ---------------------------------------------------------------------------
# WeightedInterleavedDataset
# ---------------------------------------------------------------------------


def _build_weighted(tmp_path, n_docs_per_bucket=12) -> WeightedInterleavedDataset:
    buckets = {}
    for name in ("web", "code", "math"):
        bin_path = _make_shard(tmp_path / name, name, n_docs=n_docs_per_bucket)
        buckets[name] = PreTokenizedDataset(
            bin_files=[bin_path],
            max_length=32,
            seed=42,
            pad_token_id=0,
            eod_token_id=0,
            pack_sequences=True,
            packing_buffer_size=4,
        )
    data_mod.BUCKET_TO_ID, data_mod.ID_TO_BUCKET = data_mod.build_bucket_mappings(list(buckets.keys()))
    weights = {"web": 0.5, "code": 0.3, "math": 0.2}
    return WeightedInterleavedDataset(buckets, weights, seed=123)


def _prepare_bucket_dirs(tmp_path):
    for name in ("web", "code", "math"):
        (tmp_path / name).mkdir(exist_ok=True)


class TestWeightedInterleaved:
    def test_each_item_carries_cursor(self, tmp_path):
        _prepare_bucket_dirs(tmp_path)
        wid = _build_weighted(tmp_path)
        items = _materialise(wid, limit=10)
        for it in items:
            assert "_cursor" in it
            wid_id, ws = it["_cursor"]
            assert wid_id == 0
            assert isinstance(ws, WorkerDataState)
            for cur in ws.bucket_cursors.values():
                assert isinstance(cur, PerBucketCursor)

    def test_deterministic_bucket_sequence(self, tmp_path):
        _prepare_bucket_dirs(tmp_path)
        wid1 = _build_weighted(tmp_path)
        wid2 = _build_weighted(tmp_path)
        a = _materialise(wid1, limit=20)
        b = _materialise(wid2, limit=20)
        assert [int(it["bucket_id"]) for it in a] == [int(it["bucket_id"]) for it in b]

    def test_counter_monotone(self, tmp_path):
        _prepare_bucket_dirs(tmp_path)
        wid = _build_weighted(tmp_path)
        items = _materialise(wid, limit=15)
        counters = [it["_cursor"][1].counter for it in items]
        assert counters == sorted(counters)
        assert counters[0] == 1
        assert all(b - a == 1 for a, b in zip(counters, counters[1:]))

    def test_resume_from_midstream_matches_tail(self, tmp_path):
        _prepare_bucket_dirs(tmp_path)
        wid1 = _build_weighted(tmp_path)
        full = _materialise(wid1, limit=20)
        cut = len(full) // 2
        _, mid_state = full[cut - 1]["_cursor"]

        wid2 = _build_weighted(tmp_path)
        wid2.set_worker_state(0, 0, mid_state)
        resumed = _materialise(wid2, limit=20)

        for orig, new in zip(full[cut:], resumed):
            assert int(orig["bucket_id"]) == int(new["bucket_id"])
            assert torch.equal(orig["input_ids"], new["input_ids"])

    def test_exhausted_bucket_remembered_in_cursor(self, tmp_path):
        _prepare_bucket_dirs(tmp_path)
        wid = _build_weighted(tmp_path)
        items = _materialise(wid, limit=200)
        if not items:
            pytest.skip("dataset trivially empty")
        _, last_state = items[-1]["_cursor"]
        assert len(last_state.exhausted_buckets) >= 1


# ---------------------------------------------------------------------------
# Cursor-aware collator
# ---------------------------------------------------------------------------


class TestCursorAwareCollate:
    def _make_item(self, val: int, cursor) -> Dict:
        return {
            "input_ids": torch.tensor([val, val, val], dtype=torch.long),
            "labels": torch.tensor([val, val, -100], dtype=torch.long),
            "_cursor": cursor,
        }

    def test_strips_per_item_cursors(self):
        items = [
            self._make_item(1, "cursor_A"),
            self._make_item(2, "cursor_B"),
        ]
        batch = cursor_aware_collate(items)
        assert isinstance(batch["input_ids"], torch.Tensor)
        assert batch["input_ids"].shape == (2, 3)
        assert batch["_cursor"] == "cursor_B"

    def test_empty_input(self):
        assert cursor_aware_collate([]) == {}

    def test_default_collate_compatible(self):
        items = [
            {"input_ids": torch.tensor([1, 2]), "_cursor": "x"},
            {"input_ids": torch.tensor([3, 4]), "_cursor": "y"},
        ]
        batch = cursor_aware_collate(items)
        assert torch.equal(batch["input_ids"], torch.tensor([[1, 2], [3, 4]]))


# ---------------------------------------------------------------------------
# StreamingDataModule.apply_dataloader_state
# ---------------------------------------------------------------------------


def _make_data_module(tmp_path, num_workers=0, world_size=1, n_docs=64, n_shards=1):
    """Build a minimal StreamingDataModule with one bucket on disk."""
    for split in ("train", "val"):
        bdir = tmp_path / split / "web"
        bdir.mkdir(parents=True, exist_ok=True)
        for s in range(n_shards):
            _make_shard(bdir, f"shard_{s}", n_docs=n_docs)

    from gpt_simple.config import DataConfig

    class _Tok:
        pad_token_id = 0
        eod_token_id = 0
        vocab_size = 100

    cfg = DataConfig(
        path=str(tmp_path),
        tokenizer="gpt2",
        format="pretokenized",
        max_length=32,
        overlap_size=4,
        num_workers=num_workers,
    )
    dm = StreamingDataModule(
        config=cfg,
        tokenizer=_Tok(),
        world_size=world_size,
        attention_mode="causal",
        per_device_batch_size=2,
    )
    dm.prepare_data()
    return dm


class TestApplyDataloaderState:
    def test_apply_same_topology_restores_counter(self, tmp_path):
        dm = _make_data_module(tmp_path)
        state = dm.make_dataloader_state(
            worker_states={
                0: WorkerDataState(
                    seed=42, counter=5,
                    bucket_cursors={"web": PerBucketCursor(file_progress={0: 3})},
                )
            },
            rank=0,
        )
        dm.apply_dataloader_state([state])
        wid = dm.train_dataset
        assert (0, 0) in wid.worker_states
        assert wid.worker_states[(0, 0)].counter == 5
        # Underlying dataset has the file progress merged.
        assert wid.datasets["web"].file_progress.get(0) == 3

    def test_apply_different_topology_does_not_restore_counter(self, tmp_path, caplog):
        dm = _make_data_module(tmp_path)
        # State saved as if it came from a 2x1 topology (world=2, workers=1).
        state = {
            "schema_version": DATALOADER_STATE_SCHEMA_VERSION,
            "world_size": 2,
            "num_workers": 1,
            "rank": 0,
            "worker_states": {
                0: WorkerDataState(
                    seed=42, counter=7,
                    bucket_cursors={"web": PerBucketCursor(file_progress={0: 4})},
                )
            },
        }
        with caplog.at_level(logging.INFO, logger="gpt_simple"):
            dm.apply_dataloader_state([state])
        assert "topology change" in caplog.text.lower()
        wid = dm.train_dataset
        # No per-slot counter restoration:
        assert (0, 0) not in wid.worker_states
        # But file progress IS applied (exactly-once preserved):
        assert wid.datasets["web"].file_progress.get(0) == 4

    def test_apply_with_wrong_schema_warns(self, tmp_path, caplog):
        dm = _make_data_module(tmp_path)
        state = {
            "schema_version": 99,
            "world_size": 1,
            "num_workers": 1,
            "rank": 0,
            "worker_states": {},
        }
        with caplog.at_level(logging.WARNING, logger="gpt_simple"):
            dm.apply_dataloader_state([state])
        assert "schema" in caplog.text.lower()

    def test_apply_empty_is_noop(self, tmp_path):
        dm = _make_data_module(tmp_path)
        dm.apply_dataloader_state(None)
        dm.apply_dataloader_state([])
        assert not dm.train_dataset.worker_states

    def test_make_and_apply_round_trip(self, tmp_path):
        dm = _make_data_module(tmp_path)
        original = {
            0: WorkerDataState(
                seed=42, counter=17,
                bucket_cursors={"web": PerBucketCursor(file_progress={0: 11})},
            )
        }
        state = dm.make_dataloader_state(original, rank=0)
        assert state["schema_version"] == DATALOADER_STATE_SCHEMA_VERSION
        assert state["rank"] == 0
        assert state["world_size"] == 1
        assert state["num_workers"] == 1
        dm.apply_dataloader_state([state])
        assert dm.train_dataset.worker_states[(0, 0)].counter == 17
        assert dm.train_dataset.datasets["web"].file_progress[0] == 11

    def test_state_records_effective_not_requested_workers(self, tmp_path):
        """Saved topology records the effective (post-clamp) worker count."""
        # 2 train shards / world_size=1 -> max_safe = 2 -> clamp 4 to 2.
        dm = _make_data_module(tmp_path, num_workers=4, n_shards=2, world_size=1)
        assert dm._effective_num_workers == 2
        state = dm.make_dataloader_state(
            worker_states={0: WorkerDataState(seed=42, counter=1)}, rank=0,
        )
        assert state["num_workers"] == 2


# ---------------------------------------------------------------------------
# Topology-change exactly-once invariant
#
# We simulate "run 1" with N old slots and a partial save, then "run 2"
# with M != N new slots that resumes from that state.  We verify that
# the UNION of items yielded across run 1 + run 2 equals the items
# yielded by a single continuous run with no topology change — no
# document is processed twice and no document is missed.
# ---------------------------------------------------------------------------


def _build_pretok_ds(tmp_path, name="web", n_shards=4, n_docs=32) -> PreTokenizedDataset:
    d = tmp_path / name
    d.mkdir(parents=True, exist_ok=True)
    paths = [_make_shard(d, f"shard_{i}", n_docs=n_docs) for i in range(n_shards)]
    return PreTokenizedDataset(
        bin_files=paths,
        max_length=32,
        seed=42,
        pad_token_id=0,
        eod_token_id=0,
        pack_sequences=True,
        packing_buffer_size=4,
    )


def _materialise_slot(ds: PreTokenizedDataset, rank: int, world_size: int, worker_id: int, num_workers: int) -> List[Dict]:
    """Materialise the items a given slot would yield, simulating worker_info."""
    # Mock get_worker_info so the dataset thinks it's running in worker_id of num_workers.
    from unittest.mock import patch

    class _WI:
        def __init__(self, wid, nw):
            self.id = wid
            self.num_workers = nw

    with patch("torch.utils.data.get_worker_info", return_value=_WI(worker_id, num_workers)):
        # Mock distributed so rank/world_size are honoured.  We simulate
        # by patching at the module-level _slot_info function we use.
        with patch.object(data_mod, "_slot_info", return_value=(rank, world_size, worker_id, num_workers)):
            return list(ds)


def _hash_items(items: List[Dict]) -> List[bytes]:
    """A deterministic content-hash per item, for set equality checks."""
    out = []
    for it in items:
        h = it["input_ids"].numpy().tobytes() + b"|" + it["labels"].numpy().tobytes()
        out.append(h)
    return out


class TestExactlyOnceUnderTopologyChange:
    """Run 1 with W1 slots ⊕ Run 2 with W2 slots == continuous run with any topology."""

    @pytest.mark.parametrize("old_topology,new_topology", [
        # (world_size, num_workers) tuples
        ((1, 1), (1, 2)),  # grow workers
        ((1, 2), (1, 1)),  # shrink workers
        ((2, 1), (1, 2)),  # rebalance world<->workers
        ((1, 4), (2, 2)),  # grow world, shrink workers
    ])
    def test_split_then_resume_matches_continuous(self, tmp_path, old_topology, new_topology):
        old_w, old_nw = old_topology
        new_w, new_nw = new_topology

        # --- Continuous baseline: one slot owns everything (1, 1) ----------
        ds_baseline = _build_pretok_ds(tmp_path / "baseline", n_shards=4, n_docs=24)
        baseline_items: List[Dict] = []
        for r in range(1):
            for w in range(1):
                baseline_items.extend(_materialise_slot(ds_baseline, r, 1, w, 1))
        baseline_hashes = set(_hash_items(baseline_items))

        # --- Run 1 with OLD topology: materialise EVERY slot's stream
        # up to a per-slot "cut point" (half of what they would emit).
        ds_run1 = _build_pretok_ds(tmp_path / "run1", n_shards=4, n_docs=24)
        run1_items: List[Dict] = []
        run1_progress_per_bucket: Dict[int, int] = {}  # file_idx -> items_emitted
        for r in range(old_w):
            for w in range(old_nw):
                slot_items = _materialise_slot(ds_run1, r, old_w, w, old_nw)
                cut = max(1, len(slot_items) // 2)
                taken = slot_items[:cut]
                run1_items.extend(taken)
                if taken:
                    # Merge the last item's file_progress into our running map.
                    last_cursor = taken[-1]["_cursor"]
                    _merge_file_progress(run1_progress_per_bucket, last_cursor.file_progress)

        # --- Run 2 with NEW topology: inject the run-1 progress and let
        # every new slot finish iteration.
        ds_run2 = _build_pretok_ds(tmp_path / "run2", n_shards=4, n_docs=24)
        ds_run2.set_file_progress(run1_progress_per_bucket)
        run2_items: List[Dict] = []
        for r in range(new_w):
            for w in range(new_nw):
                slot_items = _materialise_slot(ds_run2, r, new_w, w, new_nw)
                run2_items.extend(slot_items)

        # --- Exactly-once check ----------------------------------------
        combined_hashes = _hash_items(run1_items) + _hash_items(run2_items)
        # No duplicates (no document processed twice)
        assert len(combined_hashes) == len(set(combined_hashes)), (
            "exactly-once violated: some items appear more than once"
        )
        # All baseline items are covered (no document missed)
        assert set(combined_hashes) == baseline_hashes, (
            f"coverage mismatch: missing {len(baseline_hashes - set(combined_hashes))} items, "
            f"extra {len(set(combined_hashes) - baseline_hashes)} items"
        )


# ---------------------------------------------------------------------------
# CheckpointManager.load_all_dataloader_states
# ---------------------------------------------------------------------------


class TestLoadAllDataloaderStates:
    def test_reads_every_rank_file(self, tmp_path):
        from gpt_simple._checkpoint import CheckpointManager

        ckpt = tmp_path / "checkpoint-1"
        (ckpt / "dataloader_state").mkdir(parents=True)
        for r in range(3):
            state = {
                "schema_version": DATALOADER_STATE_SCHEMA_VERSION,
                "world_size": 3,
                "num_workers": 1,
                "rank": r,
                "worker_states": {0: WorkerDataState(seed=42, counter=r + 1)},
            }
            torch.save(state, ckpt / "dataloader_state" / f"rank_{r}.pkl")

        states = CheckpointManager.load_all_dataloader_states(ckpt)
        assert len(states) == 3
        assert {s["rank"] for s in states} == {0, 1, 2}

    def test_empty_when_missing(self, tmp_path):
        from gpt_simple._checkpoint import CheckpointManager

        ckpt = tmp_path / "checkpoint-1"
        ckpt.mkdir()
        assert CheckpointManager.load_all_dataloader_states(ckpt) == []

    def test_falls_back_to_legacy_single_file(self, tmp_path):
        from gpt_simple._checkpoint import CheckpointManager

        ckpt = tmp_path / "checkpoint-1"
        ckpt.mkdir()
        legacy_state = {
            "schema_version": DATALOADER_STATE_SCHEMA_VERSION,
            "world_size": 1,
            "num_workers": 1,
            "rank": 0,
            "worker_states": {},
        }
        torch.save(legacy_state, ckpt / "dataloader_state.pt")
        states = CheckpointManager.load_all_dataloader_states(ckpt)
        assert len(states) == 1


# ---------------------------------------------------------------------------
# End-to-end via a DataLoader (single-process)
# ---------------------------------------------------------------------------


class TestDataLoaderResume:
    def test_dataloader_cursor_round_trip(self, tmp_path):
        dm = _make_data_module(tmp_path, num_workers=0)
        loader = dm.train_dataloader()

        original_batches = []
        for i, batch in enumerate(loader):
            original_batches.append(batch)
            if i + 1 >= 4:
                break

        cursor_pair = original_batches[1]["_cursor"]
        assert cursor_pair is not None
        worker_id, state = cursor_pair

        dm2 = _make_data_module(tmp_path, num_workers=0)
        dm2.apply_dataloader_state([
            dm2.make_dataloader_state({worker_id: state}, rank=0)
        ])

        loader2 = dm2.train_dataloader()
        resumed_batches = []
        for i, batch in enumerate(loader2):
            resumed_batches.append(batch)
            if i + 1 >= 2:
                break

        assert torch.equal(
            original_batches[2]["input_ids"],
            resumed_batches[0]["input_ids"],
        )


# ---------------------------------------------------------------------------
# val_dataloader collation over the pretokenized path
# ---------------------------------------------------------------------------


def _make_multibucket_dm(tmp_path, buckets=("web", "code", "math"), num_workers=0,
                         n_docs=32, n_shards=2):
    """A pretokenized StreamingDataModule with several buckets (train + val)."""
    for split in ("train", "val"):
        for b in buckets:
            bdir = tmp_path / split / b
            bdir.mkdir(parents=True, exist_ok=True)
            for s in range(n_shards):
                _make_shard(bdir, f"shard_{s}", n_docs=n_docs)

    from gpt_simple.config import DataConfig

    class _Tok:
        pad_token_id = 0
        eod_token_id = 0
        vocab_size = 100

    cfg = DataConfig(
        path=str(tmp_path),
        tokenizer="gpt2",
        format="pretokenized",
        max_length=32,
        overlap_size=4,
        num_workers=num_workers,
    )
    dm = StreamingDataModule(
        config=cfg,
        tokenizer=_Tok(),
        world_size=1,
        attention_mode="causal",
        per_device_batch_size=2,
    )
    dm.prepare_data()
    return dm


class TestValDataloaderCollate:
    def test_multibucket_val_dataloader_collates(self, tmp_path):
        """Multi-bucket val (InterleavedDataset) yields collatable batches."""
        dm = _make_multibucket_dm(tmp_path, buckets=("web", "code", "math"))
        loader = dm.val_dataloader()
        batch = next(iter(loader))
        assert isinstance(batch, dict)
        assert isinstance(batch["input_ids"], torch.Tensor)
        assert batch["input_ids"].ndim == 2

    def test_single_bucket_val_dataloader_collates(self, tmp_path):
        """Single-bucket val (bare PreTokenizedDataset) yields collatable batches."""
        dm = _make_multibucket_dm(tmp_path, buckets=("web",))
        loader = dm.val_dataloader()
        batch = next(iter(loader))
        assert isinstance(batch["input_ids"], torch.Tensor)
        assert batch["input_ids"].ndim == 2

    def test_val_batch_is_eval_consumable(self, tmp_path):
        """After stripping ``_cursor``/``bucket_id``, every batch field is a tensor."""
        dm = _make_multibucket_dm(tmp_path, buckets=("web", "code"))
        loader = dm.val_dataloader()
        batch = dict(next(iter(loader)))
        if "_cursor" in batch:
            assert not isinstance(batch["_cursor"], list)
        # Mirror _evaluate: strip non-tensor metadata before forward.
        batch.pop("_cursor", None)
        batch.pop("bucket_id", None)
        assert batch, "batch has no tensor fields"
        for key, value in batch.items():
            assert isinstance(value, torch.Tensor), (
                f"batch[{key!r}] is {type(value).__name__}, not a tensor"
            )


# ---------------------------------------------------------------------------
# shard_by_rank: training shards across ranks, validation does not
# ---------------------------------------------------------------------------


class TestValNotRankSharded:
    def test_shard_by_rank_false_every_rank_sees_full_set(self, tmp_path):
        """shard_by_rank=False: every rank yields the identical stream."""
        d = tmp_path / "val"
        d.mkdir(parents=True)
        paths = [_make_shard(d, f"shard_{i}", n_docs=16) for i in range(4)]

        def _ds():
            return PreTokenizedDataset(
                bin_files=paths, max_length=32, seed=43,
                pad_token_id=0, eod_token_id=0,
                pack_sequences=True, packing_buffer_size=4,
                shard_by_rank=False,
            )

        r0 = _materialise_slot(_ds(), rank=0, world_size=4, worker_id=0, num_workers=1)
        r3 = _materialise_slot(_ds(), rank=3, world_size=4, worker_id=0, num_workers=1)
        assert len(r0) == len(r3) > 0
        assert _hash_items(r0) == _hash_items(r3)

    def test_shard_by_rank_true_partitions_across_ranks(self, tmp_path):
        """shard_by_rank=True: ranks get disjoint shards."""
        d = tmp_path / "train"
        d.mkdir(parents=True)
        paths = [_make_shard(d, f"shard_{i}", n_docs=16) for i in range(4)]

        def _ds():
            return PreTokenizedDataset(
                bin_files=paths, max_length=32, seed=42,
                pad_token_id=0, eod_token_id=0,
                pack_sequences=True, packing_buffer_size=4,
            )

        r0 = set(_hash_items(_materialise_slot(_ds(), 0, 4, 0, 1)))
        r1 = set(_hash_items(_materialise_slot(_ds(), 1, 4, 0, 1)))
        assert r0 and r1
        assert r0.isdisjoint(r1)
