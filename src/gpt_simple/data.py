"""
Data loading for gpt_simple.

Contains:
  - ``PreTokenizedDataset`` — mmap-backed IterableDataset over ``.bin/.idx`` shards
  - ``InterleavedDataset`` — round-robin interleaving of multiple IterableDatasets
  - ``WeightedInterleavedDataset`` — weighted mixing with counter-based PRNG for
    deterministic resumability
  - ``StreamingDataModule`` — top-level data module that creates loaders from a
    ``DataConfig`` (pretokenized or JSONL fallback)

Resume protocol
---------------
The pretokenized path supports deterministic resume of the training data
stream.  Every item produced carries a ``_cursor`` field describing the
state of the dataset AFTER that item was emitted.  The cursor is a small
serialisable structure (no tensors); the cursor-aware collator
(``cursor_aware_collate``) strips per-item cursors and attaches the last
one to the batch.  The training loop captures these cursors and commits
them to ``dataloader_state/rank_{N}.pkl`` at checkpoint time.

On resume, ``StreamingDataModule.apply_dataloader_state(state)`` injects
the cursors back into the datasets, which then start the next iteration
from the saved positions.  Any batches that had been prefetched but not
yet consumed are silently re-created; the cursor guarantees no document
is consumed twice.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset

from gpt_simple.config import DataConfig
from gpt_simple.errors import DataError
from gpt_simple.pretokenize import read_idx
from gpt_simple.tokenizer import SimpleLLMTokenizer

logger = logging.getLogger("gpt_simple")


# ---------------------------------------------------------------------------
# Bucket name <-> ID mappings (shared with _streaming.py)
# ---------------------------------------------------------------------------

BUCKET_TO_ID: Dict[str, int] = {}
ID_TO_BUCKET: Dict[int, str] = {}


def build_bucket_mappings(bucket_names: List[str]) -> Tuple[Dict[str, int], Dict[int, str]]:
    bucket_to_id = {name: idx for idx, name in enumerate(sorted(bucket_names))}
    id_to_bucket = {idx: name for name, idx in bucket_to_id.items()}
    return bucket_to_id, id_to_bucket


# ---------------------------------------------------------------------------
# Cursor dataclasses — small, serialisable, no tensors.
# ---------------------------------------------------------------------------

DATALOADER_STATE_SCHEMA_VERSION = 2

# Sentinel inside ``PerBucketCursor.file_progress`` meaning "this file has
# been fully consumed; skip on resume".  Any non-negative value means
# "this many items have been emitted from this file".
DONE = -1


@dataclass
class PerBucketCursor:
    """Per-file progress for one bucket.

    ``file_progress`` maps ``file_idx`` (index into ``bin_files``) to:
      - ``DONE`` (``-1``) when the file has been fully consumed.
      - A non-negative integer = items already emitted from that file.

    Files **absent** from the dict are entirely untouched.  This format
    is **topology-agnostic**: a file's identity does not depend on which
    rank or worker happened to process it in a previous run, so the
    cursor remains valid under any change of ``world_size`` or
    ``num_workers``.  On resume, each new slot inspects its assigned
    files and looks them up in this dict; whatever progress was made
    in a previous run is preserved exactly.
    """

    file_progress: Dict[int, int] = field(default_factory=dict)


@dataclass
class WorkerDataState:
    """Resume state for one (rank, worker_id) of WeightedInterleavedDataset.

    Contains both per-slot fields (``counter``, ``exhausted_buckets``) and
    the slot's contribution to per-bucket file progress
    (``bucket_cursors``).  The slot-level fields are only meaningful
    under the same topology; the file-progress is topology-agnostic and
    gets unioned across all slots at load time.
    """

    seed: int = 42
    counter: int = 0
    bucket_cursors: Dict[str, PerBucketCursor] = field(default_factory=dict)
    exhausted_buckets: List[str] = field(default_factory=list)


def _slot_info() -> Tuple[int, int, int, int]:
    """Return ``(rank, world_size, worker_id, num_workers)``.

    Works both inside and outside DataLoader workers, and with or
    without ``torch.distributed`` initialised.
    """
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            rank = dist.get_rank()
            world_size = dist.get_world_size()
        else:
            rank, world_size = 0, 1
    except ImportError:
        rank, world_size = 0, 1

    worker_info = torch.utils.data.get_worker_info()
    if worker_info is not None:
        return rank, world_size, int(worker_info.id), int(worker_info.num_workers)
    return rank, world_size, 0, 1


def _merge_file_progress(dst: Dict[int, int], src: Dict[int, int]) -> None:
    """Merge ``src`` file-progress entries into ``dst`` in-place.

    Merge rules:
      - ``DONE`` (``-1``) wins over any positive count.
      - Otherwise, the larger items-emitted count wins.
    """
    for fi, val in src.items():
        cur = dst.get(fi)
        if cur is None:
            dst[fi] = val
        elif cur == DONE or val == DONE:
            dst[fi] = DONE
        else:
            dst[fi] = max(cur, val)


def _worker_state_from_dict(d: Dict[str, Any]) -> WorkerDataState:
    """Reconstruct a ``WorkerDataState`` from a plain-dict form (e.g. JSON)."""
    bc_raw = d.get("bucket_cursors") or {}
    bc: Dict[str, PerBucketCursor] = {}
    for name, c in bc_raw.items():
        if isinstance(c, PerBucketCursor):
            bc[name] = c
        else:
            fp_raw = c.get("file_progress") or {}
            fp = {int(k): int(v) for k, v in fp_raw.items()}
            bc[name] = PerBucketCursor(file_progress=fp)
    return WorkerDataState(
        seed=int(d.get("seed", 42)),
        counter=int(d.get("counter", 0)),
        bucket_cursors=bc,
        exhausted_buckets=list(d.get("exhausted_buckets") or []),
    )


def _counter_rng(seed: int, worker_id: int, counter: int) -> random.Random:
    """Stateless PRNG seeded by (seed, worker_id, counter).

    Used by :class:`WeightedInterleavedDataset` so the choice of bucket
    at any point in the stream depends only on the counter — making the
    sequence reproducible across runs/resumes.
    """
    mix = (seed * 0x9E3779B97F4A7C15) ^ (worker_id * 0xBF58476D1CE4E5B9) ^ (counter * 0x94D049BB133111EB)
    return random.Random(mix & 0xFFFFFFFFFFFFFFFF)


# ---------------------------------------------------------------------------
# PreTokenizedDataset
# ---------------------------------------------------------------------------

class PreTokenizedDataset(IterableDataset):
    """IterableDataset over pre-tokenized binary shards.

    Parameters
    ----------
    bin_files : list of Path
        Paths to ``.bin`` files.  Each must have a matching ``.idx`` sibling.
    max_length : int
        Target sequence length (including padding).
    seed : int
        Base seed for deterministic shuffling.
    pad_token_id : int
        Token id used for padding.
    eod_token_id : int
        End-of-document token id (used for label masking).
    attention_mode : str
        ``"causal"`` skips ``doc_ids`` computation.
    pack_sequences : bool
        When *True* (default), multiple documents are packed into each sequence.
    packing_buffer_size : int
        Number of documents to buffer for first-fit-decreasing packing.
    shard_by_rank : bool
        Partition shards across distributed ranks (default).  Set to
        *False* for validation so every rank iterates the full set.

    Resume protocol
    ---------------
    The dataset holds a *global* ``file_progress`` map (file_idx ->
    items_emitted, or ``DONE``) populated at resume time via
    :meth:`set_file_progress` or :meth:`update_file_progress`.  The map
    contains entries for every file touched in any previous run,
    regardless of which slot processed it — making the format invariant
    under topology changes.

    Each worker, when iterating, filters this map down to its own
    assigned shards.  For every shard the worker plans to visit:
      - if marked ``DONE``, skip entirely;
      - if marked with N items, re-run the packer and skip the first N
        items (deterministic — same shuffle + same packing);
      - otherwise (not in the map), process from scratch.

    Each yielded item carries a ``_cursor`` field of type
    :class:`PerBucketCursor` whose ``file_progress`` reflects the
    *slot's contribution* up to and including that item.  Cursors are
    stripped before model forward (see ``cursor_aware_collate``).
    """

    def __init__(
        self,
        bin_files: List[Path],
        max_length: int,
        seed: int,
        pad_token_id: int,
        eod_token_id: int,
        attention_mode: str = "causal",
        pack_sequences: bool = True,
        packing_buffer_size: int = 256,
        shard_by_rank: bool = True,
    ):
        super().__init__()
        self.bin_files = sorted(bin_files)
        self.max_length = max_length
        self.seed = seed
        self.pad_token_id = pad_token_id
        self.eod_token_id = eod_token_id
        self.attention_mode = attention_mode
        self.pack_sequences = pack_sequences
        self.packing_buffer_size = packing_buffer_size
        self.shard_by_rank = shard_by_rank

        # Global per-file progress (file_idx -> items_emitted or DONE).
        # All workers see this same dict (via pickling); each filters by
        # its assigned shards at iteration time.  Empty = start fresh.
        self.file_progress: Dict[int, int] = {}

        self._idx_cache: Dict[int, Tuple[int, np.ndarray, np.ndarray]] = {}
        for i, bf in enumerate(self.bin_files):
            idx_path = bf.with_suffix(".idx")
            if not idx_path.exists():
                raise FileNotFoundError(f"Missing .idx file for {bf}")
            self._idx_cache[i] = read_idx(idx_path)

    def set_file_progress(self, progress: Dict[int, int]) -> None:
        """Replace the global file-progress map with ``progress``."""
        self.file_progress = {int(k): int(v) for k, v in progress.items()}

    def update_file_progress(self, progress: Dict[int, int]) -> None:
        """Merge ``progress`` into the global map (DONE wins; else max)."""
        _merge_file_progress(self.file_progress, {int(k): int(v) for k, v in progress.items()})

    # -- helpers ------------------------------------------------------------

    def _assigned_file_indices(self, rank: int, world_size: int, worker_id: int, num_workers: int) -> List[int]:
        """Return the file indices this slot is responsible for.

        Uses load-balanced bin-packing across ``world_size * num_workers``
        slots based on total token counts.  Deterministic given the same
        ``(world_size, num_workers, bin_files)``.
        """
        total_slots = world_size * num_workers
        global_slot = rank * num_workers + worker_id

        shard_sizes = []
        for i in range(len(self.bin_files)):
            _, offsets, _ = self._idx_cache[i]
            shard_sizes.append((offsets[-1], i))

        shard_sizes.sort(reverse=True)

        slot_totals = [0] * total_slots
        slot_assignments: List[List[int]] = [[] for _ in range(total_slots)]
        for token_count, shard_idx in shard_sizes:
            lightest = min(range(total_slots), key=lambda s: slot_totals[s])
            slot_assignments[lightest].append(shard_idx)
            slot_totals[lightest] += token_count

        return sorted(slot_assignments[global_slot])

    def _resolve_slot(self) -> Tuple[int, int, int, int]:
        """Slot info honouring ``shard_by_rank``.

        With ``shard_by_rank=False`` the rank/world_size are forced to
        ``(0, 1)``; worker-level sharding still applies.
        """
        rank, world_size, worker_id, num_workers = _slot_info()
        if not self.shard_by_rank:
            rank, world_size = 0, 1
        return rank, world_size, worker_id, num_workers

    def _get_worker_file_indices(self) -> List[int]:
        rank, world_size, worker_id, num_workers = self._resolve_slot()
        return self._assigned_file_indices(rank, world_size, worker_id, num_workers)

    def _shuffled_doc_indices(self, num_docs: int, file_idx: int) -> np.ndarray:
        # No epoch term: the pipeline visits each shard once.
        rng = np.random.RandomState(seed=hash((self.seed, file_idx)) % (2**31))
        indices = np.arange(num_docs)
        rng.shuffle(indices)
        return indices

    def _read_doc_tokens(self, mmap_arr: np.ndarray, offsets: np.ndarray, doc_idx: int) -> np.ndarray:
        return mmap_arr[offsets[doc_idx] : offsets[doc_idx + 1]]

    # -- label masking ------------------------------------------------------

    def _finalize_sequence(
        self,
        token_lists: List[np.ndarray],
        overlap_lens: List[int],
    ) -> Dict[str, torch.Tensor]:
        """Build the batch dict for one packed sequence.

        Label-masking rules:
          1. Padding positions -> -100
          2. First token after each EOD -> -100
          3. Overlap prefix regions -> -100

        The EOD token itself is deliberately left as a real target: with the
        model's ``shift_labels = labels[1:]`` convention, keeping ``labels`` at
        the EOD index is what teaches the model to *predict* EOD after a
        document's last content token (i.e. to terminate generation).  Masking
        it here would zero that gradient and the model would never learn to
        stop.
        """
        tokens = np.concatenate(token_lists)
        content_len = len(tokens)
        if content_len > self.max_length:
            tokens = tokens[: self.max_length]
            content_len = self.max_length

        padded = np.full(self.max_length, self.pad_token_id, dtype=np.int64)
        padded[:content_len] = tokens

        input_ids = torch.from_numpy(padded.copy())
        attention_mask = torch.zeros(self.max_length, dtype=torch.long)
        attention_mask[:content_len] = 1
        labels = input_ids.clone()

        labels[content_len:] = -100

        valid = input_ids[:content_len]
        eod_positions = (valid == self.eod_token_id).nonzero(as_tuple=True)[0]
        if eod_positions.numel() > 0:
            # NOTE: do NOT mask labels[eod_positions] — the EOD token must stay a
            # real target so the model learns to emit it (see docstring). We only
            # mask the first token *after* each EOD: with packing that token is
            # the start of an unrelated next document, so predicting it from EOD
            # is noise.
            post_eod = eod_positions + 1
            post_eod = post_eod[post_eod < content_len]
            if post_eod.numel() > 0:
                labels[post_eod] = -100

        pos = 0
        for tok_arr, ovl in zip(token_lists, overlap_lens):
            if ovl > 0:
                end = min(pos + ovl, content_len)
                labels[pos:end] = -100
            pos += len(tok_arr)
            if pos >= self.max_length:
                break

        position_ids = torch.zeros(self.max_length, dtype=torch.long)
        position_ids[:content_len] = torch.arange(content_len, dtype=torch.long)

        result: Dict[str, torch.Tensor] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "position_ids": position_ids,
        }

        if self.attention_mode != "causal":
            doc_ids = torch.zeros(self.max_length, dtype=torch.long)
            eod_mask = input_ids[:content_len] == self.eod_token_id
            if eod_mask.any():
                eod_indices = eod_mask.nonzero(as_tuple=True)[0].tolist()
                doc_id = 0
                start = 0
                for eod_idx in eod_indices:
                    doc_ids[start : eod_idx + 1] = doc_id
                    doc_id += 1
                    start = eod_idx + 1
                if start < content_len:
                    doc_ids[start:content_len] = doc_id
            result["doc_ids"] = doc_ids

        return result

    # -- packing ------------------------------------------------------------

    def _pack_and_yield(
        self,
        doc_iter: Iterator[Tuple[np.ndarray, int]],
    ) -> Iterator[Dict[str, torch.Tensor]]:
        buffer: List[Tuple[np.ndarray, int]] = []

        def _drain_buffer():
            if not buffer:
                return None
            sorted_buf = sorted(buffer, key=lambda x: len(x[0]), reverse=True)
            sequences: List[Tuple[List[np.ndarray], List[int], int]] = []
            for doc_tokens, ovl in sorted_buf:
                doc_len = len(doc_tokens)
                if doc_len > self.max_length:
                    doc_tokens = doc_tokens[: self.max_length]
                    doc_len = self.max_length
                placed = False
                for i, (tok_lists, ovl_list, cur_len) in enumerate(sequences):
                    if cur_len + doc_len <= self.max_length:
                        tok_lists.append(doc_tokens)
                        ovl_list.append(ovl)
                        sequences[i] = (tok_lists, ovl_list, cur_len + doc_len)
                        placed = True
                        break
                if not placed:
                    sequences.append(([doc_tokens], [ovl], doc_len))
            return sequences

        for doc_tokens, ovl in doc_iter:
            buffer.append((doc_tokens, ovl))
            if len(buffer) >= self.packing_buffer_size:
                packed = _drain_buffer()
                buffer.clear()
                if packed:
                    for tok_lists, ovl_list, _ in packed:
                        yield self._finalize_sequence(tok_lists, ovl_list)

        if buffer:
            packed = _drain_buffer()
            buffer.clear()
            if packed:
                for tok_lists, ovl_list, _ in packed:
                    yield self._finalize_sequence(tok_lists, ovl_list)

    def _sequential_yield(
        self,
        doc_iter: Iterator[Tuple[np.ndarray, int]],
    ) -> Iterator[Dict[str, torch.Tensor]]:
        for doc_tokens, ovl in doc_iter:
            yield self._finalize_sequence([doc_tokens], [ovl])

    # -- main iteration -----------------------------------------------------

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        rank, world_size, worker_id, num_workers = self._resolve_slot()
        file_indices = self._assigned_file_indices(rank, world_size, worker_id, num_workers)

        # Slot-local view of progress (restricted to this slot's files).
        # Updated in-place as items are emitted; stamped on each item so
        # the trainer / WID downstream can observe it.
        live: Dict[int, int] = {
            fi: int(self.file_progress[fi])
            for fi in file_indices
            if fi in self.file_progress
        }

        for fi in file_indices:
            p = live.get(fi, 0)
            if p == DONE:
                continue
            skip_n = max(0, p)

            bin_path = self.bin_files[fi]
            dtype_code, offsets, overlap_lengths = self._idx_cache[fi]

            num_docs = len(overlap_lengths)
            if num_docs == 0:
                # Empty shard (0-byte .bin / 0-document .idx): nothing to read.
                # np.memmap cannot map an empty file, so skip before opening it.
                live[fi] = DONE
                continue

            np_dtype = np.uint16 if dtype_code == 2 else np.uint32
            mmap_arr = np.memmap(str(bin_path), dtype=np_dtype, mode="r")

            shuffled = self._shuffled_doc_indices(num_docs, fi)

            def doc_iter(mmap_arr=mmap_arr):
                for idx in shuffled:
                    doc_tokens = self._read_doc_tokens(mmap_arr, offsets, idx)
                    ovl = int(overlap_lengths[idx])
                    yield doc_tokens, ovl

            if self.pack_sequences:
                inner = self._pack_and_yield(doc_iter())
            else:
                inner = self._sequential_yield(doc_iter())

            # Re-pack from scratch then skip the first `skip_n` items —
            # packing is deterministic so the skipped items match exactly
            # what the previous run produced.
            skipped = 0
            items_emitted = skip_n
            for item in inner:
                if skipped < skip_n:
                    skipped += 1
                    continue
                items_emitted += 1
                live[fi] = items_emitted
                # Snapshot post-yield state for this slot.
                item["_cursor"] = PerBucketCursor(file_progress=dict(live))
                yield item

            del mmap_arr
            # The shard's inner iterator is exhausted: mark it DONE.
            # (We won't stamp this on any item from this shard, but if
            # any subsequent shard yields, that item's cursor will carry
            # the DONE.  If no later shard yields, the previous item's
            # cursor stays at items_emitted == total — which on resume
            # causes the packer to re-run and find 0 items left, i.e.
            # equivalent to DONE.)
            live[fi] = DONE


# ---------------------------------------------------------------------------
# InterleavedDataset
# ---------------------------------------------------------------------------

class InterleavedDataset(IterableDataset):
    """Round-robin interleaving of multiple IterableDataset instances."""

    def __init__(self, datasets: List[IterableDataset]):
        super().__init__()
        self.datasets = datasets

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        iterators = [iter(ds) for ds in self.datasets]
        active = list(range(len(iterators)))
        while active:
            next_active = []
            for i in active:
                try:
                    yield next(iterators[i])
                    next_active.append(i)
                except StopIteration:
                    continue
            active = next_active


# ---------------------------------------------------------------------------
# WeightedInterleavedDataset
# ---------------------------------------------------------------------------

class WeightedInterleavedDataset(IterableDataset):
    """Sample from per-bucket datasets according to weighted mix ratios.

    Uses a *counter-based PRNG* so the choice of bucket at any point in
    the stream depends only on ``(seed, worker_id, counter)``.  This
    makes the sequence reproducible across resumes: the trainer just
    needs to restore ``counter`` and the set of active buckets to
    replay the exact same sequence.

    When a bucket exhausts its data, it is removed and the remaining
    weights are renormalised.  Iteration stops when all buckets are
    exhausted.
    """

    def __init__(
        self,
        datasets: Dict[str, IterableDataset],
        weights: Dict[str, float],
        seed: int = 42,
    ):
        super().__init__()
        self.datasets = datasets
        self.weights = weights
        self.seed = seed
        # Per-slot resume state (counter + exhausted_buckets only — the
        # file-level progress lives on each underlying PreTokenizedDataset
        # and is topology-agnostic).
        self.worker_states: Dict[Tuple[int, int], WorkerDataState] = {}

    def set_worker_state(self, rank: int, worker_id: int, state: WorkerDataState) -> None:
        """Inject per-slot resume state for one ``(rank, worker_id)`` pair.

        The slot-level fields (``counter``, ``exhausted_buckets``) are
        stored on the WID; the per-bucket ``file_progress`` entries are
        merged into the corresponding ``PreTokenizedDataset`` 's global
        progress map.
        """
        key = (int(rank), int(worker_id))
        self.worker_states[key] = state
        for bname, bcursor in state.bucket_cursors.items():
            ds = self.datasets.get(bname)
            if isinstance(ds, PreTokenizedDataset):
                ds.update_file_progress(bcursor.file_progress)

    def reset_all_progress(self) -> None:
        """Wipe all in-memory cursor state (used before re-applying)."""
        self.worker_states.clear()
        for ds in self.datasets.values():
            if isinstance(ds, PreTokenizedDataset):
                ds.set_file_progress({})

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        rank, _, worker_id, _ = _slot_info()
        saved = self.worker_states.get((rank, worker_id))

        all_buckets = {
            name: ds for name, ds in self.datasets.items()
            if name in self.weights and self.weights[name] > 0
        }
        if not all_buckets:
            return

        if saved is not None:
            already_exhausted = set(saved.exhausted_buckets)
            counter = int(saved.counter)
        else:
            already_exhausted = set()
            counter = 0

        iterators: Dict[str, Iterator] = {}
        for name, ds in all_buckets.items():
            if name in already_exhausted:
                continue
            iterators[name] = iter(ds)

        active_weights = {k: self.weights[k] for k in iterators}

        # Outgoing cursor state.  Per-bucket file_progress here is the
        # SLOT's contribution — only entries for files this slot has
        # touched in the current iteration (or carried over from saved
        # state).  The union across slots happens at load time.
        state = WorkerDataState(
            seed=int(self.seed),
            counter=counter,
            bucket_cursors={
                k: PerBucketCursor(file_progress=dict(v.file_progress))
                for k, v in (saved.bucket_cursors.items() if saved else [])
            },
            exhausted_buckets=list(already_exhausted),
        )

        while iterators:
            names = list(active_weights.keys())
            w = [active_weights[n] for n in names]

            rng = _counter_rng(state.seed, worker_id, state.counter)
            chosen = rng.choices(names, weights=w, k=1)[0]

            try:
                item = next(iterators[chosen])
            except StopIteration:
                del iterators[chosen]
                del active_weights[chosen]
                state.exhausted_buckets.append(chosen)
                # The drained bucket is recorded on ``exhausted_buckets`` (and
                # so propagates to the trainer via the per-item cursor).  The
                # training loop decides what to do with that signal — halt or
                # renormalise — based on ``data.allow_bucket_exhaustion``; see
                # ``run_training_loop``.
                # Counter NOT incremented — same counter picks again next loop.
                continue

            # Pull the bucket's per-item cursor (stamped by PreTokenizedDataset).
            bucket_cursor = item.pop("_cursor", None)
            if isinstance(bucket_cursor, PerBucketCursor):
                state.bucket_cursors[chosen] = bucket_cursor

            state.counter += 1

            if chosen in BUCKET_TO_ID:
                item["bucket_id"] = torch.tensor(BUCKET_TO_ID[chosen], dtype=torch.long)

            # Stamp a fresh snapshot on the item (deep-copy of file_progress
            # so subsequent mutations don't bleed back into yielded items).
            item["_cursor"] = (worker_id, WorkerDataState(
                seed=state.seed,
                counter=state.counter,
                bucket_cursors={
                    k: PerBucketCursor(file_progress=dict(v.file_progress))
                    for k, v in state.bucket_cursors.items()
                },
                exhausted_buckets=list(state.exhausted_buckets),
            ))
            yield item


# ---------------------------------------------------------------------------
# Cursor-aware collation
# ---------------------------------------------------------------------------


def cursor_aware_collate(batch_items: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Default-collate tensor fields and forward the last item's ``_cursor``.

    Each item produced by :class:`WeightedInterleavedDataset` carries a
    ``_cursor`` field which is a ``(worker_id, WorkerDataState)``
    tuple.  We:

      1. Pop ``_cursor`` from every item (it would break ``default_collate``).
      2. ``default_collate`` the remaining (tensor) fields.
      3. Attach the LAST item's cursor to the batch.

    The batch's ``_cursor`` thus describes the dataset state immediately
    after the last item of this batch was emitted — exactly what the
    trainer needs to record at checkpoint time.
    """
    if not batch_items:
        return {}

    cursors = [item.pop("_cursor", None) for item in batch_items]
    batch = torch.utils.data.default_collate(batch_items)
    if isinstance(batch, dict):
        batch["_cursor"] = cursors[-1]
    return batch


# ---------------------------------------------------------------------------
# StreamingDataModule
# ---------------------------------------------------------------------------

def _detect_rank() -> Tuple[int, bool]:
    """Return (rank, is_main_process)."""
    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            rank = dist.get_rank()
            return rank, rank == 0
    except ImportError:
        pass
    return 0, True


class StreamingDataModule:
    """Top-level data module that creates DataLoaders from a ``DataConfig``.

    Supports two data formats:
      - ``"pretokenized"`` (default): mmap-backed .bin/.idx shards
      - ``"jsonl"``: legacy on-the-fly tokenization (delegates to ``_streaming.py``)

    When curriculum is configured, :meth:`set_phase` swaps the training dataset
    to a :class:`WeightedInterleavedDataset` with the new phase's mix weights.
    """

    def __init__(
        self,
        config: DataConfig,
        tokenizer: SimpleLLMTokenizer,
        world_size: int = 1,
        attention_mode: str = "causal",
        per_device_batch_size: int = 4,
    ):
        self.config = config
        self.tokenizer = tokenizer
        self.world_size = world_size
        self.attention_mode = attention_mode
        self.per_device_batch_size = per_device_batch_size
        self.train_dataset: Optional[IterableDataset] = None
        self.val_dataset: Optional[IterableDataset] = None

        self._bucket_train_datasets: Dict[str, PreTokenizedDataset] = {}
        self._bucket_val_datasets: Dict[str, PreTokenizedDataset] = {}
        self._bucket_names: List[str] = []

        # Populated by ``_prepare_pretokenized``: the number of DataLoader
        # workers that's actually safe to use given the per-bucket shard
        # counts.  We may clamp below ``config.num_workers`` so that every
        # (rank, worker) slot is non-empty — otherwise an empty slot raises
        # StopIteration immediately and (via the MAX all-reduce of the
        # exhaustion flag in train.py) collapses the whole job at step 0.
        self._effective_num_workers: Optional[int] = None

    def prepare_data(self) -> None:
        """Build datasets from ``self.config.path``."""
        if self.config.format == "pretokenized":
            self._prepare_pretokenized()
        elif self.config.format == "jsonl":
            self._prepare_jsonl()
        else:
            raise DataError(f"Unknown data format: {self.config.format!r}")

    # -- pretokenized -------------------------------------------------------

    def _prepare_pretokenized(self) -> None:
        data_path = Path(self.config.path)
        _, is_main = _detect_rank()

        train_dir = data_path / "train"
        val_dir = data_path / "val"

        for d, label in [(data_path, "dataset"), (train_dir, "train"), (val_dir, "val")]:
            if not d.is_dir():
                raise DataError(f"Expected {label} directory not found: {d}")

        bucket_names = sorted(p.name for p in train_dir.iterdir() if p.is_dir())
        if not bucket_names:
            raise DataError(f"No bucket subdirectories found in {train_dir}")
        self._bucket_names = bucket_names

        # Validate curriculum bucket names against discovered directories
        if self.config.curriculum is not None:
            curriculum_buckets: set[str] = set()
            for phase in self.config.curriculum:
                curriculum_buckets.update(phase.mix.keys())
            missing = curriculum_buckets - set(bucket_names)
            if missing:
                raise DataError(
                    f"Curriculum references buckets not found on disk: {sorted(missing)}. "
                    f"Discovered buckets: {bucket_names}"
                )
            unused = set(bucket_names) - curriculum_buckets
            if unused and is_main:
                logger.warning(
                    f"Buckets discovered but never used in curriculum: {sorted(unused)}"
                )

        global BUCKET_TO_ID, ID_TO_BUCKET
        BUCKET_TO_ID, ID_TO_BUCKET = build_bucket_mappings(bucket_names)

        # Collect bin_files per (split, bucket) first so we can validate
        # the shard-count vs. world_size constraint before constructing
        # any PreTokenizedDataset instance.
        bin_files_by_split_bucket: Dict[Tuple[str, str], List[Path]] = {}
        for bucket_name in bucket_names:
            for split, parent in [("train", train_dir), ("val", val_dir)]:
                bucket_dir = parent / bucket_name
                if not bucket_dir.is_dir():
                    raise DataError(
                        f"Expected bucket directory not found: {bucket_dir}\n"
                        f"Structure must be: {data_path}/{split}/{bucket_name}/"
                    )
                bin_files = sorted(bucket_dir.glob("*.bin"))
                if not bin_files:
                    raise DataError(
                        f"No .bin files found in {split} bucket: {bucket_dir}"
                    )
                bin_files_by_split_bucket[(split, bucket_name)] = bin_files

        # --- Slot-vs-shard validation + num_workers auto-clamp -------------
        # PreTokenizedDataset._assigned_file_indices distributes shards
        # across ``world_size * num_workers`` slots using a load-balanced
        # bin-packing.  If there are fewer shards than slots, later slots
        # get nothing and StopIteration fires immediately at step 0.  Two
        # cases to handle:
        #   1. num_shards < world_size — a rank gets zero shards no matter
        #      what we do with num_workers.  Hard error.
        #   2. world_size <= num_shards < world_size * num_workers — we
        #      can rescue this by reducing num_workers per rank to
        #      ``num_shards // world_size``.  Silent clamp + info log.
        # We apply the strictest constraint across all (train) buckets so
        # the worker count is uniform across buckets / phases.
        train_shard_counts = [
            len(bin_files_by_split_bucket[("train", b)]) for b in bucket_names
        ]
        min_train_shards = min(train_shard_counts)
        if min_train_shards < self.world_size:
            shortest = bucket_names[train_shard_counts.index(min_train_shards)]
            raise DataError(
                f"Training bucket {shortest!r} has only {min_train_shards} "
                f"shard(s) but world_size={self.world_size}.  Every rank "
                f"needs at least one shard or it will exhaust at step 0 "
                f"and (via the coordinated-stop all-reduce) crash the run. "
                f"Either reduce the number of GPUs or split this bucket "
                f"into more shards via `gpt-simple tokenize ...`."
            )

        requested_num_workers = max(0, int(self.config.num_workers))
        if requested_num_workers > 0:
            max_safe_workers = min_train_shards // self.world_size
            effective = min(requested_num_workers, max(1, max_safe_workers))
            if effective < requested_num_workers and is_main:
                shortest = bucket_names[
                    train_shard_counts.index(min_train_shards)
                ]
                logger.warning(
                    f"Clamping num_workers from {requested_num_workers} to "
                    f"{effective} so every (rank, worker) slot has data "
                    f"(bucket {shortest!r} has {min_train_shards} shards "
                    f"across world_size={self.world_size}).  Add more "
                    f"shards or reduce num_workers in the config to silence "
                    f"this warning."
                )
        else:
            effective = 0
        self._effective_num_workers = effective

        # Now actually build the per-bucket datasets.
        for bucket_name in bucket_names:
            for split, dest in [
                ("train", self._bucket_train_datasets),
                ("val", self._bucket_val_datasets),
            ]:
                bin_files = bin_files_by_split_bucket[(split, bucket_name)]
                ds = PreTokenizedDataset(
                    bin_files=bin_files,
                    max_length=self.config.max_length,
                    seed=42 if split == "train" else 43,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eod_token_id=self.tokenizer.eod_token_id,
                    attention_mode=self.attention_mode,
                    pack_sequences=self.config.packing,
                    shard_by_rank=(split == "train"),
                )
                dest[bucket_name] = ds
                if is_main:
                    logger.info(f"  {split}/{bucket_name}: {len(bin_files)} shard(s)")

        # Build initial train dataset
        if self.config.curriculum is not None:
            initial_weights = self.config.curriculum[0].mix
        else:
            w = 1.0 / len(bucket_names)
            initial_weights = {b: w for b in bucket_names}
        self.train_dataset = WeightedInterleavedDataset(
            self._bucket_train_datasets, initial_weights,
        )

        # Validation always uses uniform round-robin
        val_list = list(self._bucket_val_datasets.values())
        self.val_dataset = InterleavedDataset(val_list) if len(val_list) > 1 else val_list[0]

        if is_main:
            logger.info(f"Pretokenized datasets ready ({len(bucket_names)} buckets)")

    # -- curriculum phase transitions ----------------------------------------

    def set_phase(self, phase_idx: int) -> None:
        """Rebuild the training dataset with the mix weights from *phase_idx*.

        The caller is responsible for creating a new DataLoader afterwards via
        :meth:`train_dataloader` and ``accelerator.prepare()``.
        """
        if self.config.curriculum is None:
            return
        if phase_idx < 0 or phase_idx >= len(self.config.curriculum):
            raise DataError(
                f"Invalid phase index {phase_idx} "
                f"(curriculum has {len(self.config.curriculum)} phases)"
            )
        mix = self.config.curriculum[phase_idx].mix
        self.train_dataset = WeightedInterleavedDataset(
            self._bucket_train_datasets, mix,
        )

    # -- jsonl (legacy) -----------------------------------------------------

    def _prepare_jsonl(self) -> None:
        """Delegate to the legacy streaming pipeline in ``_streaming.py``."""
        if self.config.curriculum is not None:
            raise DataError(
                "Curriculum is only supported with format='pretokenized'. "
                "Pre-tokenize your data first: gpt-simple tokenize ..."
            )

        from gpt_simple._streaming import (
            CombinedStreamingDataset,
            SequentialBucketDataset,
            build_bucket_mappings as _build,
        )

        data_path = Path(self.config.path)
        _, is_main = _detect_rank()

        train_dir = data_path / "train"
        val_dir = data_path / "val"

        for d, label in [(data_path, "dataset"), (train_dir, "train"), (val_dir, "val")]:
            if not d.is_dir():
                raise DataError(f"Expected {label} directory not found: {d}")

        if is_main:
            logger.info(f"Setting up JSONL streaming datasets from {self.config.path}")

        bucket_names = sorted(p.name for p in train_dir.iterdir() if p.is_dir())
        if not bucket_names:
            raise DataError(f"No bucket subdirectories found in {train_dir}")
        self._bucket_names = bucket_names

        global BUCKET_TO_ID, ID_TO_BUCKET
        BUCKET_TO_ID, ID_TO_BUCKET = _build(bucket_names)

        bucket_train_files: Dict[str, List[Path]] = {}
        bucket_val_files: Dict[str, List[Path]] = {}

        for bucket_name in bucket_names:
            for split, parent, dest in [
                ("train", train_dir, bucket_train_files),
                ("val", val_dir, bucket_val_files),
            ]:
                bucket_dir = parent / bucket_name
                if not bucket_dir.is_dir():
                    raise DataError(
                        f"Expected bucket directory not found: {bucket_dir}\n"
                        f"Structure must be: {data_path}/{split}/{bucket_name}/"
                    )
                files = sorted(bucket_dir.glob("*.jsonl"))
                if not files:
                    raise DataError(f"No .jsonl files found in {split} bucket: {bucket_dir}")
                dest[bucket_name] = files

        self.train_dataset = CombinedStreamingDataset(
            bucket_to_files=bucket_train_files,
            tokenizer=self.tokenizer,
            max_length=self.config.max_length,
            seed=42,
            world_size=self.world_size,
            attention_mode=self.attention_mode,
            overlap_size=self.config.overlap_size,
        )
        self.val_dataset = SequentialBucketDataset(
            bucket_to_files=bucket_val_files,
            tokenizer=self.tokenizer,
            max_length=self.config.max_length,
            seed=43,
            attention_mode=self.attention_mode,
            overlap_size=self.config.overlap_size,
        )

        if is_main:
            logger.info(f"JSONL streaming datasets ready ({len(bucket_names)} buckets)")

    # -- dataloaders --------------------------------------------------------

    def _resolved_num_workers(self) -> int:
        """DataLoader worker count actually in effect (post-clamp)."""
        if self._effective_num_workers is not None:
            return self._effective_num_workers
        return self.config.num_workers

    def train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise DataError("Call prepare_data() before train_dataloader()")
        # The cursor-aware collator is only meaningful for the pretokenized
        # path; the JSONL legacy path emits plain tensor dicts that
        # ``default_collate`` already handles.
        collate_fn = (
            cursor_aware_collate
            if isinstance(self.train_dataset, WeightedInterleavedDataset)
            else None
        )
        n_workers = self._resolved_num_workers()
        return DataLoader(
            self.train_dataset,
            batch_size=self.per_device_batch_size,
            num_workers=n_workers,
            persistent_workers=n_workers > 0,
            prefetch_factor=4 if n_workers > 0 else None,
            pin_memory=True,
            drop_last=True,
            collate_fn=collate_fn,
        )

    def val_dataloader(self) -> DataLoader:
        if self.val_dataset is None:
            raise DataError("Call prepare_data() before val_dataloader()")
        # Pretokenized items carry per-sample cursors; JSONL items do not.
        collate_fn = (
            cursor_aware_collate
            if self.config.format == "pretokenized"
            else None
        )
        val_workers = min(2, self._resolved_num_workers())
        return DataLoader(
            self.val_dataset,
            batch_size=self.per_device_batch_size,
            num_workers=val_workers,
            persistent_workers=val_workers > 0,
            pin_memory=True,
            drop_last=False,
            collate_fn=collate_fn,
        )

    # -- resume API ---------------------------------------------------------

    def apply_dataloader_state(
        self,
        states: Optional[List[Dict[str, Any]]],
    ) -> None:
        """Inject resume cursors from a *list* of per-rank state dicts.

        Each element is the dict written by
        :meth:`make_dataloader_state` (one per saved rank).  Shape::

            {
                "schema_version": int,
                "world_size": int,
                "num_workers": int,
                "rank": int,
                "worker_states": {worker_id: WorkerDataState, ...},
            }

        Resume protocol:
          1. We **union** all ``bucket_cursors[*].file_progress``
             entries across every saved slot into a global per-bucket
             file-progress map, and push it down to the underlying
             :class:`PreTokenizedDataset` s.  This map is topology-
             agnostic — the exactly-once invariant is preserved
             regardless of any change in ``world_size`` or
             ``num_workers``.
          2. If the saved topology matches the current one
             ``(world_size, num_workers)``, we additionally restore
             each slot's PRNG ``counter`` and ``exhausted_buckets``,
             so the data sequence after resume is bit-identical to
             what a continuous run would have produced.  If the
             topologies differ, counters restart at 0 (the resumed
             data order will differ, but every document is still
             consumed exactly once across runs).

        Files with an unsupported ``schema_version`` are skipped with a
        warning.
        """
        if not states:
            return
        if not isinstance(self.train_dataset, WeightedInterleavedDataset):
            return  # Only pretokenized path supports cursor resume.

        # Filter out states with unrecognised schema versions.
        good: List[Dict[str, Any]] = []
        for s in states:
            v = s.get("schema_version", 0)
            if v != DATALOADER_STATE_SCHEMA_VERSION:
                logger.warning(
                    f"Discarding dataloader state with unsupported schema "
                    f"version {v} (expected {DATALOADER_STATE_SCHEMA_VERSION})."
                )
                continue
            good.append(s)
        if not good:
            return

        current_workers = max(1, self._resolved_num_workers())
        topology_match = all(
            s.get("world_size") == self.world_size
            and s.get("num_workers") == current_workers
            for s in good
        )

        if not topology_match:
            saved_topos = sorted({
                (s.get("world_size"), s.get("num_workers")) for s in good
            })
            logger.info(
                f"Dataloader topology change: saved {saved_topos} "
                f"-> current ({self.world_size}, {current_workers}).  "
                "Redistributing file-level progress; counters restart at 0; "
                "exactly-once invariant preserved."
            )

        wid = self.train_dataset
        wid.reset_all_progress()

        # 1. Union all bucket_cursors.file_progress into the underlying datasets.
        for s in good:
            for w_id, ws in (s.get("worker_states") or {}).items():
                if not isinstance(ws, WorkerDataState):
                    ws = _worker_state_from_dict(ws)
                for bname, bcursor in ws.bucket_cursors.items():
                    ds = wid.datasets.get(bname)
                    if isinstance(ds, PreTokenizedDataset):
                        ds.update_file_progress(bcursor.file_progress)

        # 2. If topology matches, restore per-slot counter + exhausted_buckets
        #    (also the slot's bucket_cursors so the WID's outgoing stamps
        #    carry over correctly on the very first batch after resume).
        if topology_match:
            for s in good:
                saved_rank = int(s.get("rank", 0))
                for w_id, ws in (s.get("worker_states") or {}).items():
                    if not isinstance(ws, WorkerDataState):
                        ws = _worker_state_from_dict(ws)
                    # Direct assignment (don't call set_worker_state — its
                    # update_file_progress would double-merge what we just
                    # unioned above).
                    wid.worker_states[(saved_rank, int(w_id))] = ws

    def make_dataloader_state(
        self,
        worker_states: Dict[int, WorkerDataState],
        rank: int,
    ) -> Dict[str, Any]:
        """Build the dict that should be written to ``dataloader_state/rank_{N}.pkl``."""
        return {
            "schema_version": DATALOADER_STATE_SCHEMA_VERSION,
            "world_size": self.world_size,
            "num_workers": max(1, self._resolved_num_workers()),
            "rank": int(rank),
            "worker_states": dict(worker_states),
        }

