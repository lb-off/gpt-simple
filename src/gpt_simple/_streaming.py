#!/usr/bin/env python3
"""
Legacy JSONL streaming datasets for gpt_simple.

This module provides on-the-fly tokenization from raw JSONL files.  It is the
fallback path when ``DataConfig(format="jsonl")`` is used.  For production
training, prefer the pretokenized format (see ``data.py``).
"""

import json
import logging
import os
import random
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple, Union

import torch
from torch.utils.data import IterableDataset

from gpt_simple.tokenizer import SimpleLLMTokenizer

logger = logging.getLogger("gpt_simple")

BUCKET_TO_ID: Dict[str, int] = {}
ID_TO_BUCKET: Dict[int, str] = {}


def build_bucket_mappings(
    bucket_names: List[str],
) -> Tuple[Dict[str, int], Dict[int, str]]:
    bucket_to_id = {name: idx for idx, name in enumerate(sorted(bucket_names))}
    id_to_bucket = {idx: name for name, idx in bucket_to_id.items()}
    return bucket_to_id, id_to_bucket


class StreamingTextDataset(IterableDataset):
    """Streaming dataset that tokenizes raw text on the fly.

    Reads ``.jsonl`` files (one JSON object per line, each with a ``text`` field),
    using constant memory and example-level work distribution across workers.
    """
    
    def __init__(
        self,
        data_path: Union[str, Path],
        tokenizer: SimpleLLMTokenizer,
        max_length: int = 2048,
        min_text_length: int = 200,
        seed: int = 42,
        file_list: Optional[List[Union[str, Path]]] = None,
        shard_files: bool = True,
        overlap_size: int = 256,
        probabilistic_overlap: bool = False,
        overlap_probability: float = 0.7,
        use_shared_queue: bool = True,
        shared_work_state: Optional[Dict] = None,
        attention_mode: str = "causal",
    ):
        """
        Initialize streaming dataset.
        
        Args:
            data_path: Path to bucket directory (used as reference only)
            tokenizer: Tokenizer instance
            max_length: Maximum sequence length
            min_text_length: Minimum text length to keep
            seed: Random seed for shuffling
            file_list: List of .jsonl files to stream from (required)
            shard_files: If True, shard files across ranks/workers. Set False for validation.
            overlap_size: Overlap size for windowing long documents (default: 256)
            probabilistic_overlap: If True, use overlap probabilistically (default: False)
            overlap_probability: Probability of using overlap when probabilistic (default: 0.7)
            use_shared_queue: If True, use shared work state for dynamic example distribution (default: True)
            shared_work_state: Dict with {'file_idx': mp.Value, 'line_num': mp.Value, 'lock': mp.Lock} for example-level work distribution
            attention_mode: Attention backend ("causal", "sdpa_mask", "flex"). When "causal", doc_ids are omitted from batches.
        """
        self.data_path = Path(data_path)
        self.attention_mode = attention_mode
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.min_text_length = min_text_length
        self.seed = seed
        self.epoch = 0
        self.shard_files = shard_files
        # Windowing configuration for long documents
        self.overlap_size = min(overlap_size, max_length // 2)
        self.probabilistic_overlap = probabilistic_overlap
        self.overlap_probability = overlap_probability
        # Shared state for dynamic example-level work distribution
        self.use_shared_queue = use_shared_queue
        self.shared_work_state = shared_work_state
        
        # Check if we're in a distributed setting for print management
        self._is_main_process = self._check_distributed_rank()
        
        # Validate and store file list
        if file_list is None:
            raise ValueError("file_list is required. StreamingTextDataset must be provided with explicit .jsonl files.")
        
        self.data_files = self._validate_files(file_list)
        if not self.data_files:
            raise ValueError("No valid .jsonl files found in provided file_list")
        
        if self._is_main_process:
            logger.debug(f"[DATA] Loaded {len(self.data_files)} .jsonl files")
        
        # Estimate dataset size (for progress tracking)
        self._estimate_size()
    
    def set_epoch(self, epoch: int):
        """Set the current epoch to ensure shuffling changes across epochs."""
        self.epoch = epoch
    
    def _check_distributed_rank(self) -> bool:
        """Check if we're in a distributed setting and return if this is rank 0.
        
        Note: This only checks rank, not worker_id (which isn't available at __init__ time).
        For iteration-time logging, use _is_main_worker() which checks both rank and worker.
        """
        try:
            import torch.distributed as dist
            if dist.is_available() and dist.is_initialized():
                return dist.get_rank() == 0
            else:
                return True
        except ImportError:
            return True
    
    def _is_main_worker(self) -> bool:
        """Check if this is the main worker (rank 0, worker 0).
        
        Can only be called from __iter__() where worker_info is available.
        """
        try:
            import torch.distributed as dist
            if dist.is_available() and dist.is_initialized():
                is_rank_0 = dist.get_rank() == 0
            else:
                is_rank_0 = True
        except ImportError:
            is_rank_0 = True
        
        worker_info = torch.utils.data.get_worker_info()
        is_worker_0 = (worker_info is None) or (worker_info.id == 0)
        
        return is_rank_0 and is_worker_0
    
    def _validate_files(self, file_list: List[Union[str, Path]]) -> List[Path]:
        """Validate that all provided files exist and are .jsonl files."""
        files: List[Path] = []
        
        for p in file_list:
            pp = Path(p)
            
            # Check file exists
            if not pp.exists():
                raise ValueError(f"File not found: {pp}")
            
            # Check it's a file
            if not pp.is_file():
                raise ValueError(f"Path is not a file: {pp}")
            
            # Check it's a .jsonl file
            if pp.suffix != '.jsonl':
                raise ValueError(f"Only .jsonl files are supported, got: {pp}")
            
            files.append(pp)
        
        return sorted(files)  # Deterministic order
    
    def _estimate_size(self):
        """Estimate dataset size for progress tracking."""
        try:
            # Quick estimate based on first file
            if self.data_files:
                sample_file = self.data_files[0]
                sample_count = 0
                sample_bytes = 0  # Track actual bytes of sampled lines
                
                with open(sample_file, 'r', encoding='utf-8') as f:
                    for _ in range(1000):  # Sample first 1000 lines
                        line = f.readline()
                        if not line:
                            break
                        sample_count += 1
                        # Count bytes of this line (including newline)
                        sample_bytes += len(line.encode('utf-8'))
                
                # Total size of all files
                total_size = sum(f.stat().st_size for f in self.data_files)
                
                # Estimate based on lines-per-byte ratio from sample
                if sample_bytes > 0:
                    # Use SAMPLED bytes, not entire file size
                    self.estimated_size = int((sample_count / sample_bytes) * total_size)
                else:
                    # Fallback: assume all files have similar line count as sample
                    self.estimated_size = sample_count * len(self.data_files)
                
                if self._is_main_process:
                    logger.debug(f"[DATA] Estimated dataset size: ~{self.estimated_size:,} examples (sampled {sample_count} lines)")
        except Exception as e:
            logger.warning(f"[DATA] Error estimating dataset size: {e}")
            self.estimated_size = None
    
    def _read_file_stream(self, file_path: Path, shard_offset: int = 0, shard_stride: int = 1, worker_id: int = 0, global_rank: int = 0) -> Iterator[str]:
        """Stream texts from a single .jsonl file with optional example-level sharding.
        
        Expects each line to be a JSON object with a 'text' field.
        
        Args:
            file_path: Path to .jsonl file
            shard_offset: Starting line offset for this shard (e.g., rank_id)
            shard_stride: Number of lines to skip between reads (e.g., world_size)
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f):
                    # Example-level sharding: only process lines matching this shard
                    if (line_num % shard_stride) != shard_offset:
                        continue
                    
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        data = json.loads(line)
                        
                        # Expect 'text' field
                        if 'text' not in data:
                            logger.warning(f"[WORKER R{global_rank}W{worker_id}] {file_path.name}:{line_num} missing 'text' field")
                            continue
                        
                        text = str(data['text']).strip()
                        if len(text) >= self.min_text_length:
                            yield text
                        else:
                            logger.debug(f"[WORKER R{global_rank}W{worker_id}] {file_path.name}:{line_num} text too short ({len(text)} < {self.min_text_length})")
                    
                    except json.JSONDecodeError as e:
                        logger.warning(f"[WORKER R{global_rank}W{worker_id}] {file_path.name}:{line_num} invalid JSON: {e}")
                        continue
        
        except Exception as e:
            logger.warning(f"[DATA] Error reading {file_path}: {e}")
    
    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """
        Iterate over the dataset with length-binned packing for ≥0.90 efficiency.
        
        Uses token length binning and greedy filling to avoid mixing short/long docs.
        """
        
        # Get distributed info for proper sharding across ranks
        try:
            import torch.distributed as dist
            if dist.is_available() and dist.is_initialized():
                global_rank = dist.get_rank()
                world_size = dist.get_world_size()
            else:
                global_rank = 0
                world_size = 1
        except ImportError:
            global_rank = 0
            world_size = 1
        
        # Get worker info for multi-process data loading within each rank
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
        else:
            worker_id = 0
            num_workers = 1
        
        logger.debug(f"[WORKER R{global_rank}W{worker_id}/{num_workers}] Starting iteration")
        
        # Determine file distribution strategy
        # 1. Shared work state (best): Dynamic example-level distribution for perfect load balancing
        # 2. File sharding: Pre-assign files to workers (can cause imbalance)
        # 3. Example sharding: All workers see all files, shard by line number (memory intensive)
        
        if self.use_shared_queue and self.shared_work_state is not None:
            # Dynamic work state: workers pull examples atomically
            if worker_id == 0:
                logger.debug(f"[WORKER R{global_rank}] Using shared work state for dynamic distribution")
            
            # Note: shuffled_files is not used in shared work state mode
            # Files are pre-shuffled and stored in shared state
            shuffled_files = []  # Placeholder for compatibility
            shard_offset = 0
            shard_stride = 1
            
        elif self.shard_files:
            # First shard by global rank to ensure each rank sees different data
            files_per_rank = len(self.data_files) // world_size
            rank_start = global_rank * files_per_rank
            rank_end = rank_start + files_per_rank if global_rank < world_size - 1 else len(self.data_files)
            rank_files = self.data_files[rank_start:rank_end]
            
            # Log file sharding (only from worker 0 to reduce spam)
            if worker_id == 0:
                logger.debug(f"[WORKER R{global_rank}W{worker_id}] File sharding: {len(self.data_files)} files, rank gets [{rank_start}:{rank_end}] = {len(rank_files)} files")
            
            # Then shard within rank by worker
            if num_workers > 1:
                files_per_worker = len(rank_files) // num_workers
                worker_start = worker_id * files_per_worker
                worker_end = worker_start + files_per_worker if worker_id < num_workers - 1 else len(rank_files)
                worker_files = rank_files[worker_start:worker_end]
            else:
                worker_files = rank_files
        else:
            # No file sharding - all ranks/workers see all files, but we'll shard examples
            worker_files = self.data_files
            if self._is_main_worker():
                total_shards = world_size * num_workers
                logger.debug(f"[DATA] Example-level sharding: {total_shards} parallel streams ({world_size} ranks x {num_workers} workers)")
        
        # Per-rank deterministic seeding for reproducible resumes
        # Combine global seed with rank, worker, and epoch info for deterministic but different sequences
        rank_worker_seed = hash((self.seed, global_rank, worker_id, self.epoch)) % (2**32)
        random.seed(rank_worker_seed)
        
        # Shuffle files for each epoch with deterministic per-rank/worker seeding
        # (Skip if using shared work state - files are pre-shuffled there)
        if not (self.use_shared_queue and self.shared_work_state is not None):
            shuffled_files = worker_files.copy()
            random.shuffle(shuffled_files)
        
        # Determine example-level sharding parameters
        # (Skip if using shared work state - it handles distribution atomically)
        if not (self.use_shared_queue and self.shared_work_state is not None):
            if self.shard_files:
                # Files are already sharded, no need to shard examples
                shard_offset = 0
                shard_stride = 1
            else:
                # Files are not sharded, so shard examples across ranks AND workers
                # With R ranks and W workers per rank, total shards = R * W
                # Each worker gets: lines at offset (rank * W + worker), (rank * W + worker) + R*W, ...
                total_workers_across_ranks = world_size * num_workers
                worker_global_id = global_rank * num_workers + worker_id
                shard_offset = worker_global_id
                shard_stride = total_workers_across_ranks
                
                if self._is_main_worker():
                    logger.debug(f"[WORKER R{global_rank}W{worker_id}] Processing lines {shard_offset}, {shard_offset + shard_stride}, ... (stride={shard_stride})")
        
        # Length-binned packing state with bounded reservoirs
        length_bins = {}  # bin_index -> [(tokenized_window, overlap_mask, window_doc_id)]
        max_bin_size = 512  # Fixed-size reservoir per bin to prevent RAM bloat
        current_sequence = []
        current_overlap_masks = []  # Track overlap masks for each token in current_sequence
        current_doc_ids_list = []  # Track window doc IDs for each token in current_sequence
        current_length = 0
        padding_waste = 0
        total_tokens = 0
        sequences_created = 0
        documents_processed = 0
        next_window_id = 0  # Counter for assigning unique IDs to each window
        
        # Windowing statistics
        windowed_docs_count = 0
        total_windows_created = 0
        overlapped_tokens_total = 0
        
        # EOD token for document boundaries
        eod_token_id = getattr(self.tokenizer, 'eod_token_id', self.tokenizer.eos_token_id)
        
        # Define length bins for optimal packing (64 bins covering 1-2048 tokens)
        num_bins = 64
        max_doc_length = self.max_length - 1  # Leave space for EOD
        bin_ranges = []
        for i in range(num_bins):
            # Exponential distribution: more bins for shorter docs
            start = int((i / num_bins) ** 1.5 * max_doc_length) + 1
            end = int(((i + 1) / num_bins) ** 1.5 * max_doc_length)
            bin_ranges.append((start, min(end, max_doc_length)))
        
        def get_length_bin(doc_length: int) -> int:
            """Get the appropriate bin index for a document length."""
            for i, (start, end) in enumerate(bin_ranges):
                if start <= doc_length <= end:
                    return i
            return num_bins - 1  # Fallback to largest bin
        
        def try_greedy_fill_from_bins():
            """Greedily fill a sequence using length-binned documents."""
            nonlocal current_sequence, current_overlap_masks, current_doc_ids_list, current_length, length_bins
            
            if current_length == 0:
                current_sequence = []
                current_overlap_masks = []
                current_doc_ids_list = []
                current_length = 0
            
            # Greedily pack as many documents as fit; 90% is only a minimum target.
            filled = True
            while filled:
                filled = False
                remaining_space = self.max_length - current_length
                
                # Stop if we don't have meaningful space left (at least 64 tokens for a small doc)
                if remaining_space < 64:
                    break
                
                # Find best fitting documents from bins
                best_bin = None
                best_doc_idx = None
                best_fit_score = float('inf')
                
                for bin_idx, docs in length_bins.items():
                    if not docs:
                        continue
                    
                    for doc_idx, (doc_tokens, overlap_mask, win_doc_id) in enumerate(docs):
                        doc_length = len(doc_tokens)
                        
                        # Perfect fit or good fit within space
                        if doc_length <= remaining_space:
                            # Prefer docs that use more space (better packing)
                            fit_score = remaining_space - doc_length
                            if fit_score < best_fit_score:
                                best_fit_score = fit_score
                                best_bin = bin_idx
                                best_doc_idx = doc_idx
                
                # Add best fitting document
                if best_bin is not None and best_doc_idx is not None:
                    doc_tokens, overlap_mask, win_doc_id = length_bins[best_bin].pop(best_doc_idx)
                    doc_length = len(doc_tokens)
                    current_sequence.extend(doc_tokens)
                    # Extend overlap masks (empty mask if not provided)
                    if not overlap_mask:
                        overlap_mask = [0] * doc_length
                    current_overlap_masks.extend(overlap_mask)
                    # Track original doc ID for each token
                    current_doc_ids_list.extend([win_doc_id] * doc_length)
                    current_length += doc_length
                    filled = True
                    
                    # Clean up empty bins
                    if not length_bins[best_bin]:
                        del length_bins[best_bin]
            
        def finalize_sequence():
            """Finalize current sequence with minimal padding."""
            nonlocal current_sequence, current_overlap_masks, current_doc_ids_list, current_length, padding_waste, total_tokens, sequences_created
            
            if current_length == 0:
                return None
            
            # Pad up to max_length.
            padding_needed = self.max_length - current_length
            current_sequence.extend([self.tokenizer.pad_token_id] * padding_needed)
            
            # Create tensors
            input_ids = torch.tensor(current_sequence, dtype=torch.long)
            attention_mask = torch.zeros(self.max_length, dtype=torch.long)
            attention_mask[:current_length] = 1
            
            # Create labels: mask padding and the first token after each EOD.
            # The EOD token itself is kept as a real target so the model learns
            # to emit it (with shift_labels = labels[1:], masking it would zero
            # the gradient that teaches termination).
            labels = input_ids.clone()
            labels[current_length:] = -100  # Mask padding positions

            # Mask positions immediately after EOD tokens (important for RoPE)
            valid_tokens = input_ids[:current_length]
            eod_positions = (valid_tokens == eod_token_id).nonzero(as_tuple=True)[0]
            for eod_pos in eod_positions:
                if eod_pos + 1 < current_length:
                    labels[eod_pos + 1] = -100

            # Mask overlap regions from windowed documents to avoid double-counting.
            if len(current_overlap_masks) == current_length:
                for i in range(current_length):
                    if current_overlap_masks[i] == 1:
                        labels[i] = -100

            # Continuous position_ids (0..T-1) across the packed sequence; do not reset
            # at EOD. Document boundaries are handled by the attention mask.
            position_ids = torch.zeros(self.max_length, dtype=torch.long)
            position_ids[:current_length] = torch.arange(current_length, dtype=torch.long)
            
            # Create doc_ids for block-diagonal attention (sdpa_mask / flex modes).
            # Skipped entirely for "causal" mode where cross-document masking is not used.
            need_doc_ids = self.attention_mode != "causal"
            doc_ids = None

            if need_doc_ids:
                doc_ids = torch.zeros(self.max_length, dtype=torch.long)

                if current_length > 0 and len(current_doc_ids_list) == current_length:
                    doc_ids[:current_length] = torch.tensor(current_doc_ids_list, dtype=torch.long)
                elif current_length > 0:
                    valid_tokens = input_ids[:current_length]
                    eod_mask = (valid_tokens == eod_token_id)

                    if eod_mask.any():
                        eod_indices = eod_mask.nonzero(as_tuple=True)[0].tolist()
                        doc_id = 0
                        start = 0
                        for eod_idx in eod_indices:
                            doc_ids[start:eod_idx + 1] = doc_id
                            doc_id += 1
                            start = eod_idx + 1
                        if start < current_length:
                            doc_ids[start:current_length] = doc_id
                    else:
                        doc_ids[:current_length] = 0

            padding_waste += padding_needed
            total_tokens += self.max_length
            sequences_created += 1

            # Reset for next sequence
            current_sequence = []
            current_overlap_masks = []
            current_doc_ids_list = []
            current_length = 0
            
            result = {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': labels,
                'position_ids': position_ids,
            }
            if doc_ids is not None:
                result['doc_ids'] = doc_ids
            return result
        
        def create_windows_from_long_doc(tokens: List[int], doc_rng: random.Random) -> List[Tuple[List[int], List[int]]]:
            """
            Window a long document into overlapping chunks.
            
            Returns list of (token_ids, overlap_mask) tuples where:
            - token_ids: tokens for this window (including EOD only for last window)
            - overlap_mask: 1 for overlap tokens (to be masked in labels), 0 otherwise
            
            Note: Each window will be assigned a unique doc_id later to prevent
            cross-window attention (even within the same original document).
            """
            if len(tokens) < self.max_length:
                # Document fits, no windowing needed
                return [(tokens + [eod_token_id], [])]
            
            # Determine overlap for this document
            if self.probabilistic_overlap and doc_rng.random() > self.overlap_probability:
                # 30% chance: no overlap
                actual_overlap = 0
            else:
                # 70% chance: use configured overlap
                actual_overlap = self.overlap_size
            
            # Calculate stride based on overlap
            # With overlap: stride = max_length - overlap (e.g., 2048 - 512 = 1536)
            # No overlap: stride = max_length (e.g., 2048)
            stride = self.max_length - actual_overlap
            stride = max(stride, self.max_length // 2)
            
            windows = []
            start = 0  # Always start at the beginning of the document
            window_idx = 0
            
            while start < len(tokens):
                end = min(start + self.max_length, len(tokens))
                window_tokens = tokens[start:end]
                
                # Determine overlap region for loss masking
                # First window: no overlap masking
                # Other windows: mask first actual_overlap tokens
                if window_idx > 0 and actual_overlap > 0:
                    # Subsequent windows - mask overlap region
                    overlap_len = min(actual_overlap, len(window_tokens))
                    overlap_mask = [1] * overlap_len + [0] * (len(window_tokens) - overlap_len)
                else:
                    # No overlap masking needed
                    overlap_mask = [0] * len(window_tokens)
                
                # Add EOD token only to the last window
                is_last_window = (end >= len(tokens))
                if is_last_window:
                    window_tokens = window_tokens + [eod_token_id]
                    overlap_mask = overlap_mask + [0]  # EOD not in overlap
                
                windows.append((window_tokens, overlap_mask))
                
                # Move to next window with consistent stride
                start += stride
                window_idx += 1
                
                # Safety check to prevent infinite loops
                if window_idx > 1000:
                    logger.warning(f"[DATA] Too many windows ({window_idx}) for doc with {len(tokens)} tokens, breaking")
                    break
            
            return windows
        
        # Collect and bin documents before packing
        # Use per-worker RNG for deterministic windowing decisions
        windowing_rng = random.Random(rank_worker_seed)
        
        # Example-level work distribution using shared state
        def get_next_example():
            """Atomically claim the next example (file, line_number) to process.
            
            Returns (file_path, line_number) or (None, None) if all work is done.
            """
            if self.use_shared_queue and self.shared_work_state is not None:
                # Atomic work claiming with lock
                state = self.shared_work_state
                lock = state['lock']
                
                with lock:
                    file_idx = state['file_idx'].value
                    line_num = state['line_num'].value
                    
                    # Check if we've exhausted all files
                    if file_idx >= state['total_files']:
                        return None, None
                    
                    # Get current file and line
                    current_file = self.data_files[file_idx]
                    current_line = line_num
                    
                    # Increment line number for next worker
                    # Note: We don't know file length yet, so we increment optimistically
                    # If the line doesn't exist, the worker will detect it and advance the file
                    state['line_num'].value = line_num + 1
                    
                    return current_file, current_line
            else:
                # Fallback: no shared state, shouldn't happen with use_shared_queue=True
                return None, None
        
        def advance_to_next_file():
            """Move shared state to next file when current file is exhausted."""
            if self.use_shared_queue and self.shared_work_state is not None:
                state = self.shared_work_state
                lock = state['lock']
                
                with lock:
                    # Move to next file, reset line number
                    state['file_idx'].value += 1
                    state['line_num'].value = 0
                    
                    if state['file_idx'].value < state['total_files']:
                        if worker_id == 0:
                            logger.debug(f"[WORKER R{global_rank}W{worker_id}] Advanced to file {state['file_idx'].value}/{state['total_files']}")
        
        # File cache to avoid reopening same file repeatedly
        current_open_file = None
        current_open_file_path = None
        current_file_lines = []
        
        # Process examples using shared work state
        if self.use_shared_queue and self.shared_work_state is not None:
            while True:
                file_path, line_num = get_next_example()
                
                if file_path is None:
                    # All work exhausted
                    break
                
                # Read specific line from file
                try:
                    # Cache file contents if this is the same file
                    if file_path != current_open_file_path:
                        # Close previous file if open
                        if current_open_file is not None:
                            current_open_file.close()
                        
                        # Open new file and read all lines (for efficient random access)
                        current_open_file = open(file_path, 'r', encoding='utf-8')
                        current_file_lines = current_open_file.readlines()
                        current_open_file_path = file_path
                    
                    # Check if line exists
                    if line_num >= len(current_file_lines):
                        # File exhausted, advance to next file
                        advance_to_next_file()
                        continue
                    
                    # Get the line
                    line = current_file_lines[line_num].strip()
                    if not line:
                        continue
                    
                    # Parse JSON and extract text
                    try:
                        data = json.loads(line)
                        if 'text' not in data:
                            continue
                        
                        text = str(data['text']).strip()
                        if len(text) < self.min_text_length:
                            continue
                        
                        # Process this example (tokenize, window, bin, pack)
                        # Same logic as fallback path
                        try:
                            tokens = self.tokenizer.encode(text, add_special_tokens=False)
                            if len(tokens) < 10:  # Skip very short texts
                                continue
                            
                            # Window long documents or process short ones directly
                            windows = create_windows_from_long_doc(tokens, windowing_rng)
                            documents_processed += 1
                            
                            # Track windowing statistics
                            if len(windows) > 1:
                                windowed_docs_count += 1
                                total_windows_created += len(windows)
                                for _, overlap_mask in windows:
                                    overlapped_tokens_total += sum(overlap_mask) if overlap_mask else 0
                            
                            # Add each window to bins for packing
                            for window_tokens, overlap_mask in windows:
                                bin_idx = get_length_bin(len(window_tokens))
                                if bin_idx not in length_bins:
                                    length_bins[bin_idx] = []
                                
                                # If bin is full, force a flush before adding new window
                                if len(length_bins[bin_idx]) >= max_bin_size:
                                    try_greedy_fill_from_bins()
                                    if current_length >= int(self.max_length * 0.90):
                                        result = finalize_sequence()
                                        if result:
                                            if sequences_created % 500 == 0:
                                                logger.debug(f"[WORKER R{global_rank}W{worker_id}] Produced {sequences_created} sequences so far")
                                            yield result
                                
                                # Assign unique window ID and store with overlap mask
                                window_doc_id = next_window_id
                                next_window_id += 1
                                length_bins[bin_idx].append((window_tokens, overlap_mask, window_doc_id))
                            
                            # Periodically try to create sequences from bins
                            if documents_processed % 100 == 0 and length_bins:
                                try_greedy_fill_from_bins()
                                if current_length >= int(self.max_length * 0.90):
                                    result = finalize_sequence()
                                    if result:
                                        if sequences_created % 500 == 0:
                                            logger.debug(f"[WORKER R{global_rank}W{worker_id}] Produced {sequences_created} sequences so far")
                                        yield result
                        
                        except Exception as e:
                            logger.warning(f"[DATA] Error tokenizing/processing text: {e}")
                            continue
                    
                    except json.JSONDecodeError:
                        continue
                
                except Exception as e:
                    logger.warning(f"[WORKER W{worker_id}] Error reading {file_path}:{line_num}: {e}")
                    continue
            
            # Clean up
            if current_open_file is not None:
                current_open_file.close()
        
        else:
            # Fallback to old file-based iteration for backward compatibility
            for file_path in shuffled_files:
                for text in self._read_file_stream(file_path, shard_offset=shard_offset, shard_stride=shard_stride):
                    try:
                        tokens = self.tokenizer.encode(text, add_special_tokens=False)
                        if len(tokens) < 10:  # Skip very short texts
                            continue
                        
                        # Window long documents or process short ones directly
                        windows = create_windows_from_long_doc(tokens, windowing_rng)
                        documents_processed += 1
                        
                        # Track windowing statistics
                        if len(windows) > 1:
                            windowed_docs_count += 1
                            total_windows_created += len(windows)
                            # Count overlapped tokens in this document
                            for _, overlap_mask in windows:
                                overlapped_tokens_total += sum(overlap_mask) if overlap_mask else 0
                            
                            # Log very infrequently for windowed documents (only first 10k, every 5k)
                            if windowed_docs_count <= 10000 and windowed_docs_count % 5000 == 0 and global_rank == 0 and worker_id == 0:
                                avg_windows_per_doc = total_windows_created / windowed_docs_count if windowed_docs_count > 0 else 0
                                logger.debug(f"[DATA] Windowing R{global_rank}W{worker_id}: {windowed_docs_count} long docs -> {total_windows_created} windows (avg {avg_windows_per_doc:.2f}/doc)")
                        
                        # Add each window to bins for packing
                        # Each window gets a unique doc_id to prevent cross-window attention
                        for window_tokens, overlap_mask in windows:
                            bin_idx = get_length_bin(len(window_tokens))
                            if bin_idx not in length_bins:
                                length_bins[bin_idx] = []
                            
                            # If bin is full, force a flush before adding new window
                            if len(length_bins[bin_idx]) >= max_bin_size:
                                try_greedy_fill_from_bins()
                                if current_length >= int(self.max_length * 0.90):
                                    result = finalize_sequence()
                                    if result:
                                        yield result
                            
                            # Assign unique window ID and store with overlap mask
                            window_doc_id = next_window_id
                            next_window_id += 1
                            length_bins[bin_idx].append((window_tokens, overlap_mask, window_doc_id))
                        
                        # Periodically drain bins into finished sequences.
                        if documents_processed % 100 == 0 and length_bins:
                            try_greedy_fill_from_bins()
                            if current_length >= int(self.max_length * 0.90):
                                result = finalize_sequence()
                                if result:
                                    if sequences_created % 500 == 0:
                                        logger.debug(f"[WORKER R{global_rank}W{worker_id}] Produced {sequences_created} sequences so far")
                                    yield result
                    
                    except Exception as e:
                        logger.warning(f"[DATA] Error processing text: {e}")
                        continue
        
        # Final packing: drain all remaining bins
        while length_bins:
            try_greedy_fill_from_bins()
            if current_length > 0:
                result = finalize_sequence()
                if result:
                    yield result
            else:
                # No more documents can fit, clear remaining bins
                break
        
        if sequences_created > 0:
            final_efficiency = 1.0 - (padding_waste / total_tokens)
            logger.debug(
                f"[WORKER R{global_rank}W{worker_id}/{num_workers}] Done: {sequences_created} sequences, "
                f"{final_efficiency:.3f} efficiency, {documents_processed} docs"
            )
            if self._is_main_process:
                logger.debug(f"[STATS] Streaming: {sequences_created} sequences, {final_efficiency:.3f} efficiency, {documents_processed} docs")
                if windowed_docs_count > 0:
                    avg_windows_per_doc = total_windows_created / windowed_docs_count
                    duplication_factor = overlapped_tokens_total / (total_tokens - padding_waste) if (total_tokens - padding_waste) > 0 else 0
                    logger.debug(
                        f"[STATS] Windowing: {windowed_docs_count} long docs -> {total_windows_created} windows "
                        f"(avg {avg_windows_per_doc:.2f}/doc, {duplication_factor:.2%} overlap)"
                    )
        else:
            logger.warning(f"[WORKER R{global_rank}W{worker_id}/{num_workers}] Produced 0 sequences - worker may not be utilized")




class CombinedStreamingDataset(IterableDataset):
    """Uniform interleaving of per-bucket JSONL StreamingTextDatasets.

    Curriculum is NOT supported on the JSONL path; use pretokenized format instead.
    """

    def __init__(
        self,
        bucket_to_files: Dict[str, List[Path]],
        tokenizer: SimpleLLMTokenizer,
        max_length: int,
        seed: int,
        world_size: Optional[int] = None,
        attention_mode: str = "causal",
        overlap_size: int = 256,
    ):
        self.seed = seed

        if world_size is not None:
            self._world_size = world_size
        else:
            try:
                import torch.distributed as dist
                if dist.is_available() and dist.is_initialized():
                    self._world_size = dist.get_world_size()
                else:
                    self._world_size = int(os.environ.get("WORLD_SIZE", "1"))
            except (ImportError, ValueError):
                self._world_size = 1

        self.bucket_datasets: Dict[str, StreamingTextDataset] = {}
        for bucket, files in bucket_to_files.items():
            if not files:
                continue
            bucket_rng = random.Random((seed + hash(bucket)) % (2**32))
            shuffled = files.copy()
            bucket_rng.shuffle(shuffled)

            self.bucket_datasets[bucket] = StreamingTextDataset(
                data_path=files[0].parent,
                tokenizer=tokenizer,
                max_length=max_length,
                seed=(seed + hash(bucket)) % (2**32),
                file_list=shuffled,
                shard_files=False,
                attention_mode=attention_mode,
                overlap_size=overlap_size,
            )

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0

        try:
            import torch.distributed as dist
            if dist.is_available() and dist.is_initialized():
                global_rank = dist.get_rank()
            else:
                global_rank = int(os.environ.get("RANK", "0"))
        except (ImportError, ValueError):
            global_rank = 0

        rng = random.Random(hash((self.seed, global_rank, worker_id)) % (2**32))

        bucket_iters: Dict[str, Iterator] = {
            b: iter(ds) for b, ds in self.bucket_datasets.items()
        }
        active = list(bucket_iters.keys())

        while active:
            chosen = rng.choice(active)
            try:
                item = next(bucket_iters[chosen])
                if chosen in BUCKET_TO_ID:
                    item["bucket_id"] = torch.tensor(BUCKET_TO_ID[chosen], dtype=torch.long)
                yield item
            except StopIteration:
                del bucket_iters[chosen]
                active = [b for b in active if b in bucket_iters]


class SequentialBucketDataset(IterableDataset):
    """Chain through bucket JSONL datasets sequentially (for validation)."""

    def __init__(
        self,
        bucket_to_files: Dict[str, List[Path]],
        tokenizer: SimpleLLMTokenizer,
        max_length: int,
        seed: int = 42,
        attention_mode: str = "causal",
        overlap_size: int = 256,
    ):
        self.bucket_datasets: List[Tuple[str, StreamingTextDataset]] = []
        for bucket_name, files in sorted(bucket_to_files.items()):
            if not files:
                continue
            dataset = StreamingTextDataset(
                data_path=files[0].parent,
                tokenizer=tokenizer,
                max_length=max_length,
                seed=(seed + hash(bucket_name)) % (2**32),
                file_list=files,
                shard_files=False,
                attention_mode=attention_mode,
                overlap_size=overlap_size,
            )
            self.bucket_datasets.append((bucket_name, dataset))

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        for _, dataset in self.bucket_datasets:
            yield from dataset

