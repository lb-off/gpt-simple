"""
Checkpoint management for gpt_simple.

Owns the on-disk layout of checkpoints, the atomic save protocol, retention
policy (``keep_last_k``, ``keep_milestone_every``), and the
``resume: auto | scratch | <path>`` resolution logic.

On-disk layout::

    output_dir/
    |-- config.json                   # the Config that started this run
    |-- .run_state.json               # live process snapshot
    |-- tokenizer/                    # written once at run start
    |-- logs/
    |-- checkpoints/
        |-- checkpoint-1000/
        |   |-- trainer_state.json    # step, tokens, curriculum, wandb_run_id, hashes
        |   |-- dataloader_state.pt   # populated by Phase C (per-worker cursors)
        |   |-- model/
        |   |   |-- pytorch_model.bin # canonical model weights (no DDP/compile prefixes)
        |   |   |-- config.json       # ModelConfig
        |   |-- optimizer.bin
        |   |-- scheduler.bin
        |   |-- rng/rank_0.pkl
        |   |-- rng/rank_1.pkl
        |-- checkpoint-2734-shutdown/
        |-- final/

The model state dict is stored ONCE under ``model/pytorch_model.bin`` as a
plain ``state_dict()`` of the unwrapped module.  No Accelerate-format copy
is written (set via the design decision to avoid duplication on large
models).  This means resume must load the model BEFORE
``accelerator.prepare()`` (so wrapping prefixes never appear in the keys).
"""

from __future__ import annotations

import json
import logging
import os
import random
import shutil
import tempfile
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from gpt_simple.errors import CheckpointError

logger = logging.getLogger("gpt_simple")


SCHEMA_VERSION = 1
_CHECKPOINTS_DIRNAME = "checkpoints"
_TOKENIZER_DIRNAME = "tokenizer"
_PARTIAL_SUFFIX = ".partial"


# ---------------------------------------------------------------------------
# TrainerState
# ---------------------------------------------------------------------------


@dataclass
class CurriculumState:
    phase_idx: int = 0
    phase_tokens_consumed: int = 0
    current_mix: Dict[str, float] = field(default_factory=dict)


@dataclass
class TrainingTiming:
    wallclock_seconds_elapsed: float = 0.0
    last_save_duration_seconds: float = 0.0
    saved_at: str = ""


@dataclass
class TrainingMetrics:
    loss: float = float("inf")
    learning_rate: float = 0.0
    grad_norm: float = 0.0
    tokens_per_sec: float = 0.0


@dataclass
class Lineage:
    resumed_from: Optional[str] = None
    is_shutdown_checkpoint: bool = False


@dataclass
class TrainerState:
    """Canonical per-checkpoint training state.

    Serialised as ``trainer_state.json`` next to every checkpoint.  This is
    the source of truth for resume: ``global_step``, ``tokens_trained``,
    curriculum progress, and the W&B run id all live here (NOT in run
    state or checkpoint directory names).
    """

    schema_version: int = SCHEMA_VERSION
    step: int = 0
    tokens_trained: int = 0
    wandb_run_id: Optional[str] = None
    config_hash: str = ""
    model_arch_hash: str = ""
    curriculum: CurriculumState = field(default_factory=CurriculumState)
    timing: TrainingTiming = field(default_factory=TrainingTiming)
    metrics: TrainingMetrics = field(default_factory=TrainingMetrics)
    lineage: Lineage = field(default_factory=Lineage)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "TrainerState":
        """Build a TrainerState from a dict, tolerating missing nested keys."""
        version = data.get("schema_version", SCHEMA_VERSION)
        if version != SCHEMA_VERSION:
            raise CheckpointError(
                f"Unsupported trainer_state schema version: {version} "
                f"(this build supports {SCHEMA_VERSION})"
            )
        curriculum = CurriculumState(**(data.get("curriculum") or {}))
        timing = TrainingTiming(**(data.get("timing") or {}))
        metrics = TrainingMetrics(**(data.get("metrics") or {}))
        lineage = Lineage(**(data.get("lineage") or {}))
        return cls(
            schema_version=version,
            step=int(data.get("step", 0)),
            tokens_trained=int(data.get("tokens_trained", 0)),
            wandb_run_id=data.get("wandb_run_id"),
            config_hash=data.get("config_hash", ""),
            model_arch_hash=data.get("model_arch_hash", ""),
            curriculum=curriculum,
            timing=timing,
            metrics=metrics,
            lineage=lineage,
        )

    @classmethod
    def load(cls, ckpt_dir: Union[str, Path]) -> "TrainerState":
        path = Path(ckpt_dir) / "trainer_state.json"
        if not path.is_file():
            raise CheckpointError(f"trainer_state.json not found in {ckpt_dir}")
        try:
            with open(path) as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            raise CheckpointError(f"Cannot read trainer_state.json: {exc}")
        return cls.from_dict(data)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _atomic_write_json(path: Path, data: dict) -> None:
    """Write JSON atomically: tmp file + rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2, sort_keys=False)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _parse_checkpoint_step(name: str) -> Optional[int]:
    """Return the step number from a checkpoint directory name, or None.

    Recognised forms:
      - ``checkpoint-1000``     -> 1000
      - ``checkpoint-1000-shutdown`` -> 1000
      - ``final``               -> None (we rank ``final`` separately)
      - anything else (e.g. partials) -> None
    """
    if name.endswith(_PARTIAL_SUFFIX):
        return None
    if name == "final":
        return None
    if not name.startswith("checkpoint-"):
        return None
    rest = name[len("checkpoint-"):]
    head = rest.split("-", 1)[0]
    try:
        return int(head)
    except ValueError:
        return None


def _collect_rng_state() -> Dict[str, Any]:
    """Snapshot Python, NumPy, and PyTorch RNG state on the current rank."""
    state: Dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        try:
            state["torch_cuda"] = torch.cuda.get_rng_state_all()
        except Exception:
            state["torch_cuda"] = torch.cuda.get_rng_state()
    return state


def _restore_rng_state(state: Dict[str, Any]) -> None:
    try:
        random.setstate(state["python"])
    except (KeyError, TypeError, ValueError) as exc:
        logger.warning(f"Could not restore Python RNG state: {exc}")
    try:
        np.random.set_state(state["numpy"])
    except (KeyError, TypeError, ValueError) as exc:
        logger.warning(f"Could not restore NumPy RNG state: {exc}")
    try:
        torch.set_rng_state(state["torch_cpu"])
    except (KeyError, TypeError, RuntimeError) as exc:
        logger.warning(f"Could not restore PyTorch CPU RNG state: {exc}")
    if torch.cuda.is_available() and "torch_cuda" in state:
        try:
            cuda_state = state["torch_cuda"]
            if isinstance(cuda_state, list):
                torch.cuda.set_rng_state_all(cuda_state)
            else:
                torch.cuda.set_rng_state(cuda_state)
        except (RuntimeError, TypeError) as exc:
            logger.warning(f"Could not restore PyTorch CUDA RNG state: {exc}")


def _rank_of(accelerator) -> int:
    """Best-effort rank lookup that works without distributed init."""
    if accelerator is not None and hasattr(accelerator, "process_index"):
        try:
            return int(accelerator.process_index)
        except (AttributeError, TypeError):
            pass
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            return dist.get_rank()
    except ImportError:
        pass
    return int(os.environ.get("RANK", "0"))


# ---------------------------------------------------------------------------
# CheckpointManager
# ---------------------------------------------------------------------------


class CheckpointManager:
    """Owns the on-disk checkpoint layout, save/load, and retention.

    Parameters
    ----------
    output_dir : str or Path
        Run output directory.  Checkpoints live under
        ``output_dir / "checkpoints"``.
    keep_last_k : int or None
        Rolling buffer of most-recent regular checkpoints.  ``None``
        disables retention (keeps everything).  ``final/`` and the most
        recent ``-shutdown`` checkpoint are always preserved.
    keep_milestone_every : int or None
        In addition, preserve one checkpoint per N global steps.  ``None``
        disables the milestone rule.
    """

    def __init__(
        self,
        output_dir: Union[str, Path],
        keep_last_k: Optional[int] = 3,
        keep_milestone_every: Optional[int] = None,
    ):
        self.output_dir = Path(output_dir)
        self.keep_last_k = keep_last_k
        self.keep_milestone_every = keep_milestone_every

    # -- paths --------------------------------------------------------------

    @property
    def checkpoints_dir(self) -> Path:
        return self.output_dir / _CHECKPOINTS_DIRNAME

    @property
    def tokenizer_dir(self) -> Path:
        return self.output_dir / _TOKENIZER_DIRNAME

    @staticmethod
    def _build_name(step: int, tag: Optional[str]) -> str:
        if tag == "final":
            return "final"
        if tag is None:
            return f"checkpoint-{step}"
        return f"checkpoint-{step}-{tag}"

    # -- listing / discovery ------------------------------------------------

    def list_checkpoints(self) -> List[Tuple[int, str, Path]]:
        """Return ``[(step, name, path), ...]`` sorted by step ascending.

        Only directories containing ``trainer_state.json`` are returned
        (incomplete saves are skipped).  ``final/`` is always sorted last
        regardless of step.
        """
        if not self.checkpoints_dir.is_dir():
            return []

        regular: List[Tuple[int, str, Path]] = []
        final_entry: Optional[Tuple[int, str, Path]] = None

        for d in self.checkpoints_dir.iterdir():
            if not d.is_dir():
                continue
            if d.name.endswith(_PARTIAL_SUFFIX):
                continue
            if not (d / "trainer_state.json").is_file():
                continue

            try:
                ts = TrainerState.load(d)
                step = ts.step
            except CheckpointError as exc:
                logger.warning(f"Skipping unreadable checkpoint {d}: {exc}")
                continue

            if d.name == "final":
                final_entry = (step, d.name, d)
            else:
                regular.append((step, d.name, d))

        regular.sort(key=lambda x: x[0])
        if final_entry is not None:
            regular.append(final_entry)
        return regular

    # -- resume resolution --------------------------------------------------

    def resolve_resume(self, hint: str) -> Optional[Path]:
        """Resolve ``training.resume`` to a checkpoint path or None.

        ``hint`` values:
          - ``"auto"``: latest checkpoint by step, else None
          - ``"scratch"``: always None (caller may still error if checkpoints exist)
          - any other string: absolute or relative path to a checkpoint dir

        Returns
        -------
        Path or None
            Path to the checkpoint to resume from, or None to start fresh.
        """
        if hint == "scratch":
            return None
        if hint == "auto":
            ckpts = self.list_checkpoints()
            if not ckpts:
                return None
            return ckpts[-1][2]

        # Explicit path
        p = Path(hint)
        if not p.is_absolute():
            # Try relative to cwd, then relative to output_dir
            if not p.is_dir():
                alt = self.output_dir / p
                if alt.is_dir():
                    p = alt
        if not p.is_dir():
            raise CheckpointError(f"resume path does not exist: {hint}")
        if not (p / "trainer_state.json").is_file():
            raise CheckpointError(
                f"resume path is missing trainer_state.json: {p}"
            )
        return p

    def has_any_checkpoint(self) -> bool:
        """True if at least one valid checkpoint exists under ``checkpoints/``."""
        return len(self.list_checkpoints()) > 0

    # -- save ---------------------------------------------------------------

    def save(
        self,
        *,
        accelerator,
        model,
        optimizer,
        scheduler,
        trainer_state: TrainerState,
        model_config_dict: Dict[str, Any],
        tag: Optional[str] = None,
        dataloader_state: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Save a checkpoint atomically.

        Writes into ``checkpoints/<name>.partial/`` first, fsyncs, then
        renames to ``checkpoints/<name>/``.  Mid-save crashes leave a
        ``.partial`` directory behind which ``list_checkpoints`` skips.

        Parameters
        ----------
        tag : str or None
            Suffix for the checkpoint name (e.g. ``"shutdown"`` produces
            ``checkpoint-N-shutdown``).  Special value ``"final"`` produces
            ``final/`` (no step prefix).
        dataloader_state : dict or None
            Optional per-worker dataloader cursor map.  Populated by Phase C;
            in Phase A this is always ``None`` and no file is written.

        Returns
        -------
        Path
            Path to the finalised checkpoint directory.
        """
        save_start = time.monotonic()
        name = self._build_name(trainer_state.step, tag)
        ckpt_dir = self.checkpoints_dir / name
        partial = self.checkpoints_dir / f"{name}{_PARTIAL_SUFFIX}"

        accelerator.wait_for_everyone()

        if accelerator.is_main_process:
            # Wipe any leftover partial from a previous failed save
            if partial.exists():
                shutil.rmtree(partial)
            partial.mkdir(parents=True, exist_ok=True)
            (partial / "model").mkdir(exist_ok=True)
            (partial / "rng").mkdir(exist_ok=True)

        accelerator.wait_for_everyone()

        # Model: gathered state dict (handles FSDP transparently)
        # accelerator.get_state_dict() unwraps DDP and torch.compile
        try:
            model_state_dict = accelerator.get_state_dict(model)
        except Exception:
            unwrapped = accelerator.unwrap_model(model)
            model_state_dict = unwrapped.state_dict()

        # get_state_dict() does not reliably strip torch.compile's
        # "_orig_mod." prefix, so normalise it here.  Checkpoints are then
        # canonical (raw nn.Module names), loadable for both resume and
        # inference without depending on whether the run was compiled.
        from torch.nn.modules.utils import (
            consume_prefix_in_state_dict_if_present,
        )

        consume_prefix_in_state_dict_if_present(model_state_dict, "_orig_mod.")

        if accelerator.is_main_process:
            torch.save(
                model_state_dict,
                partial / "model" / "pytorch_model.bin",
            )
            _atomic_write_json(
                partial / "model" / "config.json",
                model_config_dict,
            )

            # Optimizer + scheduler (rank-0 only under DDP)
            torch.save(optimizer.state_dict(), partial / "optimizer.bin")
            torch.save(scheduler.state_dict(), partial / "scheduler.bin")

            # Pre-create per-rank directories so non-main ranks have a
            # place to write below.
            (partial / "dataloader_state").mkdir(exist_ok=True)

        accelerator.wait_for_everyone()

        # RNG and dataloader state are per-rank.
        rank = _rank_of(accelerator)
        torch.save(
            _collect_rng_state(),
            partial / "rng" / f"rank_{rank}.pkl",
        )
        if dataloader_state is not None:
            torch.save(
                dataloader_state,
                partial / "dataloader_state" / f"rank_{rank}.pkl",
            )

        accelerator.wait_for_everyone()

        # Update timing on rank 0 and write trainer_state.json
        save_duration = time.monotonic() - save_start
        trainer_state.timing.last_save_duration_seconds = save_duration
        trainer_state.timing.saved_at = _now_iso()
        trainer_state.lineage.is_shutdown_checkpoint = tag == "shutdown"

        if accelerator.is_main_process:
            _atomic_write_json(
                partial / "trainer_state.json",
                trainer_state.to_dict(),
            )

            # Atomic rename of the entire directory
            if ckpt_dir.exists():
                shutil.rmtree(ckpt_dir)
            os.rename(partial, ckpt_dir)

            # Apply retention AFTER finalising the new checkpoint
            self.apply_retention(latest_step=trainer_state.step)

            label = (
                "Final checkpoint" if tag == "final"
                else "Shutdown checkpoint" if tag == "shutdown"
                else "Checkpoint"
            )
            logger.info(
                f"{label} saved to {ckpt_dir} "
                f"(step {trainer_state.step}, "
                f"{save_duration:.1f}s)"
            )

        accelerator.wait_for_everyone()
        return ckpt_dir

    # -- load ---------------------------------------------------------------

    @staticmethod
    def load_model_state(model, ckpt_dir: Union[str, Path]) -> None:
        """Load model weights into a raw ``nn.Module``.

        Call BEFORE ``accelerator.prepare()`` to avoid prefix mismatches.
        """
        path = Path(ckpt_dir) / "model" / "pytorch_model.bin"
        if not path.is_file():
            raise CheckpointError(f"Model weights not found: {path}")
        state_dict = torch.load(path, map_location="cpu", weights_only=False)

        # torch.compile wraps the model in an OptimizedModule that prefixes
        # every parameter with "_orig_mod.".  Checkpoints saved from a compiled
        # model therefore carry that prefix, but we load into the raw (not-yet-
        # compiled) nn.Module — so strip it, or every key mismatches and the
        # model silently resumes from random weights under strict=False.
        from torch.nn.modules.utils import (
            consume_prefix_in_state_dict_if_present,
        )

        consume_prefix_in_state_dict_if_present(state_dict, "_orig_mod.")

        try:
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
        except Exception as exc:
            raise CheckpointError(
                f"Cannot load model weights from {path}: {exc}"
            )

        # Catastrophic-mismatch guard: a correct resume for this architecture
        # loads (almost) every parameter.  If most are missing, the checkpoint
        # key format doesn't match the model (e.g. an un-stripped wrapper
        # prefix) and we'd be training from scratch — fail loudly rather than
        # waste compute, since strict=False would otherwise hide it.
        n_model = len(model.state_dict())
        if n_model and len(missing) > n_model // 2:
            raise CheckpointError(
                f"Resume loaded almost no weights: {len(missing)}/{n_model} "
                f"parameters missing from {path}.  This is a checkpoint "
                f"key-format mismatch, not a normal resume.  Missing (up to 5): "
                f"{list(missing)[:5]}; unexpected (up to 5): {list(unexpected)[:5]}"
            )

        if missing:
            logger.warning(
                f"Resuming model: {len(missing)} missing parameter(s) "
                f"(showing up to 5): {list(missing)[:5]}"
            )
        if unexpected:
            logger.warning(
                f"Resuming model: {len(unexpected)} unexpected parameter(s) "
                f"(showing up to 5): {list(unexpected)[:5]}"
            )

    @staticmethod
    def load_optimizer_state(optimizer, ckpt_dir: Union[str, Path]) -> None:
        path = Path(ckpt_dir) / "optimizer.bin"
        if not path.is_file():
            raise CheckpointError(f"Optimizer state not found: {path}")
        optimizer.load_state_dict(torch.load(path, map_location="cpu", weights_only=False))

    @staticmethod
    def load_scheduler_state(scheduler, ckpt_dir: Union[str, Path]) -> None:
        path = Path(ckpt_dir) / "scheduler.bin"
        if not path.is_file():
            raise CheckpointError(f"Scheduler state not found: {path}")
        scheduler.load_state_dict(torch.load(path, map_location="cpu", weights_only=False))

    @staticmethod
    def load_rng_state(rank: int, ckpt_dir: Union[str, Path]) -> None:
        path = Path(ckpt_dir) / "rng" / f"rank_{rank}.pkl"
        if not path.is_file():
            # A fresh rank that wasn't part of the original run still works,
            # just without restored RNG; warn loudly.
            logger.warning(
                f"No RNG state for rank {rank} in {ckpt_dir}; "
                "RNG will not be restored on this rank."
            )
            return
        state = torch.load(path, map_location="cpu", weights_only=False)
        _restore_rng_state(state)

    @staticmethod
    def load_dataloader_state(
        rank: int,
        ckpt_dir: Union[str, Path],
    ) -> Optional[Dict[str, Any]]:
        """Return ONE rank's dataloader state, or ``None``.

        Convenience for the same-topology case where each rank only
        cares about its own file.  Topology-change-safe resume should
        use :meth:`load_all_dataloader_states` instead.
        """
        ckpt_path = Path(ckpt_dir)
        per_rank = ckpt_path / "dataloader_state" / f"rank_{int(rank)}.pkl"
        if per_rank.is_file():
            return torch.load(per_rank, map_location="cpu", weights_only=False)
        # Backwards compatibility with the single-file Phase A layout.
        legacy = ckpt_path / "dataloader_state.pt"
        if legacy.is_file():
            return torch.load(legacy, map_location="cpu", weights_only=False)
        return None

    @staticmethod
    def load_all_dataloader_states(
        ckpt_dir: Union[str, Path],
    ) -> List[Dict[str, Any]]:
        """Return every saved per-rank dataloader state in ``ckpt_dir``.

        Used at resume time to support arbitrary changes of
        ``(world_size, num_workers)``: the caller (the data module)
        unions all per-rank ``bucket_cursors[*].file_progress`` entries
        into a topology-agnostic global progress map and redistributes
        it to the new slots.
        """
        ckpt_path = Path(ckpt_dir)
        out: List[Dict[str, Any]] = []
        per_rank_dir = ckpt_path / "dataloader_state"
        if per_rank_dir.is_dir():
            for f in sorted(per_rank_dir.glob("rank_*.pkl")):
                out.append(torch.load(f, map_location="cpu", weights_only=False))
            if out:
                return out
        # Backwards compatibility with the Phase A single-file layout.
        legacy = ckpt_path / "dataloader_state.pt"
        if legacy.is_file():
            out.append(torch.load(legacy, map_location="cpu", weights_only=False))
        return out

    # -- retention ----------------------------------------------------------

    def apply_retention(self, latest_step: Optional[int] = None) -> List[str]:
        """Delete checkpoints that don't satisfy retention rules.

        Always preserved:
          - ``final/``
          - the most-recent ``checkpoint-N-shutdown``
          - the most-recent ``checkpoint-N`` (the implicit resume candidate)

        Additionally preserved if rules are enabled:
          - the ``keep_last_k`` most-recent regular checkpoints
          - any checkpoint whose step is divisible by ``keep_milestone_every``

        Returns
        -------
        list of str
            Names of checkpoints that were deleted.
        """
        if self.keep_last_k is None and self.keep_milestone_every is None:
            return []

        ckpts = self.list_checkpoints()
        if not ckpts:
            return []

        regular = [c for c in ckpts if c[1] != "final" and not c[1].endswith("-shutdown")]
        shutdowns = [c for c in ckpts if c[1].endswith("-shutdown")]

        keep_names: set[str] = set()

        # Always keep final
        for step, name, _ in ckpts:
            if name == "final":
                keep_names.add(name)

        # Most recent shutdown
        if shutdowns:
            keep_names.add(shutdowns[-1][1])

        # Most recent regular
        if regular:
            keep_names.add(regular[-1][1])

        # keep_last_k most recent regular checkpoints
        if self.keep_last_k is not None and self.keep_last_k > 0 and regular:
            for _, name, _ in regular[-self.keep_last_k:]:
                keep_names.add(name)

        # Milestones
        if self.keep_milestone_every is not None and self.keep_milestone_every > 0:
            interval = self.keep_milestone_every
            for step, name, _ in regular:
                if step > 0 and step % interval == 0:
                    keep_names.add(name)

        deleted: List[str] = []
        for _, name, path in ckpts:
            if name in keep_names:
                continue
            try:
                shutil.rmtree(path)
                deleted.append(name)
                logger.debug(f"Retention: deleted {name}")
            except OSError as exc:
                logger.warning(f"Could not delete checkpoint {path}: {exc}")

        return deleted

    # -- output directory housekeeping --------------------------------------

    def save_tokenizer(self, tokenizer) -> Path:
        """Write the tokenizer to ``output_dir/tokenizer/`` (idempotent).

        Tokenizer is run-stable, so this is called once at run start.  If
        the directory already exists with non-empty contents, we leave it
        alone.
        """
        self.tokenizer_dir.mkdir(parents=True, exist_ok=True)
        # Check if anything is in there already
        if any(self.tokenizer_dir.iterdir()):
            return self.tokenizer_dir
        tokenizer.save_pretrained(str(self.tokenizer_dir))
        return self.tokenizer_dir

    def assert_can_train_from_scratch(self) -> None:
        """Raise CheckpointError if existing checkpoints would be clobbered.

        Used when ``resume == "scratch"`` to refuse silent destruction of
        prior work.  The CLI translates ``--force`` into deleting the
        ``checkpoints/`` directory before calling ``train()``.
        """
        existing = self.list_checkpoints()
        if existing:
            names = [n for _, n, _ in existing[:5]]
            raise CheckpointError(
                f"Cannot start from scratch: output directory "
                f"{self.output_dir} already contains checkpoints "
                f"({names}{'...' if len(existing) > 5 else ''}). "
                f"Pass --force to clobber, --training.resume auto to "
                f"continue, or pick a different --training.output_dir."
            )


__all__ = [
    "CheckpointManager",
    "CurriculumState",
    "Lineage",
    "TrainerState",
    "TrainingMetrics",
    "TrainingTiming",
    "SCHEMA_VERSION",
]
