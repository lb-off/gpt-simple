"""
Setup / build phases for :func:`gpt_simple.train.train`.

This module holds everything that runs *before* the training loop: config
resolution, resume handling, accelerator/model/optimizer/data construction,
and the small validation + run-state helpers.  The public ``train()``
orchestrator in ``train.py`` calls these in sequence; the loop itself lives
in ``_train_loop.py``.

Nothing here imports from ``train.py`` or ``_train_loop.py`` so there is no
import cycle (``train.py`` and ``_train_loop.py`` both depend on this module).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union

import torch
from accelerate import Accelerator
from accelerate.utils import set_seed

from gpt_simple._checkpoint import CheckpointManager, TrainerState, _rank_of
from gpt_simple._run_state import RunState, new_run_state
from gpt_simple.config import ARCH_KEYS, Config, DataConfig, ModelConfig, OptimizerConfig, TrainingConfig
from gpt_simple.data import StreamingDataModule
from gpt_simple.errors import CheckpointError, ConfigError, GptSimpleError
from gpt_simple.model import SimpleLLM
from gpt_simple.tokenizer import SimpleLLMTokenizer

logger = logging.getLogger("gpt_simple")

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False

try:
    from transformers import get_cosine_with_min_lr_schedule_with_warmup
except ImportError:
    try:
        from transformers.optimization import get_cosine_with_min_lr_schedule_with_warmup
    except ImportError:
        get_cosine_with_min_lr_schedule_with_warmup = None


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

@dataclass
class TrainingResult:
    """Returned by ``train()``."""
    final_loss: float
    checkpoint_path: str
    total_tokens: int
    total_steps: int


# ---------------------------------------------------------------------------
# Loop context — everything ``run_training_loop`` needs, built by the phases
# below and assembled by the ``train()`` orchestrator.
# ---------------------------------------------------------------------------

@dataclass
class TrainContext:
    cfg: Config
    accelerator: Any
    is_main: bool
    rank: int
    ckpt_mgr: CheckpointManager
    resume_path: Optional[Path]
    trainer_state: TrainerState
    tokenizer: Any
    model: Any
    opt: Any
    lr_scheduler: Any
    data_module: Any
    train_dl: Any
    eval_dl: Any
    curriculum: Any
    phase_idx: int
    phase_tokens_consumed: Any
    use_wandb: bool
    shutdown: Any
    run_state: Any
    starting_step: int


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _resolve_config(
    model: Optional[ModelConfig],
    data: Optional[DataConfig],
    optimizer: Optional[OptimizerConfig],
    training: Optional[TrainingConfig],
    config: Optional[Union[Config, str]],
) -> Config:
    """Build a Config from the various argument forms.

    If *config* is provided (path or Config object), use it as base.
    Explicitly-passed sub-configs override the corresponding section.
    """
    if config is not None:
        if isinstance(config, (str, Path)):
            base = Config.from_file(config)
        else:
            base = config
    else:
        base = Config()

    if model is not None:
        base.model = model
    if data is not None:
        base.data = data
    if optimizer is not None:
        base.optimizer = optimizer
    if training is not None:
        base.training = training

    # Re-run validation after overrides
    base._validate_schedule()
    return base


def _config_hash(cfg: Config) -> str:
    """Stable hash of the full config (for drift detection on resume)."""
    payload = json.dumps(cfg.to_dict(), sort_keys=True).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()[:16]


def _model_arch_hash(model_cfg: ModelConfig) -> str:
    """Stable hash of the model architecture only.

    Used to detect incompatible weights on resume (any change here will
    almost certainly invalidate ``pytorch_model.bin``).
    """
    payload = {k: getattr(model_cfg, k, None) for k in ARCH_KEYS}
    blob = json.dumps(payload, sort_keys=True).encode("utf-8")
    return "sha256:" + hashlib.sha256(blob).hexdigest()[:16]


def _is_rank0() -> bool:
    """Best-effort rank-0 check that works pre-Accelerator-init."""
    try:
        import torch.distributed as _d
        if _d.is_available() and _d.is_initialized():
            return _d.get_rank() == 0
    except ImportError:
        pass
    return int(os.environ.get("RANK", "0")) == 0


def _preflight_checks(cfg: Config) -> None:
    """Validate environment and config before heavy setup.

    Delegates to :func:`gpt_simple.validate.run_offline_validation` so the
    same report rendered by ``gpt-simple validate`` is what the trainer
    sees.  On rank 0 we log the formatted summary; on every rank we
    raise a typed exception if there are blocking findings, so failure
    is detected uniformly (avoids only-one-rank-crashes-then-NCCL-hangs).

    Strictness for bucket-exhaustion and curriculum-vs-loop-budget
    findings is driven by ``cfg.data.allow_bucket_exhaustion`` and
    ``cfg.data.allow_budget_mismatch`` (set them in the YAML, or via the
    matching CLI overrides on ``gpt-simple train``).
    """
    from gpt_simple.errors import DataError as _DataError
    from gpt_simple.validate import (
        Severity,
        format_report,
        run_offline_validation,
    )

    rank0 = _is_rank0()

    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    report = run_offline_validation(cfg, world_size=world_size)

    if rank0:
        for line in format_report(report).split("\n"):
            logger.info(line)

    # Stale run state detection — lives here because it needs PID liveness
    # (the offline validator can't poll the OS process table) and is a
    # different class of problem from "the config / data is invalid".
    output = cfg.training.output_dir
    prev_state = RunState.read(output)
    if prev_state is not None and prev_state.status == "running":
        pid = prev_state.pid
        try:
            os.kill(pid, 0)
            raise GptSimpleError(
                f"Another training job (PID {pid}) appears to be running "
                f"in {output}. Stop it first with 'gpt-simple stop'."
            )
        except (OSError, ProcessLookupError):
            if rank0:
                logger.warning(
                    f"Previous run (PID {pid}) in '{output}' has status 'running' "
                    "but the process is dead. It may have crashed."
                )

    # Translate findings into the existing typed-exception hierarchy so
    # the CLI exit-code logic in cli/main.py keeps working.
    if report.has_errors() or report.has_budget_issues():
        codes = [
            f.code for f in report.findings
            if f.severity in (Severity.ERROR, Severity.BUDGET)
        ]
        first = next(
            f for f in report.findings
            if f.severity in (Severity.ERROR, Severity.BUDGET)
        )
        if first.code.startswith("data.") or first.code.startswith("curriculum."):
            raise _DataError(
                f"Validation failed ({len(codes)} blocking issue(s)). "
                f"First: {first.message}"
            )
        if first.code.startswith("resume."):
            raise CheckpointError(
                f"Validation failed ({len(codes)} blocking issue(s)). "
                f"First: {first.message}"
            )
        raise ConfigError(
            f"Validation failed ({len(codes)} blocking issue(s)). "
            f"First: {first.message}"
        )

    # wandb credential check — not config-blocking, just informational.
    if cfg.training.wandb_project and rank0:
        try:
            import wandb as _wandb
            if _wandb.api.api_key is None:
                logger.warning(
                    "wandb_project is set but no API key found. "
                    "Run 'wandb login' or set WANDB_API_KEY. "
                    "Training will continue without wandb."
                )
        except ImportError:
            pass


def _runtime_preflight(
    cfg: Config,
    accelerator,
    *,
    skip_runtime_probe: bool = False,
) -> None:
    """Hardware + synthetic-batch probe; runs after accelerator init.

    Only rank 0 performs the actual probe (to avoid every rank claiming
    a second copy of the model in GPU memory simultaneously).  Failures
    are broadcast as a barrier-guarded raise so all ranks exit together.
    """
    from gpt_simple.validate import (
        format_report,
        run_runtime_validation,
        ValidationReport,
    )

    is_main = accelerator.is_main_process
    rt_report = ValidationReport()
    failed = False
    if is_main:
        try:
            run_runtime_validation(cfg, rt_report, skip_probe=skip_runtime_probe)
        except Exception as exc:
            logger.error(f"Runtime preflight crashed: {exc}")
            failed = True
        for line in format_report(rt_report).split("\n"):
            logger.info(line)

    # Broadcast pass/fail to every rank so we don't leave non-main ranks
    # hanging in the next collective op.
    fail_tensor = torch.tensor(
        [1 if (failed or rt_report.has_errors()) else 0],
        device=accelerator.device, dtype=torch.long,
    )
    try:
        import torch.distributed as _dist
        if _dist.is_available() and _dist.is_initialized():
            _dist.all_reduce(fail_tensor, op=_dist.ReduceOp.MAX)
    except ImportError:
        pass
    if int(fail_tensor.item()) > 0:
        raise ConfigError(
            "Runtime preflight failed — see the report above for details."
        )


def _validate_resume_compatibility(
    trainer_state: TrainerState,
    resume_dir: Path,
    cfg: Config,
) -> None:
    """Warn (or error) if the resume checkpoint is incompatible with cfg."""
    # Architecture hash drift: the model weights would not load cleanly.
    current_arch = _model_arch_hash(cfg.model)
    if trainer_state.model_arch_hash and trainer_state.model_arch_hash != current_arch:
        if _is_rank0():
            logger.warning(
                f"Model arch hash differs from checkpoint at {resume_dir}: "
                f"checkpoint={trainer_state.model_arch_hash} vs current={current_arch}. "
                "Weight loading is likely to fail."
            )

    # Full config drift: just informational.
    current_cfg_hash = _config_hash(cfg)
    if trainer_state.config_hash and trainer_state.config_hash != current_cfg_hash:
        if _is_rank0():
            logger.warning(
                "Config hash differs from checkpoint (some non-arch field changed). "
                "This is usually fine (e.g. you bumped max_steps to extend training)."
            )


def _resolve_mixed_precision(requested: Optional[str]) -> str:
    """Translate ``training.mixed_precision`` into an Accelerator string.

    Resolution:
      * If ``requested`` is one of {"bf16", "fp16", "no"}, use it (validated
        earlier by ``TrainingConfig.__post_init__``).
      * Otherwise auto-detect:
        - CUDA + bf16-supported (Ampere+): "bf16"
        - CUDA, no bf16 (V100, T4, …):     "fp16"
        - No CUDA (CPU or distributed gloo): "no"
    """
    if requested in {"bf16", "fp16", "no"}:
        return requested
    if not torch.cuda.is_available():
        return "no"
    return "bf16" if torch.cuda.is_bf16_supported() else "fp16"


def trainer_state_schema_version() -> int:
    """Indirection so tests can monkey-patch SCHEMA_VERSION."""
    from gpt_simple._checkpoint import SCHEMA_VERSION
    return SCHEMA_VERSION


# ---------------------------------------------------------------------------
# Build phases — each returns the object(s) it constructs.
# ---------------------------------------------------------------------------

def setup_resume(cfg: Config):
    """Build the checkpoint manager, resolve the resume path, and load
    (or initialize) the ``TrainerState``."""
    ckpt_mgr = CheckpointManager(
        output_dir=cfg.training.output_dir,
        keep_last_k=cfg.training.keep_last_k,
        keep_milestone_every=cfg.training.keep_milestone_every,
    )

    if cfg.training.resume == "scratch":
        ckpt_mgr.assert_can_train_from_scratch()
        resume_path: Optional[Path] = None
    else:
        resume_path = ckpt_mgr.resolve_resume(cfg.training.resume)

    # Load TrainerState early so we can use it everywhere downstream
    if resume_path is not None:
        trainer_state = TrainerState.load(resume_path)
        trainer_state.lineage.resumed_from = str(resume_path)
        _validate_resume_compatibility(trainer_state, resume_path, cfg)
    else:
        trainer_state = TrainerState(
            schema_version=trainer_state_schema_version(),
        )

    return ckpt_mgr, resume_path, trainer_state


def build_accelerator(cfg: Config):
    """Init the distributed process group (CPU/gloo path) if needed and
    build the ``Accelerator``.  Returns ``(accelerator, is_main, rank,
    mixed_precision)``."""
    if not torch.cuda.is_available() and "RANK" in os.environ:
        import torch.distributed as _dist_init
        if _dist_init.is_available() and not _dist_init.is_initialized():
            _dist_init.init_process_group(backend="gloo")

    _mixed_precision = _resolve_mixed_precision(cfg.training.mixed_precision)
    # We shard data per-rank ourselves inside ``StreamingDataModule`` (using
    # ``(rank, world_size, worker_id, num_workers)`` to assign disjoint
    # shards), so Accelerate must NOT dispatch batches from rank-0.  The
    # default ``dispatch_batches=None`` would gather + scatter our cursor
    # objects, which contain Python ints/dicts (not tensors) and so crash
    # Accelerate's batch concatenation.  This is also semantically wrong:
    # we want each rank to consume only its own shards.
    from accelerate import DataLoaderConfiguration
    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        mixed_precision=_mixed_precision,
        dataloader_config=DataLoaderConfiguration(
            dispatch_batches=False,
        ),
    )
    set_seed(cfg.training.seed)

    is_main = accelerator.is_main_process
    rank = _rank_of(accelerator)
    return accelerator, is_main, rank, _mixed_precision


def prepare_output_and_hashes(
    cfg: Config,
    trainer_state: TrainerState,
    resume_path: Optional[Path],
    is_main: bool,
) -> None:
    """Create the output dir, persist the resolved config, and stamp the
    config/arch hashes onto ``trainer_state``."""
    if is_main:
        os.makedirs(cfg.training.output_dir, exist_ok=True)
        cfg.save(os.path.join(cfg.training.output_dir, "config.json"))

    # Stamp the new run's config + arch hashes on trainer_state.
    # If resuming, we only OVERWRITE config_hash (the user may have bumped
    # something like max_steps).  Arch hash is preserved from the
    # checkpoint so future resumes can detect drift across the chain.
    if resume_path is None:
        trainer_state.config_hash = _config_hash(cfg)
        trainer_state.model_arch_hash = _model_arch_hash(cfg.model)
    else:
        trainer_state.config_hash = _config_hash(cfg)
        # Keep trainer_state.model_arch_hash as-is.


def load_tokenizer(cfg: Config, ckpt_mgr: CheckpointManager, is_main: bool):
    """Load the tokenizer, pad ``cfg.model.vocab_size`` to a multiple of
    128, and persist the tokenizer once per run."""
    if is_main:
        logger.info(f"Loading tokenizer: {cfg.data.tokenizer}")
    tokenizer = SimpleLLMTokenizer(cfg.data.tokenizer)

    cfg.model.vocab_size = ((tokenizer.vocab_size + 127) // 128) * 128

    # Persist tokenizer once per run (no-op if already written).
    if is_main:
        try:
            ckpt_mgr.save_tokenizer(tokenizer.tokenizer)
        except Exception as exc:
            logger.warning(f"Could not save tokenizer to {ckpt_mgr.tokenizer_dir}: {exc}")

    return tokenizer


def build_model(cfg: Config, tokenizer, resume_path: Optional[Path], is_main: bool):
    """Construct the model, load resume weights (if any), and optionally
    compile it."""
    if is_main:
        logger.info("Creating model...")
        logger.info(f"  Vocab size: {cfg.model.vocab_size} (tokenizer: {tokenizer.vocab_size}, padded to 128)")
        logger.info(f"  Hidden size: {cfg.model.n_embd}")
        logger.info(f"  Layers: {cfg.model.n_layer}")
        logger.info(f"  Attention heads: {cfg.model.n_head}")

    llm = SimpleLLM(cfg.model, gradient_checkpointing=cfg.training.gradient_checkpointing)

    # Resume: load model weights BEFORE compile and BEFORE accelerator.prepare()
    # so the state-dict keys match the raw module (no DDP/compile prefixes).
    if resume_path is not None:
        if is_main:
            logger.info(f"Loading model weights from {resume_path}")
        CheckpointManager.load_model_state(llm, resume_path)

    if cfg.training.compile:
        # Dynamo's DDPOptimizer splits the graph at DDP gradient-bucket
        # boundaries and compiles each subgraph separately, but it does NOT
        # support the higher-order ops that activation/gradient checkpointing
        # lowers to -> BackendCompilerFailed ("RuntimeError: val") on the
        # attention subgraph under DDP.  Disable the DDP graph-splitter so the
        # whole module compiles as a single graph; checkpointing then works.
        # See pytorch/pytorch#104674.  No-op without DDP (single process); the
        # only cost under DDP is slightly less comm/compute overlap (one bucket
        # for the whole graph), which is minor on a single NVLink node.
        if cfg.training.gradient_checkpointing:
            import torch._dynamo as _dynamo
            _dynamo.config.optimize_ddp = False
            if is_main:
                logger.info(
                    "Disabled Dynamo DDPOptimizer (optimize_ddp=False) for "
                    "torch.compile + gradient-checkpointing compatibility."
                )
        try:
            if is_main:
                logger.info("Compiling model with torch.compile...")
            llm = torch.compile(llm, mode="default", fullgraph=False)
            if is_main:
                logger.info("Model compiled successfully!")
        except Exception as e:
            if is_main:
                logger.warning(f"Could not compile model: {e}")

    return llm


def init_wandb(cfg: Config, trainer_state: TrainerState, is_main: bool) -> bool:
    """Initialize Weights & Biases if configured.  Returns whether wandb
    logging is active for this run."""
    use_wandb = bool(cfg.training.wandb_project)
    if use_wandb and is_main:
        if not _WANDB_AVAILABLE:
            logger.warning("wandb not installed. Continuing without logging.")
            use_wandb = False
        else:
            run_name = cfg.training.wandb_run_name
            if run_name is None:
                from datetime import datetime
                run_name = f"gpt_simple_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            init_kwargs = dict(
                project=cfg.training.wandb_project,
                name=run_name,
                config=cfg.to_dict(),
            )
            if trainer_state.wandb_run_id:
                init_kwargs["id"] = trainer_state.wandb_run_id
                init_kwargs["resume"] = "allow"
                logger.info(f"Resuming W&B run: {trainer_state.wandb_run_id}")
            run = wandb.init(**init_kwargs)
            trainer_state.wandb_run_id = getattr(run, "id", trainer_state.wandb_run_id)
            logger.info(f"Weights & Biases initialized: {cfg.training.wandb_project}/{run_name}")

    return use_wandb


def build_optimizer_and_scheduler(cfg: Config, llm):
    """Build the AdamW optimizer (with weight-decay param grouping) and the
    cosine LR schedule.  Returns ``(opt, lr_scheduler, decay_steps)``."""
    decay_params = [p for _, p in llm.named_parameters() if p.requires_grad and p.dim() >= 2]
    no_decay_params = [p for _, p in llm.named_parameters() if p.requires_grad and p.dim() < 2]
    opt = torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": cfg.optimizer.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=cfg.optimizer.learning_rate,
        betas=(cfg.optimizer.beta1, cfg.optimizer.beta2),
        eps=cfg.optimizer.eps,
    )

    decay_steps = (
        cfg.optimizer.decay_steps
        if cfg.optimizer.decay_steps is not None
        else cfg.training.max_steps - cfg.optimizer.warmup_steps
    )
    total_schedule_steps = cfg.optimizer.warmup_steps + decay_steps

    if get_cosine_with_min_lr_schedule_with_warmup is None:
        raise GptSimpleError("transformers >= 4.35 is required for the LR scheduler")

    # We step the (unprepared) scheduler exactly once per training step
    # (gated by ``accelerator.sync_gradients`` in the loop), so the
    # underlying schedule counts in *training-step* units — no
    # multiplication by gradient_accumulation_steps or num_processes here.
    lr_scheduler = get_cosine_with_min_lr_schedule_with_warmup(
        opt,
        num_warmup_steps=cfg.optimizer.warmup_steps,
        num_training_steps=total_schedule_steps,
        min_lr_rate=cfg.optimizer.min_lr_ratio,
    )

    return opt, lr_scheduler, decay_steps


def build_data(
    cfg: Config,
    tokenizer,
    accelerator,
    trainer_state: TrainerState,
    resume_path: Optional[Path],
    is_main: bool,
):
    """Build the streaming data module + train/val dataloaders, advancing
    the curriculum to the resumed phase if needed.  Returns
    ``(data_module, train_dl, eval_dl, curriculum, phase_idx,
    phase_tokens_consumed)``."""
    data_module = StreamingDataModule(
        config=cfg.data,
        tokenizer=tokenizer,
        world_size=accelerator.num_processes,
        attention_mode=cfg.model.attention_mode,
        per_device_batch_size=cfg.training.per_device_batch_size,
    )
    data_module.prepare_data()

    # If resuming and we have a curriculum, advance the data module to the
    # right phase BEFORE building the dataloader.
    curriculum = cfg.data.curriculum
    phase_idx = trainer_state.curriculum.phase_idx if curriculum else 0
    phase_tokens_consumed = trainer_state.curriculum.phase_tokens_consumed if curriculum else 0

    if curriculum and resume_path is not None and phase_idx > 0:
        data_module.set_phase(phase_idx)
        if is_main:
            logger.info(
                f"Resumed curriculum: phase {phase_idx}, "
                f"{phase_tokens_consumed:,} tokens into phase"
            )

    train_dl = data_module.train_dataloader()
    eval_dl = data_module.val_dataloader()

    return data_module, train_dl, eval_dl, curriculum, phase_idx, phase_tokens_consumed


def restore_after_prepare(
    cfg: Config,
    ckpt_mgr: CheckpointManager,
    data_module,
    opt,
    lr_scheduler,
    resume_path: Optional[Path],
    rank: int,
    is_main: bool,
    train_dl,
):
    """Restore optimizer/scheduler/RNG/dataloader state from the resume
    checkpoint (must run AFTER ``accelerator.prepare``).  Returns the
    (possibly rebuilt) train dataloader."""
    if resume_path is not None:
        if is_main:
            logger.info(f"Loading optimizer/scheduler/RNG from {resume_path}")
        CheckpointManager.load_optimizer_state(opt, resume_path)
        CheckpointManager.load_scheduler_state(lr_scheduler, resume_path)
        CheckpointManager.load_rng_state(rank, resume_path)

        # Load *every* rank's dataloader state so we can support arbitrary
        # changes of (world_size, num_workers) between runs: the data
        # module unions all per-rank file-progress entries into a
        # topology-agnostic global map and redistributes it to the new
        # slots.  Returns an empty list if no state is on disk yet.
        all_dl_states = CheckpointManager.load_all_dataloader_states(resume_path)
        if all_dl_states:
            data_module.apply_dataloader_state(all_dl_states)
            if is_main:
                total_workers = sum(
                    len(s.get("worker_states") or {}) for s in all_dl_states
                )
                saved_topos = sorted({
                    (s.get("world_size"), s.get("num_workers")) for s in all_dl_states
                })
                logger.info(
                    f"Restored dataloader cursors from checkpoint "
                    f"({len(all_dl_states)} rank file(s), {total_workers} worker slot(s), "
                    f"saved topology={saved_topos})."
                )
            # Rebuild the train DataLoader so it picks up the newly-injected
            # per-worker cursors.  We deliberately do NOT prepare it
            # (see the long-form note next to the main prepare call above).
            train_dl = data_module.train_dataloader()

    return train_dl


def print_training_summary(
    cfg: Config,
    accelerator,
    resume_path: Optional[Path],
    starting_step: int,
    decay_steps: int,
    is_main: bool,
) -> None:
    """Log the pre-loop run summary (rank 0 only)."""
    if not is_main:
        return
    eff_bs = (
        cfg.training.per_device_batch_size
        * cfg.training.gradient_accumulation_steps
        * accelerator.num_processes
    )
    logger.info("=" * 60)
    logger.info("GPT-SIMPLE TRAINING")
    logger.info("=" * 60)
    logger.info(f"Dataset:              {cfg.data.path}")
    logger.info(f"Output:               {cfg.training.output_dir}")
    logger.info(f"Devices:              {accelerator.num_processes}")
    logger.info(f"Per-device batch size: {cfg.training.per_device_batch_size}")
    logger.info(f"Gradient accumulation: {cfg.training.gradient_accumulation_steps}")
    logger.info(f"Effective batch size:  {eff_bs}")
    logger.info(f"Max steps:            {cfg.training.max_steps}")
    logger.info(f"LR: {cfg.optimizer.learning_rate}  (warmup={cfg.optimizer.warmup_steps}, decay={decay_steps})")
    if resume_path is not None:
        logger.info(f"Resuming from step {starting_step}  (checkpoint: {resume_path})")
    logger.info("=" * 60)


def maybe_already_complete_result(
    cfg: Config,
    trainer_state: TrainerState,
    resume_path: Optional[Path],
    is_main: bool,
    starting_step: int,
) -> Optional[TrainingResult]:
    """If the resumed checkpoint already reached ``max_steps``, refresh the
    on-disk run_state and return a terminal ``TrainingResult``.  Otherwise
    return ``None`` so the caller proceeds into the loop."""
    if starting_step < cfg.training.max_steps:
        return None

    if is_main:
        logger.info(
            f"Training already complete: step {starting_step} >= max_steps "
            f"{cfg.training.max_steps}.  Bump max_steps to continue."
        )
        # Refresh the on-disk run_state so `gpt-simple status` reports
        # the current reality.  Without this, any stale error state
        # left behind by a previous failed attempt would persist (the
        # short-circuit is the one return path in train() that
        # otherwise never touches run_state).
        completed_state = new_run_state(
            max_steps=cfg.training.max_steps,
            config_path=os.path.join(cfg.training.output_dir, "config.json"),
        )
        completed_state.status = "completed"
        completed_state.global_step = starting_step
        completed_state.tokens_trained = trainer_state.tokens_trained
        completed_state.loss = trainer_state.metrics.loss
        completed_state.learning_rate = trainer_state.metrics.learning_rate
        completed_state.tokens_per_sec = trainer_state.metrics.tokens_per_sec
        if resume_path is not None:
            completed_state.latest_checkpoint = str(resume_path)
        try:
            completed_state.write(cfg.training.output_dir)
        except OSError as exc:
            logger.warning(f"Could not refresh run_state: {exc}")

    return TrainingResult(
        final_loss=trainer_state.metrics.loss,
        checkpoint_path=str(resume_path) if resume_path is not None else "",
        total_tokens=trainer_state.tokens_trained,
        total_steps=starting_step,
    )


def init_run_state(
    cfg: Config,
    trainer_state: TrainerState,
    resume_path: Optional[Path],
    is_main: bool,
    starting_step: int,
) -> Optional[RunState]:
    """Create and persist the initial ``RunState`` (rank 0 only)."""
    run_state: Optional[RunState] = None
    if is_main:
        run_state = new_run_state(
            max_steps=cfg.training.max_steps,
            config_path=os.path.join(cfg.training.output_dir, "config.json"),
        )
        run_state.global_step = starting_step
        run_state.tokens_trained = trainer_state.tokens_trained
        if resume_path is not None:
            run_state.latest_checkpoint = str(resume_path)
        run_state.write(cfg.training.output_dir)
    return run_state
