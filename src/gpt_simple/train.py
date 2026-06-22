"""
Training entry point for gpt_simple.

Public API::

    import gpt_simple
    result = gpt_simple.train(config="config.yaml")

    # or programmatically
    result = gpt_simple.train(
        model=gpt_simple.ModelConfig(n_embd=768),
        data=gpt_simple.DataConfig(path="/data/tokenized"),
        optimizer=gpt_simple.OptimizerConfig(learning_rate=3e-4),
        training=gpt_simple.TrainingConfig(max_steps=1000),
    )

The default behavior is auto-resume: rerunning the same command picks up
from the latest checkpoint under ``output_dir/checkpoints/`` if one
exists, otherwise it starts fresh.  See ``TrainingConfig.resume`` to
control this.

This module is a thin orchestrator: ``train()`` resolves the config, runs
the build phases (in ``_train_setup.py``), then hands a ``TrainContext`` to
``run_training_loop`` (in ``_train_loop.py``).  The names those modules
define are re-exported here so ``gpt_simple.train.<name>`` keeps working.
"""

from __future__ import annotations

import sys
import time
from typing import Optional, Union

from gpt_simple._run_state import RunState
from gpt_simple._shutdown import ShutdownCoordinator
from gpt_simple.config import Config, DataConfig, ModelConfig, OptimizerConfig, TrainingConfig
from gpt_simple.errors import DataError

from gpt_simple._train_setup import (
    TrainContext,
    TrainingResult,
    _config_hash,
    _is_rank0,
    _model_arch_hash,
    _preflight_checks,
    _resolve_config,
    _resolve_mixed_precision,
    _runtime_preflight,
    _validate_resume_compatibility,
    build_accelerator,
    build_data,
    build_model,
    build_optimizer_and_scheduler,
    init_run_state,
    init_wandb,
    load_tokenizer,
    maybe_already_complete_result,
    prepare_output_and_hashes,
    print_training_summary,
    restore_after_prepare,
    setup_resume,
    trainer_state_schema_version,
)
from gpt_simple._train_loop import (
    _compute_loss,
    _evaluate,
    _generate_samples,
    _move_to_device,
    run_training_loop,
)

__all__ = [
    "train",
    "TrainingResult",
    "TrainContext",
    "write_error_state",
    "run_training_loop",
    # Re-exported build phases / helpers (kept on ``gpt_simple.train`` for
    # backward compatibility with callers and tests).
    "_resolve_config",
    "_config_hash",
    "_model_arch_hash",
    "_is_rank0",
    "_preflight_checks",
    "_runtime_preflight",
    "_validate_resume_compatibility",
    "_resolve_mixed_precision",
    "trainer_state_schema_version",
    "setup_resume",
    "build_accelerator",
    "prepare_output_and_hashes",
    "load_tokenizer",
    "build_model",
    "init_wandb",
    "build_optimizer_and_scheduler",
    "build_data",
    "restore_after_prepare",
    "print_training_summary",
    "maybe_already_complete_result",
    "init_run_state",
    "_move_to_device",
    "_compute_loss",
    "_generate_samples",
    "_evaluate",
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def train(
    model: Optional[ModelConfig] = None,
    data: Optional[DataConfig] = None,
    optimizer: Optional[OptimizerConfig] = None,
    training: Optional[TrainingConfig] = None,
    config: Optional[Union[Config, str]] = None,
    *,
    skip_runtime_probe: bool = False,
) -> TrainingResult:
    """Run a pretraining job.

    Parameters
    ----------
    model, data, optimizer, training
        Individual sub-configs.  Override the corresponding section in *config*.
    config
        A ``Config`` object or a path to a YAML/JSON file.  Provides defaults
        for any sub-config not explicitly passed.

    Returns
    -------
    TrainingResult
        Summary of the completed run.
    """
    cfg = _resolve_config(model, data, optimizer, training, config)

    if not cfg.data.path:
        raise DataError("DataConfig.path is required")

    _preflight_checks(cfg)

    # -- build phases -------------------------------------------------------

    ckpt_mgr, resume_path, trainer_state = setup_resume(cfg)
    accelerator, is_main, rank, _mixed_precision = build_accelerator(cfg)

    # Hardware probe + synthetic forward/backward.  Runs before any real
    # model is built so that, if it OOMs, the OOM error is the LAST
    # thing in the log instead of being followed by a misleading torch
    # stack from the real training loop.
    _runtime_preflight(cfg, accelerator, skip_runtime_probe=skip_runtime_probe)

    prepare_output_and_hashes(cfg, trainer_state, resume_path, is_main)

    tokenizer = load_tokenizer(cfg, ckpt_mgr, is_main)
    llm = build_model(cfg, tokenizer, resume_path, is_main)
    use_wandb = init_wandb(cfg, trainer_state, is_main)
    opt, lr_scheduler, decay_steps = build_optimizer_and_scheduler(cfg, llm)
    data_module, train_dl, eval_dl, curriculum, phase_idx, phase_tokens_consumed = build_data(
        cfg, tokenizer, accelerator, trainer_state, resume_path, is_main
    )

    # -- accelerator prepare ------------------------------------------------
    # We only prepare the model and optimizer.  We deliberately do NOT route
    # the dataloaders or the LR scheduler through ``accelerator.prepare``.
    #
    # Why:
    #   - Dataloaders: ``StreamingDataModule`` already shards by
    #     ``(rank, worker_id, world_size, num_workers)`` inside
    #     ``PreTokenizedDataset``.  Accelerate's ``IterableDatasetShard``
    #     wrapper would *re-shard* the already-sharded stream by buffering
    #     ``batch_size * num_processes`` items and yielding only
    #     ``batch_size`` of them — discarding most of the data (7/8 on 8
    #     GPUs) and exhausting the dataset far too early.
    #   - Scheduler: ``AcceleratedScheduler`` advances the underlying
    #     scheduler by ``num_processes`` substeps per ``.step()`` call when
    #     ``split_batches=False``, on the assumption that the effective
    #     batch size scaled with ``num_processes``.  But our per-rank batch
    #     size is independent, so we'd race through warmup/decay
    #     ``num_processes``-times too fast.
    # We move batches to device manually inside the loop, and call
    # ``lr_scheduler.step()`` once per sync-gradients step.
    llm, opt = accelerator.prepare(llm, opt)

    train_dl = restore_after_prepare(
        cfg, ckpt_mgr, data_module, opt, lr_scheduler, resume_path, rank, is_main, train_dl
    )

    # -- graceful shutdown coordinator --------------------------------------

    loop_start_monotonic = time.monotonic()
    shutdown = ShutdownCoordinator(
        accelerator=accelerator,
        output_dir=cfg.training.output_dir,
        max_runtime_seconds=cfg.training.max_runtime_seconds,
        walltime_reserve_seconds=cfg.training.walltime_reserve_seconds,
        loop_start_monotonic=loop_start_monotonic,
    )
    shutdown.install_signal_handlers()
    shutdown.clear_flag_file()  # wipe any stale flag from a previous run

    starting_step = trainer_state.step
    print_training_summary(cfg, accelerator, resume_path, starting_step, decay_steps, is_main)

    # Already-complete short-circuit on resume
    already = maybe_already_complete_result(cfg, trainer_state, resume_path, is_main, starting_step)
    if already is not None:
        return already

    run_state = init_run_state(cfg, trainer_state, resume_path, is_main, starting_step)

    # -- run the loop -------------------------------------------------------

    ctx = TrainContext(
        cfg=cfg,
        accelerator=accelerator,
        is_main=is_main,
        rank=rank,
        ckpt_mgr=ckpt_mgr,
        resume_path=resume_path,
        trainer_state=trainer_state,
        tokenizer=tokenizer,
        model=llm,
        opt=opt,
        lr_scheduler=lr_scheduler,
        data_module=data_module,
        train_dl=train_dl,
        eval_dl=eval_dl,
        curriculum=curriculum,
        phase_idx=phase_idx,
        phase_tokens_consumed=phase_tokens_consumed,
        use_wandb=use_wandb,
        shutdown=shutdown,
        run_state=run_state,
        starting_step=starting_step,
    )
    return run_training_loop(ctx)


def write_error_state(output_dir: str, error: Exception) -> None:
    """Write an error run-state to *output_dir*.

    Called by the CLI layer when ``train()`` raises.
    """
    import traceback as _tb

    state = RunState.read(output_dir)
    if state is None:
        state = RunState()
    state.status = "error"
    state.error = "".join(_tb.format_exception(type(error), error, error.__traceback__))
    state.write(output_dir)


# ---------------------------------------------------------------------------
# Module CLI — the implementation behind ``python -m gpt_simple``.
# The actual entry point lives in gpt_simple/__main__.py, which keeps this
# module a pure library import (re-exported from __init__). The distributed
# launcher shells out to ``-m gpt_simple`` (see cli/train_cmd.py).
# ---------------------------------------------------------------------------

def _module_main() -> None:
    """CLI for ``python -m gpt_simple`` (and the distributed launcher)."""
    from gpt_simple.cli.train_cmd import _ensure_hostname_resolves
    _ensure_hostname_resolves()

    import argparse

    parser = argparse.ArgumentParser(description="gpt_simple training (module entry)")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML/JSON config")

    parser.add_argument("--training.max_steps", type=int, default=None)
    parser.add_argument("--training.output_dir", type=str, default=None)
    parser.add_argument("--training.resume", type=str, default=None)
    parser.add_argument("--training.seed", type=int, default=None)
    parser.add_argument("--training.keep_last_k", type=int, default=None)
    parser.add_argument("--training.max_runtime_seconds", type=int, default=None)
    parser.add_argument("--training.walltime_reserve_seconds", type=int, default=None)
    parser.add_argument("--training.wandb_project", type=str, default=None)
    parser.add_argument("--training.wandb_run_name", type=str, default=None)
    parser.add_argument("--optimizer.learning_rate", type=float, default=None)
    parser.add_argument("--data.path", type=str, default=None)
    parser.add_argument("--data.tokenizer", type=str, default=None)
    parser.add_argument("--data.format", type=str, default=None)

    # Boolean overrides for the two curriculum-validation policy fields.
    # These set cfg.data.allow_* = True for this run (config remains the
    # source of truth; this is the override channel).
    parser.add_argument("--allow-bucket-exhaustion", action="store_true")
    parser.add_argument("--allow-budget-mismatch", action="store_true")
    parser.add_argument("--skip-runtime-probe", action="store_true")

    args = parser.parse_args()

    from gpt_simple._logging import setup_logging
    setup_logging()

    cfg = Config.from_file(args.config)

    for section_name, section_obj in [
        ("training", cfg.training),
        ("optimizer", cfg.optimizer),
        ("data", cfg.data),
    ]:
        for attr in vars(section_obj):
            key = f"{section_name}.{attr}"
            val = getattr(args, key, None)
            if val is not None:
                setattr(section_obj, attr, val)

    cfg._validate_schedule()

    if getattr(args, "allow_bucket_exhaustion", False):
        cfg.data.allow_bucket_exhaustion = True
    if getattr(args, "allow_budget_mismatch", False):
        cfg.data.allow_budget_mismatch = True

    try:
        train(
            config=cfg,
            skip_runtime_probe=bool(getattr(args, "skip_runtime_probe", False)),
        )
    except Exception:
        try:
            write_error_state(cfg.training.output_dir, sys.exc_info()[1])
        except Exception:
            pass
        raise
