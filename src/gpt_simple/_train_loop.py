"""
Training loop for :func:`gpt_simple.train.train`.

Holds the per-step helpers, the checkpoint-snapshot helpers, and the main
``run_training_loop`` while-loop.  The loop's rapidly-changing state (step
counter, token totals, interval accumulators, dataloader cursors, ...) lives
in the :class:`_LoopState` dataclass so the snapshot helpers, which used to be
closures inside ``train()``, can be plain module-level functions.

Depends only on ``_train_setup`` (for :class:`TrainContext` /
:class:`TrainingResult`); nothing here imports ``train.py``.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional

import torch

from gpt_simple import data as _data_mod
from gpt_simple._checkpoint import TrainerState
from gpt_simple._train_setup import TrainContext, TrainingResult
from gpt_simple.errors import DataError

logger = logging.getLogger("gpt_simple")

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False


# ---------------------------------------------------------------------------
# Bucket-exhaustion policy (see ``data.allow_bucket_exhaustion``)
# ---------------------------------------------------------------------------

def _local_bucket_code(newly_exhausted, bucket_to_id: Dict[str, int]) -> int:
    """Encode this rank's newly-exhausted buckets as a single int for an
    ``all_reduce(MAX)``.

    Returns ``0`` when nothing new is exhausted, otherwise ``bucket_id + 1``
    of the highest-id exhausted bucket.  The ``+1`` keeps ``0`` reserved for
    "none", and taking the MAX lets every rank agree on a representative
    bucket even when it ran dry on a different rank first.
    """
    return max((bucket_to_id.get(b, -1) for b in newly_exhausted), default=-1) + 1


def _bucket_for_code(reduced_code: int, id_to_bucket: Dict[int, str]) -> Optional[str]:
    """Inverse of :func:`_local_bucket_code` after the all-reduce.

    Returns the bucket name when some rank reported an exhaustion, else
    ``None``.  An out-of-range code maps to ``"<unknown>"`` rather than
    failing — the halt still fires, just with a vaguer name.
    """
    if reduced_code <= 0:
        return None
    return id_to_bucket.get(reduced_code - 1, "<unknown>")


def _shutdown_status(reason: Optional[str]) -> str:
    """Map a shutdown reason to the terminal run-state status.

    A bucket exhaustion is ``"halted"`` — terminal and needing attention, so
    the resume chain must NOT auto-resubmit.  Every other reason (walltime,
    signals, flag file) is an ordinary ``"stopped"`` that the chain resumes.
    """
    return "halted" if (reason or "").startswith("bucket_exhausted") else "stopped"


# ---------------------------------------------------------------------------
# Per-step helpers
# ---------------------------------------------------------------------------

def _move_to_device(batch: dict, device) -> dict:
    """Move tensor values in *batch* to *device*, pass non-tensors through.

    We do this manually instead of via ``accelerator.prepare(dataloader)``
    because that path would also install a re-sharding wrapper
    (``IterableDatasetShard``) that double-shards the already-rank-specific
    stream emitted by ``StreamingDataModule``.  See the long comment next
    to the ``accelerator.prepare`` call in ``train()`` for the full
    rationale.

    Non-tensor entries (the ``_cursor`` Python object, ``bucket_id`` ints,
    etc.) are returned unchanged.
    """
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


def _compute_loss(model, batch):
    outputs = model(**batch, return_dict=True)
    return outputs.loss


@torch.no_grad()
def _generate_samples(model, tokenizer, accelerator, max_new_tokens=100):
    if not accelerator.is_main_process:
        return []

    prompts = [
        "The future of artificial intelligence is",
        "Once upon a time in a distant land,",
        "The most important thing in life is",
        "#This function computes the nth Fibonacci number\n",
    ]

    model.eval()
    samples = []

    for prompt in prompts:
        inputs = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=True)
        inputs = inputs.to(accelerator.device)

        try:
            unwrapped = model
            if hasattr(unwrapped, "module"):
                unwrapped = unwrapped.module
            if hasattr(unwrapped, "_orig_mod"):
                unwrapped = unwrapped._orig_mod

            output_ids = unwrapped.generate(
                inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.8,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            samples.append((prompt, text))
        except Exception as e:
            samples.append((prompt, f"[Generation failed: {e}]"))

    model.train()
    return samples


@torch.no_grad()
def _evaluate(model, eval_dataloader, accelerator, max_eval_steps=100):
    model.eval()
    losses = []

    for step, batch in enumerate(eval_dataloader):
        if step >= max_eval_steps:
            break
        # Eval batches travel through the same unprepared DataLoader path
        # as training, so we device-place them manually.  Cursors / int
        # bucket ids are stripped before forward.
        batch = _move_to_device(batch, accelerator.device)
        batch.pop("_cursor", None)
        batch.pop("bucket_id", None)
        loss = _compute_loss(model, batch)
        losses.append(accelerator.gather(loss.unsqueeze(0)))

    model.train()

    if losses:
        return torch.cat(losses).mean().item()
    return float("inf")


# ---------------------------------------------------------------------------
# Loop state + checkpoint snapshot helpers (formerly closures in train())
# ---------------------------------------------------------------------------

@dataclass
class _LoopState:
    """Mutable state carried across iterations of the training loop."""
    global_step: int
    avg_loss: float
    total_tokens_trained: Any
    phase_idx: int
    phase_tokens_consumed: Any
    step_start_time: float
    loop_start_time: float
    last_checkpoint_path: str
    interval_loss: float = 0.0
    interval_grad_norm: float = 0.0
    interval_non_pad: int = 0
    interval_possible: int = 0
    interval_steps: int = 0          # actual steps since last log (≠ logging_steps after a resume)
    step_non_pad: int = 0
    pending_worker_states: Dict[int, Any] = field(default_factory=dict)
    committed_worker_states: Dict[int, Any] = field(default_factory=dict)
    # Buckets already seen exhausted (via the dataloader cursor), so each one
    # is acted on / logged only once.
    seen_exhausted_buckets: set = field(default_factory=set)


def _build_trainer_state(
    ctx: TrainContext,
    ls: _LoopState,
    loss: float,
    lr: float,
    grad_norm: float,
    tps: float,
) -> TrainerState:
    """Snapshot current loop state into a TrainerState for saving."""
    wallclock = (
        ctx.trainer_state.timing.wallclock_seconds_elapsed
        + (time.monotonic() - ls.loop_start_time)
    )
    ts = TrainerState(
        step=ls.global_step,
        tokens_trained=int(ls.total_tokens_trained),
        wandb_run_id=ctx.trainer_state.wandb_run_id,
        config_hash=ctx.trainer_state.config_hash,
        model_arch_hash=ctx.trainer_state.model_arch_hash,
    )
    ts.curriculum.phase_idx = ls.phase_idx
    ts.curriculum.phase_tokens_consumed = int(ls.phase_tokens_consumed)
    if ctx.curriculum and ls.phase_idx < len(ctx.curriculum):
        ts.curriculum.current_mix = dict(ctx.curriculum[ls.phase_idx].mix)
    ts.timing.wallclock_seconds_elapsed = wallclock
    ts.metrics.loss = float(loss)
    ts.metrics.learning_rate = float(lr)
    ts.metrics.grad_norm = float(grad_norm)
    ts.metrics.tokens_per_sec = float(tps)
    ts.lineage.resumed_from = ctx.trainer_state.lineage.resumed_from
    return ts


def _save_now(ctx: TrainContext, ls: _LoopState, tag: Optional[str]) -> str:
    """Save a checkpoint and update run_state.  Returns the new path."""
    ts = _build_trainer_state(
        ctx,
        ls,
        loss=ls.avg_loss,
        lr=ctx.lr_scheduler.get_last_lr()[0],
        grad_norm=0.0,
        tps=0.0,
    )
    dl_state = ctx.data_module.make_dataloader_state(ls.committed_worker_states, rank=ctx.rank)
    ckpt_dir = ctx.ckpt_mgr.save(
        accelerator=ctx.accelerator,
        model=ctx.model,
        optimizer=ctx.opt,
        scheduler=ctx.lr_scheduler,
        trainer_state=ts,
        model_config_dict=asdict(ctx.cfg.model),
        tag=tag,
        dataloader_state=dl_state,
    )
    return str(ckpt_dir)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run_training_loop(ctx: TrainContext) -> TrainingResult:
    """Run the training loop to completion (or until a coordinated stop) and
    return the :class:`TrainingResult`."""
    cfg = ctx.cfg
    accelerator = ctx.accelerator
    is_main = ctx.is_main
    llm = ctx.model
    opt = ctx.opt
    lr_scheduler = ctx.lr_scheduler
    data_module = ctx.data_module
    tokenizer = ctx.tokenizer
    shutdown = ctx.shutdown
    run_state = ctx.run_state
    curriculum = ctx.curriculum
    use_wandb = ctx.use_wandb
    eval_dl = ctx.eval_dl
    train_dl = ctx.train_dl

    llm.train()
    ls = _LoopState(
        global_step=ctx.starting_step,
        avg_loss=ctx.trainer_state.metrics.loss,
        total_tokens_trained=ctx.trainer_state.tokens_trained,
        phase_idx=ctx.phase_idx,
        phase_tokens_consumed=ctx.phase_tokens_consumed,
        step_start_time=time.time(),
        loop_start_time=time.monotonic(),
        last_checkpoint_path=str(ctx.resume_path) if ctx.resume_path is not None else "",
    )

    train_iter = iter(train_dl)

    while ls.global_step < cfg.training.max_steps:
        # -- shutdown check (signals + walltime + flag file + all-reduce) ---
        if shutdown.should_shutdown():
            ls.last_checkpoint_path = _save_now(ctx, ls, tag="shutdown")

            # A bucket-exhaustion halt is terminal: it needs user attention
            # (the data ran short), so it must NOT trigger an auto-resume in
            # the chain.  ``status="halted"`` is what the resume-chain script
            # greps for to stop resubmitting.
            status = _shutdown_status(shutdown.reason)
            if is_main:
                logger.info(
                    f"{'Halted' if status == 'halted' else 'Stopped'} at step "
                    f"{ls.global_step} ({shutdown.reason}). "
                    f"Resume with: --training.resume {ls.last_checkpoint_path}"
                )
                if run_state is not None:
                    run_state.status = status
                    run_state.global_step = ls.global_step
                    run_state.tokens_trained = int(ls.total_tokens_trained)
                    run_state.latest_checkpoint = ls.last_checkpoint_path
                    run_state.write(cfg.training.output_dir)
            shutdown.clear_flag_file()
            break

        # Note: a single bucket running dry does NOT stop training — the
        # sampler drops it and renormalises over the survivors (logging the
        # drift once).  Training ends only when *all* buckets are exhausted,
        # which surfaces below as ``StopIteration`` from ``next(train_iter)``.

        # -- forward / backward ---------------------------------------------
        # NOTE: NCCL all_reduce requires CUDA tensors.  Building these on the
        # accelerator's device keeps both backends happy: gloo accepts CPU
        # tensors, NCCL needs them on the local GPU.
        _reduce_device = accelerator.device
        try:
            batch = next(train_iter)
            # The DataLoader is NOT routed through accelerator.prepare (see
            # the rationale next to the prepare call), so it doesn't move
            # tensors to the accelerator device for us.  We do it here.
            batch = _move_to_device(batch, _reduce_device)
            local_exhausted = torch.tensor(0, dtype=torch.int32, device=_reduce_device)
        except StopIteration:
            local_exhausted = torch.tensor(1, dtype=torch.int32, device=_reduce_device)

        try:
            import torch.distributed as _dist_exhaust
            if _dist_exhaust.is_available() and _dist_exhaust.is_initialized():
                _dist_exhaust.all_reduce(local_exhausted, op=_dist_exhaust.ReduceOp.MAX)
        except ImportError:
            pass

        if local_exhausted.item() > 0:
            if ls.global_step == 0:
                # Step 0 exhaustion is never legitimate — it means at least
                # one rank's DataLoader was empty from the start
                raise DataError(
                    "Training exhausted before completing a single step. "
                    "This usually means at least one (rank, worker) slot "
                    "got zero shards.  Check that every train bucket has "
                    "at least `world_size` shards and that `num_workers` "
                    "isn't larger than `num_shards // world_size`."
                )
            if is_main:
                logger.info("Dataset exhausted (coordinated stop across all ranks).")
            break

        batch.pop("bucket_id", None)

        # Extract the dataloader cursor (if present) before forwarding to the
        # model.  ``cursor_aware_collate`` puts a (worker_id, WorkerDataState)
        # tuple at batch["_cursor"]; pretokenized path only.
        _batch_cursor = batch.pop("_cursor", None)
        _wstate = None
        if _batch_cursor is not None:
            _worker_id, _wstate = _batch_cursor
            if _worker_id is not None:
                ls.pending_worker_states[int(_worker_id)] = _wstate

        # -- bucket-exhaustion policy ---------------------------------------
        # A drained bucket arrives via the cursor's ``exhausted_buckets``.
        #   * allow_bucket_exhaustion=True  -> renormalise & keep training
        #     (the loader already dropped it); just log the mix drift once.
        #   * allow_bucket_exhaustion=False -> halt with a checkpoint rather
        #     than silently change the mix.  This is coordinated across ranks
        #     (a bucket may run dry on a non-zero rank first) so every rank
        #     requests the same shutdown and writes a consistent reason.
        _exhausted_now = list(_wstate.exhausted_buckets) if _wstate is not None else []
        _newly = [b for b in _exhausted_now if b not in ls.seen_exhausted_buckets]
        ls.seen_exhausted_buckets.update(_newly)
        if cfg.data.allow_bucket_exhaustion:
            if _newly and is_main:
                logger.warning(
                    "Bucket(s) %s exhausted; mix renormalised over remaining "
                    "buckets (data.allow_bucket_exhaustion=true).", _newly,
                )
        else:
            code_t = torch.tensor(
                _local_bucket_code(_newly, _data_mod.BUCKET_TO_ID),
                dtype=torch.int32, device=accelerator.device,
            )
            try:
                import torch.distributed as _dist_bucket
                if _dist_bucket.is_available() and _dist_bucket.is_initialized():
                    _dist_bucket.all_reduce(code_t, op=_dist_bucket.ReduceOp.MAX)
            except ImportError:
                pass
            rep = _bucket_for_code(int(code_t.item()), _data_mod.ID_TO_BUCKET)
            if rep is not None:
                shutdown.request_shutdown(f"bucket_exhausted:{rep}")
                if is_main:
                    logger.warning(
                        "Bucket %r exhausted and data.allow_bucket_exhaustion=false: "
                        "halting with a checkpoint instead of silently renormalising "
                        "the mix.  To continue with a renormalised mix, resume with "
                        "--data.allow_bucket_exhaustion true.", rep,
                    )
                # Skip the optimizer step for this (already-renormalised) batch;
                # the shutdown check at the top of the loop saves and breaks.
                continue

        with accelerator.accumulate(llm):
            loss = _compute_loss(llm, batch)
            accelerator.backward(loss)

            grad_norm = None
            if accelerator.sync_gradients:
                grad_norm = accelerator.clip_grad_norm_(llm.parameters(), cfg.optimizer.max_grad_norm)

            opt.step()
            # ``opt`` is wrapped by Accelerate, so ``opt.step()`` and
            # ``opt.zero_grad()`` already no-op on non-sync micro-steps.
            # ``lr_scheduler`` is NOT prepared (see the rationale next to
            # the prepare call), so we must guard it ourselves.
            if accelerator.sync_gradients:
                lr_scheduler.step()
            opt.zero_grad()

        # -- token counting -------------------------------------------------
        if "labels" in batch:
            batch_non_pad = (batch["labels"] != -100).sum().item()
            ls.interval_non_pad += batch_non_pad
            ls.interval_possible += batch["labels"].numel()
            ls.step_non_pad += batch_non_pad

        if not accelerator.sync_gradients:
            continue

        # Optimizer stepped: commit the cursors we accumulated since the
        # previous step.  This is the only safe moment — pending cursors
        # from a partial accumulation should NOT be persisted, otherwise
        # resume would skip data we never actually trained on.
        if ls.pending_worker_states:
            ls.committed_worker_states.update(ls.pending_worker_states)
            ls.pending_worker_states.clear()

        ls.global_step += 1
        ls.interval_steps += 1
        ls.interval_loss += loss.detach().item()
        if grad_norm is not None:
            ls.interval_grad_norm += grad_norm.item()

        # -- per-step global token count (for curriculum) -------------------
        # See note above: build on accelerator.device so NCCL is happy too.
        step_tokens_t = torch.tensor(
            ls.step_non_pad, dtype=torch.float64, device=accelerator.device
        )
        try:
            import torch.distributed as _dist
            if _dist.is_available() and _dist.is_initialized():
                _dist.all_reduce(step_tokens_t, op=_dist.ReduceOp.SUM)
        except ImportError:
            pass
        step_tokens_global = step_tokens_t.item()
        ls.total_tokens_trained += step_tokens_global
        ls.step_non_pad = 0

        # -- curriculum phase transition ------------------------------------
        if curriculum and ls.phase_idx < len(curriculum) - 1:
            ls.phase_tokens_consumed += step_tokens_global
            if ls.phase_tokens_consumed >= curriculum[ls.phase_idx].duration_tokens:
                ls.phase_idx += 1
                ls.phase_tokens_consumed = 0
                if is_main:
                    logger.info(
                        f"Curriculum: advancing to phase {ls.phase_idx} "
                        f"(at {ls.total_tokens_trained:,} total tokens)"
                    )
                data_module.set_phase(ls.phase_idx)
                # No accelerator.prepare here — see prepare-call note above.
                train_dl = data_module.train_dataloader()
                train_iter = iter(train_dl)
                # The new phase has a fresh bucket mix; per-worker cursors
                # from the previous phase no longer apply.
                ls.pending_worker_states.clear()
                ls.committed_worker_states.clear()
        elif curriculum:
            ls.phase_tokens_consumed += step_tokens_global

        # -- logging --------------------------------------------------------
        if ls.global_step % cfg.training.logging_steps == 0:
            current_lr = lr_scheduler.get_last_lr()[0]

            # See note above: build on accelerator.device so NCCL is happy too.
            _rd = accelerator.device
            loss_t = torch.tensor(ls.interval_loss, dtype=torch.float64, device=_rd)
            gnorm_t = torch.tensor(ls.interval_grad_norm, dtype=torch.float64, device=_rd)
            npad_t = torch.tensor(ls.interval_non_pad, dtype=torch.float64, device=_rd)
            poss_t = torch.tensor(ls.interval_possible, dtype=torch.float64, device=_rd)

            try:
                import torch.distributed as dist
                if dist.is_available() and dist.is_initialized():
                    ws = dist.get_world_size()
                    dist.all_reduce(loss_t, op=dist.ReduceOp.SUM)
                    dist.all_reduce(gnorm_t, op=dist.ReduceOp.SUM)
                    loss_t /= ws
                    gnorm_t /= ws
                    dist.all_reduce(npad_t, op=dist.ReduceOp.SUM)
                    dist.all_reduce(poss_t, op=dist.ReduceOp.SUM)
            except ImportError:
                pass

            # Divide by the number of steps ACTUALLY accumulated this interval.
            steps_this_interval = max(ls.interval_steps, 1)
            ls.avg_loss = loss_t.item() / steps_this_interval
            avg_gnorm = gnorm_t.item() / steps_this_interval
            actual_tokens = npad_t.item()

            elapsed = time.time() - ls.step_start_time
            tok_per_sec = actual_tokens / elapsed if elapsed > 0 else 0
            time_per_step = elapsed / steps_this_interval
            packing_eff = (actual_tokens / poss_t.item() * 100) if poss_t.item() > 0 else 0

            if is_main:
                logger.info(
                    f"Step {ls.global_step:6d} | Loss {ls.avg_loss:.4f} | LR {current_lr:.2e} | "
                    f"GradNorm {avg_gnorm:.3f} | {time_per_step:.2f}s/step | "
                    f"{tok_per_sec:,.0f} tok/s | Packing {packing_eff:.1f}%"
                )

                sample_interval = cfg.training.logging_steps * 10
                if ls.global_step < 200 or ls.global_step % sample_interval == 0:
                    samples = _generate_samples(llm, tokenizer, accelerator)
                    logger.debug("=" * 60)
                    logger.debug(f"Generated Samples at Step {ls.global_step}:")
                    logger.debug("=" * 60)
                    for prompt, text in samples:
                        logger.debug(f"Prompt: {prompt}")
                        logger.debug(f"Output: {text}")
                    logger.debug("=" * 60)

                if use_wandb:
                    log_dict = {
                        "train/loss": ls.avg_loss,
                        "train/learning_rate": current_lr,
                        "train/grad_norm": avg_gnorm,
                        "throughput/tokens_per_sec": tok_per_sec,
                        "throughput/time_per_step": time_per_step,
                        "throughput/total_tokens": ls.total_tokens_trained,
                        "throughput/packing_efficiency": packing_eff,
                    }
                    if curriculum:
                        log_dict["curriculum/phase"] = ls.phase_idx
                    wandb.log(log_dict, step=ls.global_step)

                if run_state is not None:
                    run_state.global_step = ls.global_step
                    run_state.loss = ls.avg_loss
                    run_state.learning_rate = current_lr
                    run_state.tokens_trained = int(ls.total_tokens_trained)
                    run_state.tokens_per_sec = tok_per_sec
                    run_state.latest_checkpoint = ls.last_checkpoint_path or None
                    run_state.write(cfg.training.output_dir)

            ls.interval_loss = 0.0
            ls.interval_grad_norm = 0.0
            ls.interval_non_pad = 0
            ls.interval_possible = 0
            ls.interval_steps = 0
            ls.step_start_time = time.time()

        # -- evaluation -----------------------------------------------------
        if ls.global_step % cfg.training.eval_steps == 0:
            max_eval = cfg.training.max_eval_batches or 100
            eval_loss = _evaluate(llm, eval_dl, accelerator, max_eval_steps=max_eval)
            if is_main:
                logger.info(f"Eval Loss: {eval_loss:.4f}")
                if use_wandb:
                    wandb.log({
                        "eval/loss": eval_loss,
                        "eval/perplexity": math.exp(min(eval_loss, 20)),
                    }, step=ls.global_step)

        # -- checkpoint -----------------------------------------------------
        if ls.global_step % cfg.training.save_steps == 0:
            ls.last_checkpoint_path = _save_now(ctx, ls, tag=None)

    # -- final save ---------------------------------------------------------

    if shutdown.reason is None:
        ls.last_checkpoint_path = _save_now(ctx, ls, tag="final")

        if is_main:
            logger.info("=" * 60)
            logger.info("TRAINING COMPLETE!")
            logger.info("=" * 60)
            if run_state is not None:
                run_state.status = "completed"
                run_state.global_step = ls.global_step
                run_state.latest_checkpoint = ls.last_checkpoint_path
                run_state.tokens_trained = int(ls.total_tokens_trained)
                run_state.write(cfg.training.output_dir)

    if use_wandb and is_main:
        wandb.finish()

    # Tear down the distributed process group cleanly so non-zero ranks
    # don't SIGABRT during interpreter shutdown.  Accelerator doesn't do
    # this for us when we initialized dist ourselves (CPU gloo path).
    try:
        import torch.distributed as _dist_teardown
        if _dist_teardown.is_available() and _dist_teardown.is_initialized():
            _dist_teardown.destroy_process_group()
    except Exception:  # noqa: BLE001
        pass

    return TrainingResult(
        final_loss=ls.avg_loss,
        checkpoint_path=ls.last_checkpoint_path,
        total_tokens=int(ls.total_tokens_trained),
        total_steps=ls.global_step,
    )
