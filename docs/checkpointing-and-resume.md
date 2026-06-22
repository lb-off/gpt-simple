# Checkpointing & resume

GPT-Simple is built so that **N short jobs equal one long job**. A run can
stop — on purpose, on a walltime deadline, or on a signal — and resume
bit-for-bit, even on a different number of GPUs. This is what makes it
practical on clusters with hard per-job time limits.

## On-disk layout

Everything for a run lives under `training.output_dir`:

```
output_dir/
├── config.json                 # the Config that started the run
├── .run_state.json             # live status snapshot (for `gpt-simple status`)
├── tokenizer/                  # written once at run start
├── logs/
└── checkpoints/
    ├── checkpoint-1000/
    │   ├── trainer_state.json   # step, tokens, curriculum, wandb id, arch hash
    │   ├── dataloader_state/    # per-worker data cursors
    │   ├── model/
    │   │   ├── pytorch_model.bin  # plain state_dict of the unwrapped model
    │   │   └── config.json        # ModelConfig
    │   ├── optimizer.bin
    │   ├── scheduler.bin
    │   └── rng/rank_*.pkl         # per-rank RNG state
    ├── checkpoint-2734-shutdown/  # saved when stopping early
    └── final/                     # saved at end of training
```

Model weights are stored once, as a plain `state_dict` of the unwrapped
module (no DDP/compile prefixes). Saves are atomic — a checkpoint
directory only becomes visible once fully written, so an interrupted save
never corrupts a run.

## Resume policy

`training.resume` controls what happens at startup:

| Value | Behavior |
| ----- | -------- |
| `auto` (default) | Resume from the newest checkpoint under `output_dir`, or start fresh if there is none. |
| `scratch` | Train from scratch; error if checkpoints already exist (use `--force` to wipe them). |
| `<path>` | Resume from a specific checkpoint directory. |

Because `auto` resumes if and only if a checkpoint exists, **the same
command works for both the first launch and every restart** — which is
exactly what an auto-resubmitting orchestrator needs.

On resume, the trainer restores model weights, optimizer and scheduler
state, per-rank RNG, and per-worker data cursors. The architecture hash
in `trainer_state.json` is checked against the current config; an
incompatible change is rejected rather than silently loading mismatched
weights.

## Retention

To bound disk usage on long runs:

- `keep_last_k` — keep only the most recent K checkpoints (null keeps
  all).
- `keep_milestone_every` — never delete checkpoints whose step is a
  multiple of this value (e.g. permanent milestones every 5000 steps).

## Graceful shutdown and walltime

The trainer stops cleanly — saving a `*-shutdown` checkpoint and exiting
0 — in response to any of:

- **A walltime budget.** `max_runtime_seconds`, or auto-detected from
  `SLURM_JOB_END_TIME` or the generic `GPT_SIMPLE_MAX_RUNTIME` env var.
  `walltime_reserve_seconds` is the margin reserved before the deadline so
  the final save completes before the orchestrator kills the job.
- **Signals.** `SIGTERM`, `SIGINT`, `SIGUSR1`, `SIGUSR2` all request a
  save-and-exit. (`SIGUSR1` is the SLURM walltime-warning convention.) A
  second `SIGINT` (Ctrl-C twice) force-exits immediately.
- **A flag file.** `gpt-simple stop` writes `output_dir/.shutdown_requested`,
  which the loop checks each step.

Across multiple GPUs, the stop decision is agreed via an all-reduce on
every check, so no rank is left mid-step while another exits.

A graceful stop from a walltime budget, signal, or flag file reports
status `stopped` — an orchestrator should resume it. One stop is instead
terminal: when a curriculum bucket runs dry with
`data.allow_bucket_exhaustion=false`, the trainer saves a checkpoint and
reports status `halted` (it needs a human decision, so the resume
orchestrators do not resubmit it). See
[Bucket exhaustion](data.md#bucket-exhaustion).

```bash
gpt-simple stop                      # graceful: save at next step, then exit
gpt-simple stop --force              # immediate SIGKILL
gpt-simple status                    # inspect progress (reads .run_state.json)
```

## Topology-agnostic data cursors

The hardest part of exact resume is the data stream. GPT-Simple tracks
progress **per data file**, not per global step: each rank/worker records
which files it has finished and how far into the current file it is. On
resume, all per-rank cursors are merged into one global map and
redistributed across the new set of workers.

The consequence: a run can stop on, say, 8 GPUs × 4 workers and resume on
4 GPUs × 8 workers (or any other layout) and still process **every
document exactly once** — no skips, no duplicates. Prefetched-but-unused
batches from before the stop are simply regenerated from the cursor.

(Deterministic data resume applies to the `pretokenized` format; see
[Data pipeline](data.md).)

---

Authoritative source: `src/gpt_simple/_checkpoint.py`,
`src/gpt_simple/_shutdown.py`, `src/gpt_simple/data.py`.
