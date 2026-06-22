# Orchestrator examples

Drop-in templates for running a long pretraining job across many short
sequential jobs.  Each one relies on the same in-library mechanics:

  - `training.resume: auto` (the default) — every restart picks up the
    latest checkpoint.
  - `training.max_runtime_seconds` (or `SLURM_JOB_END_TIME`) — bounds
    each job's wall-clock; the trainer saves a graceful checkpoint
    before the deadline.
  - `SIGTERM` / `SIGUSR1` — orchestrator-sent signals trigger the same
    graceful save (works for SLURM, k8s, Docker, etc.).

The templates below cover the most common deployments.  Pick the one that
matches your infra and edit the highlighted variables.  The generic
templates are vendor-neutral; the `jeanzay_*` files are a worked example
for the IDRIS Jean Zay cluster (see [`jeanzay.md`](jeanzay.md)) and may
contain site-specific headers you should adapt.

## Files

| File                          | Use case                                       |
| ----------------------------- | ---------------------------------------------- |
| `slurm_resume_chain.sh`       | Training: SLURM resume-chain (generic)         |
| `kubernetes_job.yaml`         | Training: Kubernetes Job (restartPolicy=OnFailure) |
| `local_loop.sh`               | Training: laptop / single workstation          |
| `jeanzay.md`                  | Notes: Jean Zay (IDRIS) partitions, modules, accounting |
| `jeanzay_resume_chain.sbatch` | Training: Jean Zay-tuned resume-chain          |
| `jeanzay_generate.sbatch`     | Inference: Jean Zay single-checkpoint JSONL generation (single GPU, supports `--array` fan-out) |
| `jeanzay_batch_generate.sbatch` | Inference: Jean Zay self-describing JSONL via `batch-generate` (per-record model + sampling; one load per checkpoint; login-node `--dry-run` gate) |

## How they share the same library code

All templates expect:

1. An installed `gpt-simple` CLI on the worker (`pip install -e .` or
   PyPI).
2. A YAML config — generate one with `gpt-simple init -o config.yaml`.
3. A persistent `output_dir` (mounted scratch, PVC, or local SSD) that
   survives between job restarts.  The trainer writes checkpoints,
   `trainer_state.json`, `.run_state.json`, and the rotated W&B run id
   there.

There is **no orchestrator-specific code in the library itself** — the
templates only translate orchestrator signals into the standard
`SIGTERM`/`SIGUSR1` that the trainer already handles.

## What happens at each restart

```
job N starts
  └─ training.resume == 'auto'
       └─ CheckpointManager.resolve_resume(...)  → checkpoints/checkpoint-XXX/
            └─ TrainerState.step, tokens_trained, curriculum, wandb_run_id loaded
            └─ Model/optimizer/scheduler/RNG/dataloader cursors restored
job N runs until SIGTERM/SIGUSR1 or walltime budget
  └─ ShutdownCoordinator drains the current step
       └─ checkpoint-{step}/  is written atomically
       └─ training exits with code 0
orchestrator re-submits job N+1 (slurm/k8s/loop) — back to top
```

The resume path is covered by the end-to-end resume tests in
`tests/test_e2e_resume.py`, which assert a stop/resume chain matches an
uninterrupted run.

## W&B across restarts

The trainer persists the W&B `run_id` in `trainer_state.json`, so every
restart reattaches to the same run.  Time-series telemetry — train/eval
loss, gradient norms, learning rate, throughput, system metrics — is
indexed by step and **appends** cleanly across restarts; the dashboard
charts look continuous.

The **Logs tab is the exception**: W&B stores stdout/stderr as a single
file resource per run, so each restart's `wandb sync` overwrites the
previous upload.  Only the most recent restart's terminal output is
visible there.  For the full per-restart history, look at the
orchestrator log files on disk (e.g. SLURM's `*-<JOBID>.out`) — they
are kept one per restart and labelled by job id.
