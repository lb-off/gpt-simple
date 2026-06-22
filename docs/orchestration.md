# Orchestration

GPT-Simple has no built-in scheduler and makes no assumptions about your
cluster. It is designed to be driven by *any* orchestrator — SLURM,
Kubernetes, a plain shell loop — by combining two features:

1. **`resume: auto`** — the same `gpt-simple train` command starts a fresh
   run or resumes the latest checkpoint, so re-running it is always
   correct.
2. **Graceful, walltime-aware shutdown** — the trainer saves and exits 0
   before a deadline or on a signal, so the orchestrator can simply
   re-queue the job.

Together these turn *N* sequential time-limited jobs into one continuous
run. The details of what is saved and restored are in
[Checkpointing & resume](checkpointing-and-resume.md).

## The contract with your orchestrator

Your orchestrator only needs to:

- **Re-run the same command** until training reaches `max_steps`. Each run
  resumes where the last left off.
- **Communicate the deadline** (optional but recommended), via either:
  - the `SLURM_JOB_END_TIME` environment variable (set automatically by
    SLURM), or
  - the generic `GPT_SIMPLE_MAX_RUNTIME` environment variable (seconds),
    or
  - the `training.max_runtime_seconds` config field.
- **Send a signal to stop early** (optional): `SIGTERM`/`SIGUSR1` trigger
  a graceful save-and-exit.

The trainer reserves `walltime_reserve_seconds` before the deadline to
finish its final checkpoint — size this to your checkpoint write time.

### When to stop vs. resume

After each run, decide what to do from `gpt-simple status`, not the exit
code (a graceful stop and a finished run both exit 0):

- `COMPLETED` — reached `max_steps`. Stop.
- `HALTED` — a curriculum bucket ran dry with
  `data.allow_bucket_exhaustion=false`. **Terminal** — do *not* resubmit,
  or the next run halts at the same point. Resume manually with
  `--data.allow_bucket_exhaustion true` to continue with a renormalized
  mix (see [Bucket exhaustion](data.md#bucket-exhaustion)).
- `ERROR` / `CRASHED` — refuse to resubmit; inspect the logs.
- Anything else (`STOPPED`, walltime, transient) — resume.

The bundled templates already implement these checks.

## Templates

Ready-to-adapt templates live in
[`examples/orchestrators/`](../examples/orchestrators/):

| Template | Use case |
| -------- | -------- |
| `slurm_resume_chain.sh` | Auto-resubmitting SLURM job (generic clusters). |
| `kubernetes_job.yaml` | Kubernetes Job with `restartPolicy: OnFailure`. |
| `local_loop.sh` | A plain bash loop on a single machine. |

The `examples/` directory may contain site-specific details (account
names, partitions, module loads). Treat those as illustrations and adapt
them to your environment — the library itself never depends on them.

## Minimal example: a shell loop

A graceful shutdown and a completed run *both* exit 0, so the loop checks
`gpt-simple status` rather than the exit code to decide when to stop:

```bash
export GPT_SIMPLE_MAX_RUNTIME=7200          # stop ~2h in, save, exit 0
while true; do
  gpt-simple train --config config.yaml
  status=$(gpt-simple status --output_dir ./outputs 2>/dev/null || true)
  echo "$status" | grep -qE "COMPLETED|HALTED" && break   # HALTED needs attention
  sleep 5                                    # otherwise resume and continue
done
```

Each iteration trains until the budget, saves a shutdown checkpoint, and
exits; the loop relaunches and `resume: auto` continues until status
reports `COMPLETED`. `examples/orchestrators/local_loop.sh` is a more
complete version (attempt cap, error detection).

---

Authoritative source: `src/gpt_simple/_shutdown.py`,
`examples/orchestrators/`.
