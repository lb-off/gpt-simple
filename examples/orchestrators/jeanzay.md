# Jean Zay (IDRIS) notes

Site-specific notes for running GPT-Simple on the IDRIS **Jean Zay**
cluster (France). This is an *example*: nothing in the library depends on
it. For vendor-neutral guidance see
[`docs/orchestration.md`](../../docs/orchestration.md) and
[`docs/hardware-tuning.md`](../../docs/hardware-tuning.md); the bundled
`jeanzay_*.sbatch` templates in this directory go with these notes.

## Partitions

| Partition | GPU type | GPUs/node | Notes |
| --------- | -------- | --------- | ----- |
| `gpu_p13` | V100 (32GB) | 4 | Volta; fp16 only, memory-efficient attention. |
| `gpu_p5`  | A100 (80GB) | 8 | AMD EPYC hosts — needs `module load arch/a100`. |
| `gpu_p6`  | H100 (80GB) | 4 | Hopper; confirm the `arch/` module with IDRIS. |

## SLURM header: A100 → H100

```diff
- #SBATCH --account=CHANGEME@a100
- #SBATCH --constraint=a100
- #SBATCH --qos=qos_gpu_a100-t3
- #SBATCH --gres=gpu:8
+ #SBATCH --account=CHANGEME@h100     # check `idracct` for the exact name
+ #SBATCH --constraint=h100
+ #SBATCH --qos=qos_gpu_h100-t3      # confirm with `sinfo -p gpu_p6`
+ #SBATCH --gres=gpu:4               # H100 nodes have 4 GPUs, not 8

  module purge
- module load arch/a100
+ module load arch/h100              # if such a module exists; else omit
  module load pytorch-gpu/py3/2.6.0
```

Before submitting:

- **4 GPUs/node on H100, not 8.** Adjust your world size (request two
  nodes for the same total) or accept a smaller global batch.
- **`arch/` module.** `arch/a100` exists because A100 nodes use AMD EPYC
  CPUs and need a matching PyTorch build (avoids SIGILL in NCCL). Check
  `module avail arch/` on a login node for the H100 equivalent.
- **Accounting.** H100 hours are billed at a different rate; run
  `idracct` to check your allocation.

## Walltime and resume

Jean Zay enforces a hard per-job wall-clock limit, which is exactly what
GPT-Simple's stop/resume is built for: `SLURM_JOB_END_TIME` is
auto-detected, the trainer saves a shutdown checkpoint before the
deadline, and `resume: auto` continues on the next job. See
`slurm_resume_chain.sh` in this directory for an auto-resubmitting chain.

## Verifying your environment

The `pytorch-gpu/py3/2.6.0` module is recent enough for the SDPA
fast path. For a custom venv:

```bash
source $WORK/venvs/gpt-simple/bin/activate
python -c "import torch; assert torch.__version__ >= '2.3', torch.__version__"
```
