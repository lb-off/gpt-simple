#!/bin/bash
# =============================================================================
# Jean Zay SLURM resume-chain for gpt_simple.
# -----------------------------------------------------------------------------
# Submits one SLURM job that trains until walltime nears.  SIGUSR1 (raised by
# SLURM 10 min before --time expires) lands on the srun launcher, which drops
# the .shutdown_requested flag the workers poll → each rank saves a graceful
# checkpoint and exits 0.  This script then re-submits itself, picking up from
# the latest checkpoint via `training.resume: auto`.
#
# Usage:
#   sbatch \
#     --export=ALL,\
# CONFIG=$(realpath examples/configs/pretrain_2.8b_15b.yaml),\
# OUTPUT_DIR=$SCRATCH/runs/pretrain_2.8b_15b_v1 \
#     examples/orchestrators/jeanzay_resume_chain.sbatch
#
# Stopping:
#   • Graceful stop of the CURRENT job only (chain continues / resubmits):
#       touch $OUTPUT_DIR/.shutdown_requested
#   • Graceful stop of the current job AND end the chain (no resubmit) —
#     the clean way to bail out mid-run if something looks wrong:
#       touch $OUTPUT_DIR/.stop_chain $OUTPUT_DIR/.shutdown_requested
#   • Hard cancel (no graceful checkpoint; chain stops because the batch
#     script is killed before the resubmit line):
#       scancel <jobid>
#
# =============================================================================

#SBATCH --job-name=gpt28b
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=gpu_p5          # Jean Zay A100 80GB partition (8 GPUs/node)
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1         # one launcher per node → torchrun spawns 8 workers
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64          # full A100 node (8 cores/GPU)
#SBATCH --hint=nomultithread
#SBATCH --time=20:00:00             # max walltime under qos_gpu-t3 (default GPU QOS)
#SBATCH --qos=qos_gpu-t3            # switch to qos_gpu-t4 for up to 100h if allocated
#SBATCH --account=CHANGEME@a100     # ← set to your Jean Zay GPU allocation
#SBATCH --signal=USR1@600           # SIGUSR1 600s before --time expires

set -euo pipefail

# ── Inputs (from --export=ALL,...) ──────────────────────────────────
CONFIG=${CONFIG:?CONFIG (path to config.yaml) must be set via --export}
OUTPUT_DIR=${OUTPUT_DIR:?OUTPUT_DIR must be set via --export}
MAX_RESUBMITS=${MAX_RESUBMITS:-30}
RESUBMIT_COUNT=${RESUBMIT_COUNT:-0}

mkdir -p logs "$OUTPUT_DIR"

# ── Jean Zay environment ────────────────────────────────────────────
# Matches the login-node env you tested with:
#   (gpt-simple-a100) on top of pytorch-gpu/py3/2.6.0
module purge
module load arch/a100
module load pytorch-gpu/py3/2.6.0

# Activate the conda/venv where gpt_simple is installed.
# Adjust the path if your env lives elsewhere.
conda activate gpt-simple-a100 || source activate gpt-simple-a100

# ── NCCL / runtime tuning ───────────────────────────────────────────
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_IB_TIMEOUT=22
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8             # 8 CPU threads per worker (= 64/8 GPUs)

# Per-job Triton cache → avoids stampede on shared $HOME
export TRITON_CACHE_DIR=${JOBSCRATCH:-/tmp}/triton_cache.${SLURM_JOB_ID}
mkdir -p "$TRITON_CACHE_DIR"

# Pick a distinct master port per submission to avoid clashes when multiple
# jobs land on the same node.
export MASTER_PORT=$((29500 + RANDOM % 1000))

echo "============================================================="
echo "Jean Zay job ${SLURM_JOB_ID} starting (link ${RESUBMIT_COUNT}/${MAX_RESUBMITS})"
echo "  node:           $(hostname)"
echo "  config:         $CONFIG"
echo "  output_dir:     $OUTPUT_DIR"
echo "  GPUs:           ${SLURM_GPUS_ON_NODE:-?}"
echo "  walltime end:   ${SLURM_JOB_END_TIME:-unset} (epoch s)"
echo "============================================================="

# ── Train ───────────────────────────────────────────────────────────
# `training.resume: auto` (set in the YAML) picks up the latest checkpoint
# under OUTPUT_DIR/checkpoints/ automatically on every restart.
#
# srun is run without aborting the script on a non-zero exit (set +e) so the
# stop-marker and status checks below always run — otherwise a signal-killed
# srun would `set -e` us out before we could honour an explicit chain stop.
set +e
srun --kill-on-bad-exit=1 \
    gpt-simple train \
        --config "$CONFIG" \
        --nproc_per_node 8 \
        --master_port "$MASTER_PORT" \
        --training.output_dir "$OUTPUT_DIR"
SRUN_RC=$?
set -e
echo "srun exited with code ${SRUN_RC}."

# ── Explicit chain stop?  (touch $OUTPUT_DIR/.stop_chain) ───────────
# Checked first so an intentional stop always wins, regardless of status.
if [ -f "$OUTPUT_DIR/.stop_chain" ]; then
    echo "Chain stop requested (.stop_chain present); not resubmitting."
    rm -f "$OUTPUT_DIR/.stop_chain"
    exit 0
fi

# ── Did the chain finish?  Resubmit if not. ─────────────────────────
STATUS_OUTPUT=$(gpt-simple status --output_dir "$OUTPUT_DIR" 2>/dev/null || true)
echo "$STATUS_OUTPUT"

if echo "$STATUS_OUTPUT" | grep -q "COMPLETED"; then
    echo "Training complete; chain done."
    exit 0
fi

# A bucket ran dry with data.allow_bucket_exhaustion=false: the trainer
# checkpointed and halted rather than silently renormalising the mix.  This
# needs a human decision, so the chain stops (it would otherwise re-halt at
# the same point forever).  To continue with the renormalised mix, resubmit
# manually with CONFIG overriding --data.allow_bucket_exhaustion true.
if echo "$STATUS_OUTPUT" | grep -q "HALTED"; then
    echo "Training halted (bucket exhausted); refusing to resubmit."
    echo "To continue with a renormalised mix, resubmit with "
    echo "  --data.allow_bucket_exhaustion true"
    exit 0
fi

if echo "$STATUS_OUTPUT" | grep -qE "ERROR|CRASHED"; then
    echo "Training reported an error; refusing to resubmit."
    echo "Inspect ${OUTPUT_DIR}/.run_state.json and the most recent logs."
    exit 1
fi

if [ "$RESUBMIT_COUNT" -ge "$MAX_RESUBMITS" ]; then
    echo "Reached MAX_RESUBMITS=$MAX_RESUBMITS; stopping chain."
    exit 0
fi

NEXT=$((RESUBMIT_COUNT + 1))
echo "Resubmitting next link in the chain ($NEXT/$MAX_RESUBMITS)..."
sbatch \
    --export=ALL,CONFIG="$CONFIG",OUTPUT_DIR="$OUTPUT_DIR",MAX_RESUBMITS="$MAX_RESUBMITS",RESUBMIT_COUNT="$NEXT" \
    "$0"
