#!/usr/bin/env bash
#
# SLURM "resume chain" template for gpt_simple.
#
# Submits one SLURM job that trains until walltime nears.  When SLURM
# raises SIGUSR1 (sent before --time runs out), the trainer saves a
# graceful checkpoint and exits 0.  After the run, this script consults
# `gpt-simple status` and re-submits itself when there is more work
# to do.
#
# Usage:
#   sbatch \
#     --export=ALL,CONFIG=/path/to/config.yaml,OUTPUT_DIR=/scratch/$USER/run1 \
#     slurm_resume_chain.sh
#
# Tweak the #SBATCH directives below for your cluster.  Jean Zay users:
# pick the right --account=, --partition= and gres for your allocation.
#
#SBATCH --job-name=gpt-pretrain
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=02:00:00
# Send SIGUSR1 to every process in the job 300 seconds before --time
# expires.  The trainer catches this and saves a final checkpoint.
#SBATCH --signal=USR1@300
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

# ── Inputs (override via --export or env) ────────────────────────────
CONFIG=${CONFIG:?CONFIG (path to config.yaml) must be set}
OUTPUT_DIR=${OUTPUT_DIR:?OUTPUT_DIR must be set}
NPROC_PER_NODE=${NPROC_PER_NODE:-${SLURM_GPUS_ON_NODE:-${SLURM_NTASKS_PER_NODE:-1}}}
MAX_RESUBMITS=${MAX_RESUBMITS:-20}
RESUBMIT_COUNT=${RESUBMIT_COUNT:-0}

mkdir -p "$(dirname "$0")/../../logs" || true

echo "=========================================================="
echo "SLURM job ${SLURM_JOB_ID} starting (link ${RESUBMIT_COUNT}/${MAX_RESUBMITS})"
echo "  config:        ${CONFIG}"
echo "  output_dir:    ${OUTPUT_DIR}"
echo "  nproc_per_node: ${NPROC_PER_NODE}"
echo "  walltime:      ${SLURM_JOB_END_TIME:-unset} (epoch s)"
echo "=========================================================="

# Resume-by-default: passing nothing here makes the trainer call
# CheckpointManager.resolve_resume("auto") and pick the latest checkpoint.
srun gpt-simple train \
    --config "${CONFIG}" \
    --nproc_per_node "${NPROC_PER_NODE}" \
    --training.output_dir "${OUTPUT_DIR}"

# ── Did we finish?  Resubmit if not. ────────────────────────────────
STATUS_OUTPUT=$(gpt-simple status --output_dir "${OUTPUT_DIR}" 2>/dev/null || true)
echo "${STATUS_OUTPUT}"

if echo "${STATUS_OUTPUT}" | grep -q "COMPLETED"; then
    echo "Training complete; chain done."
    exit 0
fi

# Bucket exhausted with data.allow_bucket_exhaustion=false: the trainer
# checkpointed and halted instead of silently renormalising the mix.  Needs a
# human decision, so the chain stops here (it would re-halt forever otherwise).
# To continue with the renormalised mix, resubmit with
# --data.allow_bucket_exhaustion true.
if echo "${STATUS_OUTPUT}" | grep -q "HALTED"; then
    echo "Training halted (bucket exhausted); refusing to resubmit."
    echo "To continue with a renormalised mix, resubmit with --data.allow_bucket_exhaustion true."
    exit 0
fi

if echo "${STATUS_OUTPUT}" | grep -qE "ERROR|CRASHED"; then
    echo "Training reported an error; refusing to resubmit."
    echo "Inspect ${OUTPUT_DIR}/.run_state.json and the most recent logs."
    exit 1
fi

if [ "${RESUBMIT_COUNT}" -ge "${MAX_RESUBMITS}" ]; then
    echo "Reached MAX_RESUBMITS=${MAX_RESUBMITS}; stopping chain."
    exit 0
fi

NEXT=$((RESUBMIT_COUNT + 1))
echo "Resubmitting next link in the chain (${NEXT}/${MAX_RESUBMITS})..."
sbatch \
    --export=ALL,CONFIG="${CONFIG}",OUTPUT_DIR="${OUTPUT_DIR}",NPROC_PER_NODE="${NPROC_PER_NODE}",MAX_RESUBMITS="${MAX_RESUBMITS}",RESUBMIT_COUNT="${NEXT}" \
    "$0"
