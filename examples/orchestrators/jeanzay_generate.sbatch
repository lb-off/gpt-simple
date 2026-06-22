#!/bin/bash
# =============================================================================
# Jean Zay SLURM batch-inference template for gpt_simple.
# -----------------------------------------------------------------------------
# Runs `gpt-simple generate` over a JSONL prompt file on a single GPU.
#
# Usage:
#   sbatch \
#     --export=ALL,\
# CHECKPOINT=$SCRATCH/runs/my_run/checkpoints/checkpoint-12000,\
# PROMPTS=$SCRATCH/prompts/dev.jsonl,\
# OUTPUT=$SCRATCH/completions/dev.jsonl \
#     examples/orchestrators/jeanzay_generate.sbatch
#
# Optional knobs (override via --export):
#   MAX_NEW_TOKENS   default 256
#   TEMPERATURE      default 0.8
#   TOP_K            default 50
#   TOP_P            default 0.95
#   DTYPE            default bf16
#   SEED             default unset (non-deterministic sampling)
#
# Fan-out across a prompt set:
#   Split your prompts file into N shards (split -n l/N prompts.jsonl) and
#   submit with --array=0-$((N-1)). The script picks up the shard index
#   from SLURM_ARRAY_TASK_ID; see the optional array-mode section below.
#
# Multiple models / per-record sampling (batch-generate):
#   When each prompt needs its own model or sampling params, use a
#   self-describing JSONL with `gpt-simple batch-generate --input setup.jsonl`
#   instead of the `generate` invocation below. It groups records by model and
#   loads each checkpoint once. GATE SUBMISSION with a GPU-free pre-flight on
#   the LOGIN node before you sbatch:
#       gpt-simple batch-generate --input setup.jsonl --output-dir RUN --dry-run
#   For --array fan-out, shard the input file BY MODEL (one model per task) so
#   no checkpoint is loaded twice.
#
# =============================================================================

#SBATCH --job-name=gptgen
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --account=CHANGEME@a100     # ← set to your Jean Zay GPU allocation
#SBATCH --constraint=a100           # A100 nodes (Jean Zay routes via constraint, not partition)
#SBATCH --qos=qos_gpu_a100-t3       # default A100 QOS (up to 20h)
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1                # batch inference fits on a single A100
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --time=02:00:00

set -euo pipefail

# ── Inputs (from --export=ALL,...) ──────────────────────────────────
CHECKPOINT=${CHECKPOINT:?CHECKPOINT (path to checkpoint dir OR output_dir) must be set}
PROMPTS=${PROMPTS:?PROMPTS (path to input JSONL) must be set}
OUTPUT=${OUTPUT:?OUTPUT (path to output JSONL) must be set}

MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-256}
TEMPERATURE=${TEMPERATURE:-0.8}
TOP_K=${TOP_K:-50}
TOP_P=${TOP_P:-0.95}
DTYPE=${DTYPE:-bf16}
SEED=${SEED:-}

# ── Optional array-mode shard handling ──────────────────────────────
# If submitted with --array=..., expect a directory of shards alongside
# PROMPTS named PROMPTS.shards/, with files prompts_NNNN.jsonl. Output
# becomes OUTPUT.shards/completions_NNNN.jsonl. Adjust as needed.
if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
    shard=$(printf "%04d" "$SLURM_ARRAY_TASK_ID")
    PROMPTS_DIR=${PROMPTS%.jsonl}.shards
    OUTPUT_DIR=${OUTPUT%.jsonl}.shards
    PROMPTS="$PROMPTS_DIR/prompts_${shard}.jsonl"
    OUTPUT="$OUTPUT_DIR/completions_${shard}.jsonl"
    mkdir -p "$OUTPUT_DIR"
fi

mkdir -p logs "$(dirname "$OUTPUT")"

# ── Jean Zay environment ────────────────────────────────────────────
module purge
module load arch/a100
module load pytorch-gpu/py3/2.6.0
conda activate gpt-simple-a100 || source activate gpt-simple-a100

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4

echo "============================================================="
echo "gpt-simple generate — job ${SLURM_JOB_ID}${SLURM_ARRAY_TASK_ID:+ (shard $SLURM_ARRAY_TASK_ID)}"
echo "  node:           $(hostname)"
echo "  checkpoint:     $CHECKPOINT"
echo "  prompts:        $PROMPTS"
echo "  output:         $OUTPUT"
echo "  sampling:       max_new=$MAX_NEW_TOKENS temp=$TEMPERATURE top_k=$TOP_K top_p=$TOP_P dtype=$DTYPE"
echo "============================================================="

# Auto-detect: does CHECKPOINT point at a specific checkpoint dir or at a run dir?
if [ -f "$CHECKPOINT/model/config.json" ]; then
    CKPT_ARG=(--checkpoint "$CHECKPOINT")
elif [ -d "$CHECKPOINT/checkpoints" ]; then
    CKPT_ARG=(--output-dir "$CHECKPOINT")
else
    echo "ERROR: $CHECKPOINT is neither a checkpoint dir (no model/config.json) "
    echo "       nor a run dir (no checkpoints/). Aborting."
    exit 2
fi

SEED_ARG=()
if [ -n "$SEED" ]; then
    SEED_ARG=(--seed "$SEED")
fi

srun --kill-on-bad-exit=1 \
    gpt-simple generate \
        "${CKPT_ARG[@]}" \
        --prompts-file "$PROMPTS" \
        --output "$OUTPUT" \
        --max-new-tokens "$MAX_NEW_TOKENS" \
        --temperature "$TEMPERATURE" \
        --top-k "$TOP_K" \
        --top-p "$TOP_P" \
        --dtype "$DTYPE" \
        "${SEED_ARG[@]}"

echo "Wrote $(wc -l < "$OUTPUT") completions to $OUTPUT"
