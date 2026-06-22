#!/bin/bash
# =============================================================================
# Jean Zay SLURM template for `gpt-simple batch-generate`.
# -----------------------------------------------------------------------------
# Runs a SELF-DESCRIBING JSONL through one or more checkpoints on a single A100.
# Each input record may carry its own `model` block and `generation` overrides;
# records are grouped by resolved checkpoint and each model is loaded once.
#
# GATE BEFORE YOU SUBMIT — run the GPU-free pre-flight on a LOGIN node so a long
# queue wait isn't wasted on a typo (validates JSON, prompts, sampling ranges,
# and that every checkpoint + tokenizer exists, without loading weights):
#
#   gpt-simple batch-generate --input INPUT.jsonl --output OUT.jsonl --dry-run \
#       [--output-dir DEFAULT_RUN]      # only if some records use a job default
#
# Submit:
#   sbatch --export=ALL,\
# INPUT=$WORK/inference/batch-input.jsonl,\
# OUTPUT=$WORK/inference/batch-completions.jsonl \
#     examples/orchestrators/jeanzay_batch_generate.sbatch
#
# Optional --export knobs (these are *defaults*; per-record values override):
#   DEFAULT_MODEL    checkpoint dir OR run dir for records with no `model` block
#   MAX_NEW_TOKENS   default 256
#   TEMPERATURE      default 0.8
#   TOP_K            default 50
#   TOP_P            default 0.95
#   DTYPE            default bf16
#   SEED             default unset (non-deterministic sampling)
#
# Fan-out: the command groups by model, so one job over a mixed file already
# loads each checkpoint once. To split across an --array, shard the input file
# BY MODEL (one model per task) — never by line, or a checkpoint loads twice.
# =============================================================================

#SBATCH --job-name=gpts-batchgen
#SBATCH --account=CHANGEME@a100      # ← set to your Jean Zay GPU allocation
#SBATCH --constraint=a100
#SBATCH --qos=qos_gpu_a100-t3        # A100 QOS (up to 20h)
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1                 # inference fits on a single A100
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --time=02:00:00              # scale with (#prompts × max_new_tokens) + load time
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

# ── Inputs (from --export=ALL,...) ──────────────────────────────────
INPUT=${INPUT:?INPUT (path to the self-describing JSONL) must be set}
OUTPUT=${OUTPUT:?OUTPUT (path to the output JSONL) must be set}

DEFAULT_MODEL=${DEFAULT_MODEL:-}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-256}
TEMPERATURE=${TEMPERATURE:-0.8}
TOP_K=${TOP_K:-50}
TOP_P=${TOP_P:-0.95}
DTYPE=${DTYPE:-bf16}
SEED=${SEED:-}

mkdir -p "$(dirname "$OUTPUT")"

# ── Environment ─────────────────────────────────────────────────────
# arch/a100 MUST be loaded BEFORE pytorch-gpu (A100 nodes are AMD EPYC;
# the Intel build SIGILLs). Same ordering as the training job.
module purge
module load arch/a100
module load pytorch-gpu/py3/2.6.0
source "$WORK/venvs/gpt-simple-a100/bin/activate"

# Compute nodes have no internet — point HF at the offline caches.
export HF_HOME="$WORK/hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

ulimit -c 0   # no core dumps (they fill the $WORK quota)

# ── Optional job-level default model ────────────────────────────────
# Records that carry their own `model` block don't need this. If DEFAULT_MODEL
# is set, auto-detect whether it's a specific checkpoint dir or a run dir.
DEFAULT_ARG=()
if [ -n "$DEFAULT_MODEL" ]; then
    if [ -f "$DEFAULT_MODEL/model/config.json" ]; then
        DEFAULT_ARG=(--checkpoint "$DEFAULT_MODEL")
    elif [ -d "$DEFAULT_MODEL/checkpoints" ]; then
        DEFAULT_ARG=(--output-dir "$DEFAULT_MODEL")
    else
        echo "ERROR: DEFAULT_MODEL=$DEFAULT_MODEL is neither a checkpoint dir"
        echo "       (no model/config.json) nor a run dir (no checkpoints/)."
        exit 2
    fi
fi

SEED_ARG=()
if [ -n "$SEED" ]; then
    SEED_ARG=(--seed "$SEED")
fi

echo "============================================================="
echo "gpt-simple batch-generate — job ${SLURM_JOB_ID}"
echo "  node:     $(hostname)"
echo "  input:    $INPUT"
echo "  output:   $OUTPUT"
echo "  default:  ${DEFAULT_MODEL:-<none — records self-describe>}"
echo "  defaults: max_new=$MAX_NEW_TOKENS temp=$TEMPERATURE top_k=$TOP_K top_p=$TOP_P dtype=$DTYPE"
echo "============================================================="

# ── Launch ──────────────────────────────────────────────────────────
# Single process on one GPU — no torchrun, no srun wrapper. batch-generate
# runs its own GPU-free pre-flight first and exits non-zero on a bad record
# before any weights are loaded.
gpt-simple batch-generate \
    --input "$INPUT" \
    --output "$OUTPUT" \
    ${DEFAULT_ARG[@]+"${DEFAULT_ARG[@]}"} \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --temperature "$TEMPERATURE" \
    --top-k "$TOP_K" \
    --top-p "$TOP_P" \
    --dtype "$DTYPE" \
    ${SEED_ARG[@]+"${SEED_ARG[@]}"}

echo "Wrote $(wc -l < "$OUTPUT") completions to $OUTPUT"
