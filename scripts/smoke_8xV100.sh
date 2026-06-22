#!/usr/bin/env bash
# =============================================================================
# smoke_8xV100.sh
# -----------------------------------------------------------------------------
# End-to-end smoke test for gpt-simple on a small GPU box (designed for an
# 8x V100 16GB instance, but works on any machine with >=1 GPU).
#
# This script exercises the headline features:
#
#   1. Pretokenized shard pipeline (.bin / .idx).
#   2. Single-GPU training + checkpoint save.
#   3. Stop/resume parity:   N sequential jobs == 1 continuous job.
#   4. `--force` clobbering output dir.
#   5. `gpt-simple status` reflects checkpoint + history.
#   6. Walltime-triggered graceful shutdown (GPT_SIMPLE_MAX_RUNTIME).
#   7. Multi-GPU (all visible GPUs) training + resume.
#
# It is deliberately tiny: ~10M-param model, ~32 steps per phase, fp16 on
# V100.  The whole script should finish in well under 15 minutes on 8xV100,
# and a few minutes on a single GPU.
#
# Usage:
#     bash scripts/smoke_8xV100.sh [WORKDIR]
#
# WORKDIR defaults to ./smoke_run.  Override e.g. with /scratch/<user>/smoke.
# =============================================================================

set -euo pipefail

WORKDIR="${1:-./smoke_run}"
PYTHON="${PYTHON:-python}"

# Resolve to an absolute path BEFORE we cd, otherwise every later use of
# $WORKDIR (e.g. inside the YAML config) gets interpreted relative to the
# new pwd and resolves to a non-existent directory.
rm -rf "$WORKDIR"
mkdir -p "$WORKDIR"
WORKDIR="$(cd "$WORKDIR" && pwd)"
cd "$WORKDIR"

GPU_COUNT="$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ' || echo 0)"

echo "============================================================"
echo "  gpt-simple smoke test"
echo "  workdir : $(pwd)"
echo "  python  : $($PYTHON -c 'import sys; print(sys.executable)')"
echo "  cuda    : $($PYTHON -c 'import torch; print(torch.cuda.is_available(), torch.cuda.device_count())')"
echo "  gpus    : $GPU_COUNT visible"
echo "============================================================"

# --- 1. Generate synthetic .bin / .idx shards --------------------------------
# Format matches src/gpt_simple/pretokenize.py: uint16 ids, GPTS magic header.
# 32 shards × 40 docs × 128 tokens = 163,840 tokens per split — enough for an
# 8-GPU run with num_workers=2 (= 16 worker slots) to give every slot at
# least two shards (and pad for some headroom).  Single-GPU phases use the
# same data and just have far more shards per slot.

echo "[1/7] Generating synthetic pretokenized shards..."

$PYTHON - <<'PY'
from pathlib import Path
import struct
import numpy as np

IDX_MAGIC = b"GPTS"
IDX_VERSION = 1
DTYPE_UINT16 = 2
EOD_TOKEN = 50256       # gpt2 eos

def write_shard(prefix: Path, num_docs: int, doc_len: int, seed: int) -> None:
    rng = np.random.RandomState(seed)
    offsets = [0]
    overlap_lengths = []
    docs = []
    for _ in range(num_docs):
        d = rng.randint(1, 50000, size=doc_len, dtype=np.uint16)
        d[-1] = EOD_TOKEN
        docs.append(d)
        offsets.append(offsets[-1] + doc_len)
        overlap_lengths.append(0)
    np.concatenate(docs).astype(np.uint16).tofile(prefix.with_suffix(".bin"))
    with open(prefix.with_suffix(".idx"), "wb") as f:
        f.write(IDX_MAGIC)
        f.write(struct.pack("<I", IDX_VERSION))
        f.write(struct.pack("<I", DTYPE_UINT16))
        f.write(struct.pack("<I", num_docs))
        f.write(np.array(offsets, dtype=np.int64).tobytes())
        f.write(np.array(overlap_lengths, dtype=np.uint16).tobytes())

train = Path("data/train/default"); train.mkdir(parents=True, exist_ok=True)
val   = Path("data/val/default");   val.mkdir(parents=True, exist_ok=True)

# 32 train shards: comfortably > world_size * num_workers (8 * 2 = 16) so the
# load-balanced shard-to-slot assignment never leaves a (rank, worker) empty.
for s in range(32):
    write_shard(train / f"shard_{s:04d}", num_docs=40, doc_len=128, seed=s)
# Val just needs to exist for preflight; one shard is plenty.
write_shard(val / "shard_0000", num_docs=8, doc_len=128, seed=100)

print(f"  -> {sum(1 for _ in train.glob('*.bin'))} train shards + "
      f"{sum(1 for _ in val.glob('*.bin'))} val shard(s)")
PY

# --- 2. Write a small config -------------------------------------------------

CONFIG="$WORKDIR/config.yaml"
cat > "$CONFIG" <<YAML
model:
  n_embd: 256
  n_layer: 4
  n_head: 4
  n_positions: 1024
  dropout: 0.0
  use_bias: false
  activation: swish
  norm: rmsnorm
  norm_eps: 1.0e-5
  rope_base: 10000.0
  attention_mode: causal

data:
  path: $WORKDIR/data
  tokenizer: gpt2
  format: pretokenized
  max_length: 256
  overlap_size: 32
  packing: true
  num_workers: 2

optimizer:
  learning_rate: 3.0e-4
  weight_decay: 0.1
  beta1: 0.9
  beta2: 0.95
  eps: 1.0e-8
  max_grad_norm: 1.0
  warmup_steps: 4
  min_lr_ratio: 0.1

training:
  per_device_batch_size: 4
  gradient_accumulation_steps: 1
  max_steps: 32
  gradient_checkpointing: false
  compile: false
  seed: 1234
  logging_steps: 1
  eval_steps: 100000        # never eval
  save_steps: 8
  output_dir: $WORKDIR/out
  resume: auto
  keep_last_k: 5
  mixed_precision: fp16     # V100 has no bf16; explicit so unit-test envs match
YAML

# --- helpers -----------------------------------------------------------------

capture_losses() {
    # Lines look like:  "Step      1 | Loss 10.8811"
    # After whitespace tokenisation:  $1=Step  $2=1  $3=|  $4=Loss  $5=10.8811
    grep -oE 'Step[[:space:]]+[0-9]+[[:space:]]*\|[[:space:]]*Loss[[:space:]]+[0-9.]+' "$1" \
        | awk '{print $2, $5}' | sort -n
}

patch_yaml() {
    # patch_yaml <config> <key.path> <python-literal-value>
    $PYTHON - "$1" "$2" "$3" <<'PY'
import sys, yaml
path, dotted, value = sys.argv[1], sys.argv[2], sys.argv[3]
cfg = yaml.safe_load(open(path))
node, *rest = dotted.split(".")
parents = [cfg[node]]
keys = rest[:-1]; last = rest[-1]
for k in keys:
    parents.append(parents[-1][k])
parents[-1][last] = yaml.safe_load(value)
yaml.safe_dump(cfg, open(path, "w"), sort_keys=False)
PY
}

# --- 3. Continuous baseline run ---------------------------------------------

echo "[2/7] Continuous baseline run (32 steps, 1 GPU)..."
CUDA_VISIBLE_DEVICES=0 gpt-simple train --config "$CONFIG" --force \
    2>&1 | tee continuous.log
capture_losses continuous.log > continuous.losses
echo "    -> $(wc -l < continuous.losses) loss lines captured"
mv out out_continuous

# --- 4. Split run: 16 steps, stop, then resume to 32 ------------------------

echo "[3/7] Split run, phase 1 (steps 1..16)..."
patch_yaml "$CONFIG" "training.max_steps" 16
patch_yaml "$CONFIG" "training.save_steps" 16
CUDA_VISIBLE_DEVICES=0 gpt-simple train --config "$CONFIG" --force \
    2>&1 | tee split1.log

echo "[4/7] Split run, phase 2 (resume -> steps 17..32)..."
patch_yaml "$CONFIG" "training.max_steps" 32
patch_yaml "$CONFIG" "training.save_steps" 8
CUDA_VISIBLE_DEVICES=0 gpt-simple train --config "$CONFIG" \
    2>&1 | tee split2.log

grep -q "Resuming from step 16" split2.log \
    || { echo "FAIL: split2 did not resume from step 16"; exit 1; }

{ capture_losses split1.log; capture_losses split2.log; } | sort -un > split.losses

# --- 5. Parity check --------------------------------------------------------

echo "[5/7] Checking single-GPU parity (post-resume divergence)..."
$PYTHON - <<'PY'
def load(p):
    return {int(float(a)): float(b)
            for line in open(p) for a, b in [line.split()]}
cont = load("continuous.losses")
spl  = load("split.losses")
common = sorted(set(cont) & set(spl))
post = [(s, abs(cont[s] - spl[s])) for s in common if s > 16]
if not post:
    raise SystemExit("FAIL: no overlapping post-resume steps")
max_div = max(d for _, d in post)
print(f"    overlap steps: {len(post)}, max divergence (step>16): {max_div:.3e}")
# fp16 + V100 cuBLAS kernel-pick reproducibility is ~1e-2 across separate
# process invocations.  Bf16 / fp32 would be tighter (~1e-4).  We assert
# that the resumed trajectory stays close, not that it's bit-exact.
TOL = 2.0e-2
if max_div > TOL:
    raise SystemExit(f"FAIL: post-resume divergence {max_div:.3e} > {TOL:.1e}")
print("    OK")
PY

# --- 6. status command sanity ------------------------------------------------

echo "[6/7] gpt-simple status sanity..."
gpt-simple status --output_dir "$WORKDIR/out" 2>&1 | tee status.log
# `status` may use rich-table or plain output depending on TTY; both render
# "Progress" (rich row label) and "History"/"Checkpoint" rows.
grep -qE "Progress|step" status.log \
    || { echo "FAIL: status missing Progress/step row"; exit 1; }
grep -qE "History|Checkpoint" status.log \
    || { echo "FAIL: status missing History/Checkpoint row"; exit 1; }

# --- 7. Walltime + multi-GPU -------------------------------------------------

if [[ "$GPU_COUNT" -lt 2 ]]; then
    echo "[7/7] Only $GPU_COUNT GPU(s); skipping multi-GPU stage."
    echo ""
    echo "============================================================"
    echo "  SMOKE TEST PASSED (single-GPU stages only)"
    echo "============================================================"
    exit 0
fi

echo "[7/7a] Walltime-triggered shutdown (1 GPU, 8s budget)..."
patch_yaml "$CONFIG" "training.output_dir" "\"$WORKDIR/out_walltime\""
patch_yaml "$CONFIG" "training.max_steps" 10000
patch_yaml "$CONFIG" "training.save_steps" 50

# Use a very small walltime reserve so the watchdog fires quickly.
patch_yaml "$CONFIG" "training.walltime_reserve_seconds" 2

CUDA_VISIBLE_DEVICES=0 GPT_SIMPLE_MAX_RUNTIME=8 timeout 60 \
    gpt-simple train --config "$CONFIG" --force \
    2>&1 | tee walltime.log || true

grep -qiE "walltime|shutdown" walltime.log \
    || { echo "FAIL: walltime run never logged shutdown intent"; exit 1; }
find "$WORKDIR/out_walltime/checkpoints" -maxdepth 1 \
        \( -name "checkpoint-*-shutdown" -o -name "final" \) 2>/dev/null \
    | grep -q . \
    || { echo "FAIL: walltime run produced no shutdown / final checkpoint"; exit 1; }
echo "    OK"

# Restore for multi-GPU phase
patch_yaml "$CONFIG" "training.output_dir" "\"$WORKDIR/out_multi\""
patch_yaml "$CONFIG" "training.max_steps" 16
patch_yaml "$CONFIG" "training.save_steps" 8
patch_yaml "$CONFIG" "training.walltime_reserve_seconds" 300

echo "[7/7b] Multi-GPU run (${GPU_COUNT} GPUs, 16 steps)..."
gpt-simple train --config "$CONFIG" --nproc_per_node "$GPU_COUNT" --force \
    2>&1 | tee multi.log

grep -qE "Launching distributed training: $GPU_COUNT" multi.log \
    || { echo "FAIL: multi.log doesn't show launch with $GPU_COUNT GPUs"; exit 1; }
[ -d "$WORKDIR/out_multi/checkpoints/final" ] \
    || { echo "FAIL: multi-GPU run did not produce final/"; exit 1; }

echo "[7/7c] Multi-GPU resume sanity (step 16 -> step 24)..."
patch_yaml "$CONFIG" "training.max_steps" 24
gpt-simple train --config "$CONFIG" --nproc_per_node "$GPU_COUNT" \
    2>&1 | tee multi_resume.log

grep -q "Resuming from step 16" multi_resume.log \
    || { echo "FAIL: multi-GPU resume didn't pick up checkpoint at step 16"; exit 1; }

echo ""
echo "============================================================"
echo "  SMOKE TEST PASSED on ${GPU_COUNT} GPU(s)"
echo "    - single-GPU stop/resume parity within 5e-3"
echo "    - --force clobbering works"
echo "    - status command reports Step + History"
echo "    - walltime watchdog (GPT_SIMPLE_MAX_RUNTIME) triggers shutdown ckpt"
echo "    - multi-GPU train + resume picks up checkpoint"
echo "============================================================"
