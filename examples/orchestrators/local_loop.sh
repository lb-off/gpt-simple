#!/usr/bin/env bash
#
# Laptop / single-workstation "resume loop" template.
#
# Runs gpt-simple in a loop until `gpt-simple status` reports COMPLETED.
# Useful for:
#   - Long unattended runs on a desktop where the trainer might be
#     killed by power events, OS sleep, or you Ctrl-C'ing it.
#   - Debugging the same resume code path you'll use on SLURM, without
#     needing a cluster.
#
# Usage:
#   CONFIG=config.yaml OUTPUT_DIR=./outputs ./local_loop.sh

set -uo pipefail

CONFIG=${CONFIG:-config.yaml}
OUTPUT_DIR=${OUTPUT_DIR:-./outputs}
NPROC_PER_NODE=${NPROC_PER_NODE:-1}
MAX_ATTEMPTS=${MAX_ATTEMPTS:-100}
SLEEP_BETWEEN_FAILURES=${SLEEP_BETWEEN_FAILURES:-30}

attempt=0
while [ "${attempt}" -lt "${MAX_ATTEMPTS}" ]; do
    attempt=$((attempt + 1))
    echo "──────────────────────────────────────────────"
    echo "Attempt ${attempt}/${MAX_ATTEMPTS} — $(date -u +%FT%TZ)"
    echo "──────────────────────────────────────────────"

    # Resume is automatic via training.resume=auto.
    gpt-simple train \
        --config "${CONFIG}" \
        --nproc_per_node "${NPROC_PER_NODE}" \
        --training.output_dir "${OUTPUT_DIR}"
    rc=$?

    STATUS_OUTPUT=$(gpt-simple status --output_dir "${OUTPUT_DIR}" 2>/dev/null || true)

    if echo "${STATUS_OUTPUT}" | grep -q "COMPLETED"; then
        echo "Training complete."
        exit 0
    fi

    # Bucket exhausted with data.allow_bucket_exhaustion=false: trainer halted
    # with a checkpoint instead of silently renormalising.  Needs a human
    # decision, so stop looping (it would re-halt forever otherwise).
    if echo "${STATUS_OUTPUT}" | grep -q "HALTED"; then
        echo "Training halted (bucket exhausted); not retrying."
        echo "To continue with a renormalised mix, rerun with --data.allow_bucket_exhaustion true."
        exit 0
    fi

    if echo "${STATUS_OUTPUT}" | grep -qE "ERROR|CRASHED"; then
        echo "Training reported an error; not retrying."
        echo "Run \`gpt-simple status --output_dir ${OUTPUT_DIR}\` for details."
        exit 1
    fi

    # Anything else (STOPPED, walltime, transient crash) — wait briefly
    # and try again from the latest checkpoint.
    echo "Exit code ${rc}; retrying in ${SLEEP_BETWEEN_FAILURES}s..."
    sleep "${SLEEP_BETWEEN_FAILURES}"
done

echo "Hit MAX_ATTEMPTS=${MAX_ATTEMPTS}; giving up."
exit 1
