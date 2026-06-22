#!/usr/bin/env bash
# Single-node sequential fallback: trains all 3 baselines on one box.
# Prefer scripts/train.sh per node when you have multiple nodes.
#
# Usage:
#   bash scripts/run_all.sh
#   ARCHS="gated_deltanet_200m mamba2_200m" bash scripts/run_all.sh
#   bash scripts/run_all.sh train.lr=8e-4    # extra hydra overrides
set -euo pipefail
cd "$(dirname "$0")/.."

ARCHS=${ARCHS:-"gated_deltanet_200m delta_net_200m mamba2_200m transformer_200m kata_200m"}
NUM_GPUS=${NUM_GPUS:-8}

# Run preflight once up front so we fail fast.
bash scripts/preflight.sh --require-data

mkdir -p logs
START=$(date +%s)

for ARCH in $ARCHS; do
    echo
    echo "##############################"
    echo "# Training: ${ARCH}"
    echo "##############################"
    LOG=logs/${ARCH}_$(date +%Y%m%d_%H%M%S).log
    NUM_GPUS="${NUM_GPUS}" bash scripts/train.sh "${ARCH}" "$@" 2>&1 | tee "${LOG}"
done

DUR=$(( $(date +%s) - START ))
echo
echo "##############################"
echo "# Done. Total wall: $((DUR / 3600))h $(((DUR % 3600) / 60))m"
echo "# Checkpoints: runs/<arch>_fineweb_edu_10bt/"
echo "##############################"
