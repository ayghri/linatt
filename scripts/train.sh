#!/usr/bin/env bash
# Train one arch on this node. Runs preflight first; aborts on failure.
#
# Usage:  scripts/train.sh <model> [extra hydra overrides...]
#   scripts/train.sh gated_deltanet_200m
#   scripts/train.sh delta_net_200m train.lr=4e-4
#   scripts/train.sh mamba2_200m
#
# Multi-node usage: launch one arch per node, e.g.
#   node-A:  scripts/train.sh gated_deltanet_200m
#   node-B:  scripts/train.sh delta_net_200m
#   node-C:  scripts/train.sh mamba2_200m
# (Each node uses its own 8 GPUs via DDP; runs are independent and log
# separately to the same W&B project.)
#
# Single-node sequential alternative: scripts/run_all.sh
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL=${1:?"first arg = model name (gated_deltanet_200m | delta_net_200m | mamba2_200m | <your_arch>)"}
shift || true

NUM_GPUS=${NUM_GPUS:-8}
MASTER_PORT=${MASTER_PORT:-29500}

# Preflight (env, GPU, wandb, tokenizer, data cache, model build).
bash scripts/preflight.sh --require-data

accelerate launch \
    --num_processes "${NUM_GPUS}" \
    --num_machines 1 \
    --mixed_precision bf16 \
    --main_process_port "${MASTER_PORT}" \
    train.py \
    model="${MODEL}" \
    "$@"
