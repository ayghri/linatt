#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

# Usage: scripts/eval.sh <ckpt_dir> [extra hydra overrides...]
CKPT=${1:?"first arg = checkpoint dir"}
shift || true

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} \
    python eval.py +ckpt="${CKPT}" "$@"
