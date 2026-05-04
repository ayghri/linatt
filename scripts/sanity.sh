#!/usr/bin/env bash
# End-to-end sanity check on local node (e.g. 2x3090). ~4 min.
# Tests: prepare -> train (50 steps) -> inline lm-eval on piqa -> standalone eval.
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL=${1:-gated_deltanet_200m}
NUM_GPUS=${NUM_GPUS:-2}

RUN=sanity_${MODEL}
CKPT=runs/${RUN}

echo "==> [1/3] Tokenize wikitext-2 (tiny)"
python prepare.py data=sanity

echo "==> [2/3] Train 50 steps on ${NUM_GPUS} GPUs"
accelerate launch \
    --num_processes "${NUM_GPUS}" \
    --num_machines 1 \
    --mixed_precision bf16 \
    --main_process_port 29501 \
    train.py \
    model="${MODEL}" \
    data=sanity \
    train=sanity \
    eval.fractions=[1.0] \
    eval.tasks=[piqa] \
    eval.batch_size=4 \
    run_name="${RUN}" \
    output_dir="${CKPT}"

echo "==> [3/3] Standalone eval on saved ckpt"
CUDA_VISIBLE_DEVICES=0 python eval.py \
    +ckpt="${CKPT}" \
    eval.tasks=[piqa] \
    eval.batch_size=4

echo "==> Sanity passed."
