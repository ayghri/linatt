#!/usr/bin/env bash
# Smoke test the full pipeline before launching the real run.
# Each arch: tokenize wikitext-2 -> 50 train steps -> inline lm-eval (FULL
# task list from conf/config.yaml) -> save checkpoint -> standalone eval
# (FULL list) on the saved ckpt. Every eval task is exercised end-to-end so
# dataset/loader breakage surfaces here, not in the production run.
#
# Default: runs all 4 baselines sequentially (~5-15 min total on H100,
# ~30 min on 2x3090).
#
# Usage:
#   bash scripts/sanity.sh                       # all 4 archs
#   bash scripts/sanity.sh transformer_200m      # one arch only
#
# Tunables:
#   NUM_GPUS=4 bash scripts/sanity.sh            # default = nvidia-smi count, capped at 8
set -euo pipefail
cd "$(dirname "$0")/.."

ARCHS=("$@")
if [ "${#ARCHS[@]}" -eq 0 ]; then
    ARCHS=(transformer_200m gated_deltanet_200m delta_net_200m mamba2_200m)
fi

if [ -z "${NUM_GPUS:-}" ]; then
    NGPU=$(nvidia-smi --query-gpu=count --format=csv,noheader 2>/dev/null | head -1 || echo 1)
    NUM_GPUS=$(( NGPU > 8 ? 8 : NGPU ))
fi
echo "==> Sanity sweep: ${#ARCHS[@]} arch(s) on ${NUM_GPUS} GPU(s)"

# 1) Tokenize the tiny dev dataset once (cached by HF datasets).
echo
echo "==> [shared] Tokenize wikitext-2 (cached after first run)"
python prepare.py data=sanity

START=$(date +%s)
for MODEL in "${ARCHS[@]}"; do
    echo
    echo "##############################"
    echo "# Sanity: ${MODEL}"
    echo "##############################"
    RUN=sanity_${MODEL}
    CKPT=runs/${RUN}

    echo "-- [1/2] Train 50 steps + inline FULL eval suite"
    # Uses default eval.tasks list from conf/config.yaml (all 8 tasks).
    # eval.fractions=[1.0] -> only the final eval (saves time vs 4 evals).
    # eval.batch_size=4 to fit on small dev GPUs.
    accelerate launch \
        --num_processes "${NUM_GPUS}" --num_machines 1 \
        --mixed_precision bf16 --main_process_port 29501 \
        train.py \
        model="${MODEL}" \
        data=sanity train=sanity \
        eval.fractions=[1.0] eval.batch_size=4 \
        run_name="${RUN}" output_dir="${CKPT}"

    echo "-- [2/2] Standalone FULL eval suite on saved ckpt"
    CUDA_VISIBLE_DEVICES=0 python eval.py \
        +ckpt="${CKPT}" eval.batch_size=4
done

DUR=$(( $(date +%s) - START ))
echo
echo "##############################"
echo "# Sanity sweep PASSED for ${#ARCHS[@]} arch(s) in $((DUR / 60))m $((DUR % 60))s"
echo "##############################"
