#!/usr/bin/env bash
# Sweep micro_batch_size on local GPUs, find max that fits while leaving
# headroom for the inline lm-eval forward pass.
set -uo pipefail
cd "$(dirname "$0")/.."

MODEL=${1:-gated_deltanet_200m}
NUM_GPUS=${NUM_GPUS:-2}
STEPS=${STEPS:-15}

# Target peak memory ceiling per GPU. Leave room for inline lm-eval (HFLM
# forward over the live model with eval batch_size=16).
# 3090 (24GB): cap at 21GB. H100 (80GB): cap at 72GB. Auto-detect.
GPU_MEM_GB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
GPU_MEM_GB=$((GPU_MEM_GB / 1024))
HEADROOM=3
if [ "$GPU_MEM_GB" -gt 40 ]; then HEADROOM=8; fi
CEIL=$((GPU_MEM_GB - HEADROOM))
echo "GPU has ${GPU_MEM_GB}GB; targeting peak ≤ ${CEIL}GB per GPU."

OUT=runs/bench_${MODEL}.log
rm -f "$OUT"

BEST_BS=0
BEST_TPS=0
for BS in 4 8 12 16 20 24 32 40 48 64; do
    echo "--- bs=${BS} ---" | tee -a "$OUT"
    set +e
    accelerate launch \
        --num_processes "${NUM_GPUS}" --num_machines 1 \
        --mixed_precision bf16 --main_process_port 29503 \
        bench.py \
        model="${MODEL}" \
        train.micro_batch_size="${BS}" \
        train.max_steps="${STEPS}" \
        wandb.mode=disabled 2>&1 | tee -a "$OUT" | grep -E "params|step_time|tokens/sec|peak_mem|10B-token|OutOfMemory|CUDA out of memory" 2>/dev/null
    EC=${PIPESTATUS[0]}
    set -e
    if [ "$EC" -ne 0 ]; then
        echo "  -> OOM/fail at bs=${BS}" | tee -a "$OUT"
        break
    fi
    PEAK=$(grep -oP "peak_mem/gpu\s*:\s*\K[0-9.]+" "$OUT" | tail -1)
    TPS=$(grep -oP "tokens/sec\s*:\s*\K[0-9.]+" "$OUT" | tail -1)
    if [ -n "$PEAK" ] && [ -n "$TPS" ]; then
        # awk for float compare
        OK=$(awk -v p="$PEAK" -v c="$CEIL" 'BEGIN{print (p<=c)?1:0}')
        if [ "$OK" -eq 1 ]; then
            BEST_BS=$BS
            BEST_TPS=$TPS
        else
            echo "  -> peak ${PEAK}GB exceeds ${CEIL}GB ceiling" | tee -a "$OUT"
            break
        fi
    fi
done

echo
echo "============== SWEEP RESULT =============="
echo "best_bs (within headroom) : $BEST_BS"
echo "best_tps (k tok/sec)      : $BEST_TPS"
if [ "$BEST_TPS" != "0" ]; then
    awk -v t="$BEST_TPS" 'BEGIN { printf "10B tokens ETA            : %.1f h\n", 1e10/(t*1e3)/3600 }'
fi
echo "Full log: $OUT"
