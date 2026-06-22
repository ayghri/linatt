#!/usr/bin/env bash
# Preflight checks. Verify everything the training run needs, fast.
# Usage: bash scripts/preflight.sh [--require-data]
#   --require-data: also require the tokenized FineWeb-Edu cache to exist
#
# Run this before training (run_all.sh calls it automatically).
set -uo pipefail
cd "$(dirname "$0")/.."

REQUIRE_DATA=0
for arg in "$@"; do
    case "$arg" in
        --require-data) REQUIRE_DATA=1 ;;
    esac
done

OK=1
fail() { echo "  [FAIL] $1"; OK=0; }
pass() { echo "  [ OK ] $1"; }

echo "=== preflight ==="

# 1. python env
echo "-- python deps"
python - <<'PY' 2>/dev/null
import sys
missing = []
for mod in ('torch', 'fla', 'transformers', 'datasets', 'accelerate',
            'hydra', 'wandb', 'lm_eval', 'omegaconf', 'einops'):
    try:
        __import__(mod)
    except ImportError:
        missing.append(mod)
sys.exit(1 if missing else 0)
print(','.join(missing), file=sys.stderr)
PY
if [ $? -eq 0 ]; then
    pass "torch / fla / transformers / datasets / accelerate / hydra / wandb / lm_eval"
else
    fail "missing python deps - run scripts/setup.sh"
fi

# 2. transformers <5 (fla compat)
TF_OK=$(python -c "import transformers as t; print(int(int(t.__version__.split('.')[0])<5))" 2>/dev/null)
if [ "$TF_OK" = "1" ]; then
    pass "transformers <5 (fla _tied_weights_keys compat)"
else
    fail "transformers >=5; pip install 'transformers<5'"
fi

# 2b. mamba2 fast-kernel availability
echo "-- mamba2 kernels"
KERN_OK=$(python -c "
try:
    import causal_conv1d, mamba_ssm
    print(1)
except ImportError:
    print(0)
" 2>/dev/null)
if [ "$KERN_OK" = "1" ]; then
    pass "causal-conv1d + mamba-ssm (Mamba2 fast path)"
else
    echo "  [warn] causal-conv1d / mamba-ssm missing; Mamba2 will run on Triton fallback (slower but correct)"
fi

# 2c. tilelang (required on Hopper for gated_deltanet bwd correctness)
echo "-- tilelang (Hopper bwd correctness)"
TL_OK=$(python -c "
try:
    import tilelang
    print(1)
except ImportError:
    print(0)
" 2>/dev/null)
GPU_NAME=$(python -c "
try:
    import torch
    print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')
except Exception:
    print('')
" 2>/dev/null)
if [[ "$GPU_NAME" == *H100* || "$GPU_NAME" == *H200* || "$GPU_NAME" == *Hopper* ]]; then
    if [ "$TL_OK" = "1" ]; then
        pass "tilelang available (Hopper detected)"
    else
        fail "tilelang missing AND running on Hopper - gated chunk bwd will produce incorrect grads (fla #640). Install: pip install tilelang"
    fi
else
    if [ "$TL_OK" = "1" ]; then
        pass "tilelang available"
    else
        echo "  [info] tilelang not installed; not required on this GPU"
    fi
fi

# 3. GPUs
N_GPU=$(python -c "import torch; print(torch.cuda.device_count() if torch.cuda.is_available() else 0)" 2>/dev/null)
if [ "${N_GPU:-0}" -ge 1 ]; then
    GPU_NAME=$(python -c "import torch; print(torch.cuda.get_device_name(0))" 2>/dev/null)
    pass "${N_GPU}x ${GPU_NAME}"
else
    fail "no CUDA GPUs visible"
fi

# 4. W&B auth + project/entity reachable
echo "-- wandb"
PROJECT=$(grep -E '^\s*project:' conf/config.yaml | awk '{print $2}')
ENTITY=$(grep -E '^\s*entity:' conf/config.yaml | awk '{print $2}')
WANDB_OUT=$(python - <<PY 2>&1
import os, sys
try:
    import wandb
    api = wandb.Api()
    v = api.viewer
    user = v.username
    teams = list(v.teams)
    print(f'user={user}')
    print(f'teams={teams}')
    proj = '${PROJECT}'
    ent  = '${ENTITY}'
    if ent and ent != 'null' and ent not in teams and ent != user:
        print(f'ERROR: entity={ent!r} not in user teams {teams}', file=sys.stderr)
        sys.exit(2)
    sys.exit(0)
except Exception as e:
    print(f'ERROR: {e}', file=sys.stderr)
    sys.exit(1)
PY
)
WANDB_RC=$?
if [ "$WANDB_RC" -eq 0 ]; then
    pass "wandb auth ok ($(echo "$WANDB_OUT" | grep ^user=))  project=${PROJECT}  entity=${ENTITY}"
else
    fail "wandb: $WANDB_OUT"
fi

# 5. Tokenizer reachable (no auth required for ungated mirror)
echo "-- tokenizer"
TOK=$(grep -E '^tokenizer:' conf/data/fineweb_edu_10bt.yaml | awk '{print $2}')
python - <<PY 2>/dev/null
from transformers import AutoTokenizer
AutoTokenizer.from_pretrained('${TOK}', trust_remote_code=True)
PY
if [ $? -eq 0 ]; then
    pass "tokenizer ${TOK}"
else
    fail "tokenizer ${TOK} not reachable - check HF_TOKEN or switch tokenizer"
fi

# 6. Pretokenized data cache
echo "-- data cache"
CACHE=data/HuggingFaceFW/fineweb-edu/sample-10BT/train
if [ -d "$CACHE" ] && [ -f "$CACHE/dataset_info.json" ]; then
    SIZE=$(du -sh "$CACHE" 2>/dev/null | cut -f1)
    pass "tokenized FineWeb-Edu cache present (${SIZE})"
else
    if [ "$REQUIRE_DATA" = "1" ]; then
        fail "tokenized cache missing at $CACHE - run scripts/prepare.sh first"
    else
        echo "  [warn] tokenized cache missing (run scripts/prepare.sh) - skipping (preflight not strict)"
    fi
fi

# 7. All three model configs build (cheap)
echo "-- model configs build"
PYTHONPATH=. python - <<'PY' 2>/dev/null
import yaml, torch, fla, fla_patches
from transformers import AutoConfig, AutoModelForCausalLM
sizes = {}
for f in ('gated_deltanet_200m', 'delta_net_200m', 'mamba2_200m', 'transformer_200m', 'kata_200m', 'kata_spd_200m'):
    y = yaml.safe_load(open(f'conf/model/{f}.yaml'))
    cfg = AutoConfig.for_model(**y['hf_kwargs'])
    m = AutoModelForCausalLM.from_config(cfg, dtype=torch.bfloat16)
    sizes[f] = sum(p.numel() for p in m.parameters()) / 1e6
print('  ' + ' | '.join(f'{k}={v:.0f}M' for k, v in sizes.items()))
PY
if [ $? -eq 0 ]; then
    pass "all baselines build"
else
    fail "one or more model configs failed to build"
fi

echo
if [ "$OK" -eq 1 ]; then
    echo "=== preflight: PASS ==="
    exit 0
else
    echo "=== preflight: FAIL ==="
    exit 1
fi
