#!/usr/bin/env bash
# One-shot environment bootstrap for the H100 node.
# Creates a mamba env "linatt", installs torch + fla + training stack,
# verifies the model builds end-to-end.
#
# Usage (from repo root or LinAtt/):
#   bash scripts/setup.sh
#
# Re-running is safe: the env is idempotent.
set -euo pipefail

ENV_NAME=${ENV_NAME:-linatt}
PY_VERSION=${PY_VERSION:-3.11}
TORCH_VERSION=${TORCH_VERSION:-2.10.0}
# Pinned to match the causal-conv1d wheel: cu12 + torch 2.10 + cxx11_abi=TRUE + cp311
CAUSAL_CONV1D_WHL=${CAUSAL_CONV1D_WHL:-https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.6.1.post4/causal_conv1d-1.6.1+cu12torch2.10cxx11abiTRUE-cp311-cp311-linux_x86_64.whl}
MAMBA_SSM_WHL=${MAMBA_SSM_WHL:-https://github.com/state-spaces/mamba/releases/download/v2.2.5/mamba_ssm-2.2.5+cu12torch2.10cxx11abiTRUE-cp311-cp311-linux_x86_64.whl}

# Resolve repo root (one above LinAtt/).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LINATT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${LINATT_DIR}/.." && pwd)"

if ! command -v mamba >/dev/null 2>&1 && ! command -v conda >/dev/null 2>&1; then
    echo "ERROR: need mamba or conda on PATH." >&2
    exit 1
fi
PKGMGR=$(command -v mamba >/dev/null 2>&1 && echo mamba || echo conda)

echo "==> Creating ${ENV_NAME} (python=${PY_VERSION})"
$PKGMGR create -n "${ENV_NAME}" "python=${PY_VERSION}" -y -c conda-forge

# All subsequent installs go through pip in the new env.
PIP="$PKGMGR run -n ${ENV_NAME} pip install --quiet"

echo "==> torch ${TORCH_VERSION} + triton (CUDA 12.x build, H100 compatible)"
# torch 2.10 wheels live on cu128/cu129 channels (cu121 stopped at 2.5).
TORCH_INDEX=${TORCH_INDEX:-https://download.pytorch.org/whl/cu128}
$PIP "torch==${TORCH_VERSION}" triton --index-url "${TORCH_INDEX}"

echo "==> fla (editable, from repo root: ${REPO_ROOT})"
$PKGMGR run -n "${ENV_NAME}" pip install --quiet -e "${REPO_ROOT}"

echo "==> training stack"
# transformers<5 because fla uses the legacy _tied_weights_keys list contract.
$PIP "transformers<5" datasets accelerate hydra-core wandb lm-eval einops

echo "==> mamba2 fast kernels"
# Pinned wheel: cu12 + torch 2.10 + cxx11_abi=TRUE + cp311. Matches our env above.
$PIP "${CAUSAL_CONV1D_WHL}" || \
    echo "  [warn] causal-conv1d wheel install failed; Mamba2 will use Triton fallback (slower)."
# mamba-ssm: same ABI/cu/torch/python pins as causal-conv1d above.
# Note: only used by Mamba2 inference-time `selective_state_update`. Training
# is unaffected if this fails — causal-conv1d alone covers the training fast path.
$PIP "${MAMBA_SSM_WHL}" || \
    echo "  [warn] mamba-ssm wheel install failed; selective_state_update fast path off (training unaffected)."

echo "==> sanity import"
$PKGMGR run -n "${ENV_NAME}" python - <<'PY'
import sys, torch, fla, transformers, datasets, accelerate, hydra, wandb, lm_eval
print(f"python       {sys.version.split()[0]}")
print(f"torch        {torch.__version__}  cuda={torch.version.cuda}")
print(f"transformers {transformers.__version__}")
print(f"datasets     {datasets.__version__}")
print(f"accelerate   {accelerate.__version__}")
print(f"hydra-core   {hydra.__version__}")
print(f"wandb        {wandb.__version__}")
print(f"lm_eval      {lm_eval.__version__}")
print(f"fla          {getattr(fla, '__version__', 'editable')}")
try:
    import causal_conv1d, mamba_ssm
    print(f"causal_conv1d {causal_conv1d.__version__}  mamba_ssm {mamba_ssm.__version__}")
except Exception as e:
    print(f"mamba2 fast kernels: NOT installed ({e})")
print(f"GPUs         {torch.cuda.device_count()} x {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}")
PY

echo
echo "==> Setup complete. Activate with:  ${PKGMGR} activate ${ENV_NAME}"
echo "    Next: scripts/prepare.sh        (tokenizes FineWeb-Edu sample-10BT, ~2-4h)"
echo "          scripts/run_all.sh        (trains all 3 baselines sequentially)"
