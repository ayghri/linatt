#!/usr/bin/env bash
# One-shot environment bootstrap.  Does NOT require mamba or conda.
#
# Strategy: use `uv` (single static binary) to install python 3.11 + a venv
# rooted at LinAtt/.venv. uv pulls a managed CPython if the system has none.
#
# Activate with:  source LinAtt/.venv/bin/activate
#
# Usage (from repo root or LinAtt/):  bash scripts/setup.sh
# Re-running is safe.
set -euo pipefail

PY_VERSION=${PY_VERSION:-3.11}
TORCH_VERSION=${TORCH_VERSION:-2.10.0}
TORCH_INDEX=${TORCH_INDEX:-https://download.pytorch.org/whl/cu128}
# Prebuilt wheels (cu12 + torch 2.10 + cxx11abi=TRUE + cp311). Match the env above.
CAUSAL_CONV1D_WHL=${CAUSAL_CONV1D_WHL:-https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.6.1.post4/causal_conv1d-1.6.1+cu12torch2.10cxx11abiTRUE-cp311-cp311-linux_x86_64.whl}
MAMBA_SSM_WHL=${MAMBA_SSM_WHL:-https://github.com/state-spaces/mamba/releases/download/v2.3.1/mamba_ssm-2.3.1+cu12torch2.10cxx11abiTRUE-cp311-cp311-linux_x86_64.whl}

# Resolve dirs.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LINATT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${LINATT_DIR}/.." && pwd)"
VENV_DIR="${LINATT_DIR}/.venv"

# 1) Ensure uv is on PATH (install to ~/.local/bin if missing).
if ! command -v uv >/dev/null 2>&1; then
    echo "==> Installing uv (no system package needed)"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
echo "uv: $(uv --version)"

# 2) Create a venv with Python 3.11 (uv downloads a managed CPython if needed).
echo "==> Creating venv at ${VENV_DIR} (python ${PY_VERSION})"
uv venv -p "${PY_VERSION}" "${VENV_DIR}"

# Use the venv's pip via uv pip for everything below.
PIP="uv pip install --python ${VENV_DIR}/bin/python"

echo "==> torch ${TORCH_VERSION} + triton (CUDA 12.x)"
$PIP "torch==${TORCH_VERSION}" triton --index-url "${TORCH_INDEX}"

echo "==> fla (editable, from ${REPO_ROOT})"
$PIP -e "${REPO_ROOT}"

echo "==> training stack"
# transformers<5: fla uses the legacy _tied_weights_keys list contract.
$PIP "transformers<5" datasets accelerate hydra-core wandb lm-eval einops

echo "==> mamba2 fast kernels"
$PIP "${CAUSAL_CONV1D_WHL}" || \
    echo "  [warn] causal-conv1d wheel install failed; Mamba2 will use Triton fallback (slower)."
$PIP "${MAMBA_SSM_WHL}" || \
    echo "  [warn] mamba-ssm wheel install failed; selective_state_update fast path off (training unaffected)."

echo "==> sanity import"
"${VENV_DIR}/bin/python" - <<'PY'
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
echo "==> Setup complete."
echo "    Activate with:  source ${VENV_DIR}/bin/activate"
echo "    Then:           wandb login"
echo "                    bash scripts/prepare.sh"
echo "                    NUM_GPUS=4 bash scripts/train.sh gated_deltanet_200m"
