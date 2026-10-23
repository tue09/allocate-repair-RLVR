#!/usr/bin/env bash
# Create the conda environment for allocate-repair (verl v0.9.1 + vLLM 0.24.0 + torch 2.11.0).
#
#   bash scripts/setup_env.sh                        # auto-detect the CUDA build from the NVIDIA driver
#   CUDA_VARIANT=cu129 bash scripts/setup_env.sh     # force a variant (cu130 | cu129)
#   ENV_NAME=myenv bash scripts/setup_env.sh
#
# cu130 (driver >= 580): exactly verl's tested stack -- torch 2.11.0+cu130, vLLM 0.24.0 (PyPI) and the prebuilt
#                        flash-attn 2.8.3 wheel from verl's wheelhouse. No compilation.
# cu129 (driver >= 525): torch 2.11.0+cu129, the vLLM 0.24.0+cu129 release wheel, and flash-attn 2.8.3 built from
#                        source for A100 only (sm80, ~10-30 min). nvcc 12.x comes from $CUDA_HOME or conda.
# All other packages are pinned to verl's uv.lock through requirements/constraints-common.txt.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="${ENV_NAME:-arrlvr}"
PYTHON_VERSION=3.12
CUDA_VARIANT="${CUDA_VARIANT:-auto}"
VERL_REPO="https://github.com/verl-project/verl.git"
VERL_TAG="v0.9.1"
VERL_COMMIT="1876b06d0a3e4e71e06230be10af14492ca8a75b"
TORCH_VERSION=2.11.0
TORCHVISION_VERSION=0.26.0
TORCHAUDIO_VERSION=2.11.0
VLLM_VERSION=0.24.0
FLASH_ATTN_VERSION=2.8.3
VERL_WHEELHOUSE="https://verl-project.github.io/verl-wheelhouse/simple/"
CONSTRAINTS="$REPO_ROOT/requirements/constraints-common.txt"

log() { echo -e "\n\033[1;32m[setup]\033[0m $*"; }
die() { echo -e "\033[1;31m[setup] ERROR:\033[0m $*" >&2; exit 1; }

# ---------------------------------------------------------------- conda
command -v conda >/dev/null 2>&1 || die "conda not found. Install Miniconda/Miniforge first."
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"

# ---------------------------------------------------------------- CUDA variant
if [[ "$CUDA_VARIANT" == "auto" ]]; then
    command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi not found; run on a GPU node or set CUDA_VARIANT"
    DRIVER="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n1 | tr -d ' ')"
    DRIVER_MAJOR="${DRIVER%%.*}"
    if (( DRIVER_MAJOR >= 580 )); then
        CUDA_VARIANT=cu130
    elif (( DRIVER_MAJOR >= 525 )); then
        CUDA_VARIANT=cu129
    else
        die "NVIDIA driver $DRIVER is too old (need >= 525 for CUDA 12.x)"
    fi
    log "NVIDIA driver $DRIVER -> CUDA_VARIANT=$CUDA_VARIANT"
fi
case "$CUDA_VARIANT" in
    cu130 | cu129) ;;
    *) die "CUDA_VARIANT must be cu130 or cu129, got $CUDA_VARIANT" ;;
esac
TORCH_INDEX="https://download.pytorch.org/whl/${CUDA_VARIANT}"

# ---------------------------------------------------------------- verl source (pinned)
VERL_DIR="$REPO_ROOT/third_party/verl"
if [[ ! -f "$VERL_DIR/setup.py" ]]; then
    if git -C "$REPO_ROOT" rev-parse --git-dir >/dev/null 2>&1 && [[ -f "$REPO_ROOT/.gitmodules" ]]; then
        log "initializing the verl submodule"
        git -C "$REPO_ROOT" submodule update --init third_party/verl
    else
        log "cloning verl $VERL_TAG"
        git clone "$VERL_REPO" "$VERL_DIR"
    fi
fi
if [[ "$(git -C "$VERL_DIR" rev-parse HEAD 2>/dev/null || echo none)" != "$VERL_COMMIT" ]]; then
    log "checking out verl $VERL_TAG ($VERL_COMMIT)"
    git -C "$VERL_DIR" fetch --tags origin || true
    git -C "$VERL_DIR" checkout --quiet "$VERL_COMMIT"
fi

# ---------------------------------------------------------------- environment
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    log "reusing conda env '$ENV_NAME'"
else
    log "creating conda env '$ENV_NAME' (python $PYTHON_VERSION)"
    conda create -y -n "$ENV_NAME" "python=$PYTHON_VERSION" pip git
fi
conda activate "$ENV_NAME"
python -m pip install --upgrade pip setuptools wheel packaging ninja psutil

TORCH_PINS="$(mktemp)"
trap 'rm -f "$TORCH_PINS"' EXIT
printf "torch==%s+%s\ntorchvision==%s+%s\ntorchaudio==%s+%s\n" \
    "$TORCH_VERSION" "$CUDA_VARIANT" "$TORCHVISION_VERSION" "$CUDA_VARIANT" "$TORCHAUDIO_VERSION" "$CUDA_VARIANT" \
    >"$TORCH_PINS"
PIP_PINNED=(pip install -c "$CONSTRAINTS" -c "$TORCH_PINS" --extra-index-url "$TORCH_INDEX")

log "installing torch $TORCH_VERSION+$CUDA_VARIANT"
pip install "torch==${TORCH_VERSION}+${CUDA_VARIANT}" "torchvision==${TORCHVISION_VERSION}+${CUDA_VARIANT}" \
    "torchaudio==${TORCHAUDIO_VERSION}+${CUDA_VARIANT}" --index-url "$TORCH_INDEX"

log "installing vLLM $VLLM_VERSION ($CUDA_VARIANT)"
if [[ "$CUDA_VARIANT" == "cu130" ]]; then
    "${PIP_PINNED[@]}" "vllm==${VLLM_VERSION}"
else
    "${PIP_PINNED[@]}" "vllm @ https://github.com/vllm-project/vllm/releases/download/v${VLLM_VERSION}/vllm-${VLLM_VERSION}+cu129-cp38-abi3-manylinux_2_28_x86_64.whl"
fi

log "installing flash-attn $FLASH_ATTN_VERSION"
if [[ "$CUDA_VARIANT" == "cu130" ]]; then
    pip install --no-deps "flash-attn==${FLASH_ATTN_VERSION}" --index-url "$VERL_WHEELHOUSE"
else
    if [[ -z "${CUDA_HOME:-}" ]] || ! "${CUDA_HOME}/bin/nvcc" --version 2>/dev/null | grep -q "release 12"; then
        log "installing nvcc 12.9 into the conda env (needed to build flash-attn)"
        conda install -y -c "nvidia/label/cuda-12.9.1" cuda-toolkit
        export CUDA_HOME="$CONDA_PREFIX"
    fi
    log "building flash-attn for sm80 with $CUDA_HOME/bin/nvcc (this takes a while)"
    pip install einops
    FLASH_ATTN_CUDA_ARCHS=80 FLASH_ATTENTION_FORCE_BUILD=TRUE MAX_JOBS="${MAX_JOBS:-16}" \
        pip install --no-build-isolation --no-deps "flash-attn==${FLASH_ATTN_VERSION}"
fi

log "installing verl $VERL_TAG (editable) and remaining dependencies"
"${PIP_PINNED[@]}" -e "$VERL_DIR"
"${PIP_PINNED[@]}" liger-kernel trl qwen-vl-utils mathruler torchdata cachetools nvtx codetiming \
    "ray[default]" tensorboard fastapi uvicorn pytest pytest-asyncio
pip install "math-verify" -c "$CONSTRAINTS"
if [[ "${INSTALL_TRANSFERQUEUE:-0}" == "1" ]]; then
    pip install "transferqueue @ git+https://github.com/Ascend/TransferQueue.git@434f8c476b4be24bc087e6e95070e64efcc739f9" ||
        log "TransferQueue install failed (optional; not used by allocate-repair)"
fi

log "installing allocate-repair (editable)"
pip install --no-deps -e "$REPO_ROOT"

log "checking the environment"
python "$REPO_ROOT/scripts/check_env.py"
log "done. Activate with: conda activate $ENV_NAME"
