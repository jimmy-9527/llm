#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export CUDA_HOME=/opt/pytorch/lib/python3.13/site-packages/nvidia/cu13
export PATH="$CUDA_HOME/bin:$PATH"

export UV_CACHE_DIR=/opt/dlami/nvme/.cache/uv
export HF_HOME=/opt/dlami/nvme/.cache/huggingface

if ! command -v uv &>/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source $HOME/.local/bin/env
fi

uv run --with huggingface_hub huggingface-cli download \
    Qwen/Qwen2.5-Math-1.5B \
    --local-dir "$SCRIPT_DIR/data/models/Qwen2.5-Math-1.5B"
