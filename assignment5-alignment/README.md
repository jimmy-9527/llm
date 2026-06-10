# CS336 Spring 2025 Assignment 5: Alignment

For a full description of the assignment, see the assignment handout at
[cs336_spring2025_assignment5_alignment.pdf](./cs336_spring2025_assignment5_alignment.pdf)

We include a supplemental (and completely optional) assignment on safety alignment, instruction tuning, and RLHF at [cs336_spring2025_assignment5_supplement_safety_rlhf.pdf](./cs336_spring2025_assignment5_supplement_safety_rlhf.pdf)

If you see any issues with the assignment handout or code, please feel free to
raise a GitHub issue or open a pull request with a fix.

## Environment Setup

### Prerequisites

- NVIDIA GPU (tested on Tesla T4, 15 GB VRAM, Compute Capability 7.5)
- CUDA driver installed (`nvidia-smi` should work)
- Python 3.11–3.12

### 1. Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
```

### 2. Persist CUDA_HOME

`/usr/local/cuda` is the standard CUDA toolkit path on this instance:

```bash
echo 'export CUDA_HOME=/usr/local/cuda' >> ~/.bashrc
echo 'export PATH=$CUDA_HOME/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

`flash-attn` requires `CUDA_HOME` to be set when compiling from source.

### 3. Create the venv and install dependencies

```bash
uv sync
```

This creates `.venv/` and installs all dependencies from `pyproject.toml` / `uv.lock`,
including `flash-attn==2.7.4.post1` (compiled from source).

### 4. Download the base model

```bash
.venv/bin/python - <<'EOF'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="Qwen/Qwen2.5-Math-1.5B",
    local_dir="data/models/Qwen2.5-Math-1.5B",
)
print("done")
EOF
```

The model is saved to `data/models/Qwen2.5-Math-1.5B/` — the default path expected by
all training scripts.

### 5. Run a script

```bash
uv run python scripts/grpo_experiment.py
```

### Environment details (as of 2026-06-08)

| Item | Value |
|------|-------|
| GPU | Tesla T4 |
| VRAM | 15,360 MiB |
| Compute Capability | 7.5 |
| CUDA toolkit | 12.6 (`/usr/local/cuda`) |
| PyTorch | 2.5.1+cu124 |
| Python | 3.12 |

---

## Setup

As in previous assignments, we use `uv` to manage dependencies.

1. Install all packages except `flash-attn`, then all packages (`flash-attn` is weird)
```
uv sync --no-install-package flash-attn
uv sync
```

2. Run unit tests:

``` sh
uv run pytest
```

Initially, all tests should fail with `NotImplementedError`s.
To connect your implementation to the tests, complete the
functions in [./tests/adapters.py](./tests/adapters.py).

