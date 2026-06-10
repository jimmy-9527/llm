#!/usr/bin/env bash
set -euo pipefail

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

LOG_DIR="logs/ei_experiment"
mkdir -p "${LOG_DIR}"

N_EI_STEPS=5
G=4
EPOCHS=1
EVAL_MAX_EXAMPLES=500
SAMPLING_MAX_TOKENS=1024
SAMPLING_MIN_TOKENS=4
SEED=0

TRAIN_DEVICE="cuda:0"
VLLM_DEVICE="cuda:0"
VLLM_GPU_MEM=0.5
DTYPE="float16"
ATTN="sdpa"

declare -a RUNS=(
  "d128   --D_i 128"
  "d256   --D_i 256"
  "d512   --D_i 512"
  "d1024  --D_i 1024"
  "dfull  --D_i 2048"
)

echo "=== Starting EI sweep at $(date) ==="
echo "Logs: ${LOG_DIR}"
echo "n_ei_steps=${N_EI_STEPS}  G=${G}  epochs=${EPOCHS}  dtype=${DTYPE}  attn=${ATTN}"
echo

for item in "${RUNS[@]}"; do
  name=$(echo "$item" | awk '{print $1}')
  args=$(echo "$item" | cut -d' ' -f2-)

  ts=$(date +"%Y%m%d_%H%M%S")
  log_file="${LOG_DIR}/${ts}_${name}.log"

  echo "=== Run: ${name} @ $(date) ==="
  echo "CMD: uv run python scripts/expert_iteration_experiment.py ${args} ..."
  echo "LOG: ${log_file}"
  echo

  set +e
  {
    echo "===== BEGIN ${name} $(date) ====="
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uv run python scripts/expert_iteration_experiment.py \
      ${args} \
      --n_ei_steps "${N_EI_STEPS}" \
      --G "${G}" \
      --epochs "${EPOCHS}" \
      --eval_max_examples "${EVAL_MAX_EXAMPLES}" \
      --sampling_max_tokens "${SAMPLING_MAX_TOKENS}" \
      --sampling_min_tokens "${SAMPLING_MIN_TOKENS}" \
      --seed "${SEED}" \
      --train_device "${TRAIN_DEVICE}" \
      --vllm_device "${VLLM_DEVICE}" \
      --vllm_gpu_memory_utilization "${VLLM_GPU_MEM}" \
      --dtype "${DTYPE}" \
      --attn_implementation "${ATTN}" \
      --micro_batch_size 1 \
      --lr 2e-6 \
      --offload_vllm \
      --out_dir "runs/ei_experiment"
    exit_code=$?
    echo
    echo "EXIT_CODE: ${exit_code}"
    echo "===== END ${name} $(date) ====="
    exit ${exit_code}
  } 2>&1 | tee "${log_file}"
  exit_code=${PIPESTATUS[0]}
  set -e

  if [[ "${exit_code}" -ne 0 ]]; then
    echo "!!! Run ${name} failed (exit ${exit_code}). Stopping sweep."
    echo "See: ${log_file}"
    exit "${exit_code}"
  fi

  echo "=== Run ${name} DONE @ $(date) ==="
  echo
done

echo "=== All EI runs finished @ $(date) ==="
