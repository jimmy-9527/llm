#!/usr/bin/env bash
set -euo pipefail
#
# Resume the EI sweep after the compute_entropy OOM fix.
# Reruns only the configs that did NOT complete on the first pass:
#   - G2_E1_D2048  (crashed mid EI-step-3 on CUDA OOM)
#   - G8_E3_*      (never started; sweep aborts on first failure)
# G2_E1_D512 and G2_E1_D1024 already saved model_final and are skipped.
#
#   bash scripts/resume_ei_sweep.sh

cd "$(dirname "$0")/.."
source scripts/lib_sweep.sh

# expandable_segments reduces fragmentation on the shared T4s (belt-and-braces
# alongside the chunked compute_entropy fix).
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

COMMON="--n_ei_steps 5 --eval_max_examples 500 \
--sampling_max_tokens 256 --sampling_min_tokens 4 --seed 0 \
--train_device cuda:2 --vllm_device cuda:3"

RUNS=(
  "G2_E1_D2048  --G 2 --epochs 1 --D_i 2048"
  "G8_E3_D512   --G 8 --epochs 3 --D_i 512"
  "G8_E3_D1024  --G 8 --epochs 3 --D_i 1024"
  "G8_E3_D2048  --G 8 --epochs 3 --D_i 2048"
)

run_sweep scripts/ei_experiment.py "logs/expert_iteration" "${COMMON}" "${RUNS[@]}"
