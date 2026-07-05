#!/usr/bin/env bash
set -euo pipefail
#
# Expert-iteration sweep over (G rollouts, epochs, D_i questions/step).
#   bash scripts/run_ei_sweep.sh

cd "$(dirname "$0")/.."
source scripts/lib_sweep.sh

COMMON="--n_ei_steps 5 --eval_max_examples 500 \
--sampling_max_tokens 256 --sampling_min_tokens 4 --seed 0 \
--train_device cuda:2 --vllm_device cuda:3"

# "<name> <per-run args>"
RUNS=(
  "G2_E1_D512   --G 2 --epochs 1 --D_i 512"
  "G2_E1_D1024  --G 2 --epochs 1 --D_i 1024"
  "G2_E1_D2048  --G 2 --epochs 1 --D_i 2048"
  "G8_E3_D512   --G 8 --epochs 3 --D_i 512"
  "G8_E3_D1024  --G 8 --epochs 3 --D_i 1024"
  "G8_E3_D2048  --G 8 --epochs 3 --D_i 2048"
)

run_sweep scripts/ei_experiment.py "logs/expert_iteration" "${COMMON}" "${RUNS[@]}"
