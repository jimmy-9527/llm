#!/usr/bin/env bash
set -euo pipefail
#
# SFT sample-size sweep. Runs both variants back-to-back on GPUs 0/1:
#   1. standard (unfiltered) SFT   -> logs/sft_experiment
#   2. correctness-filtered SFT    -> logs/sft_experiment_filtered
# (the filtered variant adds --filter_correct).
#
#   bash scripts/run_sft_sweep.sh

cd "$(dirname "$0")/.."
source scripts/lib_sweep.sh

COMMON="--eval_interval 200 --eval_max_examples 500"

# "<name> <per-run args>"  (train_samples 0 = full dataset)
RUNS=(
  "s128   --train_samples 128  --max_steps 2000"
  "s256   --train_samples 256  --max_steps 2000"
  "s512   --train_samples 512  --max_steps 2000"
  "s1024  --train_samples 1024 --max_steps 2000"
  "sfull  --train_samples 0    --max_steps 4000"
)

# 1. standard SFT
run_sweep scripts/sft_experiment.py "logs/sft_experiment" \
  "${COMMON} --train_device cuda:0 --vllm_device cuda:1" "${RUNS[@]}"

# 2. correctness-filtered SFT
run_sweep scripts/sft_experiment.py "logs/sft_experiment_filtered" \
  "${COMMON} --filter_correct --train_device cuda:0 --vllm_device cuda:1" "${RUNS[@]}"
