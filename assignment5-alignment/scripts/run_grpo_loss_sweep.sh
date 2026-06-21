#!/usr/bin/env bash
set -euo pipefail

# ====== basic settings ======
MODEL_ID="${MODEL_ID:-data/models/Qwen2.5-Math-1.5B}"
TRAIN_PATH="${TRAIN_PATH:-data/MATH/train.jsonl}"
VAL_PATH="${VAL_PATH:-data/MATH/validation.jsonl}"
PROMPT_FILE="${PROMPT_FILE:-cs336_alignment/prompts/r1_zero.prompt}"
LOG_DIR="${LOG_DIR:-runs/grpo_loss_sweep}"

# ====== sweep grid ======
# LR fixed at the best value from the v2/v3 LR sweeps (1e-5 won both).
# Here we vary the loss/advantage estimator instead.
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
LOSS_TYPES=(no_baseline reinforce_with_baseline)

# ====== GRPO hypers ======
N_GRPO_STEPS="${N_GRPO_STEPS:-200}"
EVAL_INTERVAL="${EVAL_INTERVAL:-10}"
EVAL_MAX_EXAMPLES="${EVAL_MAX_EXAMPLES:-1024}"
SAVE_INTERVAL="${SAVE_INTERVAL:-25}"

ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-256}"
GROUP_SIZE="${GROUP_SIZE:-8}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-256}"
GRAD_ACC_STEPS="${GRAD_ACC_STEPS:-128}"
EPOCHS_PER_ROLLOUT="${EPOCHS_PER_ROLLOUT:-1}"

SAMPLING_TEMP="${SAMPLING_TEMP:-1.0}"
SAMPLING_MIN_TOKENS="${SAMPLING_MIN_TOKENS:-4}"
SAMPLING_MAX_TOKENS="${SAMPLING_MAX_TOKENS:-1024}"

GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.85}"
SEED="${SEED:-0}"

# Run on GPUs 2/3 (the LR sweep occupies 0/1).
POLICY_DEVICE="${POLICY_DEVICE:-cuda:2}"
VLLM_DEVICE="${VLLM_DEVICE:-cuda:3}"

# std normalization only affects the advantage-based arms; no_baseline uses raw
# rewards directly, so this flag is a no-op there. Kept on for an apples-to-apples
# reinforce_with_baseline run matching the LR sweeps.
USE_STD_NORM="${USE_STD_NORM:-1}"  # 1 => add --use-std-normalization

mkdir -p "${LOG_DIR}"

echo "=== GRPO loss-type sweep ==="
echo "model_id=${MODEL_ID}"
echo "log_dir=${LOG_DIR}"
echo "learning_rate=${LEARNING_RATE} (fixed)"
echo "loss_types=${LOSS_TYPES[*]}"
echo "steps=${N_GRPO_STEPS} rollout_bs=${ROLLOUT_BATCH_SIZE} group=${GROUP_SIZE} train_bs=${TRAIN_BATCH_SIZE} ga=${GRAD_ACC_STEPS} epochs=${EPOCHS_PER_ROLLOUT}"
echo "eval_interval=${EVAL_INTERVAL} eval_max_examples=${EVAL_MAX_EXAMPLES} save_interval=${SAVE_INTERVAL}"
echo

for loss in "${LOSS_TYPES[@]}"; do
  echo "---- Running loss_type=${loss} ----"

  run_dir="${LOG_DIR}/${loss}"

  cmd=(uv run python scripts/grpo_experiment.py
    --model-id "${MODEL_ID}"
    --train-path "${TRAIN_PATH}"
    --val-path "${VAL_PATH}"
    --prompt-file "${PROMPT_FILE}"
    --log-dir "${run_dir}"
    --seed "${SEED}"
    --learning-rate "${LEARNING_RATE}"
    --n-grpo-steps "${N_GRPO_STEPS}"
    --rollout-batch-size "${ROLLOUT_BATCH_SIZE}"
    --group-size "${GROUP_SIZE}"
    --train-batch-size "${TRAIN_BATCH_SIZE}"
    --gradient-accumulation-steps "${GRAD_ACC_STEPS}"
    --epochs-per-rollout-batch "${EPOCHS_PER_ROLLOUT}"
    --sampling-temperature "${SAMPLING_TEMP}"
    --sampling-min-tokens "${SAMPLING_MIN_TOKENS}"
    --sampling-max-tokens "${SAMPLING_MAX_TOKENS}"
    --gpu-memory-utilization "${GPU_MEM_UTIL}"
    --loss-type "${loss}"
    --eval-interval "${EVAL_INTERVAL}"
    --eval-max-examples "${EVAL_MAX_EXAMPLES}"
    --save-interval "${SAVE_INTERVAL}"
    --policy-device-str "${POLICY_DEVICE}"
    --vllm-device-str "${VLLM_DEVICE}"
  )

  if [[ "${USE_STD_NORM}" == "1" ]]; then
    cmd+=(--use-std-normalization)
  fi

  echo "${cmd[@]}"
  "${cmd[@]}"

  echo "---- Done loss_type=${loss} ----"
  echo
done

echo "=== Sweep complete ==="
