#!/usr/bin/env bash
set -euo pipefail

# ---------------------------
# Off-policy GRPO FINE sweep
# ---------------------------
# Follow-up to run_grpo_offpolicy_coarse.sh. Coarse 3x3 results (best/final acc):
#   - e2/tb256 (2 upd/rollout): 0.760 / 0.725  -> best peak AND stable
#   - e1/tb64  (4 upd/rollout): 0.732 / 0.725  -> stable runner-up
#   - e1/tb128 (2 upd/rollout): 0.773 / 0.000  -> highest peak but collapsed
#   - all >=8 upd/rollout cells collapsed to 0.000 (late-training reward death)
# So we zoom into the low-intensity region, keep the highest-peak collapser
# (e1/tb128) to see if the collapse was seed variance, train longer, and SAVE
# checkpoints + eval on the full set so the peak is trustworthy and deployable.

ROLLOUT_BS=256

MODEL_ID="${MODEL_ID:-data/models/Qwen2.5-Math-1.5B}"
TRAIN_PATH="${TRAIN_PATH:-data/MATH/train.jsonl}"
VAL_PATH="${VAL_PATH:-data/MATH/validation.jsonl}"
PROMPT_FILE="${PROMPT_FILE:-cs336_alignment/prompts/r1_zero.prompt}"

LEARNING_RATE="${LEARNING_RATE:-1e-5}"   # best lr from the v2/v3 lr sweeps
SEED="${SEED:-0}"

LOSS_TYPE="grpo_clip"
CLIPRANGE="${CLIPRANGE:-0.2}"

# Fine settings: longer than coarse (40), full eval set, save peaks.
N_GRPO_STEPS="${N_GRPO_STEPS:-100}"
EVAL_INTERVAL="${EVAL_INTERVAL:-5}"
EVAL_MAX_EXAMPLES="${EVAL_MAX_EXAMPLES:-1024}"
SAVE_INTERVAL="${SAVE_INTERVAL:-25}"     # keep checkpoints to recover the peak

SAMPLING_TEMP="${SAMPLING_TEMP:-1.0}"
SAMPLING_MIN_TOKENS="${SAMPLING_MIN_TOKENS:-4}"
SAMPLING_MAX_TOKENS="${SAMPLING_MAX_TOKENS:-1024}"

# Keep microbatch size constant for constant VRAM: grad_acc = train_bs / micro
MICRO_BS="${MICRO_BS:-2}"

USE_STD_NORM="${USE_STD_NORM:-1}"        # 1 => add --use-std-normalization

# GPU placement (cuda:0/1 free now; override for a second concurrent job).
POLICY_DEVICE="${POLICY_DEVICE:-cuda:0}"
VLLM_DEVICE="${VLLM_DEVICE:-cuda:1}"

LOG_DIR="${LOG_DIR:-runs/grpo_offpolicy_fine}"
mkdir -p "${LOG_DIR}"

# ---------------------------
# Candidate configs (productive region from coarse). Format: "epochs train_bs"
# ---------------------------
CANDIDATES=(
  "2 256"   # 2 updates/rollout  -> coarse winner (0.760, stable)
  "1 64"    # 4 updates/rollout  -> stable runner-up (0.732)
  "1 128"   # 2 updates/rollout  -> highest peak (0.773) but collapsed; retest
)

echo "=== Off-policy GRPO FINE sweep ==="
echo "rollout_bs=${ROLLOUT_BS} steps=${N_GRPO_STEPS} eval_interval=${EVAL_INTERVAL} eval_max=${EVAL_MAX_EXAMPLES} save_interval=${SAVE_INTERVAL}"
echo "candidates (epochs train_bs):"
printf '  - %s\n' "${CANDIDATES[@]}"
echo "lr=${LEARNING_RATE} micro_bs=${MICRO_BS} devices=${POLICY_DEVICE}/${VLLM_DEVICE}"
echo "log_dir=${LOG_DIR}"
echo

for cfg in "${CANDIDATES[@]}"; do
  E=$(echo "${cfg}" | awk '{print $1}')
  TB=$(echo "${cfg}" | awk '{print $2}')

  if (( ROLLOUT_BS % TB != 0 )); then
    echo "Skip epochs=${E} train_bs=${TB} (rollout_bs not divisible)"
    continue
  fi
  if (( TB % MICRO_BS != 0 )); then
    echo "Skip epochs=${E} train_bs=${TB} (train_bs not divisible by micro_bs=${MICRO_BS})"
    continue
  fi

  GA=$(( TB / MICRO_BS ))
  UPDATES_PER_EPOCH=$(( ROLLOUT_BS / TB ))
  TOTAL_UPDATES=$(( E * UPDATES_PER_EPOCH ))

  echo "---- Run epochs=${E} train_bs=${TB} grad_acc=${GA} updates/rollout=${TOTAL_UPDATES} ----"

  run_dir="${LOG_DIR}/e${E}_tb${TB}"

  cmd=(uv run python scripts/grpo_experiment.py
    --model-id "${MODEL_ID}"
    --train-path "${TRAIN_PATH}"
    --val-path "${VAL_PATH}"
    --prompt-file "${PROMPT_FILE}"
    --log-dir "${run_dir}"
    --seed "${SEED}"

    --loss-type "${LOSS_TYPE}"
    --cliprange "${CLIPRANGE}"

    --learning-rate "${LEARNING_RATE}"
    --n-grpo-steps "${N_GRPO_STEPS}"

    --rollout-batch-size "${ROLLOUT_BS}"
    --epochs-per-rollout-batch "${E}"
    --train-batch-size "${TB}"
    --gradient-accumulation-steps "${GA}"

    --sampling-temperature "${SAMPLING_TEMP}"
    --sampling-min-tokens "${SAMPLING_MIN_TOKENS}"
    --sampling-max-tokens "${SAMPLING_MAX_TOKENS}"

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

  echo
done

echo "=== FINE sweep complete ==="
