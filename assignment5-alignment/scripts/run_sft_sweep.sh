#!/usr/bin/env bash
set -euo pipefail

# SFT sample-size sweep.
#
# By default this runs the standard (unfiltered) SFT sweep, logging to
# logs/sft_experiment.
#
# Pass --filtered to run the same sweep with --filter_correct (filtered SFT),
# logging to logs/sft_experiment_filtered instead.
#
# Usage:
#   bash scripts/run_sft_sweep.sh              # standard SFT (default)
#   bash scripts/run_sft_sweep.sh --filtered   # filtered SFT

FILTERED=0
for arg in "$@"; do
  case "$arg" in
    --filtered) FILTERED=1 ;;
    *) echo "Unknown argument: $arg" >&2; exit 2 ;;
  esac
done

LOG_DIR="logs/sft_experiment"
FILTER_FLAG=""
if [[ "${FILTERED}" -eq 1 ]]; then
  LOG_DIR="logs/sft_experiment_filtered"
  FILTER_FLAG="--filter_correct"
fi
mkdir -p "${LOG_DIR}"

EVAL_INTERVAL=200
EVAL_MAX_EXAMPLES=500

# Sample sizes (0 = full dataset) and their step budgets.
declare -a RUNS=(
  "s128   --train_samples 128  --max_steps 2000"
  "s256   --train_samples 256  --max_steps 2000"
  "s512   --train_samples 512  --max_steps 2000"
  "s1024  --train_samples 1024 --max_steps 2000"
  "sfull  --train_samples 0    --max_steps 4000"
)

echo "=== Starting SFT sweep$([[ ${FILTERED} -eq 1 ]] && echo ' (filtered)') at $(date) ==="
echo "Logs will be saved to: ${LOG_DIR}"
echo

for item in "${RUNS[@]}"; do
  name=$(echo "$item" | awk '{print $1}')
  args=$(echo "$item" | cut -d' ' -f2-)

  cmd="uv run python scripts/sft_experiment.py ${FILTER_FLAG} ${args} --eval_interval ${EVAL_INTERVAL} --eval_max_examples ${EVAL_MAX_EXAMPLES}"
  ts=$(date +"%Y%m%d_%H%M%S")
  log_file="${LOG_DIR}/${ts}_${name}.log"

  echo "=== Run: ${name} @ $(date) ==="
  echo "Command: ${cmd}"
  echo "Log: ${log_file}"
  echo

  set +e
  {
    echo "===== BEGIN ${name} $(date) ====="
    echo "CMD: ${cmd}"
    echo
    ${cmd}
    exit_code=$?
    echo
    echo "EXIT_CODE: ${exit_code}"
    echo "===== END ${name} $(date) ====="
    exit ${exit_code}
  } 2>&1 | tee "${log_file}"
  exit_code=${PIPESTATUS[0]}
  set -e

  if [[ "${exit_code}" -ne 0 ]]; then
    echo
    echo "!!! Run ${name} failed with exit code ${exit_code}. Stopping sweep."
    echo "See log: ${log_file}"
    exit "${exit_code}"
  fi

  echo
  echo "=== Run ${name} finished successfully @ $(date) ==="
  echo
done

echo "=== All runs completed at $(date) ==="
