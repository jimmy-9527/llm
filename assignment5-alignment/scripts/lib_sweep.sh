#!/usr/bin/env bash
# Shared driver for experiment sweeps.
#
# Source this file, then call:
#   run_sweep <python_script> <log_dir> <common_flags> <run...>
#
# Each <run> is one string: "<name> <per-run args...>". For every run it invokes
#   uv run python <python_script> <per-run args> <common_flags>
# teeing combined stdout/stderr to <log_dir>/<timestamp>_<name>.log. The sweep
# stops (returns the failing exit code) as soon as any run exits non-zero.

run_sweep() {
  local py_script="$1"; shift
  local log_dir="$1"; shift
  local common_flags="$1"; shift
  local runs=("$@")

  mkdir -p "${log_dir}"
  echo "=== Starting sweep: ${py_script} @ $(date) ==="
  echo "Logs        -> ${log_dir}"
  echo "Common flags: ${common_flags}"
  echo

  local item name args ts log_file cmd exit_code
  for item in "${runs[@]}"; do
    name=$(echo "${item}" | awk '{print $1}')
    args=$(echo "${item}" | cut -d' ' -f2-)
    ts=$(date +"%Y%m%d_%H%M%S")
    log_file="${log_dir}/${ts}_${name}.log"
    cmd="uv run python ${py_script} ${args} ${common_flags}"

    echo "=== Run: ${name} @ $(date) ==="
    echo "Command: ${cmd}"
    echo "Log:     ${log_file}"
    echo

    set +e
    {
      echo "===== BEGIN ${name} $(date) ====="
      echo "CMD: ${cmd}"
      echo
      ${cmd}
      ec=$?
      echo
      echo "EXIT_CODE: ${ec}"
      echo "===== END ${name} $(date) ====="
      exit ${ec}
    } 2>&1 | tee "${log_file}"
    exit_code=${PIPESTATUS[0]}
    set -e

    if [[ "${exit_code}" -ne 0 ]]; then
      echo
      echo "!!! Run ${name} failed with exit code ${exit_code}. Stopping sweep."
      echo "See log: ${log_file}"
      return "${exit_code}"
    fi

    echo
    echo "=== Run ${name} finished successfully @ $(date) ==="
    echo
  done

  echo "=== All runs completed @ $(date) ==="
}
