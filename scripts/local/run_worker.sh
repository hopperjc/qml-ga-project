#!/bin/bash
set -euo pipefail
SHARD_FILE="${1:-}"
WORKER_LOG="${2:-logs/worker.log}"
[ -z "${SHARD_FILE}" ] && { echo "Uso: $0 <shard_file> [worker_log]"; exit 1; }
[ ! -f "${SHARD_FILE}" ] && { echo "Shard não encontrado: ${SHARD_FILE}"; exit 1; }
mkdir -p "$(dirname "${WORKER_LOG}")"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
while IFS= read -r CMD || [ -n "$CMD" ]; do
  [ -z "${CMD// }" ] && continue
  echo "[$(date +'%F %T')] START  ${CMD}" | tee -a "${WORKER_LOG}"
  set +e
  stdbuf -oL -eL bash -lc "$CMD" 2>&1 | tee -a "${WORKER_LOG}"
  EXIT=$?
  set -e
  echo "[$(date +'%F %T')] END    ${CMD} [exit=${EXIT}]" | tee -a "${WORKER_LOG}"
done < "${SHARD_FILE}"
