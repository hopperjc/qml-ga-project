#!/bin/bash
set -euo pipefail
RUNLIST="${1:-local/plan/runlist.txt}"
PLAN_DIR="${2:-local/plan}"
LOG_DIR="${3:-logs}"
[ ! -f "${RUNLIST}" ] && { echo "RUNLIST não encontrado em ${RUNLIST}"; exit 1; }
mkdir -p "${PLAN_DIR}" "${LOG_DIR}"
for k in 0 1 2 3; do
  awk -v m=4 -v k="$k" '((NR-1)%m)==k {print}' "${RUNLIST}" > "${PLAN_DIR}/shard_$((k+1)).txt"
done
for k in 1 2 3 4; do
  SHARD="${PLAN_DIR}/shard_${k}.txt"
  LOGF="${LOG_DIR}/worker${k}.log"
  PIDF="${LOG_DIR}/worker${k}.pid"
  echo "Iniciando worker ${k} com shard ${SHARD}; log em ${LOGF}"
  nohup bash scripts/local/run_worker.sh "${SHARD}" "${LOGF}" >/dev/null 2>&1 &
  echo $! > "${PIDF}"
done
echo "Workers iniciados."
for k in 1 2 3 4; do echo "  worker${k}: $(cat ${LOG_DIR}/worker${k}.pid)"; done
