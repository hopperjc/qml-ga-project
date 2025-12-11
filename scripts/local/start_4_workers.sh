#!/bin/bash
set -euo pipefail

RUNLIST="${1:-local/plan_smoke_all/runlist.txt}"
PLAN_DIR="${2:-local/plan_smoke_all}"
LOG_DIR="${3:-logs}"

[ ! -f "${RUNLIST}" ] && { echo "RUNLIST não encontrado em ${RUNLIST}"; exit 1; }
mkdir -p "${PLAN_DIR}" "${LOG_DIR}"

# shardear em 4
for k in 0 1 2 3; do
  awk -v m=4 -v k="$k" '((NR-1)%m)==k {print}' "${RUNLIST}" > "${PLAN_DIR}/shard_$((k+1)).txt"
done

# caminhos absolutos
RUNNER="$(realpath scripts/local/run_worker.sh)"
PLAN_DIR_ABS="$(realpath "${PLAN_DIR}")"
LOG_DIR_ABS="$(realpath "${LOG_DIR}")"

# cria (ou zera) os logs antes de iniciar
for k in 1 2 3 4; do
  : > "${LOG_DIR_ABS}/worker${k}.log"
done

# inicia os 4 workers
for k in 1 2 3 4; do
  SHARD="${PLAN_DIR_ABS}/shard_${k}.txt"
  LOGF="${LOG_DIR_ABS}/worker${k}.log"
  PIDF="${LOG_DIR_ABS}/worker${k}.pid"

  echo "Iniciando worker ${k} com shard ${SHARD}; log em ${LOGF}"
  nohup bash "${RUNNER}" "${SHARD}" "${LOGF}" >/dev/null 2>&1 &
  echo $! > "${PIDF}"
done

echo "Workers iniciados."
for k in 1 2 3 4; do echo "  worker${k}: $(cat ${LOG_DIR}/worker${k}.pid)"; done
