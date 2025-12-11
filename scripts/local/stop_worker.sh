#!/bin/bash
set -euo pipefail
LOG_DIR="${1:-logs}"
for k in 1 2 3 4; do
  PIDF="${LOG_DIR}/worker${k}.pid"
  if [ -f "${PIDF}" ]; then
    PID="$(cat "${PIDF}")"
    if ps -p "${PID}" >/dev/null 2>&1; then
      echo "Matando worker${k} (PID ${PID})"
      kill "${PID}" || true
    fi
  fi
done
