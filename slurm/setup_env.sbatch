#!/bin/bash
#SBATCH --job-name=qmlga_env_setup
#SBATCH --partition=long-complex
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/setup_%j.out
#SBATCH --exclude=cluster-node[1-5]
#SBATCH --chdir=${SLURM_SUBMIT_DIR}

# ==============================================================================
# Script Name: setup.sbatch
# Description: Submete o provisionamento do ambiente ao Slurm, mantendo a
#              mesma estrutura do script do Sérgio (executa setup_env.sh).
# Usage:       sbatch scripts/setup.sbatch
# ==============================================================================

set -euo pipefail

# 0. Path safety
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    cd "$SLURM_SUBMIT_DIR" || exit 1
fi
if [[ "$(pwd)" == */scripts ]]; then
    cd ..
fi

echo "Working Directory: $(pwd)"
mkdir -p logs

echo "=== [SETUP JOB] Initiating Environment Provisioning ==="
echo "Node: $SLURM_NODELIST"
echo "Date: $(date)"

# Normaliza finais de linha (caso tenha vindo do Windows)
if command -v dos2unix >/dev/null 2>&1; then
  dos2unix scripts/setup_env.sh || true
fi

if [ -f "scripts/setup_env.sh" ]; then
    # shellcheck disable=SC1091
    source scripts/setup_env.sh
else
    echo "CRITICAL ERROR: 'scripts/setup_env.sh' not found in $(pwd)"
    exit 1
fi

echo "=== [SETUP JOB] Completed Successfully at $(date) ==="
