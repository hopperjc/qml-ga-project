#!/bin/bash
#SBATCH --job-name=qsa_env_setup
#SBATCH --partition=long-complex
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --output=logs/setup_%j.out
#SBATCH --exclude=cluster-node[1-5]

# ==============================================================================
# Script Name: setup.sbatch
# Description: Submits the environment provisioning script to the Slurm scheduler.
#              This ensures that:
#              1. Heavy installation tasks do not burden the login node.
#              2. Hardware-specific binaries (JAX/CUDA) are verified against
#                 the actual A100 compute nodes during installation.
#
# Usage:       sbatch scripts/setup.sbatch
# ==============================================================================

set -euo pipefail

# ------------------------------------------------------------------------------
# 0. Path Safety
# ------------------------------------------------------------------------------
# Change directory to the project root to ensure relative paths in setup_env.sh work.
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    cd "$SLURM_SUBMIT_DIR" || exit 1
fi

# Navigate up one level if submitted from the 'scripts/' directory.
if [[ "$(pwd)" == */scripts ]]; then
    cd ..
fi

echo "Working Directory: $(pwd)"
mkdir -p logs

# ------------------------------------------------------------------------------
# 1. Execute Setup Script
# ------------------------------------------------------------------------------
echo "=== [SETUP JOB] Initiating Environment Provisioning ==="
echo "Node: $SLURM_NODELIST"
echo "Date: $(date)"

# We use 'source' to execute the script within the current shell context.
# This allows the setup script to modify environment variables if necessary,
# although the primary artifact (the Conda environment on disk) is persistent.
if [ -f "scripts/setup_env.sh" ]; then
    source scripts/setup_env.sh
else
    echo "CRITICAL ERROR: 'scripts/setup_env.sh' not found in $(pwd)"
    exit 1
fi

echo "=== [SETUP JOB] Completed Successfully at $(date) ==="