#!/bin/bash
set -euo pipefail

# Initialize Conda
CONDA_BASE="${HOME}/miniconda3"
if [ -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]; then
    # shellcheck disable=SC1091
    . "${CONDA_BASE}/etc/profile.d/conda.sh"
else
    CONDA_PATH="$(conda info --base 2>/dev/null || true)"
    if [ -n "${CONDA_PATH}" ] && [ -f "${CONDA_PATH}/etc/profile.d/conda.sh" ]; then
        # shellcheck disable=SC1091
        . "${CONDA_PATH}/etc/profile.d/conda.sh"
    else
        echo "ERROR: Could not locate Conda initialization script." >&2
        exit 1
    fi
fi

ENV_NAME="qsa"

echo "=== Creating environment '${ENV_NAME}' (Python 3.12, JAX CUDA 12) ==="

# Recreate environment
conda deactivate || true
conda remove -n "${ENV_NAME}" --all -y || true
conda create -n "${ENV_NAME}" python=3.12 -y
conda activate "${ENV_NAME}"

pip install --upgrade pip

# Locate requirements file
REQ_FILE="requirements.txt"
if [ ! -f "${REQ_FILE}" ]; then
    if [ -f "../requirements.txt" ]; then
        REQ_FILE="../requirements.txt"
    else
        echo "ERROR: requirements.txt not found." >&2
        exit 1
    fi
fi

# Install Python stack + JAX CUDA 12 wheels
pip install -r "${REQ_FILE}" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

echo "=== Environment '${ENV_NAME}' provisioned successfully ==="
