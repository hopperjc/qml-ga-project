#!/bin/bash
set -euo pipefail

# ==============================================================================
# Initialize Conda (mesma lógica/estrutura do script do Sérgio)
# ==============================================================================
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

ENV_NAME="qmlga"
PY_VER="3.12"

echo "=== Creating environment '${ENV_NAME}' (Python ${PY_VER}) ==="

# Recreate environment
conda deactivate || true
conda remove -n "${ENV_NAME}" --all -y || true
conda create -n "${ENV_NAME}" python=${PY_VER} -y
conda activate "${ENV_NAME}"

python -m pip install --upgrade pip wheel setuptools

# ==============================================================================
# Locate requirements file (mesma lógica do Sérgio)
# ==============================================================================
REQ_FILE="requirements.txt"
if [ ! -f "${REQ_FILE}" ]; then
    if [ -f "../requirements.txt" ]; then
        REQ_FILE="../requirements.txt"
    else
        echo "ERROR: requirements.txt not found." >&2
        exit 1
    fi
fi

# ==============================================================================
# Install Python stack (sem JAX; refletindo teu Poetry)
# ==============================================================================
python -m pip install -r "${REQ_FILE}"

# Instala o pacote em modo editável para expor os entrypoints (qmlga-*)
if [ -f "pyproject.toml" ] || [ -f "../pyproject.toml" ]; then
  PROJ_DIR="."
  if [ ! -f "pyproject.toml" ] && [ -f "../pyproject.toml" ]; then
    PROJ_DIR=".."
  fi
  python -m pip install -e "${PROJ_DIR}"
fi

echo "=== Environment '${ENV_NAME}' provisioned successfully ==="

# Quick check
python - <<'PY'
import sys, numpy, pandas, sklearn
import pennylane
print("Python:", sys.version.split()[0])
print("NumPy:", numpy.__version__)
print("Pandas:", pandas.__version__)
print("scikit-learn:", sklearn.__version__)
print("PennyLane:", pennylane.__version__)
PY
