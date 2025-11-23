#!/bin/bash
set -euo pipefail

# Inicializa conda (tenta várias opções)
if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
elif [ -f "$HOME/miniforge3/etc/profile.d/conda.sh" ]; then
  . "$HOME/miniforge3/etc/profile.d/conda.sh"
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  . "$HOME/miniconda3/etc/profile.d/conda.sh"
elif command -v module >/dev/null 2>&1; then
  module load Miniconda3 || module load Anaconda3 || true
  command -v conda >/dev/null 2>&1 && eval "$(conda shell.bash hook)"
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "Conda não encontrado. Instale Miniforge/Miniconda no HOME." >&2
  exit 1
fi

ENV_NAME="qmlga"
PY_VER="3.12"

# recria o ambiente
conda deactivate || true
conda remove -n "${ENV_NAME}" --all -y || true
conda create -n "${ENV_NAME}" python=${PY_VER} -y
conda activate "${ENV_NAME}"

python -m pip install --upgrade pip wheel setuptools

# instala do requirements.txt na raiz do repo
pip install -r requirements.txt

# instala o pacote para expor os entrypoints (qmlga-*)
pip install -e .

# smoke test
python - <<'PY'
import sys, numpy, pandas, sklearn, pennylane
print("Python:", sys.version.split()[0])
print("NumPy:", numpy.__version__)
print("Pandas:", pandas.__version__)
print("scikit-learn:", sklearn.__version__)
print("PennyLane:", pennylane.__version__)
PY
