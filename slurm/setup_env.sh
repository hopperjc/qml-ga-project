#!/bin/bash
set -euo pipefail

# 1. Inicializa o Conda do usuário
CONDA_BASE="${HOME}/miniconda3"
if [ -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]; then
  . "${CONDA_BASE}/etc/profile.d/conda.sh"
else
  CONDA_PATH="$(conda info --base 2>/dev/null || true)"
  if [ -n "${CONDA_PATH}" ] && [ -f "${CONDA_PATH}/etc/profile.d/conda.sh" ]; then
    . "${CONDA_PATH}/etc/profile.d/conda.sh"
  else
    echo "ERROR: Conda não encontrado. Instale Miniconda e tente novamente." >&2
    exit 1
  fi
fi

# 2. Nome e versão alvo do seu projeto
ENV_NAME="qmlga-py312"
PY_VER="3.12"

echo "=== Criando ambiente ${ENV_NAME} (Python ${PY_VER}) ==="

# 3. Recria o ambiente do zero
conda deactivate || true
conda remove -n "${ENV_NAME}" --all -y || true
conda create -n "${ENV_NAME}" python=${PY_VER} -y
conda activate "${ENV_NAME}"

# 4. Atualiza pip e instala Poetry dentro do ambiente Conda
python -m pip install --upgrade pip wheel setuptools
python -m pip install "poetry>=1.8,<2.0"

# 5. Vai para a raiz do projeto
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
  cd "${SLURM_SUBMIT_DIR}"
fi
if [[ "$(basename "$(pwd)")" == "scripts" ]]; then
  cd ..
fi
echo "Working dir: $(pwd)"

# 6. Faz o Poetry instalar no ambiente Conda atual
poetry config virtualenvs.create false --local

# 7. Usa o seu lockfile se existir, fixando versões
if [ -f "poetry.lock" ]; then
  echo "Instalando via poetry.lock"
  poetry install --no-interaction --no-root
else
  echo "Instalando e gerando lock pela primeira vez"
  poetry lock --no-update
  poetry install --no-interaction --no-root
fi

# 8. Instala o pacote em modo editável para expor os scripts console
pip install -e .

# 9. Verificação rápida
python - <<'PY'
import sys, pennylane, numpy, sklearn
print("Python:", sys.version.split()[0])
print("PennyLane:", pennylane.__version__)
print("NumPy:", numpy.__version__)
print("scikit-learn:", sklearn.__version__)
PY

echo "=== Ambiente ${ENV_NAME} provisionado com sucesso ==="
