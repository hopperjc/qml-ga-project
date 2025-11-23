#!/bin/bash
#SBATCH --job-name=qmlga_env_setup
#SBATCH --partition=long-complex
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --chdir=/home/CIN/mhjc/qml-ga-project       # raiz do repo (ajuste se seu path for outro)
#SBATCH --output=logs/setup_%j.out
#SBATCH --error=logs/setup_%j.err
# opcional: #SBATCH --exclude=cluster-node[1-5]

set -euo pipefail

echo "WD: $(pwd)"
mkdir -p logs

# normaliza finais de linha caso tenha editado no Windows
if command -v dos2unix >/dev/null 2>&1; then
  dos2unix slurm/setup_env.sh || true
fi

# garante permissão de execução (ou use 'source' sem precisar +x)
chmod +x slurm/setup_env.sh || true

echo "=== [SETUP JOB] Node: $SLURM_NODELIST  Date: $(date) ==="
# execute no mesmo shell (carrega conda etc.)
source slurm/setup_env.sh

echo "=== [SETUP JOB] DONE at $(date) ==="
