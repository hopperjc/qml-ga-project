#SBATCH --job-name=qmlga_setup
#SBATCH --partition=long-complex
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=logs/setup_%j.out
#SBATCH --chdir=${SLURM_SUBMIT_DIR}

set -euo pipefail

mkdir -p logs
echo "Node: $SLURM_NODELIST"
echo "Date: $(date)"
echo "Dir:  $(pwd)"

# Normaliza finais de linha caso tenha vindo do Windows
if command -v dos2unix >/dev/null 2>&1; then
  dos2unix scripts/setup_env.sh || true
fi

source scripts/setup_env.sh

echo "Setup finalizado em $(date)"
