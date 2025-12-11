#!/bin/bash
#SBATCH --job-name=qmlga-sweep-c1
#SBATCH --partition=short-simple
#SBATCH --qos=simple
#SBATCH --time=2-00:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --array=0-999%40
#SBATCH --output=logs/sweep_c1_%A_%a.out
#SBATCH --error=logs/sweep_c1_%A_%a.err

set -euo pipefail

ENV_NAME="qmlga"

CONDA_HOME="$HOME/miniconda3"
if [ -f "$CONDA_HOME/etc/profile.d/conda.sh" ]; then
    source "$CONDA_HOME/etc/profile.d/conda.sh"
else
    eval "$("$CONDA_HOME/bin/conda" shell.bash hook)"
fi
conda activate "$ENV_NAME"

[ -d "$HOME/qml-ga-project" ] || git clone --depth=1 https://github.com/hopperjc/qml-ga-project.git $HOME/qml-ga-project
cd $HOME/qml-ga-project
pip -q install -e .

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}

export QMLGA_DATA_ROOT="${SLURM_TMPDIR:-/tmp}/${SLURM_JOB_ID}.${SLURM_ARRAY_TASK_ID}/data"
mkdir -p "$QMLGA_DATA_ROOT"
rsync -a --delete data/processed/ "$QMLGA_DATA_ROOT/processed/"

python -m qml_ga.sweep.cli \
  --include_feature_maps amplitude,zz \
  --max_wires 30 \
  --trace \
  --shard_index ${SLURM_ARRAY_TASK_ID} \
  --shard_total 4500
