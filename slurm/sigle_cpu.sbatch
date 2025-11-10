#!/bin/bash
#SBATCH --job-name=qmlga-single
#SBATCH --partition=debug
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --output=logs/single_%j.out
#SBATCH --error=logs/single_%j.err

set -euo pipefail

module load Python/3.10
python -m venv $HOME/.venvs/qmlga-apuana
source $HOME/.venvs/qmlga-apuana/bin/activate
pip -q install --upgrade pip wheel setuptools

cd $HOME/qml-ga-project
pip -q install -e .

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}

export QMLGA_DATA_ROOT="${SLURM_TMPDIR:-/tmp}/${SLURM_JOB_ID}/data"
mkdir -p "$QMLGA_DATA_ROOT"
rsync -a --delete data/processed/ "$QMLGA_DATA_ROOT/processed/"

python -m qml_ga.single.cli \
  --dataset_yaml configs/datasets/banknote_authentication.yaml \
  --feature_map_yaml configs/feature_maps/amplitude.yaml \
  --ansatz_yaml configs/ansatz/ansatz_1_d15.yaml \
  --optimizer adam --lr 0.05 --epochs 5 --batch_size 16 \
  --wires 8 --val_size 0.2
