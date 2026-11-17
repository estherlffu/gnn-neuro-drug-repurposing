#!/bin/bash
#SBATCH --job-name=prep
#SBATCH --output=out_prep_%j.log
#SBATCH --partition=hopperday
#SBATCH --mem=48GB
#SBATCH --time=02:00:00

# Builds the seed-pruned graph from nodes.parquet + edges.parquet.
# DATA_DIR is the directory that contains those two files.
if [ -f /local/scratch/kkim636/miniconda3/etc/profile.d/conda.sh ]; then
  source /local/scratch/kkim636/miniconda3/etc/profile.d/conda.sh
  conda activate kg
elif [ -n "${CONDA_SH:-}" ]; then
  source "$CONDA_SH"
  conda activate "${CONDA_ENV:-kg}"
fi

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
python -u rgcn/prep.py --data_dir "${DATA_DIR:-.}" --out_dir "${OUT_DIR:-artifacts}" --force
