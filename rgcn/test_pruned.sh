#!/bin/bash
#SBATCH --job-name=rgcn_pruned
#SBATCH --output=out_rgcn_pruned_%j.log
#SBATCH --partition=hopperday
#SBATCH --gres=gpu:1
#SBATCH --mem=64GB
#SBATCH --time=08:00:00

# Pruned: the seed neighborhood (neurodegenerative diseases, their genes and
# pathways, and the drugs that touch those genes). Same trainer and split
# fractions as test_baseline.sh. Results go to artifacts/pruned.
#
#   DATA_DIR=/path/to/parquet bash rgcn/test_pruned.sh
#   FORCE=1 rebuilds okg_split.pt
set -euo pipefail

if [ -f /local/scratch/kkim636/miniconda3/etc/profile.d/conda.sh ]; then
  source /local/scratch/kkim636/miniconda3/etc/profile.d/conda.sh
  conda activate kg
elif [ -n "${CONDA_SH:-}" ]; then
  source "$CONDA_SH"
  conda activate "${CONDA_ENV:-kg}"
fi

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
OUT="${OUT_DIR:-artifacts/pruned}"
mkdir -p "$OUT"

if [ ! -f "$OUT/okg_split.pt" ] || [ "${FORCE:-0}" = 1 ]; then
  python -u rgcn/prep.py --data_dir "${DATA_DIR:-.}" --out_dir "$OUT" --force
fi

python -u rgcn/train_rgcn2.py --split "$OUT/okg_split.pt" --out_dir "$OUT" \
  --tag pruned --steps "${STEPS:-5000}" --eval_every "${EVAL_EVERY:-250}" \
  --patience "${PATIENCE:-10}" --eval_test
