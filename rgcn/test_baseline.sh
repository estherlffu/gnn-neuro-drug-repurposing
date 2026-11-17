#!/bin/bash
#SBATCH --job-name=rgcn_baseline
#SBATCH --output=out_rgcn_baseline_%j.log
#SBATCH --partition=hopperday
#SBATCH --gres=gpu:1
#SBATCH --mem=64GB
#SBATCH --time=12:00:00

# Baseline: same trainer and indication split, without the seed neighborhood cut.
# Anatomy expression and drug-drug edges are still dropped. Results go to
# artifacts/baseline so they do not overwrite the pruned run.
#
#   DATA_DIR=/path/to/parquet bash rgcn/test_baseline.sh
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
OUT="${OUT_DIR:-artifacts/baseline}"
mkdir -p "$OUT"

if [ ! -f "$OUT/okg_split.pt" ] || [ "${FORCE:-0}" = 1 ]; then
  python -u rgcn/prep.py --data_dir "${DATA_DIR:-.}" --out_dir "$OUT" --no_prune --force
fi

python -u rgcn/train_rgcn2.py --split "$OUT/okg_split.pt" --out_dir "$OUT" \
  --tag baseline --steps "${STEPS:-5000}" --eval_every "${EVAL_EVERY:-250}" \
  --patience "${PATIENCE:-10}" --eval_test
