#!/bin/bash
#SBATCH --job-name=rgcn
#SBATCH --output=out_rgcn_%j.log
#SBATCH --partition=hopperday
#SBATCH --gres=gpu:1
#SBATCH --mem=64GB
#SBATCH --time=08:00:00

if [ -f /local/scratch/kkim636/miniconda3/etc/profile.d/conda.sh ]; then
  source /local/scratch/kkim636/miniconda3/etc/profile.d/conda.sh
  conda activate kg
elif [ -n "${CONDA_SH:-}" ]; then
  source "$CONDA_SH"
  conda activate "${CONDA_ENV:-kg}"
fi

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
OUT="${OUT_DIR:-artifacts}"
[ -f "$OUT/okg_split.pt" ] || python -u rgcn/prep.py --data_dir "${DATA_DIR:-.}" --out_dir "$OUT" || exit 1
python -u rgcn/train_rgcn.py --split "$OUT/okg_split.pt" --out_dir "$OUT" \
  --tag pruned --steps 5000 --eval_every 250 --eval_test
