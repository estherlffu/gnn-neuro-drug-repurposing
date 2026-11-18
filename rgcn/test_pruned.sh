#!/bin/bash
#SBATCH --job-name=rgcn_pruned
#SBATCH --output=out_rgcn_pruned_%j.log
#SBATCH --partition=hopperday
#SBATCH --gres=gpu:1
#SBATCH --mem=64GB
#SBATCH --time=08:00:00

cd /local/scratch/elfu/KG
source ~/miniforge3/etc/profile.d/conda.sh
conda activate optimuskg312

set -eo pipefail   

echo "python: $(which python)"; python --version
nvidia-smi -L

DATA="$PWD"
OUT="$PWD/artifacts/pruned"
mkdir -p "$OUT"


#!/bin/bash
#SBATCH --job-name=rgcn_pruned_ef4
#SBATCH --output=out_rgcn_pruned_ef4_%j.log
#SBATCH --partition=hopperday
#SBATCH --gres=gpu:1
#SBATCH --mem=64GB
#SBATCH --time=08:00:00

cd /local/scratch/elfu/KG
source ~/miniforge3/etc/profile.d/conda.sh
conda activate optimuskg312

set -eo pipefail

echo "python: $(which python)"
python --version
nvidia-smi -L

DATA="$PWD"
OUT="$PWD/artifacts/pruned_ef5"
mkdir -p "$OUT"

echo "1/2 Building the corrected pruned graph from $DATA"
python -u rgcn/pruning_ef3.py \
  --data_dir "$DATA" \
  --out_dir "$OUT" \
  --force

echo "2/2 Training and scoring held-out test edges"
python -u rgcn/train_rgcn_ef3.py \
  --split "$OUT/okg_split.pt" \
  --out_dir "$OUT" \
  --tag pruned \
  --steps 5000 \
  --eval_every 250 \
  --patience 10 \
  --eval_test

echo "Done. Scores are in $OUT/res_pruned.json"
