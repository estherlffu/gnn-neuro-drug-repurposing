#!/bin/bash
#SBATCH --job-name=rgcn_pruned
#SBATCH --output=out_rgcn_pruned_%j.log
#SBATCH --partition=hopperday
#SBATCH --gres=gpu:1
#SBATCH --mem=64GB
#SBATCH --time=08:00:00

cd ..
source project/bin/activate

DATA="$PWD"
OUT="$PWD/artifacts/pruned"
mkdir -p "$OUT"

echo "1/2  Building the pruned graph from $DATA"
python -u rgcn/prep.py \
  --data_dir "$DATA" \
  --out_dir "$OUT" \
  --force

echo "2/2  Training, then scoring the held-out test edges"
python -u rgcn/train_rgcn2.py \
  --split "$OUT/okg_split.pt" \
  --out_dir "$OUT" \
  --tag pruned \
  --steps 5000 \
  --eval_every 250 \
  --patience 10 \
  --eval_test

echo "Done. Scores are in $OUT/res_pruned.json"
