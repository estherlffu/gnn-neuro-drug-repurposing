#!/bin/bash
#SBATCH --job-name=rgcn_baseline
#SBATCH --output=out_rgcn_baseline_%j.log
#SBATCH --partition=hopperday
#SBATCH --gres=gpu:1
#SBATCH --mem=64GB
#SBATCH --time=8:00:00

cd ..
source project/bin/activate

DATA="$PWD"
OUT="$PWD/artifacts/baseline"
mkdir -p "$OUT"

echo "1/2  Building the baseline graph from $DATA"
python -u rgcn/baseline_ef1.py \
  --data_dir "$DATA" \
  --out_dir "$OUT" \
  --no_prune \
  --force

echo "2/2  Training, then scoring the held-out test edges"
python -u rgcn/train_rgcn_base_ef1.py \
  --split "$OUT/okg_split.pt" \
  --out_dir "$OUT" \
  --tag baseline \
  --steps 5000 \
  --eval_every 250 \
  --patience 10 \
  --eval_test

echo "Done. Scores are in $OUT/res_baseline.json"
