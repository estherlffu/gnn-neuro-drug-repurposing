#!/bin/bash
#SBATCH --job-name=rotate
#SBATCH --output=out_rotate_%j.log
#SBATCH --partition=hopperday
#SBATCH --gres=gpu:1
#SBATCH --mem=64GB
#SBATCH --time=03:00:00

source /local/scratch/kkim636/miniconda3/etc/profile.d/conda.sh
conda activate kg
cd /local/scratch/kkim636

KEEP=keep/keep_nodes_core_h1_neurodeg.parquet

# Same eval sets as the R-GCN runs (created by job_full.sh / job_rgcn pruned job; build only if missing)
[ -f evalset_full.pt ] || python -u make_evalset.py --out evalset_full.pt
[ -f evalset_pruned_k1_neurodeg.pt ] || python -u make_evalset.py --keep $KEEP --out evalset_pruned_k1_neurodeg.pt

python -u train_rotate.py --tag full --evalset evalset_full.pt --steps 20000 --eval_every 2000
python -u train_rotate.py --tag pruned_k1_neurodeg --keep_nodes $KEEP \
  --evalset evalset_pruned_k1_neurodeg.pt --steps 20000 --eval_every 2000