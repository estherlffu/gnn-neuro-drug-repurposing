#!/bin/bash
#SBATCH --job-name=rgcn_pk1
#SBATCH --output=out_rgcn_pruned_k1_%j.log
#SBATCH --partition=hopperday
#SBATCH --gres=gpu:1
#SBATCH --mem=64GB
#SBATCH --time=06:00:00

source /local/scratch/kkim636/miniconda3/etc/profile.d/conda.sh
conda activate kg
cd /local/scratch/kkim636

KEEP=keep/keep_nodes_core_h1_neurodeg.parquet

# Eval set restricted to the pruned universe U (nodes in KEEP).
# Separate file name so it never collides with evalset_full.pt used by the full run.
python -u make_evalset.py --keep $KEEP --out evalset_pruned_k1_neurodeg.pt

python -u train_rgcn2.py --tag pruned_k1_neurodeg --keep_nodes $KEEP \
  --evalset evalset_pruned_k1_neurodeg.pt --steps 20000 --eval_every 2000