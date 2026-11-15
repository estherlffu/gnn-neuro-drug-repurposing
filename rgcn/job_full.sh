#!/bin/bash
#SBATCH --job-name=rgcn_full
#SBATCH --output=out_rgcn_full_%j.log
#SBATCH --partition=hopperday
#SBATCH --gres=gpu:1
#SBATCH --mem=64GB
#SBATCH --time=06:00:00

source /local/scratch/kkim636/miniconda3/etc/profile.d/conda.sh
conda activate kg
cd /local/scratch/kkim636
# evalset_full.pt: U = all nodes (no --keep argument), built from the new val/test (neurodeg INDICATION)
python -u make_evalset.py --out evalset_full.pt
python -u train_rgcn2.py --tag full --steps 20000 --evalset evalset_full.pt --eval_every 2000