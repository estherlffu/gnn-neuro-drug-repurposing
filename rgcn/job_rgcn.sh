#!/bin/bash
#SBATCH --job-name=rgcn
#SBATCH --output=out_rgcn_%j.log
#SBATCH --partition=hopperday
#SBATCH --gres=gpu:1
#SBATCH --mem=64GB
#SBATCH --time=06:00:00

source /local/scratch/kkim636/miniconda3/etc/profile.d/conda.sh
conda activate kg
cd /local/scratch/kkim636
python -u train_rgcn.py --steps 20000
