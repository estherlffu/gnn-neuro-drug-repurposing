#!/bin/bash
#SBATCH --job-name=disease_emb
#SBATCH --output=out_disease_emb_%j.log
#SBATCH --partition=hopperday
#SBATCH --mem=32GB
#SBATCH --time=01:00:00

cd /local/scratch/elfu/KG

source ~/miniforge3/etc/profile.d/conda.sh
conda activate optimuskg312

python rgcn/analyze_disease_embeddings.py