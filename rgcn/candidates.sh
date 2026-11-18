#!/bin/bash
#SBATCH --job-name=kg_predict
#SBATCH --output=logs/kg_predict_%j.out
#SBATCH --error=logs/kg_predict_%j.err
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

source ~/miniforge3/etc/profile.d/conda.sh
conda activate optimuskg312

cd /local/scratch/elfu/KG

echo "Starting candidate prediction..."
pwd
which python

python rgcn/predict_candidates_ef2.py

echo "Prediction finished."