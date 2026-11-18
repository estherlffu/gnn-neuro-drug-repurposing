#!/bin/bash
#SBATCH --job-name=prune_sizes
#SBATCH --output=out_prune_sizes_%j.log
#SBATCH --partition=hopperday
#SBATCH --mem=32GB
#SBATCH --time=02:00:00

set -euo pipefail

cd /local/scratch/elfu/KG

source ~/miniforge3/etc/profile.d/conda.sh
conda activate optimuskg312

python -u rgcn/pruning_sizes.py
