#!/bin/bash
#SBATCH --job-name=prep
#SBATCH --output=out_prep_%j.log
#SBATCH --partition=hopperday
#SBATCH --mem=48GB
#SBATCH --time=02:00:00

source /local/scratch/kkim636/miniconda3/etc/profile.d/conda.sh
conda activate kg
cd /local/scratch/kkim636

# keep the old split once (prep.py refuses to overwrite okg_split.pt)
[ -f okg_split_v1.pt ] || mv okg_split.pt okg_split_v1.pt

python -u prep.py || exit 1
cat split_report.csv

mkdir -p keep keep_v1
mv keep/keep_nodes_*.parquet keep_nodes_core_h1*.parquet keep_v1/ 2>/dev/null

python -u prune_stats.py --tier core --hops 3 \
  --disease_list keep/seed_diseases_neurodeg.csv --tag _neurodeg
python -u prune_stats.py --tier core --hops 2 --tag _geneonly