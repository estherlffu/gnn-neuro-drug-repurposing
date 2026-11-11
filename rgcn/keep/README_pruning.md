# Pruning stats: BPO + gene seed, k = 1

Command: python prune_stats.py --tier core --hops 1
Seeds: 19 GO/Reactome terms matched by name (neuroinflam, microglial cell activation,
astrocyte activation, NLRP3 inflammasome) -> 132 unique genes (see seed_terms_core.csv, seed_nodes_core.csv).
Genes are taken from BPO/PWY-GEN edges in the TRAIN split only (no test edges used for pruning).
Expansion: 1 hop, ANA-GEN edges excluded, nodes with degree > 2000 are not expanded.

| | full | seed only | k = 1 |
|---|---|---|---|
| nodes | 191,737 | 151 | 13,016 (6.8%) |
| train edges | 19.65M | 474 | 2.08M (10.6%) |
| DRG-DIS INDICATION train edges | 52,324 | 0 | 1,175 (2.2%) |
| drug nodes | 17,747 | 0 | 195 |
| neuro-disease nodes (EFO_0000618) | 4,155 | 0 | 1,294 |

Takeaway: with BPO + gene seeds only, almost all drug-disease edges are lost (2.2% kept), so a
drug-disease evaluation on this graph is not possible. Disease nodes need to be in the seed (setting 3).

Format for sharing a pruned graph: a parquet file with one column "id" (OptimusKG node ids).
train_rgcn2.py reads it via --keep_nodes. Same split file (okg_split.pt) is used for every setting.
