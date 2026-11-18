import torch
import polars as pl
import numpy as np
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts" / "pruned_ef5"

data = torch.load(
    ARTIFACTS / "okg_split.pt",
    map_location="cpu",
    weights_only=False,
)

ckpt = torch.load(
    ARTIFACTS / "rgcn_pruned.pt",
    map_location="cpu",
    weights_only=False,
)

node_index = pl.read_parquet(ARTIFACTS / "node_index.parquet")
nodes = pl.read_parquet(ROOT / "nodes.parquet")

# Reconstruct model
from model import RGCN, build_message_graph

src = data["src"].long()
dst = data["dst"].long()
rid = data["rid"].long()
undirected = data["undirected"].bool()
node_type = data["node_type"].long()

rel_names = list(data["rel_names"])
type_names = list(data["type_names"])

n_nodes = int(data["num_nodes"])
n_rels = int(data["num_rels"])

drug_type = type_names.index("DRG")
disease_type = type_names.index("DIS")

blocked_rids = torch.tensor(
    [
        i for i, name in enumerate(rel_names)
        if name.upper().startswith("DRG-DIS|")
    ],
    dtype=torch.long,
)

mp_s, mp_d, mp_r = build_message_graph(
    src,
    dst,
    rid,
    undirected,
    data["split"]["train"].long(),
    blocked_rids,
    n_rels,
)

model = RGCN(
    n_nodes,
    n_rels,
    len(type_names),
    dim=int(ckpt["dim"]),
    n_layers=int(ckpt["layers"]),
    dropout=0.1,
    n_bases=int(ckpt["bases"]),
)

model.load_state_dict(ckpt["state_dict"])
model.eval()

with torch.no_grad():
    hidden = model.encode(
        torch.stack([mp_s, mp_d]),
        mp_r,
        node_type,
        edge_drop=0.0,
    )

disease_nids = torch.nonzero(node_type == disease_type).flatten()

disease_embeddings = hidden[disease_nids].numpy()

similarity = cosine_similarity(disease_embeddings)

# Map node IDs to names
id_by_nid = {
    int(row["nid"]): row["id"]
    for row in node_index.select(["id", "nid"]).iter_rows(named=True)
}

name_lookup = {}

for row in nodes.select(["id", "properties"]).iter_rows(named=True):
    import json
    try:
        props = json.loads(row["properties"]) if row["properties"] else {}
    except Exception:
        props = {}
    name_lookup[row["id"]] = props.get("name") or row["id"]

disease_names = [
    name_lookup.get(id_by_nid[int(n)], id_by_nid[int(n)])
    for n in disease_nids
]

pairs = []

for i in range(len(disease_names)):
    for j in range(i + 1, len(disease_names)):
        pairs.append({
            "disease_1": disease_names[i],
            "disease_2": disease_names[j],
            "cosine_similarity": float(similarity[i, j]),
        })

result = pl.DataFrame(pairs).sort(
    "cosine_similarity",
    descending=True,
)

print("Top 30 most similar disease embeddings:")
print(result.head(30))

print("\nOverall similarity distribution:")
print(result.select([
    pl.col("cosine_similarity").mean().alias("mean"),
    pl.col("cosine_similarity").median().alias("median"),
    pl.col("cosine_similarity").min().alias("min"),
    pl.col("cosine_similarity").max().alias("max"),
    pl.col("cosine_similarity").std().alias("std"),
]))

result.write_csv(ARTIFACTS / "disease_embedding_similarity.csv")