
import json
from pathlib import Path

import numpy as np
import polars as pl
import torch

from model import RGCN, build_message_graph


# ------------------------------------------------------------
# Paths and configuration
# ------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts" / "pruned_ef5"

GRAPH_FILE = ARTIFACTS / "okg_split.pt"
CHECKPOINT_FILE = ARTIFACTS / "rgcn_pruned.pt"
NODE_INDEX_FILE = ARTIFACTS / "node_index.parquet"
NODES_FILE = ROOT / "nodes.parquet"

OUTPUT_ALL = ARTIFACTS / "drug_disease_candidates.csv"
OUTPUT_TOP = ARTIFACTS / "top_100_candidates.csv"
OUTPUT_DISEASE_TOP = ARTIFACTS / "top_drugs_per_disease.csv"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------------------------------------------------------
# Load graph and checkpoint
# ------------------------------------------------------------

print("Loading graph and checkpoint...", flush=True)

data = torch.load(
    GRAPH_FILE, map_location="cpu", weights_only=False
)

ckpt = torch.load(
    CHECKPOINT_FILE, map_location="cpu", weights_only=False
)

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

drug_nids = torch.nonzero(node_type == drug_type).flatten()
disease_nids = torch.nonzero(node_type == disease_type).flatten()

print(f"Nodes: {n_nodes}")
print(f"Drugs: {len(drug_nids)}")
print(f"Diseases: {len(disease_nids)}")
print(f"Checkpoint best step: {ckpt['best_step']}")


# ------------------------------------------------------------
# Reconstruct message-passing graph
# ------------------------------------------------------------

# Block every drug-disease relation, just as in training.
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

edge_index = torch.stack([mp_s, mp_d]).to(DEVICE)
edge_type = mp_r.to(DEVICE)
node_type_device = node_type.to(DEVICE)

print(f"Message-passing edges: {mp_s.numel()}")


# ------------------------------------------------------------
# Reconstruct and load trained model
# ------------------------------------------------------------

model = RGCN(
    n_nodes,
    n_rels,
    len(type_names),
    dim=int(ckpt["dim"]),
    n_layers=int(ckpt["layers"]),
    dropout=0.1,
    n_bases=int(ckpt["bases"]),
).to(DEVICE)

model.load_state_dict(ckpt["state_dict"])
model.eval()

print("Model loaded successfully.")


# ------------------------------------------------------------
# Encode graph and score all drug-disease pairs
# ------------------------------------------------------------

print("Encoding graph...", flush=True)

with torch.no_grad():
    hidden = model.encode(
        edge_index,
        edge_type,
        node_type_device,
        edge_drop=0.0,
    )

    drug_device = drug_nids.to(DEVICE)
    disease_device = disease_nids.to(DEVICE)

    scores = model.score_candidates(
        hidden,
        drug_device,
        0,
        disease_device,
    )

scores = scores.cpu().numpy()

print(f"Scored {scores.size:,} drug-disease pairs.")


# ------------------------------------------------------------
# Recover drug and disease names
# ------------------------------------------------------------

print("Loading node names...", flush=True)

node_index = pl.read_parquet(NODE_INDEX_FILE)
nodes = pl.read_parquet(NODES_FILE)

name_lookup = {}

for row in nodes.select(["id", "properties"]).iter_rows(named=True):
    node_id = row["id"]

    try:
        props = json.loads(row["properties"]) if row["properties"] else {}
    except (json.JSONDecodeError, TypeError):
        props = {}

    name_lookup[node_id] = props.get("name") or node_id

id_by_nid = {
    int(row["nid"]): row["id"]
    for row in node_index.select(["id", "nid"]).iter_rows(named=True)
}

drug_nids_list = drug_nids.tolist()
disease_nids_list = disease_nids.tolist()

drug_ids = [id_by_nid[n] for n in drug_nids_list]
disease_ids = [id_by_nid[n] for n in disease_nids_list]

drug_names = [name_lookup.get(x, x) for x in drug_ids]
disease_names = [name_lookup.get(x, x) for x in disease_ids]


# ------------------------------------------------------------
# Extract known drug-disease relationships
# ------------------------------------------------------------

print("Extracting known relationships...", flush=True)

def extract_pairs(relation_name):
    """Return canonical (drug node ID, disease node ID) pairs."""

    if relation_name not in rel_names:
        return set()

    relation_id = rel_names.index(relation_name)
    edge_ids = torch.nonzero(rid == relation_id).flatten()

    s = src[edge_ids]
    d = dst[edge_ids]

    forward = (
        (node_type[s] == drug_type)
        & (node_type[d] == disease_type)
    )

    backward = (
        (node_type[d] == drug_type)
        & (node_type[s] == disease_type)
    )

    pairs = set()

    for drug, disease in zip(s[forward].tolist(), d[forward].tolist()):
        pairs.add((drug, disease))

    for disease, drug in zip(s[backward].tolist(), d[backward].tolist()):
        pairs.add((drug, disease))

    return pairs


known_indications = extract_pairs("DRG-DIS|INDICATION")
known_contraindications = extract_pairs("DRG-DIS|CONTRAINDICATION")
known_off_label = extract_pairs("DRG-DIS|OFF_LABEL_USE")

print("Known indications:", len(known_indications))
print("Known contraindications:", len(known_contraindications))
print("Known off-label uses:", len(known_off_label))


# ------------------------------------------------------------
# Construct full prediction table
# ------------------------------------------------------------

print("Building prediction table...", flush=True)

n_drugs = len(drug_nids_list)
n_diseases = len(disease_nids_list)

drug_grid = np.repeat(drug_nids_list, n_diseases)
disease_grid = np.tile(disease_nids_list, n_drugs)

score_flat = scores.reshape(-1)

# Rank diseases separately for each drug.
within_drug_ranks = (
    np.argsort(np.argsort(-scores, axis=1), axis=1) + 1
).reshape(-1)

rank_for_disease = (
    np.argsort(np.argsort(-scores, axis=0), axis=0) + 1
).reshape(-1)

indication_flags = np.array([
    (int(d), int(dis)) in known_indications
    for d, dis in zip(drug_grid, disease_grid)
])

contra_flags = np.array([
    (int(d), int(dis)) in known_contraindications
    for d, dis in zip(drug_grid, disease_grid)
])

off_label_flags = np.array([
    (int(d), int(dis)) in known_off_label
    for d, dis in zip(drug_grid, disease_grid)
])

drug_id_grid = np.repeat(drug_ids, n_diseases)
disease_id_grid = np.tile(disease_ids, n_drugs)

drug_name_grid = np.repeat(drug_names, n_diseases)
disease_name_grid = np.tile(disease_names, n_drugs)

df = pl.DataFrame({
    "drug_id": drug_id_grid,
    "drug_name": drug_name_grid,
    "disease_id": disease_id_grid,
    "disease_name": disease_name_grid,
    "score": score_flat,
    "rank_for_drug": within_drug_ranks,
    "rank_for_disease": rank_for_disease.flatten(),
    "known_indication": indication_flags,
    "known_off_label_use": off_label_flags,
    "known_contraindication": contra_flags,
})

df = df.sort("score", descending=True)

# Export every scored pair, including known relationships.
df.write_csv(OUTPUT_ALL)

disease_candidates = (
    df
    .filter(
        (~pl.col("known_indication"))
        & (~pl.col("known_off_label_use"))
        & (~pl.col("known_contraindication"))
    )
    .sort(["disease_id", "rank_for_disease"])
    .group_by("disease_id", maintain_order=True)
    .head(100)
    .sort(["disease_name", "rank_for_disease"])
)

disease_candidates.write_csv(OUTPUT_DISEASE_TOP)

print(f"Saved disease-specific rankings: {OUTPUT_DISEASE_TOP}")

# Candidate shortlist excludes known indications, off-label uses,
# and known contraindications.
candidates = df.filter(
    (~pl.col("known_indication"))
    & (~pl.col("known_off_label_use"))
    & (~pl.col("known_contraindication"))
)

candidates = candidates.sort("score", descending=True)

candidates.head(100).write_csv(OUTPUT_TOP)

print("\n========== RESULTS ==========")
print("Total pairs scored:", df.height)
print("Unobserved eligible candidates:", candidates.height)

print("\nTop 20 candidates:")
print(candidates.head(20))

print("\nSaved:")
print(OUTPUT_ALL)
print(OUTPUT_TOP)
