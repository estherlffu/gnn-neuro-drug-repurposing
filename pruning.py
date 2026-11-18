import polars as pl
import json

nodes_path = "/local/scratch/elfu/KG/largest_connected_component_nodes.parquet"
edges_path = "/local/scratch/elfu/KG/largest_connected_component_edges.parquet"

# print(pl.read_parquet_schema(nodes))
# print(pl.read_parquet_schema(edges))

# print(pl.scan_parquet(nodes).head(5).collect())
# print(pl.scan_parquet(edges).head(5).collect())

nodes = pl.read_parquet(nodes_path,columns=["id", "label"])    
edges = pl.read_parquet(edges_path,columns=["from", "to", "label", "relation"])  

# Heldout: df with from, to, label for val/test DIS-GEN edges
edges_train = edges.join(heldout.select(["from", "to", "label"]),
                         on=["from", "to", "label"], how="anti")

# print(nodes.group_by("label").agg(
#     pl.col("id").first().alias("example_id"),
#     pl.len().alias("n"),
# ))

# terms = ["0150076", "0061900", "0150103", "0001774", "0048143", "0150077"]
# for fmt in ["GO:{}", "GO_{}"]:
#     ids = [fmt.format(t) for t in terms]
#     hit = nodes.filter(pl.col("id").is_in(ids))
#     print(fmt, f"{hit.height}/6 found:", hit["id"].to_list())

# All identified BPOs are present in the dataset
BPO_IDS = ["GO_0150076", "GO_0061900", "GO_0150103", "GO_0001774", "GO_0048143", "GO_0150077"]

# dis = (pl.scan_parquet(nodes_path)
#          .filter(pl.col("label") == "DIS")
#          .collect())
# dis.write_parquet("/local/scratch/elfu/KG/dis_nodes.parquet")

node_type = dict(zip(nodes["id"], nodes["label"]))

def make_adj(e):
    return pl.concat([
        e.select(a=pl.col("from"), b=pl.col("to")),
        e.select(a=pl.col("to"),   b=pl.col("from")),
    ]).unique()

def nbrs(adj, ids):
    return set(adj.filter(pl.col("a").is_in(list(ids)))["b"])

def of_type(ids, types):
    types = {types} if isinstance(types, str) else set(types)
    return {i for i in ids if node_type.get(i) in types}

def descendants(ids, e):
    hier = e.filter((pl.col("label") == "BPO-BPO") & (pl.col("relation") == "IS_A"))
    # verify: from = child, to = parent. If a known pair returns nothing, swap them.
    found, frontier = set(), set(ids)
    while frontier:
        kids = set(hier.filter(pl.col("to").is_in(list(frontier)))["from"])
        frontier = kids - found - set(ids)
        found |= kids
    return found

def resolve(groups, adj, e):
    resolved, seeds = {}, set()
    for g in groups:
        if "id" in g:
            ids = of_type(set(g["id"]), g["type"])
            print(f'{g["name"]}: {len(ids)}/{len(set(g["id"]))} ids found')
        else:
            source = set(resolved[g["derive_from"]])
            if g.get("expand_descendants"):
                source |= descendants(source, e)
            ids = of_type(nbrs(adj, source), g["type"])
        resolved[g["name"]] = ids
        seeds |= ids
    return seeds

def extract_subgraph(e, core_groups, context_groups, type_allow, k):
    adj = make_adj(e)
    core = resolve(core_groups, adj, e)

    reached, last = set(core), set(core)
    for hop in range(1, k + 1):
        frontier = of_type(nbrs(adj, last) - reached, type_allow)
        print(f"hop {hop}: +{len(frontier)}")
        reached |= frontier
        last = frontier
    neighbors = reached - core

    context  = resolve(context_groups, adj, e)
    anchored = context & nbrs(adj, core | neighbors)

    keep = list(core | neighbors | anchored)
    out_nodes = nodes.filter(pl.col("id").is_in(keep))
    out_edges = e.filter(pl.col("from").is_in(keep) & pl.col("to").is_in(keep))
    return out_nodes, out_edges