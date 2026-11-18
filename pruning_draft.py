import argparse
from pathlib import Path
import polars as pl

# Set BPO IDs, determined by relevance to neuroinflammation
BPO_IDS = ["GO_0150076", "GO_0061900", "GO_0150103",
           "GO_0001774", "GO_0048143", "GO_0150077"]

HIERARCHY = {"BPO": ("BPO-BPO", "IS_A", "from")}

TYPE_ALLOW = {"GEN", "PWY"}

def build_groups(msigdb_ids, disease_ids):
    core = [
        {"name": "bpo",      "type": "BPO", "id": BPO_IDS},
        {"name": "go_genes", "type": "GEN", "derive_from": "bpo", "expand_descendants": False},
        {"name": "msigdb",   "type": "GEN", "id": msigdb_ids},
    ]
    context = [
        {"name": "neuro_dis", "type": "DIS", "id": disease_ids},
    ]
    return core, context

    # Graph
    def _ids(values):
    return pl.DataFrame({"id": pl.Series(list(values), dtype=pl.Utf8)})


class KG:
    def __init__(self, nodes_path, edges_path, heldout_path=None):
        self.nodes_path, self.edges_path = nodes_path, edges_path
        self.nodes = pl.read_parquet(nodes_path, columns=["id", "label"])
        edges = pl.read_parquet(edges_path, columns=["from", "to", "label", "relation"])

        self.heldout = None
        if heldout_path:
            self.heldout = pl.read_parquet(heldout_path).select(["from", "to", "label"])
            edges = edges.join(self.heldout, on=["from", "to", "label"], how="anti")
            print(f"removed held-out edges: {self.heldout.height:,}")
        self.edges = edges

        # undirected adjacency used for all "connected to" steps
        self.adj = pl.concat([
            edges.select(a=pl.col("from"), b=pl.col("to")),
            edges.select(a=pl.col("to"),   b=pl.col("from")),
        ]).unique()
        print(f"nodes: {self.nodes.height:,}  training edges: {edges.height:,}")

    def neighbors(self, ids):
        if not ids:
            return set()
        s = pl.DataFrame({"a": pl.Series(list(ids), dtype=pl.Utf8)})
        return set(self.adj.join(s, on="a", how="semi")["b"])

    def of_type(self, ids, types):
        if not ids:
            return set()
        types = [types] if isinstance(types, str) else list(types)
        hit = self.nodes.join(_ids(ids), on="id", how="semi")
        return set(hit.filter(pl.col("label").is_in(types))["id"])

    def descendants(self, ids):
        """All descendants of `ids` via the hierarchy defined for each node type."""
        out = set()
        for ntype, (label, rel, child_col) in HIERARCHY.items():
            start = self.of_type(ids, ntype)
            if not start:
                continue
            parent_col = "to" if child_col == "from" else "from"
            h = (self.edges.filter((pl.col("label") == label) & (pl.col("relation") == rel))
                           .select(child=pl.col(child_col), parent=pl.col(parent_col)))
            found, frontier = set(), set(start)
            while frontier:
                s = pl.DataFrame({"parent": pl.Series(list(frontier), dtype=pl.Utf8)})
                kids = set(h.join(s, on="parent", how="semi")["child"])
                frontier = kids - found - start
                found |= kids
            out |= found
        return out

# Resolve
def resolve(self, groups):
    resolved, seeds = {}, set()
    for g in groups:
        if "id" in g:                                        # explicit
            found = self.of_type(set(g["id"]), g["type"])
            print(f'  [{g["name"]}] explicit: {len(found)}/{len(set(g["id"]))} ids found')
        else:                                                # derived
            source = set(resolved[g["derive_from"]])
            if g.get("expand_descendants"):
                source |= self.descendants(source)
            found = self.of_type(self.neighbors(source), g["type"])
            print(f'  [{g["name"]}] derived from {g["derive_from"]}: {len(found)} nodes')
        resolved[g["name"]] = found
        seeds |= found
    return seeds

# Extract subgraph
def extract(self, core_groups, context_groups, type_allow, k):
        print("core groups:")
        core = self.resolve(core_groups)
        reached = set(core)
        for hop in range(1, k + 1):
            frontier = self.of_type(self.neighbors(reached) - reached, type_allow)
            print(f"  hop {hop}: +{len(frontier):,} nodes")
            reached |= frontier
        neighbors = reached - core

        print("context groups:")
        context = self.resolve(context_groups)
        anchored = context & self.neighbors(core | neighbors)
        print(f"  anchored context nodes: {len(anchored):,}/{len(context):,}")

        keep = core | neighbors | anchored
        keep_lf = _ids(keep).lazy()

        out_nodes = (pl.scan_parquet(self.nodes_path)
                       .join(keep_lf, on="id", how="semi").collect())
        # edges of G with both ends in keep (training edges only, original columns kept)
        out_edges = (pl.scan_parquet(self.edges_path)
                       .join(keep_lf, left_on="from", right_on="id", how="semi")
                       .join(keep_lf, left_on="to",   right_on="id", how="semi")
                       .collect())
        if self.heldout is not None:
            out_edges = out_edges.join(self.heldout, on=["from", "to", "label"], how="anti")
        return out_nodes, out_edges, keep

# Main
def read_ids(path):
    return [l.strip() for l in Path(path).read_text().splitlines() if l.strip()]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nodes", required=True)
    ap.add_argument("--edges", required=True)
    ap.add_argument("--heldout", help="parquet with from,to,label of val/test edges")
    ap.add_argument("--msigdb-ensg", required=True, help="ENSG IDs, one per line")
    ap.add_argument("--diseases", required=True, help="disease node IDs (DOID_...), one per line")
    ap.add_argument("--k", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--out", default=".")
    a = ap.parse_args()

    kg = KG(a.nodes, a.edges, a.heldout)
    core_groups, context_groups = build_groups(read_ids(a.msigdb_ensg), read_ids(a.diseases))
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    for k in a.k:
        print(f"\n=== k = {k} ===")
        n, e, keep = kg.extract(core_groups, context_groups, TYPE_ALLOW, k)
        print(f"result: {n.height:,} nodes, {e.height:,} edges")
        print(n.group_by("label").len().sort("len", descending=True))

        touched = pl.concat([e["from"], e["to"]]).unique().len()
        print(f"isolated nodes: {n.height - touched:,}")
        if kg.heldout is not None:
            ks = pl.DataFrame({"id": pl.Series(list(keep), dtype=pl.Utf8)})
            surv = (kg.heldout.join(ks, left_on="from", right_on="id", how="semi")
                              .join(ks, left_on="to",   right_on="id", how="semi"))
            print(f"held-out edges with both endpoints retained: {surv.height:,}/{kg.heldout.height:,}")

        n.write_parquet(out / f"sub_k{k}_nodes.parquet")
        e.write_parquet(out / f"sub_k{k}_edges.parquet")


if __name__ == "__main__":
    main()