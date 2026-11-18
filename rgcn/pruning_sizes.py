
#!/usr/bin/env python3
"""
Compare graph sizes for BPO+GEN and BPO+GEN+DIS seed configurations.

Configurations:
    BPO+GEN:       k = 1, 2
    BPO+GEN+DIS:   k = 1, 2, 3

Important:
    - Uses 18 BPO seeds, 141 GEN seeds, and 79 DIS seeds.
    - Disease nodes are restricted to the original 79 seed diseases.
    - Expansion is undirected across the selected biological relationships.
    - All DRG nodes are added after biological expansion, matching the
      previous pruning_sizes.py behavior.
    - Counts induced edges among the selected nodes.

This is a k-hop sizing comparison, NOT an exact reproduction of
pruning_ef3.py, which applies additional relationship-specific filters
and node-selection caps.
"""

from pathlib import Path
from collections import Counter

import polars as pl


# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

NODES_FILE = ROOT / "nodes.parquet"
EDGES_FILE = ROOT / "edges.parquet"

SEED_DIR = ROOT / "pruning" / "seeds"

BPO_SEED_FILE = SEED_DIR / "seed_terms_core.csv"
GEN_SEED_FILE = SEED_DIR / "seed_nodes_core.csv"
DIS_SEED_FILE = SEED_DIR / "seed_diseases_neurodeg.csv"


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

# Relationships available in the graph.
ALLOW = {
    "DRG-DIS",
    "DRG-GEN",
    "DIS-GEN",
    "GEN-GEN",
    "DIS-DIS",
    "BPO-GEN",
    "BPO-BPO",
}

# BPO + GEN expansion.
BPO_GEN_LABELS = {
    "BPO-GEN",
    "BPO-BPO",
    "GEN-GEN",
}

BPO_GEN_TYPES = {
    "BPO",
    "GEN",
}

# BPO + GEN + DIS expansion.
BPO_GEN_DIS_LABELS = {
    "BPO-GEN",
    "BPO-BPO",
    "GEN-GEN",
    "DIS-GEN",
    "DIS-DIS",
}

BPO_GEN_DIS_TYPES = {
    "BPO",
    "GEN",
    "DIS",
}

# All DRG nodes are included after biological expansion.
ADD_ALL_DRUGS = True


# ------------------------------------------------------------
# Seed-loading helpers
# ------------------------------------------------------------

def read_ids(path: Path) -> list[str]:
    """Read IDs from a seed CSV containing an 'id' column."""

    if not path.exists():
        raise FileNotFoundError(f"Seed file not found: {path}")

    df = pl.read_csv(path)

    if "id" not in df.columns:
        raise ValueError(
            f"{path} must contain an 'id' column. "
            f"Found columns: {df.columns}"
        )

    return (
        df.select(pl.col("id").cast(pl.String))
        .drop_nulls()
        .to_series()
        .to_list()
    )


def add_id_variants(value: str) -> set[str]:
    """
    Generate common ID variants.

    Some IDs use ':' while their corresponding node IDs use '_'.
    """

    value = str(value).strip()

    if not value:
        return set()

    variants = {value}

    if ":" in value:
        variants.add(value.replace(":", "_"))

    if "_" in value:
        variants.add(value.replace("_", ":"))

    return variants


def resolve_many(
    raw_ids: list[str],
    canonical_ids: set[str],
    allowed_node_types: set[str],
    node_type: dict[str, str],
) -> tuple[set[str], list[str]]:
    """Resolve seed IDs against canonical node IDs and node types."""

    resolved = set()
    missing = []

    for raw_id in raw_ids:
        found = None

        for variant in add_id_variants(raw_id):
            if variant in canonical_ids:
                if node_type.get(variant) in allowed_node_types:
                    found = variant
                    break

        if found is None:
            missing.append(raw_id)
        else:
            resolved.add(found)

    return resolved, missing


# ------------------------------------------------------------
# Load graph and seeds
# ------------------------------------------------------------

def load_data():

    print("Loading nodes...")
    nodes = pl.read_parquet(NODES_FILE)

    print("Loading edges...")
    edges = pl.read_parquet(EDGES_FILE)

    required_node_columns = {"id", "label"}
    required_edge_columns = {"from", "to", "label"}

    if not required_node_columns.issubset(set(nodes.columns)):
        raise ValueError(
            f"nodes.parquet needs columns {required_node_columns}. "
            f"Found: {nodes.columns}"
        )

    if not required_edge_columns.issubset(set(edges.columns)):
        raise ValueError(
            f"edges.parquet needs columns {required_edge_columns}. "
            f"Found: {edges.columns}"
        )

    nodes = nodes.select(
        pl.col("id").cast(pl.String),
        pl.col("label").cast(pl.String),
    )

    edges = edges.select(
        pl.col("from").cast(pl.String),
        pl.col("to").cast(pl.String),
        pl.col("label").cast(pl.String),
    )

    node_type = dict(
        zip(
            nodes["id"].to_list(),
            nodes["label"].to_list(),
        )
    )

    canonical_ids = set(node_type.keys())

    print("Loading seed IDs...")

    bpo_raw = read_ids(BPO_SEED_FILE)
    gen_raw = read_ids(GEN_SEED_FILE)
    dis_raw = read_ids(DIS_SEED_FILE)

    bpo_seeds, bpo_missing = resolve_many(
        bpo_raw,
        canonical_ids,
        {"BPO"},
        node_type,
    )

    gen_seeds, gen_missing = resolve_many(
        gen_raw,
        canonical_ids,
        {"GEN"},
        node_type,
    )

    dis_seeds, dis_missing = resolve_many(
        dis_raw,
        canonical_ids,
        {"DIS"},
        node_type,
    )

    print("\nSeed matching:")
    print(f"  BPO: {len(bpo_seeds)}/{len(bpo_raw)}")
    print(f"  GEN: {len(gen_seeds)}/{len(gen_raw)}")
    print(f"  DIS: {len(dis_seeds)}/{len(dis_raw)}")

    if bpo_missing:
        print(f"  WARNING: {len(bpo_missing)} BPO seeds unmatched")

    if gen_missing:
        print(f"  WARNING: {len(gen_missing)} GEN seeds unmatched")

    if dis_missing:
        print(f"  WARNING: {len(dis_missing)} DIS seeds unmatched")

    if len(dis_seeds) != 79:
        raise ValueError(
            f"Expected 79 matched disease seeds, got {len(dis_seeds)}. "
            "Check seed_diseases_neurodeg.csv and ID resolution."
        )

    if not bpo_seeds or not gen_seeds:
        raise ValueError("BPO or GEN seed set is empty.")

    # Retain only allow-listed relationships.
    edges = edges.filter(
        pl.col("label").is_in(list(ALLOW))
    )

    print(f"\nAllow-listed edges: {edges.height:,}")

    return nodes, edges, node_type, bpo_seeds, gen_seeds, dis_seeds


# ------------------------------------------------------------
# k-hop expansion
# ------------------------------------------------------------

def expand_k_hops(
    seed_nodes: set[str],
    edges: pl.DataFrame,
    node_type: dict[str, str],
    allowed_labels: set[str],
    allowed_types: set[str],
    disease_seeds: set[str],
    k: int,
) -> set[str]:
    """
    Expand from seed nodes for k undirected hops.

    Disease restriction:
        A DIS node can only be included if it belongs to the
        original 79 disease seed IDs.

    This restriction applies even when a disease is encountered
    as a neighbor during expansion.
    """

    selected = set(seed_nodes)
    frontier = set(seed_nodes)

    biological_edges = edges.filter(
        pl.col("label").is_in(list(allowed_labels))
    )

    for hop in range(1, k + 1):

        if not frontier:
            break

        frontier_list = list(frontier)

        # Find edges touching the current frontier.
        touching = biological_edges.filter(
            pl.col("from").is_in(frontier_list)
            | pl.col("to").is_in(frontier_list)
        )

        neighbors = (
            set(touching["from"].to_list())
            | set(touching["to"].to_list())
        )

        new_nodes = neighbors - selected

        # Keep only allowed biological node types.
        new_nodes = {
            node
            for node in new_nodes
            if node_type.get(node) in allowed_types
        }

        # CRITICAL:
        # Never add diseases outside the original 79 seed diseases.
        new_nodes = {
            node
            for node in new_nodes
            if (
                node_type.get(node) != "DIS"
                or node in disease_seeds
            )
        }

        selected.update(new_nodes)
        frontier = new_nodes

        print(
            f"    hop {hop}: "
            f"{len(new_nodes):,} new nodes | "
            f"{len(selected):,} total biological nodes"
        )

    return selected


# ------------------------------------------------------------
# Graph-size calculation
# ------------------------------------------------------------

def summarize_graph(
    config_name: str,
    k: int,
    biological_nodes: set[str],
    nodes: pl.DataFrame,
    edges: pl.DataFrame,
    node_type: dict[str, str],
):

    selected_nodes = set(biological_nodes)

    # Match previous sizing script behavior:
    # add every DRG node in nodes.parquet.
    if ADD_ALL_DRUGS:
        all_drugs = {
            node_id
            for node_id, label in node_type.items()
            if label == "DRG"
        }

        selected_nodes.update(all_drugs)

    selected_list = list(selected_nodes)

    # Keep all allow-listed edges whose two endpoints are selected.
    induced_edges = edges.filter(
        pl.col("from").is_in(selected_list)
        & pl.col("to").is_in(selected_list)
    )

    # Count node types.
    type_counts = Counter(
        node_type[node_id]
        for node_id in selected_nodes
        if node_id in node_type
    )

    # Count indication edges (DRG-DIS).
    indication_edges = induced_edges.filter(
        pl.col("label") == "DRG-DIS"
    ).height

    result = {
        "config": config_name,
        "k": k,
        "nodes": len(selected_nodes),
        "edges": induced_edges.height,
        "drugs": type_counts.get("DRG", 0),
        "genes": type_counts.get("GEN", 0),
        "BPOs": type_counts.get("BPO", 0),
        "diseases": type_counts.get("DIS", 0),
        "indication_edges": indication_edges,
    }

    print("\n  Graph summary:")
    print(f"    Nodes:             {result['nodes']:,}")
    print(f"    Edges:             {result['edges']:,}")
    print(f"    Drugs:             {result['drugs']:,}")
    print(f"    Genes:             {result['genes']:,}")
    print(f"    BPOs:              {result['BPOs']:,}")
    print(f"    Diseases:          {result['diseases']:,}")
    print(f"    Indication edges:  {result['indication_edges']:,}")

    return result


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():

    print("=" * 65)
    print("OptimusKG graph-size comparison")
    print("=" * 65)

    (
        nodes,
        edges,
        node_type,
        bpo_seeds,
        gen_seeds,
        dis_seeds,
    ) = load_data()

    results = []

    configurations = [
        {
            "name": "BPO+GEN",
            "seed_nodes": bpo_seeds | gen_seeds,
            "labels": BPO_GEN_LABELS,
            "types": BPO_GEN_TYPES,
            "ks": [1, 2],
        },
        {
            "name": "BPO+GEN+DIS",
            "seed_nodes": bpo_seeds | gen_seeds | dis_seeds,
            "labels": BPO_GEN_DIS_LABELS,
            "types": BPO_GEN_DIS_TYPES,
            "ks": [1, 2, 3],
        },
    ]

    for config in configurations:

        print("\n" + "=" * 65)
        print(f"Configuration: {config['name']}")
        print("=" * 65)

        for k in config["ks"]:

            print(f"\nRunning k={k}...")

            biological_nodes = expand_k_hops(
                seed_nodes=config["seed_nodes"],
                edges=edges,
                node_type=node_type,
                allowed_labels=config["labels"],
                allowed_types=config["types"],
                disease_seeds=dis_seeds,
                k=k,
            )

            result = summarize_graph(
                config_name=config["name"],
                k=k,
                biological_nodes=biological_nodes,
                nodes=nodes,
                edges=edges,
                node_type=node_type,
            )

            results.append(result)

    # --------------------------------------------------------
    # Final comparison table
    # --------------------------------------------------------

    results_df = pl.DataFrame(results)

    print("\n\n" + "=" * 90)
    print("FINAL GRAPH-SIZE COMPARISON")
    print("=" * 90)

    print(results_df)

    print("\nDisease-seed restriction check:")

    for result in results:
        if result["config"] == "BPO+GEN+DIS":
            print(
                f"  {result['config']} k={result['k']}: "
                f"{result['diseases']} diseases "
                f"(maximum allowed: 79)"
            )

            if result["diseases"] > 79:
                raise RuntimeError(
                    "Disease restriction failed: more than 79 diseases "
                    "were included."
                )

    output_file = ROOT / "pruning_sizes_results.csv"
    results_df.write_csv(output_file)

    print(f"\nSaved results to: {output_file}")
    print("\nDone.")


if __name__ == "__main__":
    main()
