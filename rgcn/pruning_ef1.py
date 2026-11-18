
"""
Standalone neuroinflammation-informed OptimusKG graph pruning.

Builds a pruned graph and indication split for downstream RGCN training.

Inputs:
    nodes.parquet
    edges.parquet
    data/seeds/seed_diseases_neurodeg.csv
    data/seeds/seed_nodes_core.csv
    data/seeds/seed_terms_core.csv

Outputs:
    okg_split.pt
    node_index.parquet
    rel_index.parquet
    split_report.csv

No dependency on prep.py or pruning.py.
"""

import argparse
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import polars as pl
import torch

# The repository root contains okg_files.py.
REPO_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_DIR))

from okg_files import find_table


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

ALLOW = {
    "DRG-DIS",
    "DRG-GEN",
    "DIS-GEN",
    "GEN-GEN",
    "DIS-DIS",
    "DIS-PHE",
    "DRG-PHE",
    "PHE-GEN",
    "BPO-GEN",
    "PWY-GEN",
    "BPO-BPO",
    "PWY-PWY",
    "PHE-PHE",
}

# Large biological relations are retained when they touch a seed.
ANCHOR_LABELS = {
    "DIS-GEN",
    "DIS-PHE",
    "PHE-GEN",
    "BPO-GEN",
    "PWY-GEN",
    "BPO-BPO",
    "PWY-PWY",
    "PHE-PHE",
    "DIS-DIS",
}


# ------------------------------------------------------------
# Seed directory and ID helpers
# ------------------------------------------------------------

def seed_dir_default():
    """Find the default seed directory."""
    here = Path(__file__).resolve().parent

    candidates = [
        here.parent / "data" / "seeds",
        here.parent / "pruning" / "seeds",
        here / "seeds",
        Path("data/seeds"),
        Path("pruning/seeds"),
        Path("keep"),
    ]

    for path in candidates:
        if (path / "seed_diseases_neurodeg.csv").exists():
            return path

    return candidates[0]


def add_id_variants(lookup, raw_id, nid):
    """Register IDs and colon/underscore aliases."""
    lookup[raw_id] = nid

    if ":" in raw_id:
        lookup.setdefault(raw_id.replace(":", "_"), nid)

    if "_" in raw_id:
        lookup.setdefault(raw_id.replace("_", ":"), nid)


def read_ids(path):
    """Read IDs from a seed CSV."""
    if path is None or not Path(path).exists():
        raise SystemExit(f"Missing seed file: {path}")

    frame = pl.read_csv(path)

    if "id" not in frame.columns:
        raise SystemExit(f"{path} has no id column")

    return [
        str(value).strip()
        for value in frame["id"].to_list()
        if value is not None and str(value).strip()
    ]


# ------------------------------------------------------------
# Graph selection helpers
# ------------------------------------------------------------

def topk_per_group(group, neigh, score, k, rng):
    """Keep up to k highest-scoring neighbors per group."""
    if len(group) == 0 or k <= 0:
        return np.empty(0, dtype=np.int64)

    order = np.lexsort(
        (
            rng.random(len(group)),
            -np.asarray(score, dtype=np.float64),
            group,
        )
    )

    grouped = group[order]
    chosen = neigh[order]

    change = np.empty(len(grouped), dtype=bool)
    change[0] = True

    if len(grouped) > 1:
        change[1:] = grouped[1:] != grouped[:-1]

    starts = np.maximum.accumulate(
        np.where(change, np.arange(len(grouped)), 0)
    )

    rank = np.arange(len(grouped)) - starts

    return chosen[rank < k]


def oriented(src, dst, src_t, dst_t, type_a, type_b):
    """Orient edges consistently as type_a -> type_b."""
    forward = (src_t == type_a) & (dst_t == type_b)
    backward = (src_t == type_b) & (dst_t == type_a)

    take = forward | backward

    a = np.where(src_t[take] == type_a, src[take], dst[take])
    b = np.where(src_t[take] == type_a, dst[take], src[take])

    return a, b


def cap_undirected(src, dst, k, rng):
    """Keep edges so each endpoint has at most k neighbors."""
    if len(src) == 0:
        return np.empty(0, dtype=bool)

    edge_id = np.arange(len(src))

    group = np.concatenate([src, dst])
    neigh = np.concatenate([dst, src])
    eid = np.concatenate([edge_id, edge_id])

    score = np.zeros(len(group))

    picked = topk_per_group(group, eid, score, k, rng)

    keep = np.zeros(len(src), dtype=bool)
    keep[np.unique(picked)] = True

    return keep


# ------------------------------------------------------------
# Load graph edges
# ------------------------------------------------------------

def load_edges(edge_path, anchor_ids):
    """Load allow-listed edges and retain seed-anchored relationships."""
    scan = pl.scan_parquet(edge_path)

    try:
        columns = scan.collect_schema().names()
    except AttributeError:
        columns = scan.schema.names()

    wanted = [
        c for c in ("from", "to", "label", "relation", "undirected")
        if c in columns
    ]

    missing = {"from", "to", "label", "relation"} - set(wanted)

    if missing:
        raise SystemExit(
            f"{edge_path} is missing columns: {sorted(missing)}"
        )

    frame = (
        pl.scan_parquet(edge_path)
        .select(wanted)
        .with_columns(
            pl.col("from").cast(pl.Utf8),
            pl.col("to").cast(pl.Utf8),
            pl.col("label").cast(pl.Utf8),
            pl.col("relation").cast(pl.Utf8),
        )
        .filter(pl.col("label").is_in(sorted(ALLOW)))
        .collect()
    )

    if frame.height == 0:
        raise SystemExit(f"No allow-listed edges in {edge_path}")

    if "undirected" not in frame.columns:
        frame = frame.with_columns(
            pl.lit(False).alias("undirected")
        )

    frame = frame.with_columns(
        pl.col("undirected").fill_null(False)
    )

    anchored = (
        ~pl.col("label").is_in(sorted(ANCHOR_LABELS))
        | pl.col("from").is_in(anchor_ids)
        | pl.col("to").is_in(anchor_ids)
    )

    before = frame.height
    frame = frame.filter(anchored)

    print(
        f"Allow-listed edges: {before:,} | "
        f"seed-anchored edges: {frame.height:,}",
        flush=True,
    )

    return frame


# ------------------------------------------------------------
# Select nodes
# ------------------------------------------------------------

def select_nodes(
    src,
    dst,
    src_t,
    dst_t,
    labels,
    codes,
    seeds,
    n_nodes,
    max_genes,
    term_genes,
    max_pheno,
    gene_neighbors,
    rng,
):
    """
    Select neurodegenerative diseases, biological nodes, and drugs.

    All drug nodes are retained independently of their known
    drug-disease associations.
    """

    n = n_nodes
    code = codes

    dis = seeds["dis"]
    gene_seeds = seeds["gene"]
    term_seeds = seeds["term"]
    drug_seeds = seeds["drug"]

    def edges(label, a, b):
        empty = (
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
        )

        if a not in code or b not in code:
            return empty

        take = labels == label

        if not np.any(take):
            return empty

        return oriented(
            src[take],
            dst[take],
            src_t[take],
            dst_t[take],
            code[a],
            code[b],
        )

    # --------------------------------------------------------
    # Disease-associated genes
    # --------------------------------------------------------

    dis_node, gene_node = edges("DIS-GEN", "DIS", "GEN")

    seed_dis_edge = (
        np.isin(dis_node, dis)
        if len(dis_node)
        else np.empty(0, dtype=bool)
    )

    shared = (
        np.bincount(gene_node[seed_dis_edge], minlength=n)
        if seed_dis_edge.any()
        else np.zeros(n, dtype=np.int64)
    )

    # --------------------------------------------------------
    # Genes associated with biological terms
    # --------------------------------------------------------

    term_gene_a = []
    term_gene_b = []

    for label, term_type in (
        ("BPO-GEN", "BPO"),
        ("PWY-GEN", "PWY"),
    ):
        a, b = edges(label, term_type, "GEN")

        if len(a):
            term_gene_a.append(a)
            term_gene_b.append(b)

    if term_gene_a:
        term_ids = np.concatenate(term_gene_a)
        term_genes_arr = np.concatenate(term_gene_b)

        linked = np.zeros(n, dtype=bool)

        hit = np.isin(term_ids, term_seeds)

        if hit.any():
            linked[term_genes_arr[hit]] = True

    else:
        term_ids = np.empty(0, dtype=np.int64)
        term_genes_arr = np.empty(0, dtype=np.int64)
        linked = np.zeros(n, dtype=bool)

    # Score genes using disease overlap, term association, and
    # explicit gene seed membership.
    is_seed_gene = np.zeros(n, dtype=bool)
    is_seed_gene[gene_seeds] = True

    gene_score_table = (
        shared.astype(np.float64) * 10.0
        + linked.astype(np.float64) * 5.0
        + is_seed_gene.astype(np.float64) * 1e6
    )

    genes = set(int(g) for g in gene_seeds)

    if len(dis_node):
        keep_g = topk_per_group(
            dis_node[seed_dis_edge],
            gene_node[seed_dis_edge],
            gene_score_table[gene_node[seed_dis_edge]],
            max_genes,
            rng,
        )

        genes.update(int(g) for g in keep_g)

    if len(term_ids):
        seed_term_edge = np.isin(term_ids, term_seeds)

        extra = topk_per_group(
            term_ids[seed_term_edge],
            term_genes_arr[seed_term_edge],
            gene_score_table[term_genes_arr[seed_term_edge]],
            term_genes,
            rng,
        )

        genes.update(int(g) for g in extra)

    # --------------------------------------------------------
    # All candidate drug nodes
    # --------------------------------------------------------

    drugs = set(int(d) for d in drug_seeds)

    # Do not select drugs based on known DRG-DIS edges.
    # Do not add every drug-associated gene, which would
    # substantially expand the graph beyond the seed neighborhood.

    # --------------------------------------------------------
    # Disease-associated phenotypes
    # --------------------------------------------------------

    phe_d, phe_p = edges("DIS-PHE", "DIS", "PHE")

    phenos = set()

    if len(phe_d):
        hit = np.isin(phe_d, dis)

        shared_phe = (
            np.bincount(phe_p[hit], minlength=n)
            if hit.any()
            else np.zeros(n)
        )

        picked = topk_per_group(
            phe_d[hit],
            phe_p[hit],
            shared_phe[phe_p[hit]],
            max_pheno,
            rng,
        )

        phenos.update(int(p) for p in picked)

    # --------------------------------------------------------
    # Biological terms and their neighbors
    # --------------------------------------------------------

    terms = set(int(t) for t in term_seeds)

    if len(term_ids) and gene_seeds.size:
        hit = np.isin(term_genes_arr, gene_seeds)

        picked = topk_per_group(
            term_genes_arr[hit],
            term_ids[hit],
            np.ones(int(hit.sum())),
            10,
            rng,
        )

        terms.update(int(t) for t in picked)

    for label, term_type in (
        ("BPO-BPO", "BPO"),
        ("PWY-PWY", "PWY"),
        ("PHE-PHE", "PHE"),
    ):
        a, b = edges(label, term_type, term_type)

        if not len(a):
            continue

        anchors = (
            term_seeds
            if term_type != "PHE"
            else np.fromiter(
                phenos,
                dtype=np.int64,
                count=len(phenos),
            )
        )

        if anchors.size == 0:
            continue

        hit = np.isin(a, anchors)

        picked = topk_per_group(
            a[hit],
            b[hit],
            np.ones(int(hit.sum())),
            10,
            rng,
        )

        if term_type == "PHE":
            phenos.update(int(t) for t in picked)
        else:
            terms.update(int(t) for t in picked)

    # --------------------------------------------------------
    # Final node mask
    # --------------------------------------------------------

    mask = np.zeros(n, dtype=bool)

    for group in (dis, genes, drugs, phenos, terms):
        if len(group) == 0:
            continue

        idx = (
            np.fromiter(group, dtype=np.int64, count=len(group))
            if not isinstance(group, np.ndarray)
            else group
        )

        mask[idx] = True

    return mask, gene_neighbors


# ------------------------------------------------------------
# Edge filtering
# ------------------------------------------------------------

def apply_gene_neighbor_cap(src, dst, labels, node_mask, cap, rng):
    """Apply the gene-gene neighbor cap."""
    keep = node_mask[src] & node_mask[dst]

    gene_edge = keep & (labels == "GEN-GEN")

    if cap <= 0 or not gene_edge.any():
        return keep

    idx = np.nonzero(gene_edge)[0]

    chosen = cap_undirected(
        src[idx],
        dst[idx],
        cap,
        rng,
    )

    keep[idx] = chosen

    return keep


# ------------------------------------------------------------
# Indication splitting
# ------------------------------------------------------------

def split_indications(
    drug,
    dis,
    node_fam,
    n_fam,
    val_frac,
    test_frac,
    seed,
):
    """
    Group split indication edges by (drug, disease family).

    0 = train
    1 = validation
    2 = test

    Disease families are kept together to reduce subtype leakage.
    """

    fam = node_fam[dis]

    es = np.zeros(len(drug), dtype=np.int64)

    cand = np.nonzero(fam >= 0)[0]

    if cand.size == 0:
        return es

    g = torch.Generator().manual_seed(seed)

    de = torch.from_numpy(
        np.ascontiguousarray(drug[cand], dtype=np.int64)
    )

    fe = torch.from_numpy(
        np.ascontiguousarray(fam[cand], dtype=np.int64)
    )

    _, inv = torch.unique(
        de * n_fam + fe,
        return_inverse=True,
    )

    gsize = torch.bincount(inv)

    perm = torch.randperm(
        gsize.numel(),
        generator=g,
    )

    cum = torch.cumsum(
        gsize[perm],
        0,
    ).float()

    n_test = test_frac * float(cand.size)
    n_val = val_frac * float(cand.size)

    gs = torch.zeros(gsize.numel(), dtype=torch.long)

    gs[perm[cum <= n_test]] = 2

    gs[
        perm[
            (cum > n_test)
            & (cum <= n_test + n_val)
        ]
    ] = 1

    es[cand] = gs[inv].numpy()

    if not (es[cand] == 1).any() or not (es[cand] == 2).any():
        print(
            "Group split left validation or test empty; "
            "splitting those edges individually.",
            flush=True,
        )

        order = torch.randperm(
            cand.size,
            generator=g,
        ).numpy()

        n_te = max(1, int(round(test_frac * cand.size)))
        n_va = max(1, int(round(val_frac * cand.size)))

        if n_te + n_va >= cand.size:
            n_va = max(1, cand.size // 5)
            n_te = max(1, cand.size // 5)

        es[cand] = 0

        es[cand[order[:n_te]]] = 2

        es[cand[order[n_te:n_te + n_va]]] = 1

    return es


# ------------------------------------------------------------
# Main graph-building function
# ------------------------------------------------------------

def build(args):

    node_path = find_table(args.data_dir, "nodes")
    edge_path = find_table(args.data_dir, "edges")

    print(f"Nodes: {node_path}", flush=True)
    print(f"Edges: {edge_path}", flush=True)

    schema = pl.scan_parquet(node_path).collect_schema().names()

    node_cols = [
        c for c in ("id", "label", "properties")
        if c in schema
    ]

    if "id" not in node_cols or "label" not in node_cols:
        raise SystemExit(
            f"{node_path} must have id and label columns. "
            f"Found: {schema}"
        )

    nodes = (
        pl.read_parquet(node_path, columns=node_cols)
        .with_row_index("nid")
    )

    nodes = nodes.with_columns(
        pl.col("id").cast(pl.Utf8),
        pl.col("label").cast(pl.Utf8),
    )

    # --------------------------------------------------------
    # Build ID lookup
    # --------------------------------------------------------

    lookup = {}

    for nid, raw in zip(
        nodes["nid"].to_list(),
        nodes["id"].to_list(),
    ):
        add_id_variants(lookup, raw, nid)

    # --------------------------------------------------------
    # Load seed files
    # --------------------------------------------------------

    seed_dir = Path(args.seed_dir)

    disease_path = seed_dir / "seed_diseases_neurodeg.csv"

    if not disease_path.exists():
        raise SystemExit(f"Missing {disease_path}")

    neuro_rows = pl.read_csv(disease_path)

    required_columns = {"id", "source"}

    if not required_columns.issubset(set(neuro_rows.columns)):
        raise SystemExit(
            f"{disease_path} must contain columns "
            f"{sorted(required_columns)}"
        )

    neuro_rows = neuro_rows.with_columns(
        pl.col("id").cast(pl.Utf8),
        pl.col("source").cast(pl.Utf8),
    )

    # IMPORTANT: only the neurodegenerative disease file is used.
    # seed_diseases.csv is deliberately excluded.
    disease_ids = read_ids(disease_path)

    gene_ids = read_ids(
        seed_dir / "seed_nodes_core.csv"
    )

    term_ids = read_ids(
        seed_dir / "seed_terms_core.csv"
    )

    print(
        f"Disease entries in seed CSV: {len(set(disease_ids))}",
        flush=True,
    )

    # --------------------------------------------------------
    # Resolve seed IDs
    # --------------------------------------------------------

    def resolve_many(raw_ids):
        found = []
        missing = []

        for raw in raw_ids:
            nid = lookup.get(raw.strip())

            if nid is None:
                missing.append(raw)
            else:
                found.append(nid)

        return (
            np.unique(np.asarray(found, dtype=np.int64)),
            missing,
        )

    dis_nids, dis_missing = resolve_many(disease_ids)

    gene_nids, gene_missing = resolve_many(gene_ids)

    term_nids, term_missing = resolve_many(term_ids)

    print(
        f"Seeds matched | "
        f"diseases {dis_nids.size}/{len(set(disease_ids))} | "
        f"genes {gene_nids.size}/{len(set(gene_ids))} | "
        f"terms {term_nids.size}/{len(set(term_ids))}",
        flush=True,
    )

    if dis_missing:
        print(
            f"Unmatched disease IDs (first 10): {dis_missing[:10]}",
            flush=True,
        )

    if gene_missing:
        print(
            f"Unmatched gene IDs (first 10): {gene_missing[:10]}",
            flush=True,
        )

    if term_missing:
        print(
            f"Unmatched term IDs (first 10): {term_missing[:10]}",
            flush=True,
        )

    if dis_nids.size == 0:
        raise SystemExit(
            "None of the neurodegenerative disease seeds matched "
            "nodes.parquet. Check ID formats."
        )

    # --------------------------------------------------------
    # Identify every drug node independently
    # --------------------------------------------------------

    drug_nids = np.where(
        nodes["label"].to_numpy() == "DRG"
    )[0].astype(np.int64)

    print(
        f"Candidate drug nodes identified: {drug_nids.size:,}",
        flush=True,
    )

    # --------------------------------------------------------
    # Load anchored edges
    # --------------------------------------------------------

    anchor = [
        nodes["id"][int(i)]
        for i in (
            list(dis_nids)
            + list(gene_nids)
            + list(term_nids)
        )
    ]

    anchor_ids = []

    for raw in anchor:
        anchor_ids.append(raw)

        if ":" in raw:
            anchor_ids.append(raw.replace(":", "_"))

        if "_" in raw:
            anchor_ids.append(raw.replace("_", ":"))

    edges = load_edges(
        edge_path,
        list(set(anchor_ids)),
    )

    print(
        f"Joining {edges.height:,} edges to "
        f"{nodes.height:,} nodes",
        flush=True,
    )

    # --------------------------------------------------------
    # Join edges to node indices
    # --------------------------------------------------------

    indexed = nodes.select("id", "nid", "label")

    joined = (
        edges
        .join(
            indexed.rename({
                "id": "from",
                "nid": "src",
                "label": "src_label",
            }),
            on="from",
            how="inner",
        )
        .join(
            indexed.rename({
                "id": "to",
                "nid": "dst",
                "label": "dst_label",
            }),
            on="to",
            how="inner",
        )
    )

    print(
        f"Edges with both endpoints in nodes.parquet: "
        f"{joined.height:,}/{edges.height:,}",
        flush=True,
    )

    if joined.height == 0:
        raise SystemExit(
            "Every allow-listed edge pointed at an ID "
            "not present in nodes.parquet."
        )

    # --------------------------------------------------------
    # Encode node and relationship types
    # --------------------------------------------------------

    type_names = sorted(
        nodes["label"].unique().to_list()
    )

    type_code = {
        name: i
        for i, name in enumerate(type_names)
    }

    for required in ("DRG", "DIS", "GEN"):
        if required not in type_code:
            raise SystemExit(
                f"Node file has no {required!r} label. "
                f"Available labels: {type_names}"
            )

    src = joined["src"].to_numpy().astype(np.int64)
    dst = joined["dst"].to_numpy().astype(np.int64)

    src_t = np.asarray([
        type_code[x]
        for x in joined["src_label"].to_list()
    ], dtype=np.int64)

    dst_t = np.asarray([
        type_code[x]
        for x in joined["dst_label"].to_list()
    ], dtype=np.int64)

    labels = np.asarray(
        joined["label"].to_list()
    )

    relations = np.asarray(
        joined["relation"].to_list()
    )

    undirected_np = (
        joined["undirected"]
        .to_numpy()
        .astype(bool)
    )

    rtype = np.asarray([
        f"{a}|{b}"
        for a, b in zip(labels.tolist(), relations.tolist())
    ])

    # --------------------------------------------------------
    # Prune graph
    # --------------------------------------------------------

    rng = np.random.default_rng(args.seed)

    n_nodes_full = nodes.height

    if args.no_prune:
        print(
            "Not pruning: keeping every allow-listed edge.",
            flush=True,
        )

        node_mask = np.zeros(
            n_nodes_full,
            dtype=bool,
        )

        node_mask[src] = True
        node_mask[dst] = True

        edge_keep = np.ones(
            len(src),
            dtype=bool,
        )

    else:
        node_mask, _ = select_nodes(
            src,
            dst,
            src_t,
            dst_t,
            labels,
            type_code,
            {
                "dis": dis_nids,
                "gene": gene_nids,
                "term": term_nids,
                "drug": drug_nids,
            },
            n_nodes_full,
            args.max_genes,
            args.term_genes,
            args.max_pheno,
            args.gene_neighbors,
            rng,
        )

        # Every target disease remains a candidate.
        node_mask[dis_nids] = True

        edge_keep = apply_gene_neighbor_cap(
            src,
            dst,
            labels,
            node_mask,
            args.gene_neighbors,
            rng,
        )

        edge_keep &= node_mask[src] & node_mask[dst]

    src = src[edge_keep]
    dst = dst[edge_keep]

    labels = labels[edge_keep]
    rtype = rtype[edge_keep]

    undirected_np = undirected_np[edge_keep]

    src_t = src_t[edge_keep]
    dst_t = dst_t[edge_keep]

    touched = np.zeros(
        n_nodes_full,
        dtype=bool,
    )

    touched[src] = True
    touched[dst] = True

    if not args.no_prune:
        touched[dis_nids] = True

    node_mask = touched

    # --------------------------------------------------------
    # Reindex retained nodes
    # --------------------------------------------------------

    kept_idx = np.nonzero(node_mask)[0]

    new_id = np.full(
        n_nodes_full,
        -1,
        dtype=np.int64,
    )

    new_id[kept_idx] = np.arange(
        kept_idx.size
    )

    src = new_id[src]
    dst = new_id[dst]

    kept_nodes = (
        pl.DataFrame({
            "nid": kept_idx.tolist()
        })
        .with_row_index("new_id")
        .join(
            nodes.select([
                c for c in (
                    "nid",
                    "id",
                    "label",
                    "properties",
                )
                if c in nodes.columns
            ]),
            on="nid",
            how="left",
        )
        .sort("new_id")
    )

    node_ids = kept_nodes["id"].to_list()
    node_labels = kept_nodes["label"].to_list()

    kept_types = sorted(set(node_labels))

    kept_code = {
        name: i
        for i, name in enumerate(kept_types)
    }

    node_type = np.asarray([
        kept_code[x]
        for x in node_labels
    ], dtype=np.int64)

    # --------------------------------------------------------
    # Reindex relationships
    # --------------------------------------------------------

    rel_names = sorted(set(rtype.tolist()))

    rel_code = {
        name: i
        for i, name in enumerate(rel_names)
    }

    rid = np.asarray([
        rel_code[x]
        for x in rtype.tolist()
    ], dtype=np.int64)

    if len(src) == 0:
        raise SystemExit(
            "Pruning removed every edge. Check seed IDs and "
            "the allow-listed relationships."
        )

    order = np.lexsort((rid, dst, src))

    src = src[order]
    dst = dst[order]
    rid = rid[order]

    undirected_np = undirected_np[order]

    # Remove duplicate directed triples.
    change = np.empty(len(src), dtype=bool)

    change[0] = True

    if len(src) > 1:
        change[1:] = (
            (src[1:] != src[:-1])
            | (dst[1:] != dst[:-1])
            | (rid[1:] != rid[:-1])
        )

    src = src[change]
    dst = dst[change]
    rid = rid[change]

    undirected_np = undirected_np[change]

    # --------------------------------------------------------
    # Identify indication relation
    # --------------------------------------------------------

    ind_name = "DRG-DIS|INDICATION"

    if ind_name not in rel_code:
        close = [
            name for name in rel_names
            if (
                "INDICATION" in name.upper()
                and "DRG" in name.upper()
                and "DIS" in name.upper()
            )
        ]

        if not close:
            raise SystemExit(
                "No DRG-DIS|INDICATION edges in the pruned graph. "
                f"Available relations: {rel_names}"
            )

        ind_name = close[0]

        print(
            f"Using indication relation {ind_name}",
            flush=True,
        )

    ind_rid = rel_code[ind_name]

    # --------------------------------------------------------
    # Disease family mapping
    # --------------------------------------------------------

    fam = {}

    for row in neuro_rows.iter_rows(named=True):
        if str(row["source"]).strip() == "root":
            fam[str(row["id"]).strip()] = str(row["id"]).strip()

    for row in neuro_rows.iter_rows(named=True):
        raw = str(row["id"]).strip()

        if raw in fam:
            continue

        match = re.match(
            r"descendant of (\S+)",
            str(row["source"]).strip(),
        )

        fam[raw] = match.group(1) if match else raw

    fam_ids = sorted(set(fam.values()))

    fam_index = {
        name: i
        for i, name in enumerate(fam_ids)
    }

    id_to_new = {
        raw: i
        for i, raw in enumerate(node_ids)
    }

    for raw, i in list(id_to_new.items()):
        if ":" in raw:
            id_to_new.setdefault(
                raw.replace(":", "_"),
                i,
            )

        if "_" in raw:
            id_to_new.setdefault(
                raw.replace("_", ":"),
                i,
            )

    node_fam = np.full(
        kept_idx.size,
        -1,
        dtype=np.int64,
    )

    for raw, root in fam.items():
        nid = id_to_new.get(raw)

        if nid is not None:
            node_fam[nid] = fam_index[root]

    # --------------------------------------------------------
    # Orient indication edges drug -> disease for splitting
    # --------------------------------------------------------

    drug_t = kept_code["DRG"]
    dis_t = kept_code["DIS"]

    is_ind = rid == ind_rid

    st = node_type[src]
    dt = node_type[dst]

    drug_src = (st == drug_t) & (dt == dis_t)
    drug_dst = (dt == drug_t) & (st == dis_t)

    orient = is_ind & (drug_src | drug_dst)

    if int(is_ind.sum()) != int(orient.sum()):
        print(
            f"Dropped {int(is_ind.sum() - orient.sum())} "
            "indication edges whose endpoints are not DRG and DIS.",
            flush=True,
        )

    drug = np.where(
        drug_src[orient],
        src[orient],
        dst[orient],
    )

    dis = np.where(
        drug_src[orient],
        dst[orient],
        src[orient],
    )

    es_ind = split_indications(
        drug,
        dis,
        node_fam,
        max(len(fam_ids), 1),
        args.val_frac,
        args.test_frac,
        args.seed,
    )

    ind_idx = np.nonzero(orient)[0]

    es_full = np.zeros(
        len(src),
        dtype=np.int64,
    )

    es_full[ind_idx] = es_ind

    # --------------------------------------------------------
    # Hold out a small general indication test slice
    # --------------------------------------------------------

    g = torch.Generator().manual_seed(args.seed + 1)

    other_ind = ind_idx[node_fam[dis] < 0]

    if other_ind.size and args.gen_frac > 0:
        perm = torch.randperm(
            other_ind.size,
            generator=g,
        ).numpy()

        n_gen = int(
            args.gen_frac * other_ind.size
        )

        gen_e = other_ind[perm[:n_gen]]

    else:
        gen_e = np.empty(
            0,
            dtype=np.int64,
        )

    val_e = np.nonzero(es_full == 1)[0]
    test_e = np.nonzero(es_full == 2)[0]

    held = (
        np.concatenate([val_e, test_e, gen_e])
        if gen_e.size
        else np.concatenate([val_e, test_e])
    )

    # --------------------------------------------------------
    # Leakage guard
    # --------------------------------------------------------

    n = kept_idx.size

    pair = (
        np.minimum(src, dst).astype(np.int64) * n
        + np.maximum(src, dst).astype(np.int64)
    )

    held_pairs = (
        set(pair[held].tolist())
        if held.size
        else set()
    )

    drop = (
        np.fromiter(
            (p in held_pairs for p in pair.tolist()),
            dtype=bool,
            count=len(pair),
        )
        if held_pairs
        else np.zeros(len(pair), dtype=bool)
    )

    drop[held] = False

    print(
        f"Leak guard removed {int(drop.sum())} "
        "edges sharing a node pair with a held-out indication.",
        flush=True,
    )

    train = np.nonzero(
        ~np.isin(np.arange(len(src)), held)
        & ~drop
    )[0]

    if orient[train].any() and held.size:
        assert not bool(
            np.isin(
                pair[train][orient[train]],
                list(held_pairs),
            ).any()
        ), "Train indication shares a held-out pair."

    # --------------------------------------------------------
    # Shared-gene diagnostic
    # --------------------------------------------------------

    gene_t = kept_code["GEN"]

    drug_genes = defaultdict(set)
    dis_genes = defaultdict(set)

    for s, d, name in zip(
        src.tolist(),
        dst.tolist(),
        (rel_names[i] for i in rid.tolist()),
    ):
        if name.startswith("DRG-GEN"):
            a, b = (
                (s, d)
                if (
                    node_type[s] == drug_t
                    and node_type[d] == gene_t
                )
                else (d, s)
            )

            if (
                node_type[a] == drug_t
                and node_type[b] == gene_t
            ):
                drug_genes[a].add(b)

        elif name.startswith("DIS-GEN"):
            a, b = (
                (s, d)
                if (
                    node_type[s] == dis_t
                    and node_type[d] == gene_t
                )
                else (d, s)
            )

            if (
                node_type[a] == dis_t
                and node_type[b] == gene_t
            ):
                dis_genes[a].add(b)

    neuro_local = node_fam[dis] >= 0
    neuro_ind = ind_idx[neuro_local]

    shared_hit = 0

    for dr, di in zip(
        drug[neuro_local].tolist(),
        dis[neuro_local].tolist(),
    ):
        if drug_genes[int(dr)] & dis_genes[int(di)]:
            shared_hit += 1

    frac = shared_hit / max(1, neuro_ind.size)

    print(
        f"Neurodegenerative indications whose drug and disease "
        f"share a gene: {shared_hit}/{neuro_ind.size} ({frac:.1%})",
        flush=True,
    )

    if neuro_ind.size and frac < 0.3:
        print(
            "Warning: few shared genes. Consider raising --max_genes.",
            flush=True,
        )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    by_type = Counter(node_labels)
    by_label = Counter(labels.tolist())

    print(
        f"Pruned graph | nodes {n:,} {dict(by_type)}",
        flush=True,
    )

    print(
        f"Pruned graph | edges {len(src):,} {dict(by_label)}",
        flush=True,
    )

    # Report the retained drug-disease relations separately.
    print("\nDrug-disease relationship counts:", flush=True)

    for relation in sorted(rel_names):
        if relation.startswith("DRG-DIS|"):
            count = int((rid == rel_code[relation]).sum())

            print(
                f"  {relation}: {count:,}",
                flush=True,
            )

    print(
        {
            "train": int(train.size),
            "val": int(val_e.size),
            "test": int(test_e.size),
            "test_gen": int(gen_e.size),
            "train_indications": int(orient[train].sum()),
        },
        flush=True,
    )

    if val_e.size == 0 or test_e.size == 0:
        raise SystemExit(
            "Validation or test has no neurodegenerative "
            "indication edges."
        )

    # --------------------------------------------------------
    # Identify neuro-related nodes
    # --------------------------------------------------------

    is_neuro = np.zeros(
        n,
        dtype=bool,
    )

    if "properties" in kept_nodes.columns:
        try:
            prop = (
                kept_nodes["properties"]
                .fill_null("")
                .cast(pl.Utf8)
            )

            is_neuro = (
                prop.str.contains(
                    "EFO_0000618|EFO:0000618"
                )
                .fill_null(False)
                .to_numpy()
            )

        except Exception as exc:
            print(
                f"Could not read nervous-system tag: {exc}",
                flush=True,
            )

    for raw, root in fam.items():
        nid = id_to_new.get(raw)

        if nid is not None:
            is_neuro[nid] = True

    # --------------------------------------------------------
    # Write split report
    # --------------------------------------------------------

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    neuro_ids = set(
        neuro_rows["id"].cast(pl.Utf8).to_list()
    )

    fam_name = {}

    for root, i in fam_index.items():
        if root in neuro_ids:
            name = neuro_rows.filter(
                pl.col("id") == root
            )["name"][0]
        else:
            name = root

        fam_name[i] = name

    report_rows = []

    for k, root in enumerate(fam_ids):
        ee = (
            es_ind[neuro_local][
                node_fam[dis[neuro_local]] == k
            ]
            if neuro_local.any()
            else np.empty(0, dtype=np.int64)
        )

        report_rows.append({
            "family": root,
            "name": fam_name[k],
            "total": int(ee.size),
            "train": int((ee == 0).sum()) if ee.size else 0,
            "val": int((ee == 1).sum()) if ee.size else 0,
            "test": int((ee == 2).sum()) if ee.size else 0,
        })

    report = (
        pl.DataFrame(report_rows)
        .sort("total", descending=True)
    )

    pl.Config.set_tbl_rows(50)

    print(report, flush=True)

    report.write_csv(
        out_dir / "split_report.csv"
    )

    # --------------------------------------------------------
    # Save RGCN-compatible payload
    # --------------------------------------------------------

    payload = {
        "src": torch.from_numpy(
            src.astype(np.int64)
        ),

        "dst": torch.from_numpy(
            dst.astype(np.int64)
        ),

        "rid": torch.from_numpy(
            rid.astype(np.int64)
        ),

        "undirected": torch.from_numpy(
            undirected_np.astype(bool)
        ),

        "node_type": torch.from_numpy(
            node_type.astype(np.int64)
        ),

        "type_names": kept_types,
        "rel_names": rel_names,
        "node_ids": node_ids,

        "split": {
            "train": torch.from_numpy(
                train.astype(np.int64)
            ),

            "val": torch.from_numpy(
                val_e.astype(np.int64)
            ),

            "test": torch.from_numpy(
                test_e.astype(np.int64)
            ),

            "test_gen": torch.from_numpy(
                gen_e.astype(np.int64)
            ),
        },

        "neuro_fam": torch.from_numpy(
            node_fam.astype(np.int64)
        ),

        "is_neuro": torch.from_numpy(
            is_neuro.astype(bool)
        ),

        "num_nodes": int(n),
        "num_rels": len(rel_names),
        "num_ntypes": len(kept_types),
        "ind_rid": int(ind_rid),
        "pruned": not args.no_prune,
        "shared_gene_frac": float(frac),
    }

    out_pt = out_dir / "okg_split.pt"

    torch.save(payload, out_pt)

    pl.DataFrame({
        "id": node_ids,
        "nid": list(range(n)),
        "label": node_labels,
    }).write_parquet(
        out_dir / "node_index.parquet"
    )

    pl.DataFrame({
        "rtype": rel_names,
        "rid": list(range(len(rel_names))),
    }).write_parquet(
        out_dir / "rel_index.parquet"
    )

    print(
        f"Saved {out_pt} | {n:,} nodes, "
        f"{len(rel_names)} relations",
        flush=True,
    )

    return payload


# ------------------------------------------------------------
# Command-line arguments
# ------------------------------------------------------------

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Prune OptimusKG for neuroinflammation-informed "
            "neurodegenerative drug repurposing."
        )
    )

    parser.add_argument(
        "--data_dir",
        default=".",
        help="Directory containing nodes.parquet and edges.parquet.",
    )

    parser.add_argument(
        "--seed_dir",
        default=str(seed_dir_default()),
        help="Directory containing the three seed CSV files.",
    )

    parser.add_argument(
        "--out_dir",
        default="artifacts",
        help="Output directory.",
    )

    parser.add_argument(
        "--max_genes",
        type=int,
        default=400,
        help="Genes retained per seed disease.",
    )

    parser.add_argument(
        "--term_genes",
        type=int,
        default=150,
        help="Genes retained per seed pathway or process.",
    )

    parser.add_argument(
        "--max_pheno",
        type=int,
        default=40,
        help="Phenotypes retained per seed disease.",
    )

    parser.add_argument(
        "--gene_neighbors",
        type=int,
        default=40,
        help="Gene-gene neighbors retained per gene.",
    )

    parser.add_argument(
        "--val_frac",
        type=float,
        default=0.15,
    )

    parser.add_argument(
        "--test_frac",
        type=float,
        default=0.25,
    )

    parser.add_argument(
        "--gen_frac",
        type=float,
        default=0.02,
        help="Fraction of non-neurodegenerative indications held out as test_gen.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--no_prune",
        action="store_true",
        help="Keep every allow-listed edge.",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing output split.",
    )

    return parser.parse_args(argv)


# ------------------------------------------------------------
# Entry point
# ------------------------------------------------------------

def main(argv=None):
    args = parse_args(argv)

    out_pt = Path(args.out_dir) / "okg_split.pt"

    if out_pt.exists() and not args.force:
        raise SystemExit(
            f"{out_pt} already exists. "
            "Pass --force to rebuild it."
        )

    build(args)


if __name__ == "__main__":
    main()
